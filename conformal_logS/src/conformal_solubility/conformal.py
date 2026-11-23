"""The Random Forest model and the three interval methods compared here.

All three produce a 90% interval around the same point prediction, so any
difference in coverage comes from the calibration step and nothing else.

uncalibrated_gaussian (baseline for H1)
    Treat the spread of the forest's trees as a standard deviation and take
    +/- z * sigma. This is the common practice the project argues against: it
    assumes the errors are Gaussian and that tree disagreement is calibrated,
    and neither holds, so nothing guarantees 90% coverage.

split_conformal
    Hold out a calibration set the model never saw, take the empirical
    (1 - alpha) quantile of absolute residuals with the finite-sample
    correction, and use that one width everywhere. Under exchangeability this
    has marginal coverage of at least 1 - alpha whatever the model does.

normalized_conformal
    Same guarantee, but the non-conformity score is divided by a per-molecule
    difficulty estimate (the forest's tree spread), so hard molecules get wide
    intervals and easy ones narrow. Marginal coverage is unchanged; what
    improves is *conditional* coverage -- which is exactly what H3 asks about.

normalized_conformal_qrf
    The same idea with a better scale: the forest's conditional inter-quantile
    range instead of its tree spread. Tree spread measures how much the trees
    disagree about the mean, which is an epistemic quantity; it understates
    the irreducible noise, so it under-widens exactly where the data are noisy
    rather than unfamiliar.

cqr
    Conformalized quantile regression (Romano et al., 2019). Rather than
    scaling one width, it calibrates the forest's own conditional quantiles,
    so the interval can sit off-centre from the point prediction when the
    conditional distribution is skewed.

mondrian_conformal
    Group-conditional (Mondrian) conformal prediction. Calibration molecules
    are grouped by how structurally familiar they are, and each group gets its
    own calibrated width. The guarantee becomes conditional on the group:
    unfamiliar molecules are no longer handed a width that was averaged over
    the familiar ones. This is the method the README names as the natural next
    step, and it targets the H3 failure directly.

mondrian_normalized
    Both adaptivity mechanisms at once: a per-group calibrated quantile
    applied to a per-molecule normalized score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import Config
from .novelty import assign_similarity_bins, similarity_bin_edges
from .quantile_forest import QuantileForestReadout

LOGGER = logging.getLogger(__name__)

#: Methods that need each molecule's similarity to the training set.
NOVELTY_AWARE_METHODS = frozenset({"mondrian_conformal", "mondrian_normalized"})

#: Methods that read conditional quantiles out of the fitted forest.
QUANTILE_FOREST_METHODS = frozenset({"cqr", "normalized_conformal_qrf"})


def make_model(cfg: Config, seed: int) -> Pipeline:
    """Median imputation followed by a Random Forest.

    Imputation is inside the pipeline so the median is learned from training
    rows only -- computing it over the whole dataset would leak test
    information into the model.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "forest",
                RandomForestRegressor(
                    n_estimators=cfg.n_estimators,
                    max_features=cfg.max_features,
                    min_samples_leaf=cfg.min_samples_leaf,
                    n_jobs=cfg.n_jobs,
                    random_state=seed,
                ),
            ),
        ]
    )


def tree_predictions(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Per-tree predictions, shape ``(n_trees, n_samples)``.

    The features are pushed through the pipeline's pre-processing steps first,
    then handed to each tree directly, because the ensemble's own ``predict``
    only returns the mean.
    """
    transformed = X
    for _, step in model.steps[:-1]:
        transformed = step.transform(transformed)
    forest = model.steps[-1][1]
    return np.stack([tree.predict(transformed) for tree in forest.estimators_])


def difficulty(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Per-molecule difficulty estimate: the standard deviation of the trees.

    Where the trees disagree, the model is extrapolating -- which is the
    behaviour we expect on unfamiliar scaffolds.
    """
    return tree_predictions(model, X).std(axis=0)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal quantile with the finite-sample correction.

    Returns the ``k``-th smallest score for ``k = ceil((n + 1)(1 - alpha))``.
    That "+1" is what turns an empirical quantile into a guarantee of at least
    ``1 - alpha`` coverage on a new exchangeable point, and the rank is taken
    directly on the sorted scores rather than through ``np.quantile``, whose
    interpolation convention would land on a different (slightly too small)
    value and quietly erode the guarantee.

    With a calibration set too small for the requested alpha, ``k`` exceeds
    ``n`` and no finite width can be certified, so this raises rather than
    silently returning the largest residual.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("cannot calibrate on an empty calibration set")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        raise ValueError(
            f"calibration set of {n} points is too small for alpha={alpha}; "
            f"need at least {int(np.ceil(1.0 / alpha)) - 1} points"
        )
    return float(scores[k - 1])


@dataclass
class IntervalResult:
    """A point prediction and its interval, plus the width scale used."""

    prediction: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray  # per-molecule half-width, for diagnostics


class UncalibratedGaussianInterval:
    """+/- z * (tree standard deviation). No calibration set is used."""

    name = "uncalibrated_gaussian"

    def __init__(self, model: Pipeline, alpha: float) -> None:
        self.model = model
        self.alpha = alpha
        self.z = float(norm.ppf(1.0 - alpha / 2.0))

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, similarity: np.ndarray | None = None
    ) -> "UncalibratedGaussianInterval":
        # Deliberately ignores the calibration data: that is the point of the
        # baseline. Accepting the arguments keeps the interface uniform.
        return self

    def predict(self, X: np.ndarray, similarity: np.ndarray | None = None) -> IntervalResult:
        prediction = self.model.predict(X)
        half_width = self.z * difficulty(self.model, X)
        return IntervalResult(
            prediction=prediction,
            lower=prediction - half_width,
            upper=prediction + half_width,
            scale=half_width,
        )


class SplitConformalInterval:
    """One calibrated width for every molecule."""

    name = "split_conformal"

    def __init__(self, model: Pipeline, alpha: float) -> None:
        self.model = model
        self.alpha = alpha
        self.qhat: float | None = None

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, similarity: np.ndarray | None = None
    ) -> "SplitConformalInterval":
        residuals = np.abs(np.asarray(y_cal, dtype=float) - self.model.predict(X_cal))
        self.qhat = conformal_quantile(residuals, self.alpha)
        return self

    def predict(self, X: np.ndarray, similarity: np.ndarray | None = None) -> IntervalResult:
        if self.qhat is None:
            raise RuntimeError("calibrate() must be called before predict()")
        prediction = self.model.predict(X)
        half_width = np.full(len(prediction), self.qhat)
        return IntervalResult(
            prediction=prediction,
            lower=prediction - half_width,
            upper=prediction + half_width,
            scale=half_width,
        )


class NormalizedConformalInterval:
    """Calibrated width scaled per molecule by the forest's tree spread."""

    name = "normalized_conformal"

    def __init__(self, model: Pipeline, alpha: float, beta: float = 0.10) -> None:
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.qhat: float | None = None

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        return difficulty(self.model, X) + self.beta

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, similarity: np.ndarray | None = None
    ) -> "NormalizedConformalInterval":
        residuals = np.abs(np.asarray(y_cal, dtype=float) - self.model.predict(X_cal))
        self.qhat = conformal_quantile(residuals / self._sigma(X_cal), self.alpha)
        return self

    def predict(self, X: np.ndarray, similarity: np.ndarray | None = None) -> IntervalResult:
        if self.qhat is None:
            raise RuntimeError("calibrate() must be called before predict()")
        prediction = self.model.predict(X)
        half_width = self.qhat * self._sigma(X)
        return IntervalResult(
            prediction=prediction,
            lower=prediction - half_width,
            upper=prediction + half_width,
            scale=half_width,
        )


class NormalizedQuantileForestInterval:
    """Normalized conformal with a quantile-forest scale.

    Identical in form to :class:`NormalizedConformalInterval` -- same score,
    same guarantee -- but the per-molecule difficulty is the forest's
    conditional inter-quantile range rather than the spread of its trees.

    The distinction matters for H3. Tree spread is *epistemic*: it is large
    where the trees extrapolate differently, and it collapses toward zero on
    molecules the forest is uniformly confident about, even if the measurement
    itself is noisy. The inter-quantile range carries the aleatoric term too,
    so the resulting scale is a width in LogS units rather than an unscaled
    disagreement statistic.
    """

    name = "normalized_conformal_qrf"

    def __init__(
        self,
        model: Pipeline,
        alpha: float,
        readout: QuantileForestReadout,
        beta: float = 0.10,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.readout = readout
        self.beta = beta
        self.qhat: float | None = None

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        # Half the inter-quantile range, so the scale is comparable in
        # magnitude to a standard deviation and beta keeps its meaning.
        return self.readout.interval_width(X, self.alpha) / 2.0 + self.beta

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, similarity: np.ndarray | None = None
    ) -> "NormalizedQuantileForestInterval":
        residuals = np.abs(np.asarray(y_cal, dtype=float) - self.model.predict(X_cal))
        self.qhat = conformal_quantile(residuals / self._sigma(X_cal), self.alpha)
        return self

    def predict(self, X: np.ndarray, similarity: np.ndarray | None = None) -> IntervalResult:
        if self.qhat is None:
            raise RuntimeError("calibrate() must be called before predict()")
        prediction = self.model.predict(X)
        half_width = self.qhat * self._sigma(X)
        return IntervalResult(
            prediction=prediction,
            lower=prediction - half_width,
            upper=prediction + half_width,
            scale=half_width,
        )


class ConformalizedQuantileInterval:
    """Conformalized quantile regression (Romano et al., 2019).

    The forest's own conditional quantiles at ``alpha/2`` and ``1 - alpha/2``
    give a provisional interval. On the calibration set the conformity score

        E = max(q_lo(x) - y,  y - q_hi(x))

    measures how far outside that provisional interval the truth fell --
    negative when it fell comfortably inside. One calibrated quantile of E
    then pads (or, if the forest's quantiles were already too wide, trims)
    every interval by the same amount:

        C(x) = [q_lo(x) - Q,  q_hi(x) + Q]

    Two things distinguish this from the normalized variant. The interval need
    not be symmetric about the point prediction, so a skewed conditional
    distribution is represented honestly; and the shape of the interval comes
    from the model rather than from a scalar scale, so it adapts in a way a
    single multiplier cannot.

    The point prediction reported alongside it is still the forest mean, which
    is the same number every other method here reports -- the base model is
    held fixed, only the interval changes.
    """

    name = "cqr"

    def __init__(
        self, model: Pipeline, alpha: float, readout: QuantileForestReadout
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.readout = readout
        self.qhat: float | None = None

    def _bounds(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pair = self.readout.predict_quantiles(
            X, [self.alpha / 2.0, 1.0 - self.alpha / 2.0]
        )
        return pair[:, 0], pair[:, 1]

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, similarity: np.ndarray | None = None
    ) -> "ConformalizedQuantileInterval":
        y_cal = np.asarray(y_cal, dtype=float)
        lo, hi = self._bounds(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        self.qhat = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X: np.ndarray, similarity: np.ndarray | None = None) -> IntervalResult:
        if self.qhat is None:
            raise RuntimeError("calibrate() must be called before predict()")
        prediction = self.model.predict(X)
        lo, hi = self._bounds(X)
        lower = lo - self.qhat
        upper = hi + self.qhat

        # A strongly negative Q trims more than the provisional interval is
        # wide. Romano et al. allow it, but an inverted interval is not a
        # prediction set; collapse those to their midpoint and say so.
        inverted = upper < lower
        if np.any(inverted):
            midpoint = (lower[inverted] + upper[inverted]) / 2.0
            lower[inverted] = midpoint
            upper[inverted] = midpoint
            LOGGER.warning(
                "CQR produced %d inverted interval(s); collapsed to a point",
                int(inverted.sum()),
            )
        return IntervalResult(
            prediction=prediction,
            lower=lower,
            upper=upper,
            scale=(upper - lower) / 2.0,
        )


class MondrianConformalInterval:
    """Group-conditional conformal prediction, grouped by structural novelty.

    Split conformal gives one width to every molecule, so its guarantee is
    *marginal*: 90% of the test set is covered on average. The H3 analysis in
    this project shows what that average hides -- on a random split the
    least-familiar quartile is covered 0.837 of the time while the most
    familiar is covered 0.952. Averaging reliability across molecules of very
    different difficulty is precisely the thing a screening campaign cannot
    afford, because the unfamiliar molecules are the ones it is there to find.

    Mondrian (group-conditional) conformal prediction, the taxonomy of Vovk et
    al., fixes a partition of the input space in advance and calibrates each
    cell separately. Here the partition is structural novelty: the maximum
    Tanimoto similarity between a molecule and the training set, cut into
    equal-count bins *on the calibration molecules*. Each bin's width is the
    calibrated quantile of that bin's own residuals, so the guarantee holds
    within each novelty bin rather than only on average over all of them.

    The partition must be fixed without looking at the test labels, which it
    is: the edges come from calibration similarities, and a new molecule is
    assigned to a bin from its own features alone. A molecule less familiar
    than anything in the calibration set falls into bin 0 and receives the
    widest calibrated width, which is the conservative direction.

    What this cannot do is manufacture information that is not there. If the
    calibration molecules are all familiar -- which is exactly the situation
    under the plain ``scaffold`` regime, where calibration is a random slice
    of the training pool -- then every bin is calibrated on familiar molecules
    and no partition of them can describe an unfamiliar one. The diagnostics
    recorded here (``calibration_bin_counts_``, ``test_bin_counts_``,
    ``pooled_fallback_bins_``) are what make that visible rather than letting
    it hide inside a disappointing coverage number.
    """

    name = "mondrian_conformal"

    def __init__(
        self, model: Pipeline, alpha: float, n_bins: int = 4, min_bin: int = 20
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.n_bins = n_bins
        # A bin needs enough points for the finite-sample rank to exist at
        # all: ceil((n + 1)(1 - alpha)) <= n requires n >= ceil(1/alpha) - 1.
        self.min_bin = max(min_bin, int(np.ceil(1.0 / alpha)) - 1)
        self.edges_: np.ndarray | None = None
        self.qhat_: np.ndarray | None = None
        self.pooled_qhat_: float | None = None
        self.calibration_bin_counts_: np.ndarray | None = None
        self.test_bin_counts_: np.ndarray | None = None
        self.pooled_fallback_bins_: list[int] = []

    def _scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(y, dtype=float) - self.model.predict(X))

    def _half_width(self, X: np.ndarray, bins: np.ndarray) -> np.ndarray:
        return np.asarray(self.qhat_)[bins]

    def _require_similarity(self, similarity: np.ndarray | None, where: str) -> np.ndarray:
        if similarity is None:
            raise ValueError(
                f"{self.name} needs each molecule's similarity to the training "
                f"set at {where}; pass similarity=..."
            )
        return np.asarray(similarity, dtype=float)

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, similarity: np.ndarray | None = None
    ) -> "MondrianConformalInterval":
        similarity = self._require_similarity(similarity, "calibration")
        if len(similarity) != len(y_cal):
            raise ValueError("similarity and y_cal must have the same length")

        scores = self._scores(X_cal, y_cal)
        self.edges_ = similarity_bin_edges(similarity, self.n_bins)
        bins = assign_similarity_bins(similarity, self.edges_)

        # The pooled quantile is both the fallback for an under-filled bin and
        # the thing a bin has to beat for the partition to have been worth it.
        self.pooled_qhat_ = conformal_quantile(scores, self.alpha)

        counts = np.zeros(self.n_bins, dtype=int)
        qhat = np.empty(self.n_bins, dtype=float)
        self.pooled_fallback_bins_ = []
        for b in range(self.n_bins):
            mask = bins == b
            counts[b] = int(mask.sum())
            if counts[b] >= self.min_bin:
                qhat[b] = conformal_quantile(scores[mask], self.alpha)
            else:
                # Too few calibration molecules in this bin to certify a width
                # for it. Borrowing the pooled width keeps marginal coverage
                # roughly intact and is recorded, not silently absorbed.
                qhat[b] = self.pooled_qhat_
                self.pooled_fallback_bins_.append(b)

        self.calibration_bin_counts_ = counts
        self.qhat_ = qhat
        if self.pooled_fallback_bins_:
            LOGGER.info(
                "%s: bins %s had < %d calibration molecules and fell back to "
                "the pooled width",
                self.name,
                self.pooled_fallback_bins_,
                self.min_bin,
            )
        return self

    def predict(self, X: np.ndarray, similarity: np.ndarray | None = None) -> IntervalResult:
        if self.qhat_ is None or self.edges_ is None:
            raise RuntimeError("calibrate() must be called before predict()")
        similarity = self._require_similarity(similarity, "prediction")
        bins = assign_similarity_bins(similarity, self.edges_)
        self.test_bin_counts_ = np.bincount(bins, minlength=self.n_bins)

        prediction = self.model.predict(X)
        half_width = self._half_width(X, bins)
        return IntervalResult(
            prediction=prediction,
            lower=prediction - half_width,
            upper=prediction + half_width,
            scale=half_width,
        )

    def diagnostics(self) -> dict[str, object]:
        """Per-bin calibration counts, widths and fallbacks, for the record."""
        return {
            "edges": None if self.edges_ is None else self.edges_.tolist(),
            "qhat_by_bin": None if self.qhat_ is None else np.asarray(self.qhat_).tolist(),
            "pooled_qhat": self.pooled_qhat_,
            "calibration_bin_counts": (
                None
                if self.calibration_bin_counts_ is None
                else self.calibration_bin_counts_.tolist()
            ),
            "test_bin_counts": (
                None if self.test_bin_counts_ is None else self.test_bin_counts_.tolist()
            ),
            "pooled_fallback_bins": list(self.pooled_fallback_bins_),
        }


class MondrianNormalizedInterval(MondrianConformalInterval):
    """Mondrian bins *and* a per-molecule normalized score.

    The two adaptivity mechanisms address different things and compose: the
    bins handle the part of the difficulty that tracks structural novelty,
    the normalization handles the part that varies molecule to molecule
    within a bin. Each bin calibrates its own quantile of the normalized
    score, and the width is that quantile times the molecule's own scale.
    """

    name = "mondrian_normalized"

    def __init__(
        self,
        model: Pipeline,
        alpha: float,
        n_bins: int = 4,
        min_bin: int = 20,
        beta: float = 0.10,
    ) -> None:
        super().__init__(model, alpha, n_bins=n_bins, min_bin=min_bin)
        self.beta = beta

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        return difficulty(self.model, X) + self.beta

    def _scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return super()._scores(X, y) / self._sigma(X)

    def _half_width(self, X: np.ndarray, bins: np.ndarray) -> np.ndarray:
        return np.asarray(self.qhat_)[bins] * self._sigma(X)


def build_method(
    name: str,
    model: Pipeline,
    cfg: Config,
    *,
    quantile_readout: QuantileForestReadout | None = None,
):
    if name == "uncalibrated_gaussian":
        return UncalibratedGaussianInterval(model, cfg.alpha)
    if name == "split_conformal":
        return SplitConformalInterval(model, cfg.alpha)
    if name == "normalized_conformal":
        return NormalizedConformalInterval(model, cfg.alpha, cfg.normalized_beta)
    if name == "mondrian_conformal":
        return MondrianConformalInterval(
            model, cfg.alpha, n_bins=cfg.mondrian_bins, min_bin=cfg.mondrian_min_bin
        )
    if name == "mondrian_normalized":
        return MondrianNormalizedInterval(
            model,
            cfg.alpha,
            n_bins=cfg.mondrian_bins,
            min_bin=cfg.mondrian_min_bin,
            beta=cfg.normalized_beta,
        )
    if name in QUANTILE_FOREST_METHODS:
        if quantile_readout is None:
            raise ValueError(
                f"{name!r} needs a QuantileForestReadout; pass quantile_readout=..."
            )
        if name == "cqr":
            return ConformalizedQuantileInterval(model, cfg.alpha, quantile_readout)
        return NormalizedQuantileForestInterval(
            model, cfg.alpha, quantile_readout, cfg.normalized_beta
        )
    raise ValueError(f"unknown interval method: {name!r}")
