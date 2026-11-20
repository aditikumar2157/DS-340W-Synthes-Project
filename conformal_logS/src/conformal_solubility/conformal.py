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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import Config


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

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "UncalibratedGaussianInterval":
        # Deliberately ignores the calibration data: that is the point of the
        # baseline. Accepting the arguments keeps the interface uniform.
        return self

    def predict(self, X: np.ndarray) -> IntervalResult:
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

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalInterval":
        residuals = np.abs(np.asarray(y_cal, dtype=float) - self.model.predict(X_cal))
        self.qhat = conformal_quantile(residuals, self.alpha)
        return self

    def predict(self, X: np.ndarray) -> IntervalResult:
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
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> "NormalizedConformalInterval":
        residuals = np.abs(np.asarray(y_cal, dtype=float) - self.model.predict(X_cal))
        self.qhat = conformal_quantile(residuals / self._sigma(X_cal), self.alpha)
        return self

    def predict(self, X: np.ndarray) -> IntervalResult:
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


def build_method(name: str, model: Pipeline, cfg: Config):
    if name == "uncalibrated_gaussian":
        return UncalibratedGaussianInterval(model, cfg.alpha)
    if name == "split_conformal":
        return SplitConformalInterval(model, cfg.alpha)
    if name == "normalized_conformal":
        return NormalizedConformalInterval(model, cfg.alpha, cfg.normalized_beta)
    raise ValueError(f"unknown interval method: {name!r}")
