"""Tests for the quantile-forest read-out.

The read-out is the one piece of this project with a genuinely tricky
implementation: a ragged gather plus a grouped weighted quantile, both
vectorised for speed. Speed is not worth anything if the numbers are wrong,
so the central test here compares it against a slow, obviously-correct
transcription of Meinshausen's formula.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from conformal_solubility.conformal import difficulty, make_model
from conformal_solubility.config import Config
from conformal_solubility.quantile_forest import (
    QuantileForestReadout,
    weighted_quantiles,
)


def synthetic(n: int, seed: int, noise_scale: float = 1.0):
    """Regression data whose noise grows with the first feature."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 4))
    signal = 2.0 * X[:, 0] + X[:, 1] ** 2 - 1.5 * X[:, 2]
    noise = rng.normal(0, noise_scale * (0.2 + 0.6 * np.abs(X[:, 0])))
    return X, signal + noise


def reference_quantiles(model, X_train, y_train, X_query, quantiles):
    """Meinshausen's formula written out directly, one query at a time.

    w_i(x) = (1/T) sum_t 1[leaf_t(x_i) = leaf_t(x)] / |leaf_t(x)|

    then the weighted empirical quantiles of y under those weights. Slow and
    obviously right, which is the point.
    """
    transformed_train = X_train
    transformed_query = X_query
    for _, step in model.steps[:-1]:
        transformed_train = step.transform(transformed_train)
        transformed_query = step.transform(transformed_query)
    forest = model.steps[-1][1]
    trees = forest.estimators_

    train_leaves = np.column_stack([tree.apply(transformed_train) for tree in trees])
    query_leaves = np.column_stack([tree.apply(transformed_query) for tree in trees])

    out = np.empty((len(X_query), len(quantiles)))
    for i in range(len(X_query)):
        weights = np.zeros(len(y_train))
        for t in range(len(trees)):
            members = train_leaves[:, t] == query_leaves[i, t]
            weights[members] += 1.0 / (members.sum() * len(trees))
        out[i] = weighted_quantiles(y_train, weights, quantiles)
    return out


def test_weighted_quantiles_matches_hand_computation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    # Cumulative weight: 0.1, 0.3, 0.6, 1.0
    got = weighted_quantiles(values, weights, [0.05, 0.1, 0.25, 0.6, 0.9, 1.0])
    assert got.tolist() == [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]


def test_weighted_quantiles_with_uniform_weights_is_a_plain_quantile():
    rng = np.random.default_rng(0)
    values = rng.normal(size=201)
    weights = np.ones_like(values)
    got = weighted_quantiles(values, weights, [0.5])
    assert got[0] == pytest.approx(np.median(values))


def test_weighted_quantiles_rejects_degenerate_input():
    with pytest.raises(ValueError, match="empty"):
        weighted_quantiles(np.array([]), np.array([]), [0.5])
    with pytest.raises(ValueError, match="positive"):
        weighted_quantiles(np.array([1.0, 2.0]), np.array([0.0, 0.0]), [0.5])


@pytest.mark.parametrize("seed", [3, 17])
def test_vectorised_readout_matches_the_slow_reference(seed):
    """The test that justifies the vectorised implementation existing.

    Exact equality, not a tolerance: a quantile is an order statistic, so the
    two paths either select the same observation or they do not. Getting this
    to hold needed the cumulative weights to be rounded before comparison --
    see ``_CUMULATIVE_DECIMALS``. Without it the two disagreed on roughly 2%
    of queries, every one of them an exact tie broken differently.
    """
    # Four features only, so max_features must stay at 1.0 here: the project
    # default of 0.3 would offer each split a single feature and build trees
    # too weak for leaf membership to mean much. On the real data 0.3 of 216
    # descriptors is 64 candidate features per split, which is not the same
    # situation at all.
    cfg = Config(n_estimators=12, max_features=1.0)
    X, y = synthetic(240, seed=seed)
    model = make_model(cfg, seed=0)
    model.fit(X[:180], y[:180])

    readout = QuantileForestReadout(model, X[:180], y[:180])
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    got = readout.predict_quantiles(X[180:], levels)
    expected = reference_quantiles(model, X[:180], y[:180], X[180:], levels)
    np.testing.assert_array_equal(got, expected)


def test_quantiles_are_monotone_in_level():
    cfg = Config(n_estimators=30, max_features=1.0)
    X, y = synthetic(400, seed=5)
    model = make_model(cfg, seed=0)
    model.fit(X[:300], y[:300])
    readout = QuantileForestReadout(model, X[:300], y[:300])

    got = readout.predict_quantiles(X[300:], [0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.all(np.diff(got, axis=1) >= 0), "quantiles must not decrease with level"


def test_quantile_width_tracks_heteroscedastic_noise():
    """The conditional width must grow where the noise does.

    The synthetic noise grows with |X0|, so the conditional interval has to be
    wider for large |X0| than for small.
    """
    cfg = Config(n_estimators=200, max_features=1.0)
    X, y = synthetic(2000, seed=7)
    model = make_model(cfg, seed=0)
    model.fit(X[:1400], y[:1400])
    readout = QuantileForestReadout(model, X[:1400], y[:1400])

    X_test = X[1400:]
    width = readout.interval_width(X_test, alpha=0.10)
    quiet = np.abs(X_test[:, 0]) < 1.0
    loud = np.abs(X_test[:, 0]) > 2.0
    assert width[loud].mean() > 1.25 * width[quiet].mean()


def test_quantile_width_beats_tree_spread_as_a_noise_estimate():
    """The claim that justifies ``normalized_conformal_qrf`` existing.

    Tree spread measures disagreement between trees about the *mean*. Grow
    the leaves a little and that disagreement shrinks toward zero even where
    the data are genuinely noisy, because the trees agree -- correctly -- on
    a mean they cannot predict well. The conditional inter-quantile range
    keeps tracking the noise, because it describes the spread of the
    observations rather than the spread of the estimates.

    Measured as rank correlation against the true noise scale, which is known
    here by construction.
    """
    X, y = synthetic(2000, seed=7)
    X_test = X[1400:]
    true_noise_scale = 0.2 + 0.6 * np.abs(X_test[:, 0])

    cfg = Config(n_estimators=200, max_features=1.0, min_samples_leaf=20)
    model = make_model(cfg, seed=0)
    model.fit(X[:1400], y[:1400])
    readout = QuantileForestReadout(model, X[:1400], y[:1400])

    qrf_rank = spearmanr(readout.interval_width(X_test, 0.10), true_noise_scale).statistic
    spread_rank = spearmanr(difficulty(model, X_test), true_noise_scale).statistic

    assert qrf_rank > 0.25, f"quantile width should track the noise, got {qrf_rank:.2f}"
    assert qrf_rank > spread_rank + 0.2, (
        f"quantile width ({qrf_rank:.2f}) should track the noise better than "
        f"tree spread ({spread_rank:.2f})"
    )


def test_readout_rejects_mismatched_training_data():
    cfg = Config(n_estimators=10)
    X, y = synthetic(200, seed=11)
    model = make_model(cfg, seed=0)
    model.fit(X[:150], y[:150])
    with pytest.raises(ValueError, match="number of rows"):
        QuantileForestReadout(model, X[:150], y[:140])


def test_readout_does_not_disturb_the_fitted_model():
    """The read-out must be exactly that: the point predictions cannot move.

    This is what lets CQR and the QRF-normalized variant be compared against
    the other methods at all -- they must all be describing the same forest.
    """
    cfg = Config(n_estimators=40)
    X, y = synthetic(500, seed=13)
    model = make_model(cfg, seed=0)
    model.fit(X[:350], y[:350])

    before = model.predict(X[350:])
    readout = QuantileForestReadout(model, X[:350], y[:350])
    readout.predict_quantiles(X[350:], [0.05, 0.95])
    after = model.predict(X[350:])
    # Not assert_array_equal: the forest predicts with n_jobs=-1, and the
    # order in which threads accumulate the per-tree means is not fixed, so
    # two calls can differ in the last bit or two without anything having
    # changed. The claim being tested is that the readout leaves the fitted
    # model alone, which a 1e-12 tolerance establishes just as well.
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)
