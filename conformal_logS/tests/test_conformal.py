"""Tests for the conformal machinery.

The central claim of split conformal prediction is a coverage guarantee, so
the central test is an empirical check of that guarantee on data where the
right answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_solubility.config import Config
from conformal_solubility.conformal import (
    NormalizedConformalInterval,
    SplitConformalInterval,
    UncalibratedGaussianInterval,
    conformal_quantile,
    difficulty,
    make_model,
    tree_predictions,
)
from conformal_solubility.metrics import picp


def synthetic(n: int, seed: int, noise_scale: float = 1.0):
    """A regression problem with input-dependent noise.

    The noise grows with the first feature, which is the situation the
    normalized variant exists for: one global width must be too wide in the
    quiet region and too narrow in the loud one.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 4))
    signal = 2.0 * X[:, 0] + X[:, 1] ** 2 - 1.5 * X[:, 2]
    noise = rng.normal(0, noise_scale * (0.2 + 0.6 * np.abs(X[:, 0])))
    return X, signal + noise


def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(1.0, 11.0)  # n = 10
    # ceil((10 + 1) * 0.9) / 10 = 1.0 -> the largest score
    assert conformal_quantile(scores, 0.10) == pytest.approx(10.0)
    # ceil((10 + 1) * 0.5) / 10 = 0.6 -> the 6th of 10 scores
    assert conformal_quantile(scores, 0.50) == pytest.approx(6.0)


def test_conformal_quantile_is_monotone_in_alpha():
    scores = np.linspace(0, 5, 50)
    widths = [conformal_quantile(scores, a) for a in (0.30, 0.20, 0.10, 0.05)]
    assert widths == sorted(widths), "smaller alpha must not give a narrower width"


def test_conformal_quantile_rejects_too_small_calibration_set():
    with pytest.raises(ValueError, match="too small"):
        conformal_quantile(np.arange(5.0), 0.10)
    with pytest.raises(ValueError, match="empty"):
        conformal_quantile(np.array([]), 0.10)


def test_tree_predictions_mean_matches_pipeline_predict():
    cfg = Config(n_estimators=50, n_jobs=1)
    X, y = synthetic(200, seed=1)
    model = make_model(cfg, seed=1)
    model.fit(X, y)
    np.testing.assert_allclose(
        tree_predictions(model, X).mean(axis=0), model.predict(X), rtol=1e-10
    )


def test_difficulty_is_non_negative_and_varies():
    cfg = Config(n_estimators=60, n_jobs=1)
    X, y = synthetic(300, seed=2)
    model = make_model(cfg, seed=2)
    model.fit(X[:200], y[:200])
    sigma = difficulty(model, X[200:])
    assert np.all(sigma >= 0)
    assert sigma.std() > 0, "a useful difficulty estimate cannot be constant"


@pytest.mark.parametrize("alpha", [0.10, 0.20])
def test_split_conformal_achieves_nominal_coverage(alpha):
    """Marginal coverage must land at or above 1 - alpha on exchangeable data."""
    cfg = Config(alpha=alpha, n_estimators=100, n_jobs=1)
    X, y = synthetic(2400, seed=3)
    X_train, y_train = X[:800], y[:800]
    X_cal, y_cal = X[800:1600], y[800:1600]
    X_test, y_test = X[1600:], y[1600:]

    model = make_model(cfg, seed=3)
    model.fit(X_train, y_train)
    method = SplitConformalInterval(model, alpha).calibrate(X_cal, y_cal)
    result = method.predict(X_test)

    coverage = picp(y_test, result.lower, result.upper)
    # Guarantee is >= 1 - alpha in expectation; allow finite-sample slack below.
    assert coverage >= (1 - alpha) - 0.04
    assert coverage <= (1 - alpha) + 0.08


def test_normalized_conformal_also_covers_but_with_varying_widths():
    alpha = 0.10
    cfg = Config(alpha=alpha, n_estimators=100, n_jobs=1)
    X, y = synthetic(2400, seed=4)
    model = make_model(cfg, seed=4)
    model.fit(X[:800], y[:800])

    constant = SplitConformalInterval(model, alpha).calibrate(X[800:1600], y[800:1600])
    adaptive = NormalizedConformalInterval(model, alpha, cfg.normalized_beta).calibrate(
        X[800:1600], y[800:1600]
    )

    flat = constant.predict(X[1600:])
    varying = adaptive.predict(X[1600:])

    assert picp(y[1600:], varying.lower, varying.upper) >= (1 - alpha) - 0.04
    assert np.allclose(flat.scale, flat.scale[0]), "split conformal width must be constant"
    assert varying.scale.std() > 0, "normalized width must adapt per molecule"


def test_predict_before_calibrate_raises():
    cfg = Config(n_estimators=20, n_jobs=1)
    X, y = synthetic(120, seed=5)
    model = make_model(cfg, seed=5)
    model.fit(X, y)
    with pytest.raises(RuntimeError, match="calibrate"):
        SplitConformalInterval(model, 0.10).predict(X)
    with pytest.raises(RuntimeError, match="calibrate"):
        NormalizedConformalInterval(model, 0.10).predict(X)


def test_uncalibrated_baseline_misses_the_target_by_more_than_conformal():
    """The baseline exists to be beaten.

    Note the test is direction-agnostic. Tree spread is not a calibrated
    predictive standard deviation, and which way it errs depends on the data:
    on this synthetic problem it is far too wide (over-covering), while on
    other problems the same construction is over-confident. Either way it has
    no reason to land on the nominal level, and that -- not the sign of the
    error -- is the claim H1 makes.
    """
    alpha = 0.10
    cfg = Config(alpha=alpha, n_estimators=100, n_jobs=1)
    X, y = synthetic(2400, seed=6, noise_scale=1.5)
    model = make_model(cfg, seed=6)
    model.fit(X[:800], y[:800])

    baseline = UncalibratedGaussianInterval(model, alpha).calibrate(
        X[800:1600], y[800:1600]
    )
    conformal = SplitConformalInterval(model, alpha).calibrate(X[800:1600], y[800:1600])

    baseline_result = baseline.predict(X[1600:])
    conformal_result = conformal.predict(X[1600:])
    baseline_gap = abs(picp(y[1600:], baseline_result.lower, baseline_result.upper) - (1 - alpha))
    conformal_gap = abs(
        picp(y[1600:], conformal_result.lower, conformal_result.upper) - (1 - alpha)
    )

    assert baseline_gap > 0.02, "an uncalibrated baseline that lands on 90% is suspicious"
    assert conformal_gap < baseline_gap
