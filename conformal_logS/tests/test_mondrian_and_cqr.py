"""Tests for the novelty-conditional and quantile-based interval methods.

The guarantee these methods make is the thing worth testing, and it is
testable: build data where the right answer is known by construction, run the
method many times, and check that coverage comes out where it was promised.

Mondrian conformal prediction promises something stronger than split
conformal -- coverage *within each group*, not just on average -- so the test
for it checks the per-group coverage, which is exactly the number split
conformal is allowed to miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_solubility.config import Config
from conformal_solubility.conformal import (
    ConformalizedQuantileInterval,
    MondrianConformalInterval,
    MondrianNormalizedInterval,
    NormalizedQuantileForestInterval,
    SplitConformalInterval,
    build_method,
    make_model,
)
from conformal_solubility.metrics import picp
from conformal_solubility.novelty import (
    assign_similarity_bins,
    similarity_bin_edges,
)
from conformal_solubility.quantile_forest import QuantileForestReadout


def grouped_problem(n: int, seed: int):
    """Two kinds of molecule, one much noisier than the other.

    ``group`` plays the role structural novelty plays in the real experiment:
    a quantity known per example, correlated with how hard that example is.
    The noise in the hard group is four times the easy group's, so a single
    pooled width must be too wide for one and too narrow for the other --
    the failure Mondrian calibration exists to prevent.

    Returned ``familiarity`` is high for the easy group and low for the hard
    one, mimicking Tanimoto similarity to the training set.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 5))
    hard = rng.random(n) < 0.5
    X[:, 4] = hard.astype(float)  # the model can see which group it is in
    signal = 2.0 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2]
    noise = rng.normal(0.0, np.where(hard, 2.0, 0.5))
    familiarity = np.where(hard, 0.2, 0.8) + rng.normal(0, 0.02, size=n)
    return X, signal + noise, familiarity, hard


def fitted(cfg: Config, X, y, n_train: int):
    model = make_model(cfg, seed=0)
    model.fit(X[:n_train], y[:n_train])
    return model


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_bin_edges_split_into_equal_counts():
    values = np.linspace(0, 1, 400)
    edges = similarity_bin_edges(values, n_bins=4)
    assert len(edges) == 3
    counts = np.bincount(assign_similarity_bins(values, edges), minlength=4)
    assert counts.min() >= 95 and counts.max() <= 105


def test_bins_learned_on_one_set_apply_to_another():
    """The property Mondrian needs: the partition is a rule, not a labelling."""
    calibration = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    edges = similarity_bin_edges(calibration, n_bins=4)
    # A molecule less familiar than anything in calibration belongs to bin 0,
    # which is the bin carrying the widest calibrated interval.
    assert assign_similarity_bins(np.array([0.0]), edges)[0] == 0
    assert assign_similarity_bins(np.array([99.0]), edges)[0] == 3


def test_bin_edges_tolerate_heavy_ties():
    """Random splits leave many molecules at similarity 1.0."""
    values = np.concatenate([np.full(60, 1.0), np.linspace(0.2, 0.9, 20)])
    edges = similarity_bin_edges(values, n_bins=4)
    bins = assign_similarity_bins(values, edges)
    assert bins.min() >= 0 and bins.max() <= 3  # no out-of-range index


# --------------------------------------------------------------------------
# Mondrian conformal
# --------------------------------------------------------------------------


def test_mondrian_achieves_coverage_within_each_group():
    """The claim: group-conditional coverage, not merely marginal.

    Split conformal is run alongside as the contrast. It is allowed to -- and
    does -- hit 90% overall while missing badly inside each group.
    """
    cfg = Config(alpha=0.10, n_estimators=120, max_features=1.0)
    X, y, familiarity, hard = grouped_problem(6000, seed=1)
    model = fitted(cfg, X, y, 2500)

    cal = slice(2500, 4000)
    test = slice(4000, 6000)

    mondrian = MondrianConformalInterval(model, cfg.alpha, n_bins=2, min_bin=25)
    mondrian.calibrate(X[cal], y[cal], similarity=familiarity[cal])
    grouped = mondrian.predict(X[test], similarity=familiarity[test])

    pooled = SplitConformalInterval(model, cfg.alpha).calibrate(X[cal], y[cal])
    flat = pooled.predict(X[test])

    hard_test = hard[test]
    for mask, label in ((hard_test, "hard"), (~hard_test, "easy")):
        grouped_cov = picp(y[test][mask], grouped.lower[mask], grouped.upper[mask])
        assert 0.85 < grouped_cov < 0.95, f"{label} group coverage {grouped_cov:.3f}"

    # And the contrast: the pooled width misses in both directions at once.
    pooled_hard = picp(y[test][hard_test], flat.lower[hard_test], flat.upper[hard_test])
    pooled_easy = picp(
        y[test][~hard_test], flat.lower[~hard_test], flat.upper[~hard_test]
    )
    assert pooled_hard < 0.85, f"expected under-coverage on the hard group, got {pooled_hard:.3f}"
    assert pooled_easy > 0.95, f"expected over-coverage on the easy group, got {pooled_easy:.3f}"


def test_mondrian_gives_the_unfamiliar_group_the_wider_interval():
    cfg = Config(alpha=0.10, n_estimators=100, max_features=1.0)
    X, y, familiarity, hard = grouped_problem(4000, seed=2)
    model = fitted(cfg, X, y, 1800)
    cal, test = slice(1800, 3000), slice(3000, 4000)

    method = MondrianConformalInterval(model, cfg.alpha, n_bins=2, min_bin=25)
    method.calibrate(X[cal], y[cal], similarity=familiarity[cal])
    result = method.predict(X[test], similarity=familiarity[test])

    hard_test = hard[test]
    assert result.scale[hard_test].mean() > result.scale[~hard_test].mean()


def test_mondrian_falls_back_to_the_pooled_width_for_a_thin_bin():
    """An under-filled bin must borrow, not invent, and must say that it did."""
    cfg = Config(alpha=0.10, n_estimators=50, max_features=1.0)
    X, y, familiarity, _ = grouped_problem(1200, seed=3)
    model = fitted(cfg, X, y, 600)

    # 40 calibration molecules over 4 bins is 10 each: below min_bin.
    cal, test = slice(600, 640), slice(640, 1200)
    method = MondrianConformalInterval(model, cfg.alpha, n_bins=4, min_bin=25)
    method.calibrate(X[cal], y[cal], similarity=familiarity[cal])

    assert method.pooled_fallback_bins_, "a thin bin should have been recorded"
    diagnostics = method.diagnostics()
    assert diagnostics["pooled_qhat"] is not None
    assert sum(diagnostics["calibration_bin_counts"]) == 40

    result = method.predict(X[test], similarity=familiarity[test])
    assert np.all(np.isfinite(result.scale))


def test_mondrian_requires_similarity():
    cfg = Config(n_estimators=20, max_features=1.0)
    X, y, _, _ = grouped_problem(400, seed=4)
    model = fitted(cfg, X, y, 200)
    method = MondrianConformalInterval(model, cfg.alpha)
    with pytest.raises(ValueError, match="similarity"):
        method.calibrate(X[200:300], y[200:300])


def test_mondrian_normalized_is_both_binned_and_scaled():
    cfg = Config(alpha=0.10, n_estimators=100, max_features=1.0)
    X, y, familiarity, _ = grouped_problem(3000, seed=5)
    model = fitted(cfg, X, y, 1400)
    cal, test = slice(1400, 2200), slice(2200, 3000)

    method = MondrianNormalizedInterval(model, cfg.alpha, n_bins=2, min_bin=25)
    method.calibrate(X[cal], y[cal], similarity=familiarity[cal])
    result = method.predict(X[test], similarity=familiarity[test])

    # Binned alone gives two distinct widths; binned plus normalized gives a
    # different width to (almost) every molecule.
    assert len(np.unique(np.round(result.scale, 9))) > 100
    coverage = picp(y[test], result.lower, result.upper)
    assert 0.84 < coverage < 0.96


# --------------------------------------------------------------------------
# CQR
# --------------------------------------------------------------------------


def test_cqr_attains_marginal_coverage():
    cfg = Config(alpha=0.10, n_estimators=150, max_features=1.0)
    X, y, _, _ = grouped_problem(6000, seed=6)
    model = fitted(cfg, X, y, 2500)
    readout = QuantileForestReadout(model, X[:2500], y[:2500])
    cal, test = slice(2500, 4000), slice(4000, 6000)

    method = ConformalizedQuantileInterval(model, cfg.alpha, readout)
    method.calibrate(X[cal], y[cal])
    result = method.predict(X[test])
    assert 0.87 < picp(y[test], result.lower, result.upper) < 0.93


def test_cqr_adapts_when_the_difficulty_is_visible_to_the_forest():
    """CQR widens where the noise grows -- provided the trees split there.

    Here the noise scales with |X0| and X0 also drives the mean, so the
    partition the forest built to predict the mean happens to separate quiet
    regions from loud ones, and the leaf-conditional quantiles inherit that.
    """
    rng = np.random.default_rng(21)
    n = 6000
    X = rng.uniform(-3, 3, size=(n, 4))
    y = 2.0 * X[:, 0] + X[:, 1] - rng.normal(0.0, 0.2 + 0.8 * np.abs(X[:, 0]))

    cfg = Config(alpha=0.10, n_estimators=150, max_features=1.0)
    model = fitted(cfg, X, y, 2500)
    readout = QuantileForestReadout(model, X[:2500], y[:2500])
    cal, test = slice(2500, 4000), slice(4000, n)

    method = ConformalizedQuantileInterval(model, cfg.alpha, readout)
    method.calibrate(X[cal], y[cal])
    result = method.predict(X[test])

    width = result.upper - result.lower
    loud = np.abs(X[test][:, 0]) > 2.0
    quiet = np.abs(X[test][:, 0]) < 1.0
    assert width[loud].mean() > 1.5 * width[quiet].mean()


def test_cqr_cannot_see_difficulty_the_forest_had_no_reason_to_split_on():
    """The limitation that makes the Mondrian variant worth having.

    A quantile forest reads its conditional distribution out of a partition
    that was built to predict the *mean*. In ``grouped_problem`` the noisy and
    quiet groups have the same mean, so splitting on the group label reduces
    no squared error and the trees never do it. The leaves therefore mix both
    groups and CQR hands them near-identical widths, however well calibrated
    it is marginally.

    Mondrian calibration does not have this blind spot, because its partition
    is supplied from outside the model rather than discovered by it. This is
    exactly the situation of the real experiment: structural novelty does not
    shift a molecule's solubility, it changes how wrong the model is about it,
    so a mean-fitting forest has no reason to isolate it.
    """
    cfg = Config(alpha=0.10, n_estimators=150, max_features=1.0)
    X, y, familiarity, hard = grouped_problem(5000, seed=7)
    model = fitted(cfg, X, y, 2200)
    readout = QuantileForestReadout(model, X[:2200], y[:2200])
    cal, test = slice(2200, 3600), slice(3600, 5000)
    hard_test = hard[test]

    cqr = ConformalizedQuantileInterval(model, cfg.alpha, readout)
    cqr.calibrate(X[cal], y[cal])
    cqr_width = cqr.predict(X[test]).scale * 2.0

    mondrian = MondrianConformalInterval(model, cfg.alpha, n_bins=2, min_bin=25)
    mondrian.calibrate(X[cal], y[cal], similarity=familiarity[cal])
    mondrian_width = mondrian.predict(X[test], similarity=familiarity[test]).scale * 2.0

    cqr_ratio = cqr_width[hard_test].mean() / cqr_width[~hard_test].mean()
    mondrian_ratio = (
        mondrian_width[hard_test].mean() / mondrian_width[~hard_test].mean()
    )
    assert cqr_ratio < 1.3, f"CQR unexpectedly separated the groups ({cqr_ratio:.2f})"
    assert mondrian_ratio > 2.0, f"Mondrian should separate them ({mondrian_ratio:.2f})"


def test_cqr_reports_the_forest_mean_as_its_point_prediction():
    """CQR must not quietly change what the project calls the prediction."""
    cfg = Config(alpha=0.10, n_estimators=60, max_features=1.0)
    X, y, _, _ = grouped_problem(1500, seed=8)
    model = fitted(cfg, X, y, 700)
    readout = QuantileForestReadout(model, X[:700], y[:700])

    method = ConformalizedQuantileInterval(model, cfg.alpha, readout)
    method.calibrate(X[700:1100], y[700:1100])
    result = method.predict(X[1100:])
    np.testing.assert_allclose(
        result.prediction, model.predict(X[1100:]), rtol=1e-12, atol=1e-12
    )


def test_cqr_never_returns_an_inverted_interval():
    """A large negative calibration quantile must not flip the bounds."""
    cfg = Config(alpha=0.50, n_estimators=60, max_features=1.0)
    X, y, _, _ = grouped_problem(1500, seed=9)
    model = fitted(cfg, X, y, 700)
    readout = QuantileForestReadout(model, X[:700], y[:700])

    method = ConformalizedQuantileInterval(model, cfg.alpha, readout)
    method.calibrate(X[700:1100], y[700:1100])
    result = method.predict(X[1100:])
    assert np.all(result.upper >= result.lower)


def test_normalized_qrf_attains_marginal_coverage():
    cfg = Config(alpha=0.10, n_estimators=150, max_features=1.0)
    X, y, _, _ = grouped_problem(5000, seed=10)
    model = fitted(cfg, X, y, 2200)
    readout = QuantileForestReadout(model, X[:2200], y[:2200])
    cal, test = slice(2200, 3600), slice(3600, 5000)

    method = NormalizedQuantileForestInterval(model, cfg.alpha, readout, beta=0.1)
    method.calibrate(X[cal], y[cal])
    result = method.predict(X[test])
    assert 0.86 < picp(y[test], result.lower, result.upper) < 0.94


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_build_method_covers_every_configured_method():
    cfg = Config(n_estimators=20, max_features=1.0)
    X, y, _, _ = grouped_problem(400, seed=11)
    model = fitted(cfg, X, y, 200)
    readout = QuantileForestReadout(model, X[:200], y[:200])
    for name in cfg.methods:
        method = build_method(name, model, cfg, quantile_readout=readout)
        assert method.name == name


def test_build_method_refuses_quantile_methods_without_a_readout():
    cfg = Config(n_estimators=20, max_features=1.0)
    X, y, _, _ = grouped_problem(400, seed=12)
    model = fitted(cfg, X, y, 200)
    with pytest.raises(ValueError, match="QuantileForestReadout"):
        build_method("cqr", model, cfg)
