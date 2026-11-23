"""Tests for the screening-triage decision layer and the dataset registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_solubility.config import KNOWN_DATASETS, Config
from conformal_solubility.data import (
    DATASETS,
    curate,
    get_spec,
    largest_fragment,
)
from conformal_solubility.screening import (
    ACCEPT,
    DEFER,
    REJECT,
    point_triage,
    screening_curve,
    screening_summary,
    triage,
)


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------


def test_triage_decides_only_when_the_interval_clears_the_threshold():
    lower = np.array([-3.0, -6.0, -5.0, -4.0])
    upper = np.array([-2.0, -5.0, -3.0, -4.0])
    got = triage(lower, upper, threshold=-4.0)
    assert got.tolist() == [
        ACCEPT,  # entirely above -4
        REJECT,  # entirely below -4
        DEFER,   # straddles -4
        ACCEPT,  # degenerate interval sitting exactly on the threshold
    ]


def test_triage_defers_an_interval_that_merely_grazes_the_threshold():
    got = triage(np.array([-4.5]), np.array([-3.5]), threshold=-4.0)
    assert got[0] == DEFER


def test_point_triage_never_defers():
    prediction = np.array([-3.0, -5.0, -4.0])
    got = point_triage(prediction, threshold=-4.0)
    assert DEFER not in got.tolist()
    assert got.tolist() == [ACCEPT, REJECT, ACCEPT]


def test_screening_summary_counts_decisions_and_errors_by_hand():
    #            truth   interval          point
    # molecule 0: -3.0   [-3.5, -2.5] acc   -3.0 acc   truth above  -> both right
    # molecule 1: -5.0   [-6.0, -4.5] rej   -5.2 rej   truth below  -> both right
    # molecule 2: -3.0   [-6.0, -2.0] defer -5.0 rej   truth above  -> point wrong
    # molecule 3: -5.0   [-3.5, -3.0] acc   -3.2 acc   truth below  -> both wrong
    y_true = np.array([-3.0, -5.0, -3.0, -5.0])
    prediction = np.array([-3.0, -5.2, -5.0, -3.2])
    lower = np.array([-3.5, -6.0, -6.0, -3.5])
    upper = np.array([-2.5, -4.5, -2.0, -3.0])

    got = screening_summary(y_true, prediction, lower, upper, threshold=-4.0)

    assert got["accept_fraction"] == pytest.approx(0.5)
    assert got["reject_fraction"] == pytest.approx(0.25)
    assert got["defer_fraction"] == pytest.approx(0.25)
    assert got["decisive_fraction"] == pytest.approx(0.75)
    # One wrong call (molecule 3) out of three decisive ones.
    assert got["decisive_error_rate"] == pytest.approx(1 / 3)
    # Of the two accepted, one was truly below the threshold.
    assert got["false_accept_rate"] == pytest.approx(0.5)
    # The point predictor calls all four and gets two wrong (2 and 3).
    assert got["point_error_rate"] == pytest.approx(0.5)
    assert got["fraction_truly_above"] == pytest.approx(0.5)


def test_screening_summary_reports_nan_rather_than_zero_when_nothing_is_decided():
    """Deferring everything is a real outcome and must not read as perfect."""
    y_true = np.array([-3.0, -5.0])
    prediction = np.array([-3.0, -5.0])
    lower = np.array([-9.0, -9.0])
    upper = np.array([0.0, 0.0])
    got = screening_summary(y_true, prediction, lower, upper, threshold=-4.0)
    assert got["decisive_fraction"] == 0.0
    assert np.isnan(got["decisive_error_rate"])
    assert np.isnan(got["false_accept_rate"])


def test_wider_intervals_defer_more_and_err_less():
    """The trade the analysis exists to expose.

    Same predictions, same truth; only the width changes. Widening must move
    molecules from the decisive pile to the deferred one, and must not make a
    confident call more likely to be wrong.
    """
    rng = np.random.default_rng(0)
    y_true = rng.normal(-4.0, 1.5, size=4000)
    prediction = y_true + rng.normal(0, 0.8, size=4000)

    previous_decisive = 1.1
    previous_error = -1.0
    for half_width in (0.2, 0.6, 1.2, 2.4):
        got = screening_summary(
            y_true,
            prediction,
            prediction - half_width,
            prediction + half_width,
            threshold=-4.0,
        )
        assert got["decisive_fraction"] < previous_decisive
        assert got["decisive_error_rate"] <= previous_error or previous_error < 0
        previous_decisive = got["decisive_fraction"]
        previous_error = got["decisive_error_rate"]


def test_screening_curve_sweeps_thresholds():
    rng = np.random.default_rng(1)
    y_true = rng.normal(-4.0, 1.5, size=500)
    prediction = y_true + rng.normal(0, 0.5, size=500)
    frame = screening_curve(
        y_true, prediction, prediction - 1.0, prediction + 1.0, [-6.0, -4.0, -2.0]
    )
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert frame["threshold"].tolist() == [-6.0, -4.0, -2.0]
    # A threshold below almost everything leaves almost nothing truly above...
    assert frame.iloc[0]["fraction_truly_above"] > frame.iloc[2]["fraction_truly_above"]


# --------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------


def test_registry_and_config_agree_on_which_datasets_exist():
    assert set(DATASETS) == set(KNOWN_DATASETS)


def test_config_rejects_an_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        Config(dataset_name="not-a-dataset").validate()


def test_each_dataset_has_its_own_processed_path():
    paths = {get_spec(name).processed_path for name in DATASETS}
    assert len(paths) == len(DATASETS)


def test_largest_fragment_keeps_single_component_molecules_untouched():
    assert largest_fragment("CCO") == "CCO"
    assert largest_fragment("c1ccccc1O") == "c1ccccc1O"


def test_largest_fragment_strips_a_counter_ion():
    """AqSolDB contains salts; the counter-ion must not drive the scaffold."""
    got = largest_fragment("CC(=O)[O-].[Na+]")
    assert "Na" not in got
    assert got == "CC(=O)[O-]"


def test_curation_collapses_duplicates_to_the_median():
    frame = pd.DataFrame(
        {
            "SMILES": ["CCO", "OCC", "c1ccccc1", "bad-smiles", "CCO"],
            "Solubility": [-1.0, -3.0, -2.0, -1.0, -2.0],
        }
    )
    curated, counts = curate(frame, "aqsoldb")

    assert counts["rows_in"] == 5
    assert counts["dropped_unparseable"] == 1
    # CCO and OCC are the same molecule written two ways, plus a third copy.
    assert counts["duplicate_structures_collapsed"] == 2
    assert counts["rows_out"] == 2

    ethanol = curated[curated["smiles"] == "CCO"].iloc[0]
    assert ethanol["n_measurements"] == 3
    assert ethanol["logS"] == pytest.approx(-2.0)  # median of -1, -3, -2


def test_curation_drops_rows_with_no_usable_target():
    frame = pd.DataFrame(
        {"smiles": ["CCO", "c1ccccc1"], "measured log solubility in mols per litre": [-1.0, None]}
    )
    curated, counts = curate(frame, "esol")
    assert counts["dropped_missing_target"] == 1
    assert len(curated) == 1
