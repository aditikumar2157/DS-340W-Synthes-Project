"""Tests for the V-SYNTHES2 docking bridge.

Two things matter here and nothing else does. The module must refuse to
produce a confident-looking interval from a file too small to calibrate on,
and it must never hand the model a field that would not exist at the moment
the decision is made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformal_solubility.config import Config
from conformal_solubility.vsynthes import (
    FRAGMENT_SCORE_FIELD,
    POSE_DERIVED_FIELDS,
    TARGET_FIELD,
    baseline_fragment_score_gap,
    check_sufficient_data,
    conformal_docking_intervals,
    expansion_gap,
    feature_columns,
    load_sdf,
    minimum_calibration_size,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SDF = REPO_ROOT / "RMSD_assessment" / "list_1.sdf"


def synthetic_sdf(path: Path, n: int, seed: int = 0) -> Path:
    """Write an SDF of single-atom records carrying realistic property blocks.

    The structures are deliberately trivial -- the module reads property
    fields, and a real docking pose would make the fixture unreadable without
    testing anything extra. The *relationship* is what matters: the final
    score depends on the fragment score plus the molecule's size, with noise,
    which is the structure the predictor is meant to find.
    """
    rng = np.random.default_rng(seed)
    fragment = rng.normal(-16.0, 2.0, size=n)
    weight = rng.uniform(250.0, 420.0, size=n)
    atoms = rng.integers(30, 55, size=n)
    flexible = rng.integers(2, 8, size=n)
    final = fragment - 11.0 - 0.02 * (weight - 330.0) + rng.normal(0, 1.5, size=n)
    elements = ["C", "N", "O"]

    blocks = []
    for i in range(n):
        element = elements[i % len(elements)]
        blocks.append(
            f"mol{i}\n"
            "  TEST\n"
            "\n"
            "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            f"    0.0000    0.0000    0.0000 {element:<3} 0  0  0  0  0\n"
            "M  END\n"
            f"> <{TARGET_FIELD}>\n{final[i]:.6f}\n\n"
            f"> <{FRAGMENT_SCORE_FIELD}>\n{fragment[i]:.6f}\n\n"
            f"> <MW>\n{weight[i]:.3f}\n\n"
            f"> <Natom>\n{atoms[i]}\n\n"
            f"> <Nflex>\n{flexible[i]}\n\n"
            # A pose-derived field, present exactly as a real file would have
            # it, and perfectly correlated with the target so that a leak
            # would be unmistakable in the metrics.
            f"> <RTCNNscore>\n{final[i] * 1.5:.6f}\n\n"
            f"> <NAME>\nmol{i}\n\n"
            "$$$$\n"
        )
    path.write_text("".join(blocks))
    return path


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_reads_the_repository_example_file():
    docking = load_sdf(EXAMPLE_SDF)
    assert len(docking) == 10
    assert TARGET_FIELD in docking.frame.columns
    assert FRAGMENT_SCORE_FIELD in docking.frame.columns
    assert len(docking.smiles) == 10


def test_expansion_gap_is_fragment_minus_final():
    docking = load_sdf(EXAMPLE_SDF)
    gap = expansion_gap(docking)
    np.testing.assert_allclose(
        gap["gap"], gap["fragment_score"] - gap["final_score"], rtol=1e-12
    )


def test_baseline_gap_reports_a_spread_not_a_point():
    got = baseline_fragment_score_gap(load_sdf(EXAMPLE_SDF))
    assert got["n"] == 10
    assert got["gap_std"] > 0
    assert got["gap_min"] < got["gap_max"]
    assert -1.0 <= got["spearman_fragment_vs_final"] <= 1.0


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------


def test_pose_derived_fields_are_never_offered_as_features(tmp_path):
    docking = load_sdf(synthetic_sdf(tmp_path / "leak.sdf", 60))
    columns = feature_columns(docking.frame)
    assert "RTCNNscore" not in columns, "a rescoring field leaked into the features"
    assert TARGET_FIELD not in columns
    assert not (set(columns) & (POSE_DERIVED_FIELDS - {FRAGMENT_SCORE_FIELD}))


def test_the_fragment_score_is_kept_as_a_feature(tmp_path):
    """It is known before docking, and it is what V-SYNTHES2 decides on."""
    docking = load_sdf(synthetic_sdf(tmp_path / "keep.sdf", 60))
    assert FRAGMENT_SCORE_FIELD in feature_columns(docking.frame)


def test_constant_and_textual_fields_are_dropped():
    frame = pd.DataFrame(
        {
            "Score": [1.0, 2.0, 3.0, 4.0],
            "orig_Score": [0.5, 0.6, 0.7, 0.8],
            "always_seven": [7, 7, 7, 7],
            "NAME": ["a", "b", "c", "d"],
        }
    )
    columns = feature_columns(frame)
    assert "always_seven" not in columns
    assert "NAME" not in columns
    assert columns == ["orig_Score"]


# --------------------------------------------------------------------------
# Refusing to calibrate on too little data
# --------------------------------------------------------------------------


def test_minimum_calibration_size_matches_the_conformal_rank():
    assert minimum_calibration_size(0.10) == 9
    assert minimum_calibration_size(0.05) == 19
    assert minimum_calibration_size(0.20) == 4


def test_check_sufficient_data_rejects_a_ten_molecule_file():
    with pytest.raises(ValueError, match="too few to calibrate"):
        check_sufficient_data(10, alpha=0.10, calib_frac=0.20)


def test_the_repository_example_is_refused_rather_than_guessed_at():
    """The example SDFs are too small, and the module has to say so."""
    docking = load_sdf(EXAMPLE_SDF)
    with pytest.raises(ValueError, match="too few to calibrate"):
        conformal_docking_intervals(docking, Config(alpha=0.10))


# --------------------------------------------------------------------------
# The fit itself, on a file large enough to support one
# --------------------------------------------------------------------------


def test_intervals_are_calibrated_on_a_large_enough_file(tmp_path):
    docking = load_sdf(synthetic_sdf(tmp_path / "big.sdf", 1200, seed=1))
    outcome = conformal_docking_intervals(
        docking, Config(alpha=0.10, n_estimators=120), seed=0
    )
    assert outcome["n_train"] + outcome["n_calib"] + outcome["n_test"] == 1200
    assert 0.84 < outcome["metrics"]["picp"] < 0.96
    assert outcome["metrics"]["mpiw"] > 0
    assert len(outcome["predictions"]) == outcome["n_test"]


def test_mondrian_runs_on_a_docking_set(tmp_path):
    docking = load_sdf(synthetic_sdf(tmp_path / "mondrian.sdf", 1200, seed=2))
    outcome = conformal_docking_intervals(
        docking,
        Config(alpha=0.10, n_estimators=120, mondrian_bins=2, mondrian_min_bin=25),
        method="mondrian_conformal",
        seed=0,
    )
    assert 0.80 < outcome["metrics"]["picp"] < 0.98


def test_screening_threshold_comes_from_the_training_fold(tmp_path):
    """Setting it from the test labels would be using the answer."""
    docking = load_sdf(synthetic_sdf(tmp_path / "triage.sdf", 1200, seed=3))
    outcome = conformal_docking_intervals(
        docking, Config(alpha=0.10, n_estimators=80), seed=0
    )
    test_median = float(np.median(outcome["predictions"]["y_true"]))
    assert outcome["screening_threshold"] != pytest.approx(test_median, abs=1e-9)
    assert 0.0 <= outcome["screening"]["decisive_fraction"] <= 1.0
