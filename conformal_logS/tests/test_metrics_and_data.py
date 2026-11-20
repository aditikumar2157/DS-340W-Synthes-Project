"""Tests for metrics, curation, featurization and the novelty analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_solubility import data, metrics
from conformal_solubility.features import featurize
from conformal_solubility.novelty import (
    binned_reliability,
    max_train_similarity,
    similarity_bins,
)


# --- metrics ---------------------------------------------------------------


def test_picp_counts_boundary_values_as_covered():
    y = np.array([0.0, 1.0, 2.0])
    lower = np.array([0.0, 0.5, 5.0])
    upper = np.array([0.5, 1.0, 6.0])
    assert metrics.picp(y, lower, upper) == pytest.approx(2 / 3)


def test_mae_rmse_and_width_on_hand_computed_values():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 4.0, 3.0])
    assert metrics.mae(y, pred) == pytest.approx(2 / 3)
    assert metrics.rmse(y, pred) == pytest.approx(np.sqrt(4 / 3))
    assert metrics.mpiw(np.array([0.0, 0.0]), np.array([1.0, 3.0])) == pytest.approx(2.0)


def test_coverage_gap_sign_distinguishes_over_and_under_coverage():
    y = np.zeros(10)
    wide = metrics.coverage_gap(y, np.full(10, -1.0), np.full(10, 1.0), 0.10)
    narrow_lower = np.full(10, 0.5)
    narrow = metrics.coverage_gap(y, narrow_lower, np.full(10, 1.0), 0.10)
    assert wide == pytest.approx(0.10)   # 100% coverage, conservative
    assert narrow == pytest.approx(-0.90)  # 0% coverage, over-confident


def test_interval_score_penalizes_misses_more_than_width():
    y = np.array([0.0])
    tight_hit = metrics.interval_score(y, np.array([-0.5]), np.array([0.5]), 0.10)
    wide_hit = metrics.interval_score(y, np.array([-2.0]), np.array([2.0]), 0.10)
    tight_miss = metrics.interval_score(y, np.array([1.0]), np.array([1.5]), 0.10)
    assert tight_hit < wide_hit < tight_miss


def test_summarize_rejects_swapped_bounds_and_mismatched_lengths():
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="swapped"):
        metrics.summarize(y, y, np.array([1.0, 2.0]), np.array([0.0, 1.0]), 0.10)
    with pytest.raises(ValueError, match="same length"):
        metrics.summarize(y, y, np.array([0.0]), np.array([1.0]), 0.10)


# --- curation --------------------------------------------------------------


def test_curate_canonicalizes_drops_bad_smiles_and_collapses_duplicates():
    frame = pd.DataFrame(
        {
            data.SMILES_COLUMN: ["CCO", "OCC", "c1ccccc1", "not_a_molecule", "CCN"],
            data.TARGET_COLUMN: [-0.5, -0.7, -2.0, -1.0, np.nan],
        }
    )
    curated, counts = data.curate(frame)
    assert counts["dropped_unparseable"] == 1
    assert counts["dropped_missing_target"] == 1
    # CCO and OCC are the same molecule: one row, median of the two values.
    assert counts["duplicate_structures_collapsed"] == 1
    assert len(curated) == 2
    ethanol = curated.loc[curated["smiles"] == "CCO"].iloc[0]
    assert ethanol["logS"] == pytest.approx(-0.6)
    assert ethanol["n_measurements"] == 2


def test_curate_output_is_sorted_and_unique():
    frame = pd.DataFrame(
        {
            data.SMILES_COLUMN: ["c1ccccc1", "CCO", "CCCO"],
            data.TARGET_COLUMN: [-2.0, -0.5, -0.7],
        }
    )
    curated, _ = data.curate(frame)
    assert curated["smiles"].is_unique
    assert list(curated["smiles"]) == sorted(curated["smiles"])


def test_load_raw_reports_a_missing_file_with_the_download_command():
    with pytest.raises(FileNotFoundError, match="curl"):
        data.load_raw(path=data.RAW_DATA_DIR / "does_not_exist.csv")


# --- featurization ---------------------------------------------------------


def test_featurize_returns_finite_or_nan_matrix_with_stable_columns():
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    X, names, fps = featurize(smiles, morgan_bits=512)
    assert X.shape == (3, len(names))
    assert len(fps) == 3
    assert fps[0].GetNumBits() == 512
    assert "Ipc" not in names, "the overflowing descriptor must stay excluded"
    assert not np.isinf(X).any(), "non-finite values must be mapped to NaN"


def test_featurize_rejects_unparseable_smiles():
    with pytest.raises(ValueError, match="unparseable"):
        featurize(["CCO", "definitely_not_smiles"])


# --- novelty ---------------------------------------------------------------


def test_max_train_similarity_is_one_for_a_molecule_present_in_training():
    _, _, fps = featurize(["CCO", "c1ccccc1", "CCCCO"], morgan_bits=512)
    similarity = max_train_similarity([fps[0]], [fps[0], fps[1]])
    assert similarity[0] == pytest.approx(1.0)
    novel = max_train_similarity([fps[1]], [fps[0], fps[2]])
    assert 0.0 <= novel[0] < 1.0


def test_similarity_bins_are_equal_count_and_ordered():
    similarity = np.linspace(0.05, 0.95, 40)
    bins = similarity_bins(similarity, n_bins=4)
    assert set(bins.unique()) == {"Q1", "Q2", "Q3", "Q4"}
    assert bins.value_counts().nunique() == 1, "quartiles should be equal-count"
    assert bins.iloc[0] == "Q1" and bins.iloc[-1] == "Q4"


def test_similarity_bins_handle_heavy_ties():
    similarity = np.array([0.4] * 18 + [0.9, 0.95])
    bins = similarity_bins(similarity, n_bins=4)
    assert len(bins) == 20
    assert bins.nunique() == 4


def test_binned_reliability_reports_one_row_per_bin():
    rng = np.random.default_rng(0)
    n = 80
    y = rng.normal(size=n)
    pred = y + rng.normal(scale=0.2, size=n)
    half = np.full(n, 0.5)
    frame = binned_reliability(
        y, pred - half, pred + half, pred, np.linspace(0, 1, n), alpha=0.10, n_bins=4
    )
    assert list(frame["novelty_bin"]) == ["Q1", "Q2", "Q3", "Q4"]
    assert frame["n"].sum() == n
    assert (frame["similarity_min"] <= frame["similarity_max"]).all()
