"""Structural novelty, for the H3 analysis.

H3 claims that reliable intervals are harder to maintain for structurally
unfamiliar molecules. Testing that needs a per-molecule measure of how
unfamiliar a test compound is, which the scaffold split alone does not give:
a scaffold split tells you the test set as a whole is unfamiliar, not which
members of it are the furthest out.

The measure used here is the maximum Tanimoto similarity between a test
molecule's Morgan fingerprint and any training molecule's. Max, not mean:
what matters is whether the model has seen *anything* like this compound, not
how it compares to the library on average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs


def max_train_similarity(
    test_fps: list[DataStructs.ExplicitBitVect],
    train_fps: list[DataStructs.ExplicitBitVect],
) -> np.ndarray:
    """For each test molecule, its similarity to the nearest training molecule.

    1.0 means an identical fingerprint is present in training; values near 0
    mean nothing in training shares substructures with it.
    """
    if not train_fps:
        raise ValueError("training fingerprint list is empty")
    return np.array(
        [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in test_fps],
        dtype=float,
    )


def similarity_bins(similarity: np.ndarray, n_bins: int = 4) -> pd.Series:
    """Group molecules into equal-count similarity bins, least familiar first.

    Equal-count (quantile) bins rather than fixed cut-offs, so that each bin
    holds enough molecules for its coverage estimate to mean something -- the
    similarity distribution shifts a lot between random and scaffold splits,
    and fixed cut-offs would leave some bins nearly empty.
    """
    similarity = np.asarray(similarity, dtype=float)
    if len(similarity) < n_bins:
        raise ValueError(
            f"need at least {n_bins} molecules to form {n_bins} bins, got {len(similarity)}"
        )
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    try:
        binned = pd.qcut(similarity, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        # Too many ties to cut into n_bins; fall back to ranking.
        ranks = pd.Series(similarity).rank(method="first")
        binned = pd.qcut(ranks, q=n_bins, labels=labels)
    return pd.Series(binned, name="novelty_bin").astype(str)


def binned_reliability(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y_pred: np.ndarray,
    similarity: np.ndarray,
    alpha: float,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Per-bin accuracy and interval quality, ordered least familiar first.

    Q1 is the least familiar quartile. A method whose coverage holds in Q4 but
    collapses in Q1 is marginally valid and conditionally useless -- the
    failure mode H3 predicts.
    """
    from .metrics import summarize  # local import keeps the module graph acyclic

    bins = similarity_bins(similarity, n_bins=n_bins)
    rows = []
    for label in sorted(bins.unique()):
        mask = (bins == label).to_numpy()
        stats = summarize(y_true[mask], y_pred[mask], lower[mask], upper[mask], alpha)
        stats["novelty_bin"] = label
        stats["similarity_min"] = float(similarity[mask].min())
        stats["similarity_max"] = float(similarity[mask].max())
        stats["similarity_mean"] = float(similarity[mask].mean())
        rows.append(stats)
    return pd.DataFrame(rows)
