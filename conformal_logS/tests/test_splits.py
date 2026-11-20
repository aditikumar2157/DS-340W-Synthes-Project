"""Tests for the split logic.

A leak here would quietly invalidate every reliability number in the project,
so the invariants are checked rather than assumed: the three sets must be
disjoint, cover everything, and -- for the scaffold split -- share no scaffold.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_solubility.splits import (
    make_split,
    murcko_scaffold,
    random_split,
    scaffold_groups,
    scaffold_split,
    scaffold_split_ood_calib,
)

SMILES = [
    "CCO",
    "CCCO",
    "CC(=O)O",
    "c1ccccc1",
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCO",
    "c1ccncc1",
    "c1ccncc1C",
    "c1ccncc1CC",
    "C1CCCCC1",
    "C1CCCCC1C",
    "C1CCCCC1CC",
    "c1ccc2ccccc2c1",
    "c1ccc2ccccc2c1C",
    "c1ccc2ccccc2c1CC",
    "CCCCCC",
    "CCCCCCC",
    "C1CCNCC1",
    "C1CCNCC1C",
]


def assert_partition(split, n):
    train, calib, test = split
    combined = np.concatenate([train, calib, test])
    assert len(combined) == n, "every row must appear exactly once"
    assert set(combined.tolist()) == set(range(n))
    assert len(train) and len(calib) and len(test), "no set may be empty"


def test_random_split_is_a_partition():
    split = random_split(100, 0.6, 0.2, seed=0)
    assert_partition(split, 100)
    assert len(split[0]) == 60
    assert len(split[1]) == 20


def test_random_split_is_reproducible_and_seed_sensitive():
    a = random_split(200, 0.6, 0.2, seed=7)
    b = random_split(200, 0.6, 0.2, seed=7)
    c = random_split(200, 0.6, 0.2, seed=8)
    np.testing.assert_array_equal(a[2], b[2])
    assert not np.array_equal(a[2], c[2])


def test_random_split_rejects_impossible_fractions():
    with pytest.raises(ValueError, match="sum to < 1"):
        random_split(100, 0.8, 0.3, seed=0)


def test_murcko_scaffold_strips_side_chains():
    assert murcko_scaffold("c1ccccc1CCO") == murcko_scaffold("c1ccccc1CC")
    assert murcko_scaffold("CCO") == "", "acyclic molecules have an empty scaffold"


def test_scaffold_groups_cover_every_molecule():
    groups = scaffold_groups(SMILES)
    covered = sorted(i for indices in groups.values() for i in indices)
    assert covered == list(range(len(SMILES)))


def scaffolds_of(indices):
    return {murcko_scaffold(SMILES[i]) for i in indices}


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_scaffold_split_isolates_the_test_scaffolds(seed):
    """The test set must share no scaffold with train or calibration; train
    and calibration are allowed to share, because the pool is split at random
    on purpose -- that is the regime this split models."""
    train, calib, test = scaffold_split(SMILES, 0.5, 0.25, seed=seed)
    assert_partition((train, calib, test), len(SMILES))
    test_s = scaffolds_of(test)
    assert not (scaffolds_of(train) & test_s), "a scaffold leaked from train into test"
    assert not (
        scaffolds_of(calib) & test_s
    ), "a scaffold leaked from calibration into test"


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_scaffold_ood_calib_split_keeps_all_three_disjoint(seed):
    train, calib, test = scaffold_split_ood_calib(SMILES, 0.5, 0.25, seed=seed)
    assert_partition((train, calib, test), len(SMILES))
    train_s, calib_s, test_s = (
        scaffolds_of(train),
        scaffolds_of(calib),
        scaffolds_of(test),
    )
    assert not (train_s & test_s)
    assert not (calib_s & test_s)
    assert not (train_s & calib_s), "calibration must be unfamiliar to the model here"


def test_the_two_scaffold_splits_share_their_test_set():
    """Only the calibration source differs between the two scaffold regimes;
    if the test sets differed too, their metrics would not be comparable."""
    _, _, test_a = scaffold_split(SMILES, 0.5, 0.25, seed=1)
    _, _, test_b = scaffold_split_ood_calib(SMILES, 0.5, 0.25, seed=1)
    np.testing.assert_array_equal(test_a, test_b)


def test_deterministic_scaffold_split_trains_on_the_common_scaffolds():
    train, _, test = scaffold_split_ood_calib(SMILES, 0.5, 0.25, seed=None)
    train_sizes = [
        len(indices)
        for scaffold, indices in scaffold_groups(SMILES).items()
        if scaffold in {murcko_scaffold(SMILES[i]) for i in train}
    ]
    test_sizes = [
        len(indices)
        for scaffold, indices in scaffold_groups(SMILES).items()
        if scaffold in {murcko_scaffold(SMILES[i]) for i in test}
    ]
    assert max(train_sizes) >= max(test_sizes)


def test_make_split_dispatches_and_rejects_unknown():
    for split_type in ("random", "scaffold", "scaffold_ood_calib"):
        assert_partition(
            make_split(split_type, SMILES, 0.5, 0.25, seed=0), len(SMILES)
        )
    with pytest.raises(ValueError, match="unknown split type"):
        make_split("murcko-ish", SMILES, 0.5, 0.25, seed=0)
