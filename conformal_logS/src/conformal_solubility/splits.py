"""Train / calibration / test splits.

Three regimes. The first two are the comparison the proposal describes; the
third exists because the first two cannot tell two different failures apart.

``random``
    Molecules are assigned at random. Training, calibration and test sets all
    look alike, so this is the optimistic case and the one where conformal
    prediction's exchangeability assumption actually holds.

``scaffold``
    Molecules are grouped by Bemis-Murcko scaffold; whole groups go to the
    test set or to the train+calibration pool, and the pool is then divided at
    random. No scaffold is shared with the test set, so test molecules are
    structurally unfamiliar -- the realistic prospective-screening case, and
    the condition the parent conformalized-fusion-regression paper found
    hardest. This is also what a practitioner would naturally do: calibrate on
    whatever labelled data is to hand.

``scaffold_ood_calib``
    Same scaffold-disjoint test set, but the calibration set is its own block
    of scaffolds, unfamiliar to the *model* as well. The calibration residuals
    therefore already reflect extrapolation error.

Why the third regime matters: under ``scaffold``, calibration data is
exchangeable with the training data but not with the test data, so conformal
prediction's guarantee simply does not apply and under-coverage is expected.
Comparing ``scaffold`` with ``scaffold_ood_calib`` separates "the model is
worse out of distribution" from "the calibration set was drawn from the wrong
distribution" -- two explanations for the same lost coverage, with different
fixes.

All three return three disjoint index arrays covering every row exactly once.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Generic Bemis-Murcko scaffold as a SMILES string.

    Acyclic molecules have no ring system and yield an empty scaffold; they
    are all grouped together under ``""``, which is the standard convention
    and keeps them out of the test set unless the group lands there whole.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES in scaffold split: {smiles!r}")
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except RuntimeError as exc:  # pragma: no cover - environment-dependent
        # RDKit 2026.09 ships a broken MurckoDecompose ("bad any cast") that
        # makes every scaffold call fail. Say so plainly instead of letting a
        # C++ cast error surface from the middle of the split.
        import rdkit

        raise RuntimeError(
            "RDKit's Murcko decomposition failed "
            f"(rdkit {rdkit.__version__}): {exc}. "
            "Install a version this project is pinned against: "
            "pip install 'rdkit>=2024.3,<2026.3'"
        ) from exc
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    """Map each scaffold to the indices of the molecules that carry it."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(index)
    return dict(groups)


def _check_fracs(train_frac: float, calib_frac: float) -> float:
    test_frac = 1.0 - train_frac - calib_frac
    if train_frac <= 0 or calib_frac <= 0 or test_frac <= 0:
        raise ValueError(
            "train_frac and calib_frac must be positive and sum to < 1; "
            f"got {train_frac} and {calib_frac}"
        )
    return test_frac


def random_split(
    n: int, train_frac: float, calib_frac: float, seed: int
) -> Split:
    """Shuffle rows, then cut at the requested proportions."""
    _check_fracs(train_frac, calib_frac)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_calib = int(round(calib_frac * n))
    # Guarantee a non-empty test set even after rounding.
    n_calib = min(n_calib, n - n_train - 1)
    return (
        np.sort(order[:n_train]),
        np.sort(order[n_train : n_train + n_calib]),
        np.sort(order[n_train + n_calib :]),
    )


def _ordered_groups(
    groups: dict[str, list[int]], seed: int | None
) -> list[list[int]]:
    """Scaffold groups in assignment order.

    ``seed=None`` gives the deterministic MoleculeNet-style order -- largest
    groups first, so the common scaffolds train the model and the rare ones
    are tested. A seed shuffles the groups instead (a "random scaffold
    split"), which keeps scaffolds disjoint while letting the experiment
    repeat over several draws and report a spread rather than one number.
    """
    if seed is None:
        return [
            indices
            for _, indices in sorted(
                groups.items(), key=lambda item: (-len(item[1]), item[0])
            )
        ]
    rng = np.random.default_rng(seed)
    keys = sorted(groups)  # sort first so the shuffle is reproducible
    rng.shuffle(keys)
    return [groups[key] for key in keys]


def scaffold_split_ood_calib(
    smiles: list[str], train_frac: float, calib_frac: float, seed: int | None = None
) -> Split:
    """Assign whole scaffold groups to train, then calibration, then test.

    All three sets are scaffold-disjoint, so the calibration molecules are
    unfamiliar to the model in the same way the test molecules are.
    """
    _check_fracs(train_frac, calib_frac)
    groups = scaffold_groups(smiles)
    n = len(smiles)
    n_train_target = train_frac * n
    n_calib_target = calib_frac * n

    train: list[int] = []
    calib: list[int] = []
    test: list[int] = []
    for indices in _ordered_groups(groups, seed):
        if len(train) < n_train_target:
            train.extend(indices)
        elif len(calib) < n_calib_target:
            calib.extend(indices)
        else:
            test.extend(indices)

    if not test:
        raise ValueError(
            "scaffold split produced an empty test set; lower train_frac/calib_frac"
        )
    return np.sort(np.array(train)), np.sort(np.array(calib)), np.sort(np.array(test))


def scaffold_split(
    smiles: list[str], train_frac: float, calib_frac: float, seed: int | None = None
) -> Split:
    """Scaffold-disjoint test set; calibration drawn at random from the rest.

    Whole scaffold groups fill a train+calibration pool until it reaches
    ``train_frac + calib_frac`` of the data; everything left is the test set.
    The pool is then divided at random, which is what a practitioner does in
    practice -- the calibration set is simply a slice of the labelled data,
    with no attempt to match it to the molecules being screened.
    """
    _check_fracs(train_frac, calib_frac)
    groups = scaffold_groups(smiles)
    n = len(smiles)
    pool_target = (train_frac + calib_frac) * n

    pool: list[int] = []
    test: list[int] = []
    for indices in _ordered_groups(groups, seed):
        if len(pool) < pool_target:
            pool.extend(indices)
        else:
            test.extend(indices)

    if not test:
        raise ValueError(
            "scaffold split produced an empty test set; lower train_frac/calib_frac"
        )

    pool_array = np.array(pool)
    rng = np.random.default_rng(0 if seed is None else seed)
    shuffled = rng.permutation(pool_array)
    n_train = int(round(len(pool_array) * train_frac / (train_frac + calib_frac)))
    n_train = min(max(n_train, 1), len(pool_array) - 1)
    return (
        np.sort(shuffled[:n_train]),
        np.sort(shuffled[n_train:]),
        np.sort(np.array(test)),
    )


def make_split(
    split_type: str,
    smiles: list[str],
    train_frac: float,
    calib_frac: float,
    seed: int,
) -> Split:
    if split_type == "random":
        return random_split(len(smiles), train_frac, calib_frac, seed)
    if split_type == "scaffold":
        return scaffold_split(smiles, train_frac, calib_frac, seed)
    if split_type == "scaffold_ood_calib":
        return scaffold_split_ood_calib(smiles, train_frac, calib_frac, seed)
    raise ValueError(f"unknown split type: {split_type!r}")
