"""Molecular featurization: RDKit descriptors for the model, Morgan
fingerprints for structural similarity.

Two separate representations on purpose. The Random Forest consumes physical
descriptors (molecular weight, LogP, TPSA, counts, graph indices), which is
what the proposal specifies. The novelty analysis needs a notion of
*structural* similarity, which descriptors do not give -- two very different
molecules can share a descriptor vector -- so that uses binary Morgan
fingerprints and Tanimoto similarity.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

LOGGER = logging.getLogger(__name__)

# A handful of RDKit descriptors are unstable (they raise, or return inf) for
# otherwise valid molecules, and Ipc overflows to astronomically large values
# on medium-sized rings, which would dominate any distance computation.
EXCLUDED_DESCRIPTORS = frozenset({"Ipc"})


def descriptor_functions() -> list[tuple[str, callable]]:
    """The descriptor list, in a fixed order, minus the unstable entries."""
    return [
        (name, fn)
        for name, fn in Descriptors.descList
        if name not in EXCLUDED_DESCRIPTORS
    ]


def mols_from_smiles(smiles: list[str]) -> list[Chem.Mol]:
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES reached featurization: {smi!r}")
        mols.append(mol)
    return mols


def compute_descriptors(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    """Dense descriptor matrix.

    Failures and non-finite values become NaN rather than aborting the run;
    the modelling pipeline imputes them with the training-set median. This
    keeps one awkward molecule from dropping a row that the split already
    accounted for.
    """
    functions = descriptor_functions()
    names = [name for name, _ in functions]
    matrix = np.full((len(mols), len(functions)), np.nan, dtype=np.float64)
    failures = 0
    for i, mol in enumerate(mols):
        for j, (_, fn) in enumerate(functions):
            try:
                value = fn(mol)
            except Exception:  # noqa: BLE001 - a bad descriptor must not kill the run
                failures += 1
                continue
            matrix[i, j] = value
    matrix[~np.isfinite(matrix)] = np.nan
    n_nan = int(np.isnan(matrix).sum())
    if n_nan:
        LOGGER.info(
            "descriptor matrix: %d non-finite entries (%d raised) out of %d",
            n_nan,
            failures,
            matrix.size,
        )
    return matrix, names


def morgan_fingerprints(
    mols: list[Chem.Mol], radius: int = 2, n_bits: int = 2048
) -> list[DataStructs.ExplicitBitVect]:
    """Binary Morgan (ECFP-like) fingerprints for Tanimoto similarity."""
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(mol) for mol in mols]


def featurize(
    smiles: list[str], morgan_radius: int = 2, morgan_bits: int = 2048
) -> tuple[np.ndarray, list[str], list[DataStructs.ExplicitBitVect]]:
    """Descriptors and fingerprints for a list of SMILES, in input order."""
    mols = mols_from_smiles(smiles)
    descriptors, names = compute_descriptors(mols)
    fingerprints = morgan_fingerprints(mols, radius=morgan_radius, n_bits=morgan_bits)
    return descriptors, names, fingerprints
