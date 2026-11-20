"""Load and curate the aqueous solubility dataset.

Source: the Delaney (ESOL) set as distributed with MoleculeNet / DeepChem --
1,128 small molecules with measured log solubility in mol/L.

Curation here is deliberately explicit, because the scaffold split and the
novelty analysis are both sensitive to duplicate structures: a molecule that
appears in both the training and test halves would quietly inflate coverage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

DATASET_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)
RAW_FILENAME = "delaney-processed.csv"
SMILES_COLUMN = "smiles"
TARGET_COLUMN = "measured log solubility in mols per litre"

# RDKit prints a parse warning per bad SMILES; we count them ourselves instead.
RDLogger.DisableLog("rdApp.*")


def raw_path() -> Path:
    return RAW_DATA_DIR / RAW_FILENAME


def download_instructions() -> str:
    return (
        f"Dataset not found at {raw_path()}.\n"
        f"Download it once with:\n\n"
        f"    mkdir -p {RAW_DATA_DIR}\n"
        f"    curl -sfL -o {raw_path()} {DATASET_URL}\n"
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV without altering it."""
    path = path or raw_path()
    if not path.exists():
        raise FileNotFoundError(download_instructions())
    frame = pd.read_csv(path)
    missing = {SMILES_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {sorted(missing)}")
    return frame


def canonical_smiles(smiles: str) -> str | None:
    """Canonical SMILES, or None when RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def curate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonicalize, drop unparseable rows, and collapse duplicate structures.

    Duplicate structures are collapsed to the median of their measurements:
    the median is robust to a single badly transcribed value, and keeping one
    row per structure is what makes the train/test separation meaningful.

    Returns the curated frame (columns: ``smiles``, ``logS``, ``n_measurements``)
    and a dictionary of counts for the run log.
    """
    work = frame[[SMILES_COLUMN, TARGET_COLUMN]].rename(
        columns={SMILES_COLUMN: "smiles_raw", TARGET_COLUMN: "logS"}
    )
    counts = {"rows_in": len(work)}

    work["logS"] = pd.to_numeric(work["logS"], errors="coerce")
    work = work.dropna(subset=["logS"])
    counts["dropped_missing_target"] = counts["rows_in"] - len(work)

    work["smiles"] = work["smiles_raw"].astype(str).str.strip().map(canonical_smiles)
    before = len(work)
    work = work.dropna(subset=["smiles"])
    counts["dropped_unparseable"] = before - len(work)

    grouped = (
        work.groupby("smiles", as_index=False)
        .agg(logS=("logS", "median"), n_measurements=("logS", "size"))
        .sort_values("smiles", kind="mergesort")
        .reset_index(drop=True)
    )
    counts["duplicate_structures_collapsed"] = len(work) - len(grouped)
    counts["rows_out"] = len(grouped)
    return grouped, counts


def load_dataset(
    path: Path | None = None, cache: bool = True
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the curated dataset, writing a copy to ``data/processed`` for the record."""
    curated, counts = curate(load_raw(path))
    if cache:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        curated.to_csv(PROCESSED_DATA_DIR / "esol_curated.csv", index=False)
    LOGGER.info("curated dataset: %s", counts)
    return curated, counts
