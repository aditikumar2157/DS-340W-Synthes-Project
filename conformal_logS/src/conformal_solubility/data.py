"""Load and curate aqueous-solubility datasets.

Two datasets, behind one interface.

``esol``
    The Delaney (ESOL) set as distributed with MoleculeNet / DeepChem --
    1,128 small molecules with measured log solubility in mol/L. Small,
    clean, and the set the project's first results were produced on.

``aqsoldb``
    AqSolDB (Sorkun et al., 2019), 9,982 compounds assembled from nine
    published sources, with the same target in the same units. Roughly nine
    times the size and considerably messier, which is the point: the
    scaffold-split numbers on ESOL carry a PICP spread of up to 0.11 across
    seeds because the test set is only a handful of scaffold groups, and a
    conclusion that rests on a spread that wide is not a conclusion. AqSolDB
    also spans a far broader slice of chemical space, so the novelty analysis
    has somewhere to go.

Curation is deliberately explicit and identical for both, because the
scaffold split and the novelty analysis are both sensitive to duplicate
structures: a molecule appearing in both the training and test halves would
quietly inflate coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

# RDKit prints a parse warning per bad SMILES; we count them ourselves instead.
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset comes from and which of its columns we use."""

    name: str
    filename: str
    url: str
    smiles_column: str
    target_column: str
    citation: str

    @property
    def raw_path(self) -> Path:
        return RAW_DATA_DIR / self.filename

    @property
    def processed_path(self) -> Path:
        return PROCESSED_DATA_DIR / f"{self.name}_curated.csv"


DATASETS: dict[str, DatasetSpec] = {
    "esol": DatasetSpec(
        name="esol",
        filename="delaney-processed.csv",
        url=(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
            "delaney-processed.csv"
        ),
        smiles_column="smiles",
        target_column="measured log solubility in mols per litre",
        citation="Delaney (2004), as distributed with MoleculeNet/DeepChem",
    ),
    "aqsoldb": DatasetSpec(
        name="aqsoldb",
        filename="curated-solubility-dataset.csv",
        # The canonical home is Harvard Dataverse (doi:10.7910/DVN/OVHAW8).
        # The file is committed under data/raw/ as well, because some networks
        # cannot reach Dataverse and a dataset you cannot fetch is a result
        # nobody can reproduce.
        url="https://dataverse.harvard.edu/api/access/datafile/3407241",
        smiles_column="SMILES",
        target_column="Solubility",
        citation="Sorkun, Khetan & Er, AqSolDB, Scientific Data 6:143 (2019)",
    ),
}

DEFAULT_DATASET = "esol"

# Backwards-compatible module-level names: the first version of this module
# only knew about ESOL and exposed these directly.
DATASET_URL = DATASETS["esol"].url
RAW_FILENAME = DATASETS["esol"].filename
SMILES_COLUMN = DATASETS["esol"].smiles_column
TARGET_COLUMN = DATASETS["esol"].target_column


def get_spec(dataset: str | DatasetSpec = DEFAULT_DATASET) -> DatasetSpec:
    if isinstance(dataset, DatasetSpec):
        return dataset
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}"
        ) from None


def raw_path(dataset: str | DatasetSpec = DEFAULT_DATASET) -> Path:
    return get_spec(dataset).raw_path


def download_instructions(dataset: str | DatasetSpec = DEFAULT_DATASET) -> str:
    spec = get_spec(dataset)
    return (
        f"Dataset {spec.name!r} not found at {spec.raw_path}.\n"
        f"Download it once with:\n\n"
        f"    mkdir -p {RAW_DATA_DIR}\n"
        f"    curl -sfL -o {spec.raw_path} {spec.url}\n"
    )


def load_raw(
    path: Path | None = None, dataset: str | DatasetSpec = DEFAULT_DATASET
) -> pd.DataFrame:
    """Read the raw CSV without altering it."""
    spec = get_spec(dataset)
    path = path or spec.raw_path
    if not path.exists():
        raise FileNotFoundError(download_instructions(spec))
    frame = pd.read_csv(path)
    missing = {spec.smiles_column, spec.target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {sorted(missing)}")
    return frame


def canonical_smiles(smiles: str) -> str | None:
    """Canonical SMILES, or None when RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def curate(
    frame: pd.DataFrame, dataset: str | DatasetSpec = DEFAULT_DATASET
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonicalize, drop unparseable rows, and collapse duplicate structures.

    Duplicate structures are collapsed to the median of their measurements:
    the median is robust to a single badly transcribed value, and keeping one
    row per structure is what makes the train/test separation meaningful.
    AqSolDB needs this far more than ESOL does -- it is an aggregation of nine
    source databases, so the same compound recurs with differing values.

    Returns the curated frame (columns: ``smiles``, ``logS``, ``n_measurements``)
    and a dictionary of counts for the run log.
    """
    spec = get_spec(dataset)
    work = frame[[spec.smiles_column, spec.target_column]].rename(
        columns={spec.smiles_column: "smiles_raw", spec.target_column: "logS"}
    )
    counts = {"rows_in": len(work)}

    work["logS"] = pd.to_numeric(work["logS"], errors="coerce")
    work = work.dropna(subset=["logS"])
    counts["dropped_missing_target"] = counts["rows_in"] - len(work)

    work["smiles"] = work["smiles_raw"].astype(str).str.strip().map(canonical_smiles)
    before = len(work)
    work = work.dropna(subset=["smiles"])
    counts["dropped_unparseable"] = before - len(work)

    # A handful of AqSolDB entries are salts or mixtures written as dot-
    # separated components. They parse, but their "scaffold" is whichever
    # fragment RDKit happens to return, which would put a counter-ion in
    # charge of a split. Keep the largest organic component instead.
    before = len(work)
    work["smiles"] = work["smiles"].map(largest_fragment)
    work = work.dropna(subset=["smiles"])
    counts["dropped_no_organic_fragment"] = before - len(work)

    grouped = (
        work.groupby("smiles", as_index=False)
        .agg(logS=("logS", "median"), n_measurements=("logS", "size"))
        .sort_values("smiles", kind="mergesort")
        .reset_index(drop=True)
    )
    counts["duplicate_structures_collapsed"] = len(work) - len(grouped)
    counts["rows_out"] = len(grouped)
    return grouped, counts


def largest_fragment(smiles: str) -> str | None:
    """The heaviest covalently connected component of a structure.

    For a single-component molecule this returns it unchanged, so ESOL is
    untouched. For a salt such as a sodium carboxylate it returns the organic
    anion: the measured solubility belongs to the compound, but the scaffold
    and the descriptors should describe the part that does the chemistry.
    """
    if "." not in smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    pieces = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not pieces:
        return None
    heaviest = max(pieces, key=lambda frag: frag.GetNumHeavyAtoms())
    return Chem.MolToSmiles(heaviest)


def load_dataset(
    path: Path | None = None,
    cache: bool = True,
    dataset: str | DatasetSpec = DEFAULT_DATASET,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the curated dataset, writing a copy to ``data/processed`` for the record."""
    spec = get_spec(dataset)
    curated, counts = curate(load_raw(path, spec), spec)
    if cache:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        curated.to_csv(spec.processed_path, index=False)
    LOGGER.info("curated %s: %s", spec.name, counts)
    return curated, counts
