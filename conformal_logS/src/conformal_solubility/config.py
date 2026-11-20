"""Experiment configuration.

Every number that controls the experiment lives here so that a run is fully
described by one object, which is written alongside the results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# conformal_logS/src/conformal_solubility/config.py -> conformal_logS/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"


@dataclass
class Config:
    """Settings for one sweep of the experiment.

    Attributes
    ----------
    alpha:
        Miscoverage level. ``alpha = 0.10`` targets 90% prediction intervals,
        matching the midterm proposal.
    train_frac, calib_frac:
        Proportions of the dataset held for model fitting and for conformal
        calibration. The remainder is the test set. The calibration set is
        carved out of the same side of the split as the training data, so a
        scaffold split keeps test scaffolds unseen by both the model and the
        calibrator.
    seeds:
        One repeat of the whole pipeline per seed, so that every reported
        number comes with a spread rather than a single lucky draw.
    n_estimators, max_features, min_samples_leaf:
        Random Forest hyper-parameters. The forest needs enough trees for its
        per-tree spread to be a usable difficulty estimate, which the
        normalized conformal variant relies on.
    novelty_bins:
        Number of equal-count bins used to group test molecules by maximum
        Tanimoto similarity to the training set (the H3 analysis).
    """

    alpha: float = 0.10
    train_frac: float = 0.60
    calib_frac: float = 0.20
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    n_estimators: int = 500
    max_features: float | str = 0.3
    min_samples_leaf: int = 1
    n_jobs: int = -1

    # Difficulty estimates of exactly zero would make the normalized
    # non-conformity score blow up; beta keeps it finite.
    normalized_beta: float = 0.10

    morgan_radius: int = 2
    morgan_bits: int = 2048
    novelty_bins: int = 4

    split_types: tuple[str, ...] = ("random", "scaffold", "scaffold_ood_calib")
    methods: tuple[str, ...] = (
        "uncalibrated_gaussian",
        "split_conformal",
        "normalized_conformal",
    )

    dataset_name: str = "esol"
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.calib_frac

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.test_frac <= 0.0:
            raise ValueError(
                "train_frac + calib_frac must be < 1 so a test set remains; "
                f"got {self.train_frac} + {self.calib_frac}"
            )
        if not self.seeds:
            raise ValueError("at least one seed is required")
        unknown = set(self.methods) - {
            "uncalibrated_gaussian",
            "split_conformal",
            "normalized_conformal",
        }
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        unknown_splits = set(self.split_types) - {
            "random",
            "scaffold",
            "scaffold_ood_calib",
        }
        if unknown_splits:
            raise ValueError(f"unknown split types: {sorted(unknown_splits)}")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["split_types"] = list(self.split_types)
        payload["methods"] = list(self.methods)
        path.write_text(json.dumps(payload, indent=2) + "\n")
