"""Calibrated intervals for the decision V-SYNTHES2 actually makes blind.

The parent pipeline is hierarchical. It docks a small MEL fragment, keeps the
fragments that score well, enumerates those into full molecules, and re-docks
the result. The expensive step is the re-docking, and the decision of which
fragments to carry into it is made on the *fragment's* score -- before anyone
knows what the enumerated molecule will score.

That gap is the uncertainty V-SYNTHES2 has no account of. Its own output
files carry both numbers side by side: ``orig_Score`` is what the MEL fragment
scored, ``Score`` is what the enumerated molecule scored. The difference
between them is the error in the decision, and it is exactly the quantity a
calibrated interval can put bounds on.

So this module asks the screening question in the project's own terms:

    given a fragment's score and the cheap properties of the molecule it
    would enumerate into, what range will the final docking score fall in,
    and how confident can we be before spending the CPU time?

and then answers it with the same conformal machinery the solubility
experiment uses, so the calibration claim is the one already tested.

Leakage is the hazard here
--------------------------
An SDF written after docking carries fields computed *from the docked pose*:
the score itself, its components (van der Waals, solvation, entropy terms),
and rescoring values. Feeding any of those to a model that predicts the score
would produce a spectacular result and mean nothing, because at the moment
the decision is made none of them exist. :data:`POSE_DERIVED_FIELDS` names
them and :func:`feature_columns` refuses to return them, so the mistake has
to be made deliberately rather than by accident.

What is legitimately available before docking: the fragment's own score, and
anything computable from the enumerated structure on a laptop -- molecular
weight, LogP, atom and rotatable-bond counts, RDKit descriptors. Enumeration
is cheap; docking is not.

Status
------
The example SDFs shipped with this repository hold ten molecules each, and
``RMSD_assessment/RMSD/RMSD_MP.sdf`` is an empty placeholder, so there is no
dataset here to run this on: split conformal needs at least
``ceil(1/alpha) - 1`` calibration molecules before a 1-alpha interval exists
at all, which is 19 for the 95% case and 9 for 90%, on top of a training set.
:func:`check_sufficient_data` says so in those terms rather than returning a
confident-looking number from nine points. Running it for real needs a
V-SYNTHES2 output file from a licensed ICM-Pro+VLS installation, which the
repository's main README describes how to produce.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .config import Config
from .conformal import SplitConformalInterval, build_method, make_model
from .metrics import summarize
from .novelty import max_train_similarity
from .quantile_forest import QuantileForestReadout
from .screening import screening_summary

LOGGER = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.*")

#: The final docking score of the enumerated molecule: the target.
TARGET_FIELD = "Score"

#: The MEL fragment's score -- the number V-SYNTHES2 selects on.
FRAGMENT_SCORE_FIELD = "orig_Score"

#: Fields computed from the docked pose, or from the score itself. Available
#: only *after* paying the cost the prediction is meant to avoid, so none of
#: them may be a feature. Listed by name rather than detected, because a
#: silent leak produces a better-looking result than an honest model and
#: nothing in the metrics would reveal it.
POSE_DERIVED_FIELDS = frozenset(
    {
        "Score",
        "RTCNNscore",
        "mfScore",
        "Dsolv",
        "Eintl",
        "Hbond",
        "Hphob",
        "VwInt",
        "SolEl",
        "dTSsc",
        "CONF",
        "RecConf",
        "IX",
        "L",
        "O",
        "OBJECTS",
    }
)


@dataclass
class DockingSet:
    """Properties and structures read out of a V-SYNTHES2 SDF."""

    frame: pd.DataFrame
    smiles: list[str]
    path: str

    def __len__(self) -> int:
        return len(self.frame)


def load_sdf(path, sanitize: bool = True) -> DockingSet:
    """Read an SDF's property block into a frame, keeping record order.

    Records RDKit cannot parse are skipped rather than aborting the read -- a
    docking output can contain an awkward structure without the rest of the
    file being unusable -- and the number skipped is logged.
    """
    path = str(path)
    supplier = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=False)
    rows: list[dict] = []
    smiles: list[str] = []
    skipped = 0
    for mol in supplier:
        if mol is None:
            skipped += 1
            continue
        rows.append(mol.GetPropsAsDict())
        smiles.append(Chem.MolToSmiles(mol))
    if skipped:
        LOGGER.warning("%s: skipped %d unparseable record(s)", path, skipped)
    if not rows:
        raise ValueError(f"{path} contained no readable records")
    return DockingSet(frame=pd.DataFrame(rows), smiles=smiles, path=path)


def feature_columns(
    frame: pd.DataFrame, target: str = TARGET_FIELD, extra_exclusions=()
) -> list[str]:
    """Numeric fields usable as features, with pose-derived ones removed.

    The fragment score is kept: it is known before the expensive docking run
    and it is the number the parent pipeline already decides on, so a model
    that could not use it would not be describing V-SYNTHES2's decision.
    """
    excluded = set(POSE_DERIVED_FIELDS) | {target} | set(extra_exclusions)
    excluded.discard(FRAGMENT_SCORE_FIELD)
    usable = []
    for name in frame.columns:
        if name in excluded:
            continue
        column = pd.to_numeric(frame[name], errors="coerce")
        if column.notna().sum() < len(frame) / 2:
            continue  # mostly text or mostly missing
        if column.nunique(dropna=True) < 2:
            continue  # constant: carries nothing and upsets the imputer
        usable.append(name)
    return usable


def minimum_calibration_size(alpha: float) -> int:
    """Smallest calibration set for which a 1-alpha interval exists.

    ``ceil((n + 1)(1 - alpha)) <= n`` rearranges to ``n >= ceil(1/alpha) - 1``.
    Below it there is no k-th smallest residual to take and no finite width
    can be certified -- the method does not degrade gracefully, it stops
    applying.
    """
    return int(np.ceil(1.0 / alpha)) - 1


def check_sufficient_data(n_records: int, alpha: float, calib_frac: float) -> None:
    """Raise with an explanation when the file is too small to calibrate on."""
    needed_calib = minimum_calibration_size(alpha)
    needed_total = int(np.ceil(needed_calib / calib_frac)) + 2
    if n_records < needed_total:
        raise ValueError(
            f"{n_records} molecules is too few to calibrate a "
            f"{1 - alpha:.0%} interval: with calib_frac={calib_frac:g} that "
            f"leaves about {int(n_records * calib_frac)} calibration "
            f"molecules, and the finite-sample quantile needs at least "
            f"{needed_calib} (so roughly {needed_total} molecules in total, "
            f"before leaving any for training or testing). Run V-SYNTHES2 on "
            f"a real REAL-space subset and point this at its output, or raise "
            f"alpha if a wider miscoverage level is acceptable."
        )


def expansion_gap(docking: DockingSet) -> pd.DataFrame:
    """Fragment score, final score, and the error between them.

    This is the quantity the whole module exists to bound. A positive gap
    means the enumerated molecule scored *worse* than its fragment promised.
    """
    frame = docking.frame
    for field in (FRAGMENT_SCORE_FIELD, TARGET_FIELD):
        if field not in frame.columns:
            raise ValueError(
                f"{docking.path} has no {field!r} field; this analysis needs "
                f"both {FRAGMENT_SCORE_FIELD!r} (the MEL fragment's score) and "
                f"{TARGET_FIELD!r} (the enumerated molecule's score)"
            )
    fragment = pd.to_numeric(frame[FRAGMENT_SCORE_FIELD], errors="coerce")
    final = pd.to_numeric(frame[TARGET_FIELD], errors="coerce")
    return pd.DataFrame(
        {
            "fragment_score": fragment,
            "final_score": final,
            "gap": fragment - final,
        }
    )


def conformal_docking_intervals(
    docking: DockingSet,
    cfg: Config | None = None,
    target: str = TARGET_FIELD,
    method: str = "split_conformal",
    seed: int = 0,
) -> dict:
    """Fit, calibrate and evaluate calibrated intervals on a docking set.

    The split is random, because a docking output is one campaign against one
    target and there is no second campaign to hold out; the scaffold-split
    stress test of the solubility experiment has no counterpart here, and
    claiming otherwise would overstate what this can show.

    Returns the per-molecule intervals, the usual interval metrics, and the
    screening-triage summary at the median final score -- "which half of this
    batch is worth docking" being the decision in its own terms.
    """
    cfg = cfg or Config()
    check_sufficient_data(len(docking), cfg.alpha, cfg.calib_frac)

    columns = feature_columns(docking.frame, target=target)
    if not columns:
        raise ValueError(
            f"no usable feature columns in {docking.path}: every numeric field "
            f"was constant, mostly missing, or derived from the docked pose"
        )
    X = (
        docking.frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    y = pd.to_numeric(docking.frame[target], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    smiles = [s for s, k in zip(docking.smiles, keep) if k]
    check_sufficient_data(len(y), cfg.alpha, cfg.calib_frac)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_train = int(round(cfg.train_frac * len(y)))
    n_calib = int(round(cfg.calib_frac * len(y)))
    n_calib = min(n_calib, len(y) - n_train - 1)
    train_idx = order[:n_train]
    calib_idx = order[n_train : n_train + n_calib]
    test_idx = order[n_train + n_calib :]

    model = make_model(cfg, seed)
    model.fit(X[train_idx], y[train_idx])

    readout = None
    if method in {"cqr", "normalized_conformal_qrf"}:
        readout = QuantileForestReadout(model, X[train_idx], y[train_idx])

    similarity = None
    if method.startswith("mondrian"):
        from .features import morgan_fingerprints

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            raise ValueError("a structure failed to re-parse for fingerprinting")
        fps = morgan_fingerprints(mols, cfg.morgan_radius, cfg.morgan_bits)
        similarity = {
            "calib": max_train_similarity(
                [fps[i] for i in calib_idx], [fps[i] for i in train_idx]
            ),
            "test": max_train_similarity(
                [fps[i] for i in test_idx], [fps[i] for i in train_idx]
            ),
        }

    interval = build_method(method, model, cfg, quantile_readout=readout)
    interval.calibrate(
        X[calib_idx],
        y[calib_idx],
        similarity=None if similarity is None else similarity["calib"],
    )
    result = interval.predict(
        X[test_idx], similarity=None if similarity is None else similarity["test"]
    )

    metrics = summarize(
        y[test_idx], result.prediction, result.lower, result.upper, cfg.alpha
    )
    # "Worth docking" = in the better-scoring half of the batch. Defined on
    # the training fold so the threshold is not set using test labels.
    threshold = float(np.median(y[train_idx]))
    triage = screening_summary(
        y[test_idx], result.prediction, result.lower, result.upper, threshold
    )

    predictions = pd.DataFrame(
        {
            "smiles": [smiles[i] for i in test_idx],
            "y_true": y[test_idx],
            "y_pred": result.prediction,
            "lower": result.lower,
            "upper": result.upper,
            "half_width": result.scale,
        }
    )
    return {
        "method": method,
        "target": target,
        "feature_columns": columns,
        "n_train": len(train_idx),
        "n_calib": len(calib_idx),
        "n_test": len(test_idx),
        "metrics": metrics,
        "screening": triage,
        "screening_threshold": threshold,
        "predictions": predictions,
    }


def baseline_fragment_score_gap(docking: DockingSet) -> dict[str, float]:
    """How bad the uncalibrated status quo is, in one dictionary.

    V-SYNTHES2 ranks by the fragment score and carries the top fragments
    forward. If the fragment score were a perfect proxy the gap would be
    constant and the ranking would survive enumeration intact. Reporting its
    spread gives the conformal intervals something concrete to be compared
    against.
    """
    gap = expansion_gap(docking).dropna()
    if gap.empty:
        raise ValueError("no records carried both a fragment and a final score")
    correlation = (
        float(gap["fragment_score"].corr(gap["final_score"], method="spearman"))
        if len(gap) > 2
        else float("nan")
    )
    return {
        "n": int(len(gap)),
        "gap_mean": float(gap["gap"].mean()),
        "gap_std": float(gap["gap"].std(ddof=1)) if len(gap) > 1 else float("nan"),
        "gap_min": float(gap["gap"].min()),
        "gap_max": float(gap["gap"].max()),
        "spearman_fragment_vs_final": correlation,
    }
