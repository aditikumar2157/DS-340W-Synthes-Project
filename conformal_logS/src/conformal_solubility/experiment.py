"""The experiment runner.

One call to :func:`run_experiment` produces every number the project reports:
the headline metrics per split type and interval method, the same metrics
broken down by structural novelty, and the per-molecule predictions for the
figures. Featurization happens once and is reused across every split and
seed, which is what makes a five-seed sweep cheap.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RESULTS_DIR
from .conformal import build_method, difficulty, make_model
from .data import load_dataset
from .features import featurize
from .metrics import summarize
from .novelty import binned_reliability, max_train_similarity
from .splits import make_split

LOGGER = logging.getLogger(__name__)

HYPOTHESIS_NOTES = {
    "H1": "conformal calibration lands closer to 90% coverage than the uncalibrated baseline",
    "H2": "point accuracy is worse under scaffold splitting than random splitting",
    "H3": "coverage degrades for structurally unfamiliar molecules",
}


def _dataset_bundle(cfg: Config) -> dict:
    """Load the data and featurize once for the whole sweep."""
    frame, counts = load_dataset()
    smiles = frame["smiles"].tolist()
    X, descriptor_names, fingerprints = featurize(
        smiles, morgan_radius=cfg.morgan_radius, morgan_bits=cfg.morgan_bits
    )
    y = frame["logS"].to_numpy(dtype=float)
    LOGGER.info(
        "dataset ready: %d molecules, %d descriptors", len(smiles), len(descriptor_names)
    )
    return {
        "frame": frame,
        "smiles": smiles,
        "X": X,
        "y": y,
        "fingerprints": fingerprints,
        "descriptor_names": descriptor_names,
        "curation_counts": counts,
    }


def run_single(bundle: dict, cfg: Config, split_type: str, seed: int) -> dict:
    """One split, one seed: fit, calibrate, evaluate every method.

    The model is fitted once and shared by all three interval methods, so the
    only thing that varies between them is how the interval is formed.
    """
    X, y = bundle["X"], bundle["y"]
    train_idx, calib_idx, test_idx = make_split(
        split_type, bundle["smiles"], cfg.train_frac, cfg.calib_frac, seed
    )
    overlap = (
        set(train_idx.tolist()) & set(calib_idx.tolist())
        | set(train_idx.tolist()) & set(test_idx.tolist())
        | set(calib_idx.tolist()) & set(test_idx.tolist())
    )
    if overlap:
        raise AssertionError(f"split {split_type}/{seed} leaked {len(overlap)} indices")

    model = make_model(cfg, seed)
    model.fit(X[train_idx], y[train_idx])

    fps = bundle["fingerprints"]
    similarity = max_train_similarity(
        [fps[i] for i in test_idx], [fps[i] for i in train_idx]
    )

    metric_rows: list[dict] = []
    novelty_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for method_name in cfg.methods:
        method = build_method(method_name, model, cfg)
        method.calibrate(X[calib_idx], y[calib_idx])
        result = method.predict(X[test_idx])

        stats = summarize(
            y[test_idx], result.prediction, result.lower, result.upper, cfg.alpha
        )
        stats.update(
            split_type=split_type,
            seed=seed,
            method=method_name,
            n_train=len(train_idx),
            n_calib=len(calib_idx),
            n_test=len(test_idx),
            mean_train_similarity=float(similarity.mean()),
        )
        metric_rows.append(stats)

        binned = binned_reliability(
            y[test_idx],
            result.lower,
            result.upper,
            result.prediction,
            similarity,
            cfg.alpha,
            n_bins=cfg.novelty_bins,
        )
        binned["split_type"] = split_type
        binned["seed"] = seed
        binned["method"] = method_name
        novelty_rows.extend(binned.to_dict("records"))

        prediction_rows.append(
            pd.DataFrame(
                {
                    "split_type": split_type,
                    "seed": seed,
                    "method": method_name,
                    "smiles": [bundle["smiles"][i] for i in test_idx],
                    "y_true": y[test_idx],
                    "y_pred": result.prediction,
                    "lower": result.lower,
                    "upper": result.upper,
                    "half_width": result.scale,
                    "max_train_similarity": similarity,
                    "tree_std": difficulty(model, X[test_idx]),
                }
            )
        )

    return {
        "metrics": metric_rows,
        "novelty": novelty_rows,
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, one row per split x method."""
    value_columns = [
        "mae",
        "rmse",
        "picp",
        "mpiw",
        "median_piw",
        "coverage_gap",
        "abs_coverage_gap",
        "interval_score",
    ]
    grouped = metrics.groupby(["split_type", "method"], sort=False)[value_columns]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def evaluate_hypotheses(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Turn the summary table into an explicit verdict on H1-H3.

    Each verdict is a comparison between rows the sweep already produced; the
    point is to state the claim and the number side by side instead of
    leaving the reader to eyeball the table.
    """
    indexed = summary.set_index(["split_type", "method"])
    rows = []

    # H1: calibration closes the coverage gap, in both split regimes.
    for split_type in cfg.split_types:
        if ("uncalibrated_gaussian" not in indexed.loc[split_type].index) or (
            "split_conformal" not in indexed.loc[split_type].index
        ):
            continue
        baseline = indexed.loc[(split_type, "uncalibrated_gaussian")]
        conformal = indexed.loc[(split_type, "split_conformal")]
        rows.append(
            {
                "hypothesis": "H1",
                "claim": HYPOTHESIS_NOTES["H1"],
                "condition": f"{split_type} split",
                "baseline": f"PICP {baseline['picp_mean']:.3f} "
                f"(gap {baseline['coverage_gap_mean']:+.3f})",
                "comparison": f"PICP {conformal['picp_mean']:.3f} "
                f"(gap {conformal['coverage_gap_mean']:+.3f})",
                "supported": bool(
                    conformal["abs_coverage_gap_mean"] < baseline["abs_coverage_gap_mean"]
                ),
            }
        )

    # H2: scaffold splitting costs accuracy. All three interval methods share
    # one fitted model, so accuracy depends on the split alone -- read it off
    # whichever method is present and report it once per scaffold regime.
    reference_method = cfg.methods[0] if cfg.methods else None
    for split_type in cfg.split_types:
        if split_type == "random" or reference_method is None:
            continue
        try:
            rnd = indexed.loc[("random", reference_method)]
            scf = indexed.loc[(split_type, reference_method)]
        except KeyError:
            continue
        rows.append(
            {
                "hypothesis": "H2",
                "claim": HYPOTHESIS_NOTES["H2"],
                "condition": f"{split_type} vs random",
                "baseline": f"random RMSE {rnd['rmse_mean']:.3f}",
                "comparison": f"{split_type} RMSE {scf['rmse_mean']:.3f}",
                "supported": bool(scf["rmse_mean"] > rnd["rmse_mean"]),
            }
        )

    # H1*: does calibrating on molecules that are themselves unfamiliar
    # recover the coverage that the plain scaffold regime loses? This is the
    # question the third split regime was added to answer.
    if {"scaffold", "scaffold_ood_calib"} <= set(cfg.split_types):
        for method in ("split_conformal", "normalized_conformal"):
            try:
                naive = indexed.loc[("scaffold", method)]
                matched = indexed.loc[("scaffold_ood_calib", method)]
            except KeyError:
                continue
            rows.append(
                {
                    "hypothesis": "H1*",
                    "claim": "calibrating on unfamiliar molecules recovers coverage "
                    "lost under scaffold shift",
                    "condition": method,
                    "baseline": f"random calibration PICP {naive['picp_mean']:.3f} "
                    f"(gap {naive['coverage_gap_mean']:+.3f})",
                    "comparison": f"scaffold calibration PICP {matched['picp_mean']:.3f} "
                    f"(gap {matched['coverage_gap_mean']:+.3f})",
                    "supported": bool(
                        matched["abs_coverage_gap_mean"] < naive["abs_coverage_gap_mean"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_h3(novelty: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compare coverage in the least and most familiar novelty quartiles."""
    value_columns = ["picp", "mpiw", "mae", "rmse", "coverage_gap", "similarity_mean"]
    per_bin = (
        novelty.groupby(["split_type", "method", "novelty_bin"], sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    rows = []
    for (split_type, method), group in per_bin.groupby(
        ["split_type", "method"], sort=False
    ):
        group = group.sort_values("novelty_bin")
        least = group.iloc[0]   # Q1: least similar to training
        most = group.iloc[-1]   # Q4: most similar
        rows.append(
            {
                "hypothesis": "H3",
                "claim": HYPOTHESIS_NOTES["H3"],
                "condition": f"{split_type} split / {method}",
                "baseline": f"most familiar quartile PICP {most['picp']:.3f} "
                f"(sim {most['similarity_mean']:.2f})",
                "comparison": f"least familiar quartile PICP {least['picp']:.3f} "
                f"(sim {least['similarity_mean']:.2f})",
                "supported": bool(least["picp"] < most["picp"]),
            }
        )
    return pd.DataFrame(rows), per_bin


def environment_record() -> dict[str, str]:
    import sklearn
    import rdkit

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
    }


def run_experiment(cfg: Config, results_dir: Path | None = None) -> dict:
    """Run the full sweep and write every table to ``results/``."""
    cfg.validate()
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = _dataset_bundle(cfg)

    metric_rows: list[dict] = []
    novelty_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_type in cfg.split_types:
        for seed in cfg.seeds:
            LOGGER.info("running %s split, seed %d", split_type, seed)
            outcome = run_single(bundle, cfg, split_type, seed)
            metric_rows.extend(outcome["metrics"])
            novelty_rows.extend(outcome["novelty"])
            prediction_frames.append(outcome["predictions"])

    metrics = pd.DataFrame(metric_rows)
    novelty = pd.DataFrame(novelty_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = aggregate(metrics)
    hypotheses_12 = evaluate_hypotheses(summary, cfg)
    hypothesis_3, novelty_summary = evaluate_h3(novelty, cfg)
    verdicts = pd.concat([hypotheses_12, hypothesis_3], ignore_index=True)

    metrics.to_csv(results_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    novelty.to_csv(results_dir / "novelty_by_seed.csv", index=False)
    novelty_summary.to_csv(results_dir / "novelty_summary.csv", index=False)
    verdicts.to_csv(results_dir / "hypothesis_verdicts.csv", index=False)
    predictions.to_csv(results_dir / "test_predictions.csv.gz", index=False)
    cfg.to_json(results_dir / "config.json")
    pd.Series(environment_record()).to_json(results_dir / "environment.json", indent=2)
    pd.Series(bundle["curation_counts"]).to_json(
        results_dir / "curation_counts.json", indent=2
    )

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "summary": summary,
        "novelty": novelty,
        "novelty_summary": novelty_summary,
        "verdicts": verdicts,
        "predictions": predictions,
        "curation_counts": bundle["curation_counts"],
    }
