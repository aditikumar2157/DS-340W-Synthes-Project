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
from .conformal import (
    NOVELTY_AWARE_METHODS,
    QUANTILE_FOREST_METHODS,
    build_method,
    difficulty,
    make_model,
)
from .data import get_spec, load_dataset
from .features import featurize
from .metrics import summarize
from .novelty import binned_reliability, max_train_similarity
from .quantile_forest import QuantileForestReadout
from .screening import screening_summary
from .splits import make_split

LOGGER = logging.getLogger(__name__)

HYPOTHESIS_NOTES = {
    "H1": "conformal calibration lands closer to 90% coverage than the uncalibrated baseline",
    "H2": "point accuracy is worse under scaffold splitting than random splitting",
    "H3": "coverage degrades for structurally unfamiliar molecules",
    "H4": "calibrating within novelty bins (Mondrian) narrows the gap between "
    "the least- and most-familiar quartiles",
    "H5": "calibrated intervals make fewer confident mistakes than a point "
    "prediction when used to accept or reject screening candidates",
}


def _dataset_bundle(cfg: Config) -> dict:
    """Load the data and featurize once for the whole sweep."""
    frame, counts = load_dataset(dataset=cfg.dataset_name)
    smiles = frame["smiles"].tolist()
    X, descriptor_names, fingerprints = featurize(
        smiles, morgan_radius=cfg.morgan_radius, morgan_bits=cfg.morgan_bits
    )
    y = frame["logS"].to_numpy(dtype=float)
    LOGGER.info(
        "dataset %s ready: %d molecules, %d descriptors",
        cfg.dataset_name,
        len(smiles),
        len(descriptor_names),
    )
    return {
        "frame": frame,
        "smiles": smiles,
        "X": X,
        "y": y,
        "fingerprints": fingerprints,
        "descriptor_names": descriptor_names,
        "curation_counts": counts,
        "dataset": get_spec(cfg.dataset_name),
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
    train_fps = [fps[i] for i in train_idx]
    # Novelty is measured against the *training* molecules for both the test
    # and the calibration set. Mondrian conformal needs the calibration side
    # too: its bins have to be defined on molecules whose labels it may look
    # at, and then applied unchanged to molecules whose labels it may not.
    similarity = max_train_similarity([fps[i] for i in test_idx], train_fps)
    calib_similarity = max_train_similarity([fps[i] for i in calib_idx], train_fps)

    # One quantile read-out of the fitted forest, shared by every method that
    # needs it. Building it is the expensive part; the trees are not touched.
    readout = None
    if set(cfg.methods) & QUANTILE_FOREST_METHODS:
        readout = QuantileForestReadout(model, X[train_idx], y[train_idx])

    metric_rows: list[dict] = []
    novelty_rows: list[dict] = []
    prediction_rows: list[dict] = []
    screening_rows: list[dict] = []
    mondrian_rows: list[dict] = []

    for method_name in cfg.methods:
        method = build_method(method_name, model, cfg, quantile_readout=readout)
        needs_novelty = method_name in NOVELTY_AWARE_METHODS
        method.calibrate(
            X[calib_idx],
            y[calib_idx],
            similarity=calib_similarity if needs_novelty else None,
        )
        result = method.predict(
            X[test_idx], similarity=similarity if needs_novelty else None
        )

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
            mean_calib_similarity=float(calib_similarity.mean()),
        )
        metric_rows.append(stats)

        screening = screening_summary(
            y[test_idx],
            result.prediction,
            result.lower,
            result.upper,
            cfg.screening_threshold,
        )
        screening.update(split_type=split_type, seed=seed, method=method_name)
        screening_rows.append(screening)

        if needs_novelty:
            diagnostics = method.diagnostics()
            mondrian_rows.append(
                {
                    "split_type": split_type,
                    "seed": seed,
                    "method": method_name,
                    **{key: str(value) for key, value in diagnostics.items()},
                    "n_pooled_fallback_bins": len(diagnostics["pooled_fallback_bins"]),
                }
            )

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
        "screening": screening_rows,
        "mondrian": mondrian_rows,
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


def conditional_coverage(novelty: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-seed summaries of how *unevenly* coverage is spread across novelty.

    Marginal PICP is one number for the whole test set; these are the numbers
    it averages away. Note that the bins here are the evaluation quartiles
    computed on the test set, which are deliberately *not* the bins Mondrian
    calibrates within -- scoring a method on its own partition would flatter
    it. Every method is judged on the same fixed partition.

    ``worst_bin_picp``
        The least-covered quartile. The guarantee a screening chemist
        experiences is the one that holds for the molecules they actually
        look at, not the average over everything.
    ``conditional_gap``
        The largest absolute distance from the target across quartiles -- the
        single number H4 is about.
    ``novelty_spread``
        Coverage in the most-familiar quartile minus the least-familiar. Zero
        means reliability does not depend on familiarity, which is the goal.
    """
    target = cfg.target_coverage
    rows = []
    for (split_type, method, seed), group in novelty.groupby(
        ["split_type", "method", "seed"], sort=False
    ):
        group = group.sort_values("novelty_bin")
        picp_values = group["picp"].to_numpy(dtype=float)
        rows.append(
            {
                "split_type": split_type,
                "method": method,
                "seed": seed,
                "worst_bin_picp": float(picp_values.min()),
                "conditional_gap": float(np.max(np.abs(picp_values - target))),
                "novelty_spread": float(picp_values[-1] - picp_values[0]),
                "least_familiar_picp": float(picp_values[0]),
                "most_familiar_picp": float(picp_values[-1]),
                "mpiw": float(group["mpiw"].mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_conditional(conditional: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds of the conditional summaries."""
    value_columns = [
        "worst_bin_picp",
        "conditional_gap",
        "novelty_spread",
        "least_familiar_picp",
        "most_familiar_picp",
        "mpiw",
    ]
    grouped = conditional.groupby(["split_type", "method"], sort=False)[value_columns]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def evaluate_h4(conditional_summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """H4: does calibrating within novelty bins even coverage out?

    The comparison is each Mondrian variant against the plain method it is
    built from -- ``mondrian_conformal`` against ``split_conformal``, and
    ``mondrian_normalized`` against ``normalized_conformal``. Both members of
    a pair share a fitted forest and a calibration set, so the only thing that
    differs is whether the calibration quantile was taken within novelty bins
    or over all of them at once.
    """
    pairs = [
        ("mondrian_conformal", "split_conformal"),
        ("mondrian_normalized", "normalized_conformal"),
    ]
    indexed = conditional_summary.set_index(["split_type", "method"])
    rows = []
    for split_type in cfg.split_types:
        for mondrian, plain in pairs:
            if mondrian not in cfg.methods or plain not in cfg.methods:
                continue
            try:
                grouped = indexed.loc[(split_type, mondrian)]
                pooled = indexed.loc[(split_type, plain)]
            except KeyError:
                continue
            rows.append(
                {
                    "hypothesis": "H4",
                    "claim": HYPOTHESIS_NOTES["H4"],
                    "condition": f"{split_type} / {mondrian} vs {plain}",
                    "baseline": f"pooled worst-bin PICP {pooled['worst_bin_picp_mean']:.3f} "
                    f"(cond. gap {pooled['conditional_gap_mean']:.3f})",
                    "comparison": f"binned worst-bin PICP {grouped['worst_bin_picp_mean']:.3f} "
                    f"(cond. gap {grouped['conditional_gap_mean']:.3f})",
                    "supported": bool(
                        grouped["conditional_gap_mean"] < pooled["conditional_gap_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_screening(screening: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds of the triage outcomes."""
    value_columns = [
        "decisive_fraction",
        "decisive_error_rate",
        "defer_fraction",
        "false_accept_rate",
        "point_error_rate",
        "fraction_truly_above",
    ]
    grouped = screening.groupby(["split_type", "method"], sort=False)[value_columns]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def evaluate_h5(screening_summary_frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """H5: do the intervals buy a lower confident-error rate than a point call?

    The point predictor is decisive on every molecule, so it is the harder
    comparison to win on volume and the easier one to win on error rate. The
    verdict asks only whether a confident call from the interval is more
    trustworthy than a call from the point prediction; the deferral rate is
    reported next to it so the trade is visible rather than implied.
    """
    rows = []
    for _, row in screening_summary_frame.iterrows():
        if row["method"] == "uncalibrated_gaussian":
            continue
        rows.append(
            {
                "hypothesis": "H5",
                "claim": HYPOTHESIS_NOTES["H5"],
                "condition": f"{row['split_type']} / {row['method']} "
                f"(LogS >= {cfg.screening_threshold:g})",
                "baseline": f"point call error {row['point_error_rate_mean']:.3f} "
                f"on 100% of molecules",
                "comparison": f"interval call error {row['decisive_error_rate_mean']:.3f} "
                f"on {row['decisive_fraction_mean']:.0%} of molecules",
                "supported": bool(
                    row["decisive_error_rate_mean"] < row["point_error_rate_mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


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
    screening_rows: list[dict] = []
    mondrian_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_type in cfg.split_types:
        for seed in cfg.seeds:
            LOGGER.info("running %s split, seed %d", split_type, seed)
            outcome = run_single(bundle, cfg, split_type, seed)
            metric_rows.extend(outcome["metrics"])
            novelty_rows.extend(outcome["novelty"])
            screening_rows.extend(outcome["screening"])
            mondrian_rows.extend(outcome["mondrian"])
            prediction_frames.append(outcome["predictions"])

    metrics = pd.DataFrame(metric_rows)
    novelty = pd.DataFrame(novelty_rows)
    screening = pd.DataFrame(screening_rows)
    mondrian = pd.DataFrame(mondrian_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    summary = aggregate(metrics)
    conditional = conditional_coverage(novelty, cfg)
    conditional_summary = aggregate_conditional(conditional)
    screening_summary_frame = aggregate_screening(screening)

    hypotheses_12 = evaluate_hypotheses(summary, cfg)
    hypothesis_3, novelty_summary = evaluate_h3(novelty, cfg)
    hypothesis_4 = evaluate_h4(conditional_summary, cfg)
    hypothesis_5 = evaluate_h5(screening_summary_frame, cfg)
    verdicts = pd.concat(
        [hypotheses_12, hypothesis_3, hypothesis_4, hypothesis_5], ignore_index=True
    )

    metrics.to_csv(results_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    novelty.to_csv(results_dir / "novelty_by_seed.csv", index=False)
    novelty_summary.to_csv(results_dir / "novelty_summary.csv", index=False)
    conditional.to_csv(results_dir / "conditional_by_seed.csv", index=False)
    conditional_summary.to_csv(results_dir / "conditional_summary.csv", index=False)
    screening.to_csv(results_dir / "screening_by_seed.csv", index=False)
    screening_summary_frame.to_csv(results_dir / "screening_summary.csv", index=False)
    if not mondrian.empty:
        mondrian.to_csv(results_dir / "mondrian_diagnostics.csv", index=False)
    verdicts.to_csv(results_dir / "hypothesis_verdicts.csv", index=False)
    predictions.to_csv(results_dir / "test_predictions.csv.gz", index=False)
    cfg.to_json(results_dir / "config.json")
    pd.Series(environment_record()).to_json(results_dir / "environment.json", indent=2)
    pd.Series(bundle["curation_counts"]).to_json(
        results_dir / "curation_counts.json", indent=2
    )

    return {
        "config": asdict(cfg),
        "dataset": bundle["dataset"],
        "metrics": metrics,
        "summary": summary,
        "novelty": novelty,
        "novelty_summary": novelty_summary,
        "conditional": conditional,
        "conditional_summary": conditional_summary,
        "screening": screening,
        "screening_summary": screening_summary_frame,
        "mondrian_diagnostics": mondrian,
        "verdicts": verdicts,
        "predictions": predictions,
        "curation_counts": bundle["curation_counts"],
    }
