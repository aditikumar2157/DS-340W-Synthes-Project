#!/usr/bin/env python3
"""Run the conformal-prediction experiment end to end.

    python scripts/run_experiment.py                      # ESOL, 5 seeds
    python scripts/run_experiment.py --dataset aqsoldb    # AqSolDB, 5 seeds
    python scripts/run_experiment.py --quick              # 1 seed, 100 trees
    python scripts/run_experiment.py --alpha 0.2          # 80% intervals

Writes every table to ``results/`` and every figure to ``results/figures/``,
then prints the summary and the H1-H5 verdicts as markdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from conformal_solubility.config import KNOWN_DATASETS, Config, RESULTS_DIR
from conformal_solubility.experiment import run_experiment
from conformal_solubility.plots import make_all_figures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(KNOWN_DATASETS),
        default="esol",
        help="which solubility dataset to run on (default: esol)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="miscoverage level; 0.10 gives 90%% intervals (default: 0.10)",
    )
    parser.add_argument(
        "--screening-threshold",
        type=float,
        default=-4.0,
        help="LogS cut-off for the screening-triage analysis (default: -4.0)",
    )
    parser.add_argument(
        "--mondrian-bins",
        type=int,
        default=4,
        help="novelty bins Mondrian conformal calibrates within (default: 4)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="one repeat of the whole pipeline per seed",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=500, help="trees in the Random Forest"
    )
    parser.add_argument(
        "--novelty-bins",
        type=int,
        default=4,
        help="equal-count similarity bins for the H3 analysis",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="where tables and figures are written",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one seed and 100 trees, for a fast smoke run",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip plotting")
    return parser.parse_args(argv)


def markdown_table(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    """Render selected columns as a markdown table, renaming as we go."""
    view = frame[list(columns)].rename(columns=columns)
    return view.to_markdown(index=False, floatfmt=".3f")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    seeds = (0,) if args.quick else tuple(args.seeds)
    cfg = Config(
        dataset_name=args.dataset,
        alpha=args.alpha,
        seeds=seeds,
        n_estimators=100 if args.quick else args.n_estimators,
        novelty_bins=args.novelty_bins,
        mondrian_bins=args.mondrian_bins,
        screening_threshold=args.screening_threshold,
    )
    results = run_experiment(cfg, results_dir=args.results_dir)

    print(
        "\n## Headline results — %s, mean over %d seed(s)\n"
        % (args.dataset, len(seeds))
    )
    print(
        markdown_table(
            results["summary"],
            {
                "split_type": "Split",
                "method": "Method",
                "mae_mean": "MAE",
                "rmse_mean": "RMSE",
                "picp_mean": "PICP",
                "mpiw_mean": "MPIW",
                "coverage_gap_mean": "Coverage gap",
                "interval_score_mean": "Interval score",
            },
        )
    )

    print("\n## Coverage by structural novelty (Q1 = least familiar)\n")
    print(
        markdown_table(
            results["novelty_summary"],
            {
                "split_type": "Split",
                "method": "Method",
                "novelty_bin": "Quartile",
                "similarity_mean": "Mean Tanimoto",
                "picp": "PICP",
                "mpiw": "MPIW",
                "rmse": "RMSE",
            },
        )
    )

    print("\n## Conditional coverage across novelty quartiles (H4)\n")
    print(
        markdown_table(
            results["conditional_summary"],
            {
                "split_type": "Split",
                "method": "Method",
                "least_familiar_picp_mean": "Q1 PICP",
                "most_familiar_picp_mean": "Q4 PICP",
                "worst_bin_picp_mean": "Worst bin",
                "conditional_gap_mean": "Worst-bin gap",
                "novelty_spread_mean": "Q4 - Q1",
            },
        )
    )

    print(
        "\n## Screening triage at LogS >= %g (H5)\n" % cfg.screening_threshold
    )
    print(
        markdown_table(
            results["screening_summary"],
            {
                "split_type": "Split",
                "method": "Method",
                "decisive_fraction_mean": "Called",
                "decisive_error_rate_mean": "Error among calls",
                "false_accept_rate_mean": "False accepts",
                "point_error_rate_mean": "Point-call error",
            },
        )
    )

    print("\n## Hypotheses\n")
    verdicts = results["verdicts"].copy()
    verdicts["supported"] = verdicts["supported"].map({True: "yes", False: "no"})
    print(
        markdown_table(
            verdicts,
            {
                "hypothesis": "H",
                "condition": "Condition",
                "baseline": "Reference",
                "comparison": "Comparison",
                "supported": "Supported",
            },
        )
    )

    if not args.no_figures:
        figures = make_all_figures(results, out_dir=args.results_dir / "figures")
        print("\nFigures written:")
        for path in figures:
            print(f"  {path}")

    print(f"\nTables written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
