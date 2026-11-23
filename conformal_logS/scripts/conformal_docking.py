#!/usr/bin/env python3
"""Calibrated intervals for V-SYNTHES2 hit expansion.

V-SYNTHES2 decides which MEL fragments to enumerate and re-dock using the
fragment's own docking score, before the enumerated molecule's score exists.
This script measures how large that gap is in a given output file, and fits
calibrated prediction intervals for the final score from information that is
available *before* the expensive docking run.

    # how far the fragment score misses the final score
    python scripts/conformal_docking.py path/to/output.sdf --gap-only

    # calibrated 90% intervals for the final score
    python scripts/conformal_docking.py path/to/output.sdf \
        --method mondrian_conformal --alpha 0.10 --csv docking_intervals.csv

The repository's own example SDFs hold ten molecules, which is below the
minimum any conformal interval can be built from, so this reports what it
needs rather than producing a number. Point it at a V-SYNTHES2 run over a
real REAL-space subset (see the main README, Procedure 1) to use it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conformal_solubility.config import KNOWN_METHODS, Config
from conformal_solubility.vsynthes import (
    baseline_fragment_score_gap,
    conformal_docking_intervals,
    feature_columns,
    load_sdf,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sdf", type=Path, help="a V-SYNTHES2 output SDF")
    parser.add_argument(
        "--target",
        default="Score",
        help="the SDF field to predict (default: Score, the final docking score)",
    )
    parser.add_argument(
        "--method",
        default="split_conformal",
        choices=sorted(KNOWN_METHODS),
        help="interval method (default: split_conformal)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="miscoverage level; 0.10 gives 90%% intervals (default: 0.10)",
    )
    parser.add_argument("--seed", type=int, default=0, help="split and forest seed")
    parser.add_argument(
        "--gap-only",
        action="store_true",
        help="only report the fragment-to-final score gap, fit nothing",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="write per-molecule intervals here"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    docking = load_sdf(args.sdf)
    print(f"Read {len(docking)} molecules from {args.sdf}")

    try:
        gap = baseline_fragment_score_gap(docking)
    except ValueError as exc:
        print(f"\nFragment-to-final gap unavailable: {exc}")
    else:
        print("\n## Fragment score vs final score (the uncalibrated status quo)\n")
        print(f"  molecules with both scores : {gap['n']}")
        print(
            f"  gap (fragment - final)     : mean {gap['gap_mean']:+.2f}, "
            f"sd {gap['gap_std']:.2f}, range [{gap['gap_min']:+.2f}, {gap['gap_max']:+.2f}]"
        )
        print(
            f"  rank agreement (Spearman)  : {gap['spearman_fragment_vs_final']:.3f}"
        )
        print(
            "\n  A gap with a wide spread is the cost of deciding on the fragment\n"
            "  score alone; a rank agreement well below 1 means the ordering does\n"
            "  not survive enumeration."
        )

    if args.gap_only:
        return 0

    usable = feature_columns(docking.frame, target=args.target)
    print(f"\nFeature fields available before docking ({len(usable)}): {usable}")

    cfg = Config(alpha=args.alpha)
    try:
        outcome = conformal_docking_intervals(
            docking, cfg=cfg, target=args.target, method=args.method, seed=args.seed
        )
    except ValueError as exc:
        print(f"\nCannot fit intervals on this file:\n  {exc}")
        return 1

    print(
        f"\n## {outcome['method']} intervals for {outcome['target']} "
        f"(train {outcome['n_train']} / calibrate {outcome['n_calib']} / "
        f"test {outcome['n_test']})\n"
    )
    print(json.dumps(outcome["metrics"], indent=2))
    print(
        f"\n## Triage at the training-set median score "
        f"({outcome['screening_threshold']:.2f})\n"
    )
    print(json.dumps(outcome["screening"], indent=2))

    if args.csv:
        outcome["predictions"].to_csv(args.csv, index=False)
        print(f"\nPer-molecule intervals written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
