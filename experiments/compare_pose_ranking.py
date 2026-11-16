#!/usr/bin/env python3
"""Compare parent and experimental rankings on RMSD-assessed SDF data.

Input: an SDF whose records contain <Score> and <RMSD> properties.  RMSD is
the pose-retention outcome from the repository's RMSD assessment workflow.
Lower docking scores and lower RMSD values are treated as better.

The experimental rankings are (1) score per heavy atom and (2) a Borda-style
rank fusion of docking score and heavy-atom count. This is an analysis-only
experiment, not a replacement for CapSelect or a validated pose-quality model.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def records_from_sdf(path: Path):
    for index, block in enumerate(path.read_text(errors="replace").split("$$$$"), 1):
        lines = block.splitlines()
        if len(lines) < 5 or not lines[3].strip():
            continue
        try:
            atom_count = int(lines[3][:3])
        except ValueError:
            continue
        props = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("> <") and line.endswith(">"):
                key = line[3:-1]
                i += 1
                values = []
                while i < len(lines) and lines[i].strip():
                    values.append(lines[i].strip())
                    i += 1
                props[key] = " ".join(values)
            i += 1
        try:
            score = float(props["Score"])
            rmsd = float(props["RMSD"])
        except (KeyError, ValueError):
            continue
        heavy_atoms = sum(
            1 for atom_line in lines[4 : 4 + atom_count]
            if len(atom_line) >= 34 and atom_line[31:34].strip().upper() not in {"H", "D"}
        )
        if heavy_atoms:
            yield {"record": index, "score": score, "rmsd": rmsd, "heavy_atoms": heavy_atoms}


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for j in range(start, end):
            ranks[order[j]] = rank
        start = end
    return ranks


def spearman(x, y):
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else float("nan")


def summarize(rows, key, label, top_fraction):
    ordered = sorted(rows, key=lambda row: row[key])
    top_n = max(1, math.ceil(len(ordered) * top_fraction))
    return {
        "ranking": label,
        "n": len(rows),
        "top_n": top_n,
        "top_median_rmsd": statistics.median(row["rmsd"] for row in ordered[:top_n]),
        "spearman_rank_vs_rmsd": spearman([row[key] for row in rows], [row["rmsd"] for row in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdf", type=Path, help="RMSD-assessed SDF with Score and RMSD fields")
    parser.add_argument("--top-fraction", type=float, default=0.10, help="fraction used for top-set RMSD (default: 0.10)")
    parser.add_argument("--csv", type=Path, default=Path("ranking_comparison.csv"), help="summary CSV path")
    args = parser.parse_args()
    if not 0 < args.top_fraction <= 1:
        parser.error("--top-fraction must be in (0, 1]")
    rows = list(records_from_sdf(args.sdf))
    if len(rows) < 2:
        parser.error("need at least two valid records with numeric <Score> and <RMSD> fields")
    for row in rows:
        row["score_per_heavy_atom"] = row["score"] / row["heavy_atoms"]
    score_ranks = average_ranks([row["score"] for row in rows])
    size_ranks = average_ranks([row["heavy_atoms"] for row in rows])
    for row, score_rank, size_rank in zip(rows, score_ranks, size_ranks):
        # Smaller ranks are preferred; the two criteria contribute equally.
        row["score_size_rank_fusion"] = score_rank + size_rank
    summaries = [
        summarize(rows, "score", "Parent: docking score", args.top_fraction),
        summarize(rows, "score_per_heavy_atom", "Experiment: score / heavy atoms", args.top_fraction),
        summarize(rows, "score_size_rank_fusion", "Experiment: docking/size rank fusion", args.top_fraction),
    ]
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Records evaluated: {len(rows)}; top fraction: {args.top_fraction:.0%}")
    for result in summaries:
        print(" | ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in result.items()))
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
