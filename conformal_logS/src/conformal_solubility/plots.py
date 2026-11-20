"""Figures for the report.

Four figures, one question each:

1. ``coverage.png``       -- does each method hit the 90% target? (H1)
2. ``accuracy.png``       -- what does scaffold splitting cost in accuracy? (H2)
3. ``novelty_coverage.png`` -- does coverage survive on unfamiliar molecules? (H3)
4. ``intervals.png``      -- are the calibrated intervals usefully narrow?

Palette notes: the three methods get a fixed categorical order (orange =
uncalibrated baseline, blue = split conformal, purple = normalized conformal),
assigned by name so a figure that omits a method never repaints the others.
The hues were checked for colour-vision separation rather than chosen by eye,
and every figure also distinguishes methods by position or marker, so colour
is never the only cue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR

# Fixed per-method colours, keyed by name (never by position in a filtered list).
METHOD_COLORS = {
    "uncalibrated_gaussian": "#C2410C",
    "split_conformal": "#1B6FB5",
    "normalized_conformal": "#6D28A8",
}
METHOD_MARKERS = {
    "uncalibrated_gaussian": "o",
    "split_conformal": "s",
    "normalized_conformal": "^",
}
METHOD_LABELS = {
    "uncalibrated_gaussian": "Uncalibrated (tree SD)",
    "split_conformal": "Split conformal",
    "normalized_conformal": "Normalized conformal",
}
SPLIT_LABELS = {
    "random": "Random split",
    "scaffold": "Scaffold split\n(random calibration)",
    "scaffold_ood_calib": "Scaffold split\n(scaffold calibration)",
}
# Short forms for axes where the full label would collide with its neighbour.
SPLIT_SHORT = {
    "random": "Random",
    "scaffold": "Scaffold\nrandom calib.",
    "scaffold_ood_calib": "Scaffold\nscaffold calib.",
}
SPLIT_COLORS = {
    "random": "#1B6FB5",
    "scaffold": "#C2410C",
    "scaffold_ood_calib": "#6D28A8",
}

INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#DDDDDD"
SURFACE = "#FFFFFF"

PLOT_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
}


def _ordered(values, reference) -> list:
    """Keep ``reference``'s order, drop anything absent from ``values``."""
    present = set(values)
    return [item for item in reference if item in present]


def plot_coverage(summary: pd.DataFrame, alpha: float, out_dir: Path) -> Path:
    """Mean PICP per method and split, against the nominal target.

    Bars because the question is magnitude against a fixed reference, and the
    reference line is what the eye should land on first.
    """
    target = 1.0 - alpha
    splits = _ordered(summary["split_type"], list(SPLIT_LABELS))
    methods = _ordered(summary["method"], list(METHOD_COLORS))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        width = 0.8 / max(len(methods), 1)
        positions = np.arange(len(splits))

        for m_index, method in enumerate(methods):
            means, errors = [], []
            for split_type in splits:
                row = summary[
                    (summary["split_type"] == split_type) & (summary["method"] == method)
                ]
                means.append(float(row["picp_mean"].iloc[0]) if len(row) else np.nan)
                err = float(row["picp_std"].iloc[0]) if len(row) else np.nan
                errors.append(0.0 if np.isnan(err) else err)
            offset = (m_index - (len(methods) - 1) / 2) * width
            bars = ax.bar(
                positions + offset,
                means,
                width * 0.92,  # 2px-equivalent surface gap between adjacent bars
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                zorder=3,
            )
            ax.errorbar(
                positions + offset,
                means,
                yerr=errors,
                fmt="none",
                ecolor=INK,
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
            # Direct labels: three series, so every bar can carry its value.
            for bar, mean in zip(bars, means):
                if np.isnan(mean):
                    continue
                ax.annotate(
                    f"{mean:.3f}",
                    (bar.get_x() + bar.get_width() / 2, mean),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=8,
                    color=INK,
                )

        ax.axhline(target, color=INK, linewidth=1.2, linestyle="--", zorder=5)
        ax.annotate(
            f"target {target:.0%}",
            (-0.48, target),
            textcoords="offset points",
            xytext=(0, 5),
            ha="left",
            fontsize=9,
            color=INK,
        )

        ax.set_xticks(positions, [SPLIT_LABELS.get(s, s) for s in splits])
        ax.set_ylabel("Coverage (PICP)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Interval coverage against the 90% target")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3
        )
        fig.tight_layout()
        path = out_dir / "coverage.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def plot_accuracy(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """Per-seed RMSE and MAE under each split (H2).

    The three interval methods share one fitted model, so accuracy is a
    property of the split alone; this figure plots the split, with one dot
    per seed so the spread is visible rather than averaged away.
    """
    accuracy = (
        metrics[metrics["method"] == metrics["method"].iloc[0]]
        if len(metrics)
        else metrics
    )
    splits = _ordered(accuracy["split_type"], list(SPLIT_LABELS))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True)
        for ax, metric, label in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"]):
            for index, split_type in enumerate(splits):
                values = accuracy.loc[
                    accuracy["split_type"] == split_type, metric
                ].to_numpy()
                jitter = np.random.default_rng(0).normal(0, 0.035, len(values))
                ax.scatter(
                    np.full(len(values), index) + jitter,
                    values,
                    s=36,
                    color=SPLIT_COLORS.get(split_type, INK),
                    edgecolor=SURFACE,
                    linewidth=1.0,
                    zorder=3,
                )
                mean = float(np.mean(values)) if len(values) else np.nan
                ax.hlines(
                    mean, index - 0.22, index + 0.22, color=INK, linewidth=1.6, zorder=4
                )
                ax.annotate(
                    f"{mean:.3f}",
                    (index + 0.26, mean),
                    fontsize=9,
                    color=INK,
                    va="center",
                )
            ax.set_xticks(range(len(splits)), [SPLIT_SHORT.get(s, s) for s in splits])
            ax.set_xlim(-0.5, len(splits) - 0.05)
            ax.set_ylabel(f"{label} (log units)")
            ax.set_title(label)
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
        fig.suptitle(
            "Point accuracy by split regime (one dot per seed)",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout()
        path = out_dir / "accuracy.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def plot_novelty_coverage(
    novelty_summary: pd.DataFrame, alpha: float, out_dir: Path
) -> Path:
    """Coverage by structural-novelty quartile, one panel per split (H3)."""
    target = 1.0 - alpha
    splits = _ordered(novelty_summary["split_type"], list(SPLIT_LABELS))
    methods = _ordered(novelty_summary["method"], list(METHOD_COLORS))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            1, max(len(splits), 1), figsize=(10.5, 4.0), sharey=True, squeeze=False
        )
        for ax, split_type in zip(axes[0], splits):
            panel = novelty_summary[novelty_summary["split_type"] == split_type]
            bins = sorted(panel["novelty_bin"].unique())
            for method in methods:
                series = panel[panel["method"] == method].set_index("novelty_bin")
                values = [
                    float(series.loc[b, "picp"]) if b in series.index else np.nan
                    for b in bins
                ]
                ax.plot(
                    range(len(bins)),
                    values,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=7,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.0,
                    linewidth=2.0,
                    label=METHOD_LABELS[method],
                    zorder=3,
                )
            ax.axhline(target, color=INK, linewidth=1.1, linestyle="--", zorder=2)
            tick_labels = [
                f"{b}\n{'least' if i == 0 else 'most' if i == len(bins) - 1 else ''}"
                for i, b in enumerate(bins)
            ]
            ax.set_xticks(range(len(bins)), tick_labels)
            ax.set_xlabel("Similarity to training set (quartile)")
            ax.set_title(SPLIT_LABELS.get(split_type, split_type))
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("Coverage (PICP)")
        axes[0][0].set_ylim(0.5, 1.02)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            "Coverage against structural novelty",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = out_dir / "novelty_coverage.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def plot_interval_widths(predictions: pd.DataFrame, out_dir: Path) -> Path:
    """Distribution of interval widths per method and split.

    Coverage alone can be bought by widening, so the widths belong next to it:
    this is the "usefulness" half of the reliability claim.
    """
    splits = _ordered(predictions["split_type"], list(SPLIT_LABELS))
    methods = _ordered(predictions["method"], list(METHOD_COLORS))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            1, max(len(splits), 1), figsize=(10.5, 4.0), sharey=True, squeeze=False
        )
        for ax, split_type in zip(axes[0], splits):
            panel = predictions[predictions["split_type"] == split_type]
            data, colors = [], []
            for method in methods:
                widths = (
                    panel.loc[panel["method"] == method, "upper"]
                    - panel.loc[panel["method"] == method, "lower"]
                )
                data.append(widths.to_numpy())
                colors.append(METHOD_COLORS[method])
            boxes = ax.boxplot(
                data,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": INK, "linewidth": 1.4},
                whiskerprops={"color": MUTED, "linewidth": 1.0},
                capprops={"color": MUTED, "linewidth": 1.0},
            )
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(SURFACE)
                patch.set_linewidth(1.5)
                patch.set_alpha(0.9)
            ax.set_xticks(
                range(1, len(methods) + 1),
                [METHOD_LABELS[m].replace(" ", "\n", 1) for m in methods],
            )
            ax.set_title(SPLIT_LABELS.get(split_type, split_type))
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("Interval width (log units)")
        fig.suptitle(
            "Interval width distributions",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout()
        path = out_dir / "intervals.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def make_all_figures(results: dict, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = results["config"]["alpha"]
    paths = [
        plot_coverage(results["summary"], alpha, out_dir),
        plot_accuracy(results["metrics"], out_dir),
        plot_novelty_coverage(results["novelty_summary"], alpha, out_dir),
        plot_interval_widths(results["predictions"], out_dir),
    ]
    return paths
