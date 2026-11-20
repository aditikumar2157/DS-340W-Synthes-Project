"""Accuracy and interval-quality metrics.

Point accuracy (MAE, RMSE) says how close the prediction is. Interval quality
needs both halves of the story at once: PICP without MPIW rewards uselessly
wide intervals, and MPIW without PICP rewards confidently wrong ones.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability: the fraction of true values
    that land inside their interval. The headline reliability number."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width -- how much the interval costs in
    usefulness. In LogS units."""
    return float(np.mean(upper - lower))


def median_piw(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.median(upper - lower))


def coverage_gap(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Signed distance from nominal coverage: negative means under-covering
    (over-confident intervals), positive means conservative."""
    return float(picp(y_true, lower, upper) - (1.0 - alpha))


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Winkler interval score: width plus a penalty for each miss, scaled by
    alpha. A single number that cannot be gamed by widening or narrowing
    alone, so it breaks ties that PICP and MPIW leave open. Lower is better."""
    width = upper - lower
    below = np.clip(lower - y_true, 0.0, None)
    above = np.clip(y_true - upper, 0.0, None)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Every metric for one method on one test set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (len(y_true) == len(y_pred) == len(lower) == len(upper)):
        raise ValueError("metric inputs must all have the same length")
    if np.any(upper < lower):
        raise ValueError("found upper < lower; interval bounds are swapped")
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "picp": picp(y_true, lower, upper),
        "mpiw": mpiw(lower, upper),
        "median_piw": median_piw(lower, upper),
        "coverage_gap": coverage_gap(y_true, lower, upper, alpha),
        "abs_coverage_gap": abs(coverage_gap(y_true, lower, upper, alpha)),
        "interval_score": interval_score(y_true, lower, upper, alpha),
    }
