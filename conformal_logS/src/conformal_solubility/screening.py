"""Screening triage: what a calibrated interval is actually *for*.

Everything else in this package measures intervals as intervals -- coverage,
width, interval score. Those are the right diagnostics, but they are not the
reason anyone wants uncertainty in a screening campaign. V-SYNTHES2 does not
ask "what is this molecule's LogS"; it asks "is this molecule worth the next
stage", and it asks that of millions of molecules under a fixed budget.

So this module turns intervals into the decision they are meant to support.
Fix a property threshold ``tau`` (say, LogS >= -4: anything below is too
poorly soluble to pursue). A point prediction forces a two-way call on every
molecule. An interval permits a three-way one:

    accept   interval lies entirely above tau   -- confidently good enough
    reject   interval lies entirely below tau   -- confidently not
    defer    interval straddles tau             -- the model does not know

Deferring is not a failure; it is the output of an honest model, and in a
real campaign it is the queue that goes to an assay rather than straight to
synthesis. What matters is the arithmetic between the two things it trades:

``decisive_fraction``
    How many molecules the model is willing to call. A model that defers
    everything is useless however well calibrated it is.

``decisive_error_rate``
    How often a confident call is wrong -- accepted when truly below tau, or
    rejected when truly above. This is the number that costs money, because a
    confident wrong call is the one nobody checks.

This reframing is where calibration earns its keep in a way that PICP alone
cannot show. An over-wide uncalibrated interval covers beautifully and defers
nearly everything, so its coverage number looks excellent while it does no
work. A too-narrow one is decisive and wrong. Only an honestly calibrated
interval is decisive *and* right, and only a comparison on these two axes at
once can tell the three cases apart.

The point predictor is reported alongside as the baseline a screen would use
today: it is decisive on 100% of molecules, and its error rate is the price
of that confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACCEPT = "accept"
REJECT = "reject"
DEFER = "defer"


def triage(lower: np.ndarray, upper: np.ndarray, threshold: float) -> np.ndarray:
    """Three-way decision per molecule from its interval.

    A molecule is accepted only when the whole interval clears the threshold,
    and rejected only when the whole interval falls short. Anything touching
    the threshold is deferred, including an interval that merely grazes it --
    the conservative reading, which is the right one when the cost of a
    confident error exceeds the cost of a second look.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    decision = np.full(len(lower), DEFER, dtype=object)
    decision[lower >= threshold] = ACCEPT
    decision[upper < threshold] = REJECT
    return decision


def point_triage(prediction: np.ndarray, threshold: float) -> np.ndarray:
    """The two-way call a point prediction forces: no deferral available."""
    prediction = np.asarray(prediction, dtype=float)
    return np.where(prediction >= threshold, ACCEPT, REJECT).astype(object)


def screening_summary(
    y_true: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Decision quality at one threshold, for intervals and for points alike.

    Returns both the interval-based three-way outcome and the point-based
    two-way outcome on the same molecules, so the cost of deferring and the
    benefit of not being wrong can be read off against each other.
    """
    y_true = np.asarray(y_true, dtype=float)
    truth_above = y_true >= threshold

    decision = triage(lower, upper, threshold)
    accepted = decision == ACCEPT
    rejected = decision == REJECT
    deferred = decision == DEFER
    decisive = accepted | rejected

    wrong_accept = int(np.sum(accepted & ~truth_above))
    wrong_reject = int(np.sum(rejected & truth_above))
    n_decisive = int(decisive.sum())

    point = point_triage(prediction, threshold)
    point_wrong = int(np.sum((point == ACCEPT) & ~truth_above)) + int(
        np.sum((point == REJECT) & truth_above)
    )

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "fraction_truly_above": float(np.mean(truth_above)),
        "accept_fraction": float(np.mean(accepted)),
        "reject_fraction": float(np.mean(rejected)),
        "defer_fraction": float(np.mean(deferred)),
        "decisive_fraction": float(np.mean(decisive)),
        # Error rate *among molecules the model was willing to call*. NaN when
        # it called none, which is a real outcome and must not read as zero.
        "decisive_error_rate": (
            float((wrong_accept + wrong_reject) / n_decisive)
            if n_decisive
            else float("nan")
        ),
        "false_accept_rate": (
            float(wrong_accept / int(accepted.sum())) if accepted.any() else float("nan")
        ),
        "point_error_rate": float(point_wrong / len(y_true)),
    }


def screening_curve(
    y_true: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """:func:`screening_summary` swept over several thresholds.

    One threshold is a choice someone has to defend; a sweep shows whether
    the conclusion depends on it. Thresholds beyond the range of the data are
    degenerate (everything is on one side), so they are reported with the
    class balance alongside, which makes that visible.
    """
    return pd.DataFrame(
        [
            screening_summary(y_true, prediction, lower, upper, float(tau))
            for tau in np.asarray(thresholds, dtype=float)
        ]
    )
