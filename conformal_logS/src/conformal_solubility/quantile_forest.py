"""Quantile regression forest (Meinshausen, 2006) read off an already-fitted
Random Forest.

The README's own list of next steps asks for this: tree spread is "a
convenient proxy [that] understates the irreducible noise term", and a
quantile-regression forest should sharpen the normalized conformal variant.
It is also what conformalized quantile regression (Romano et al., 2019) needs
as its base model.

Why a read-out rather than a separate model
-------------------------------------------
The whole design of this study is that the base model is held fixed, so any
difference between interval methods comes from calibration alone. Fitting a
second forest -- even with the same hyper-parameters -- would break that: the
point predictions would move, and a change in MAE would confound every
interval comparison.

So this class takes the forest that was *already* fitted and reads a
conditional distribution out of it. The trees are untouched, the reported
point prediction is still that same forest's mean, and the only new thing is
how the leaves are interrogated.

The estimator
-------------
Meinshausen's observation is that a random forest already defines a weight
over training samples:

    w_i(x) = (1 / T) * sum_t  1[leaf_t(x_i) = leaf_t(x)] / |leaf_t(x)|

Every tree splits one unit of weight evenly across the training points that
share the query's leaf, and the forest averages over trees. The forest's mean
prediction is sum_i w_i(x) y_i; the conditional *distribution* estimate is the
weighted empirical distribution of the same y_i, and its quantiles are what we
want. Tree spread, by contrast, measures disagreement between trees about the
mean -- an epistemic quantity that says nothing about the noise around it.

Weights are computed over all training rows routed through each tree, not just
that tree's bootstrap in-bag rows. This is the common implementation (and
Meinshausen's original formulation); using out-of-bag rows only would give an
honest but much noisier estimate from ~37% of the data per tree.

Implementation note
-------------------
The gather is vectorised per tree rather than looped per query molecule: a
five-seed sweep calls this tens of times, and the obvious per-molecule loop
(500 small array concatenations per molecule) is roughly two orders of
magnitude slower. ``tests/test_quantile_forest.py`` checks the vectorised
result against a direct transcription of the formula above.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

#: Cumulative weights are resolved to this many decimal places before being
#: compared against a quantile level.
#:
#: Leaf weights are exact rationals -- 1 / (leaf size x n_trees) -- so a
#: cumulative weight frequently lands *exactly* on a round level such as 0.25,
#: and which side of the comparison it falls on decides which order statistic
#: is returned. Summation order then matters: the batched path accumulates one
#: running total across every query, so by the last query it carries ~1e-14 of
#: drift, enough to flip an exact tie that a per-query sum resolves the other
#: way. Rounding first makes the two agree.
#:
#: Ten decimals is comfortably below any real difference in weight (the
#: smallest possible is 1 / (n_train x n_trees), about 1e-6 for the sizes used
#: here) and comfortably above the drift.
_CUMULATIVE_DECIMALS = 10


def _preprocess(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Push features through every pipeline step except the forest itself."""
    transformed = X
    for _, step in model.steps[:-1]:
        transformed = step.transform(transformed)
    return transformed


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Lower weighted quantiles of a single weighted sample.

    Returns the smallest observed value whose cumulative weight reaches each
    requested level -- the inverse-CDF convention, with no interpolation
    between neighbouring observations. Interpolating would invent values the
    forest never saw, and for the conformal use below it is the ordering that
    matters, not smoothness.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if values.size == 0:
        raise ValueError("cannot take quantiles of an empty sample")

    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    fraction = np.round(cumulative / total, _CUMULATIVE_DECIMALS)
    positions = np.searchsorted(fraction, quantiles, side="left")
    return values[np.clip(positions, 0, len(values) - 1)]


class QuantileForestReadout:
    """Conditional quantiles from a fitted forest's leaf structure.

    Parameters
    ----------
    model:
        The fitted pipeline whose last step is a ``RandomForestRegressor``.
        It is read, never re-fitted.
    X_train, y_train:
        The rows the forest was fitted on. They are needed because a fitted
        scikit-learn forest keeps no copy of its training targets, and the
        quantile estimate is a weighting *of those targets*.
    """

    def __init__(self, model: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = model
        self.y_train = np.asarray(y_train, dtype=float)
        forest = model.steps[-1][1]
        self.n_trees = len(forest.estimators_)

        leaves = forest.apply(_preprocess(model, X_train))  # (n_train, n_trees)
        if leaves.shape[0] != len(self.y_train):
            raise ValueError("X_train and y_train disagree on the number of rows")

        # Per tree, group the training rows by the leaf they land in. Storing
        # the groups as a sorted index array plus (start, count) per leaf lets
        # a whole batch of queries be gathered with array arithmetic.
        self._order: list[np.ndarray] = []
        self._leaf_ids: list[np.ndarray] = []
        self._starts: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        for t in range(self.n_trees):
            column = leaves[:, t]
            order = np.argsort(column, kind="stable")
            unique, starts, counts = np.unique(
                column[order], return_index=True, return_counts=True
            )
            self._order.append(order)
            self._leaf_ids.append(unique)
            self._starts.append(starts)
            self._counts.append(counts)

    def _gather(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pooled (owner, value, weight) triples for every query row.

        ``owner[k]`` is the index of the query row that triple ``k`` belongs
        to. Each tree contributes weight ``1 / n_trees`` to every query,
        divided evenly among the training rows sharing that query's leaf.
        """
        forest = self.model.steps[-1][1]
        leaves = forest.apply(_preprocess(self.model, X))  # (n_queries, n_trees)
        n_queries = leaves.shape[0]
        query_index = np.arange(n_queries)

        owners: list[np.ndarray] = []
        values: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for t in range(self.n_trees):
            unique = self._leaf_ids[t]
            position = np.searchsorted(unique, leaves[:, t])
            position = np.clip(position, 0, len(unique) - 1)
            if not np.array_equal(unique[position], leaves[:, t]):
                # Every leaf of a fitted tree holds at least one training row,
                # so this cannot happen unless the forest was re-fitted under
                # us or a different X_train was supplied.
                raise RuntimeError(
                    "a query landed in a leaf holding no training rows; the "
                    "readout was built from different training data than the "
                    "forest was fitted on"
                )

            count = self._counts[t][position]
            start = self._starts[t][position]

            # Ragged gather: expand (start, count) pairs into a flat index.
            total = int(count.sum())
            group_start = np.repeat(np.cumsum(count) - count, count)
            ramp = np.arange(total) - group_start
            flat = self._order[t][np.repeat(start, count) + ramp]

            owners.append(np.repeat(query_index, count))
            values.append(self.y_train[flat])
            weights.append(np.repeat(1.0 / (count * self.n_trees), count))

        return (
            np.concatenate(owners),
            np.concatenate(values),
            np.concatenate(weights),
        )

    def predict_quantiles(
        self, X: np.ndarray, quantiles: list[float] | np.ndarray
    ) -> np.ndarray:
        """Conditional quantiles, shape ``(n_queries, n_quantiles)``.

        Quantiles are read with the same inverse-CDF convention as
        :func:`weighted_quantiles`, applied to every query at once.
        """
        quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError(f"quantiles must lie in [0, 1], got {quantiles}")

        owner, value, weight = self._gather(X)
        n_queries = len(X)

        # Sort by (owner, value) so each query's pooled sample is contiguous
        # and internally ordered -- the layout a cumulative sum needs.
        order = np.lexsort((value, owner))
        owner, value, weight = owner[order], value[order], weight[order]

        group_start = np.searchsorted(owner, np.arange(n_queries), side="left")
        group_end = np.searchsorted(owner, np.arange(n_queries), side="right")
        if np.any(group_end <= group_start):
            raise RuntimeError("a query row gathered no training weight")

        sizes = group_end - group_start
        cumulative = np.cumsum(weight)
        before = np.where(group_start > 0, cumulative[group_start - 1], 0.0)
        totals = cumulative[group_end - 1] - before
        within = np.round(
            (cumulative - np.repeat(before, sizes)) / np.repeat(totals, sizes),
            _CUMULATIVE_DECIMALS,
        )

        # Offsetting each group by its own index makes the key globally
        # increasing, so one searchsorted answers every query at once.
        key = owner * 2.0 + within
        out = np.empty((n_queries, len(quantiles)), dtype=float)
        for j, q in enumerate(quantiles):
            target = np.arange(n_queries) * 2.0 + q
            position = np.searchsorted(key, target, side="left")
            position = np.clip(position, group_start, group_end - 1)
            out[:, j] = value[position]
        return out

    def interval_width(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """The central ``1 - alpha`` inter-quantile range for each query.

        This is the difficulty estimate the normalized conformal variant uses
        when asked for a quantile-forest scale instead of tree spread: it is a
        width in the units of the target, and unlike tree spread it includes
        the irreducible noise the forest cannot explain away.
        """
        bounds = self.predict_quantiles(X, [alpha / 2.0, 1.0 - alpha / 2.0])
        return np.clip(bounds[:, 1] - bounds[:, 0], 0.0, None)
