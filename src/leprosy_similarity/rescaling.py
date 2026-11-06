"""Rescaling the position to the range the fitted model can reach.

The raw position is a similarity-weighted mean of the reference anchors (TT 0,
BB 0.5, LL 1). No record is similar to the reference cases of one pole only,
so no record reaches 0 or 1, and how compressed the scale is depends on the
fitted model. The reported position therefore maps the model's own reach onto
[0, 1]: every recordable combination of findings (``recordable_grid``) is
positioned, and with ``lo`` and ``hi`` the lowest and highest positions,

    rescaled = clip((position - lo) / (hi - lo), 0, 1).

The most tuberculoid recordable presentation then reads 0 and the most
lepromatous reads 1. The map is linear, so rank measures and the additivity of
per-finding contributions are kept; contributions and spreads scale by
``1 / (hi - lo)``. Each fitted model gets its own range, computed from its own
training records, so no held-out record influences its scale.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from leprosy_similarity.schema import CLASS_TARGETS, FEATURES, NUMERIC_FEATURES, categorical_blocks


def recordable_grid(records: pd.DataFrame) -> pd.DataFrame:
    """Every complete record that can be written on the intake form.

    Each numeric finding takes every value observed in ``records``; each
    one-hot block takes one of its levels. The grid is derived from the
    records, so a value the cohort never records is not invented. On the
    analytic cohort it has 58,800 rows.
    """
    numeric = [c for c in FEATURES if c in NUMERIC_FEATURES]
    blocks = categorical_blocks()
    axes = [sorted(records[c].dropna().unique()) for c in numeric]
    axes += [list(range(len(columns))) for columns in blocks.values()]
    rows = []
    for combination in itertools.product(*axes):
        record: dict[str, Any] = dict(zip(numeric, combination[: len(numeric)]))
        for columns, pick in zip(blocks.values(), combination[len(numeric):]):
            for i, column in enumerate(columns):
                record[column] = 1.0 if i == pick else 0.0
        rows.append(record)
    return pd.DataFrame(rows, columns=list(FEATURES)).astype(float)


def shared_leaf_counts(leaves_x: np.ndarray, leaves_p: np.ndarray) -> np.ndarray:
    """Number of trees in which each row of ``x`` shares a leaf with each row of ``p``.

    A sparse product of one-hot leaf indicators: exact, and fast enough to
    position the whole grid for every fitted model.
    """
    leaves_x = np.asarray(leaves_x, dtype=np.int64)
    leaves_p = np.asarray(leaves_p, dtype=np.int64)
    n_trees = leaves_x.shape[1]
    width = int(max(leaves_x.max(), leaves_p.max())) + 1
    offset = np.arange(n_trees, dtype=np.int64) * width

    def indicators(leaves: np.ndarray) -> sparse.csr_matrix:
        n = leaves.shape[0]
        columns = (leaves + offset[None, :]).ravel()
        rows = np.repeat(np.arange(n), n_trees)
        data = np.ones(columns.size, dtype=np.float64)
        return sparse.csr_matrix((data, (rows, columns)), shape=(n, n_trees * width))

    return (indicators(leaves_x) @ indicators(leaves_p).T).toarray()


@dataclass(frozen=True)
class PositionScale:
    """Linear map of the raw range ``[lo, hi]`` onto [0, 1], clipped."""

    lo: float
    hi: float
    argmin: dict[str, float]
    argmax: dict[str, float]

    def __call__(self, position: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(position, dtype=float) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "argmin": self.argmin, "argmax": self.argmax}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PositionScale:
        return cls(lo=float(data["lo"]), hi=float(data["hi"]), argmin=dict(data["argmin"]), argmax=dict(data["argmax"]))


def fit_position_scale(
    leaves_of: Callable[[pd.DataFrame], np.ndarray],
    reference_leaves: np.ndarray,
    reference_classes: np.ndarray,
    grid: pd.DataFrame,
    batch_size: int,
) -> PositionScale:
    """Position every grid record with the fitted model and take the range.

    ``leaves_of`` maps a frame of complete records to the model's leaf matrix,
    through the same missing-value strategy used everywhere else (the
    indicator strategy, for one, appends columns the model expects).
    """
    anchors = np.array([CLASS_TARGETS[c] for c in reference_classes], dtype=float)
    position = np.empty(len(grid))
    for start in range(0, len(grid), batch_size):
        counts = shared_leaf_counts(leaves_of(grid.iloc[start : start + batch_size]), reference_leaves)
        total = counts.sum(axis=1)
        position[start : start + batch_size] = np.where(
            total > 0, (counts @ anchors) / np.where(total > 0, total, 1), np.nan
        )
    low, high = int(np.nanargmin(position)), int(np.nanargmax(position))
    return PositionScale(
        lo=float(position[low]),
        hi=float(position[high]),
        argmin=grid.iloc[low].to_dict(),
        argmax=grid.iloc[high].to_dict(),
    )
