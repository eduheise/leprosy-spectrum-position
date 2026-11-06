"""The one 5-fold split every cross-validated result shares.

The configuration search and every result reported for the selected
configuration use the same stratified split of the analytic cohort, so a
record is held out in exactly one fold, and always the same one.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def assign_folds(cohort: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Fold of each record: ``record_id`` and ``fold`` (0-based), in cohort order.

    Fold k holds the k-th test set of a shuffled ``StratifiedKFold`` over the
    Ridley-Jopling labels.
    """
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(cohort), -1, dtype=int)
    for k, (_, test) in enumerate(splitter.split(cohort, cohort["class"].astype(str).to_numpy())):
        fold[test] = k
    return pd.DataFrame({"record_id": cohort["record_id"].to_numpy(), "fold": fold})


def fold_of(cohort: pd.DataFrame, folds: pd.DataFrame) -> np.ndarray:
    """Fold of each cohort row, aligned on ``record_id``."""
    lookup = folds.set_index("record_id")["fold"]
    missing = ~cohort["record_id"].isin(lookup.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} cohort records have no fold")
    return lookup.loc[cohort["record_id"]].to_numpy()


def iter_folds(cohort: pd.DataFrame, folds: pd.DataFrame) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """``(fold, train_rows, test_rows)`` positional indices, folds in order."""
    fold = fold_of(cohort, folds)
    for k in np.unique(fold):
        yield int(k), np.flatnonzero(fold != k), np.flatnonzero(fold == k)
