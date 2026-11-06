"""The selected configuration fitted once per fold of the shared split.

Every cross-validated result reads a record through the one model that did not
train on it: the model fitted on the other four folds. Each fold model carries
its own rescaling range, fitted on its own training records, so no held-out
record influences the scale it is read on.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from leprosy_similarity.config import Config
from leprosy_similarity.estimator import Configuration, SpectrumEstimator
from leprosy_similarity.folds import iter_folds


@dataclass(frozen=True)
class FoldModel:
    """The estimator fitted without fold ``fold``, and the rows it may score."""

    fold: int
    train_rows: np.ndarray
    test_rows: np.ndarray
    estimator: SpectrumEstimator


def fit_fold_models(
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    folds: pd.DataFrame,
    configuration: Configuration,
    config: Config,
) -> list[FoldModel]:
    """Fit the configuration on four folds at a time, rescaling range included."""
    models = []
    for k, train_rows, test_rows in iter_folds(cohort, folds):
        estimator = SpectrumEstimator(configuration, reference_cases, config).fit(cohort.iloc[train_rows])
        models.append(FoldModel(k, train_rows, test_rows, estimator))
    return models


def held_out_position(fold_models: Sequence[FoldModel], records: pd.DataFrame, rescaled: bool = True) -> np.ndarray:
    """Position of every row of ``records`` by the model that did not train on it.

    ``records`` is aligned row for row with the cohort the fold models were
    fitted on; its findings may be edited (a sweep sets one finding for every
    record), but row ``i`` is still scored by the model that held out record
    ``i``.
    """
    out = np.full(len(records), np.nan)
    for model in fold_models:
        held_out = records.iloc[model.test_rows]
        position = model.estimator.position if rescaled else model.estimator.raw_position
        out[model.test_rows] = position(held_out)
    return out


def fold_of_rows(fold_models: Sequence[FoldModel], n_rows: int) -> np.ndarray:
    """Fold index of each cohort row."""
    fold = np.full(n_rows, -1, dtype=int)
    for model in fold_models:
        fold[model.test_rows] = model.fold
    return fold
