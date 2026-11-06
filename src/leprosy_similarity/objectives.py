"""The two objectives the configuration search minimises.

For each fold of the shared split, the configuration is fitted on the other
four folds and the held-out fold is positioned twice: once as recorded, and
once after a random number of its observed findings is hidden and filled by
the configuration's own missing-value strategy. Pooled over the five folds:

- objective 1, class-balanced MAE: the mean, over the five Ridley-Jopling
  classes, of each class's mean distance between position and class target,
  so a small class (TT, BB) weighs as much as a large one;
- objective 2, masking drift: the mean distance between a record's position
  and its position with findings hidden.

The masking sizes and the hidden findings come from one generator that runs
through the folds in order, so a configuration is scored on the same draws
every time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leprosy_similarity.config import Config
from leprosy_similarity.estimator import Configuration, SpectrumEstimator
from leprosy_similarity.folds import iter_folds
from leprosy_similarity.masking import mask_rows
from leprosy_similarity.schema import CLASSES, FEATURES


@dataclass
class CrossValidatedPositions:
    """Held-out positions of every record, as recorded and with findings hidden."""

    position: np.ndarray
    masked_position: np.ndarray
    fold: np.ndarray
    scales: list[dict] = field(default_factory=list)


def held_out_positions(
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    folds: pd.DataFrame,
    configuration: Configuration,
    config: Config,
) -> CrossValidatedPositions:
    """Fit on four folds, position the fifth as recorded and masked, five times."""
    n = len(cohort)
    position = np.full(n, np.nan)
    masked_position = np.full(n, np.nan)
    fold_index = np.full(n, -1, dtype=int)
    scales = []
    rng = np.random.default_rng(config.search.masking_seed)
    for k, train_rows, test_rows in iter_folds(cohort, folds):
        estimator = SpectrumEstimator(configuration, reference_cases, config)
        estimator.fit(cohort.iloc[train_rows], rescale=False)
        test = cohort.iloc[test_rows][list(FEATURES)].reset_index(drop=True)
        raw = estimator.raw_position(test)
        sizes = rng.choice(config.search.masking_sizes, size=len(test_rows))
        masked, _ = mask_rows(test, sizes, seed=int(rng.integers(1 << 30)))
        raw_masked = estimator.raw_position(masked)
        scale = estimator.fit_scale(cohort.iloc[train_rows])
        position[test_rows] = scale(raw)
        masked_position[test_rows] = scale(raw_masked)
        fold_index[test_rows] = k
        scales.append(scale.to_dict())
    return CrossValidatedPositions(position, masked_position, fold_index, scales)


def class_balanced_mae(position: np.ndarray, target: np.ndarray, classes: np.ndarray) -> float:
    """Mean over classes of the mean |position - class target|."""
    ok = np.isfinite(position) & np.isfinite(target)
    per_class = [
        float(np.mean(np.abs(position[ok & (classes == c)] - target[ok & (classes == c)])))
        for c in CLASSES
        if (ok & (classes == c)).any()
    ]
    return float(np.mean(per_class)) if per_class else float("inf")


def masking_drift(position: np.ndarray, masked_position: np.ndarray) -> float:
    """Mean |position - position with findings hidden|."""
    ok = np.isfinite(position) & np.isfinite(masked_position)
    return float(np.mean(np.abs(position[ok] - masked_position[ok]))) if ok.any() else float("inf")


def evaluate_configuration(
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    folds: pd.DataFrame,
    configuration: Configuration,
    config: Config,
) -> dict:
    """Both objectives and their diagnostics for one configuration."""
    start = time.time()
    cv = held_out_positions(cohort, reference_cases, folds, configuration, config)
    target = cohort["target"].to_numpy(dtype=float)
    classes = cohort["class"].astype(str).to_numpy()
    ok = np.isfinite(cv.position) & np.isfinite(target)
    return {
        "label_mae_macro": class_balanced_mae(cv.position, target, classes),
        "masking_mae": masking_drift(cv.position, cv.masked_position),
        "label_mae_micro": float(np.mean(np.abs(cv.position[ok] - target[ok]))),
        "class_means": {
            c: float(np.mean(cv.position[ok & (classes == c)])) for c in CLASSES if (ok & (classes == c)).any()
        },
        "elapsed_s": round(time.time() - start, 2),
        "positions": cv,
    }
