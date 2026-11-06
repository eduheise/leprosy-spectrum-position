"""The range the selected configuration's position can reach, and its rescaling.

Input:  data/processed/cohort.parquet, reference_cases.parquet, folds.csv,
        results/selected_configuration.json
Output: results/position_rescaling.json

The selected configuration is fitted on the whole analytic cohort and every
recordable complete record is positioned (``rescaling.recordable_grid``);
the lowest and highest raw positions define the map onto [0, 1]. The step
records what that range is, which records attain it, how the similarity
weight splits across the TT, BB and LL reference cases at both ends, where
the reference cases, the cohort, the raw midpoint and the class anchors land
on the rescaled scale, and the range each fold model of the shared split
reaches from its own training records.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.config import Config, load_config
from leprosy_similarity.estimator import Configuration, SpectrumEstimator, load_selected_configuration
from leprosy_similarity.folds import iter_folds
from leprosy_similarity.rescaling import recordable_grid, shared_leaf_counts
from leprosy_similarity.schema import CLASS_TARGETS, FEATURES

INPUTS = (paths.COHORT, paths.REFERENCE_CASES, paths.FOLDS, paths.SELECTED_CONFIGURATION)
OUTPUTS = (paths.POSITION_RESCALING,)

REFERENCE_CLASSES = ("TT", "BB", "LL")


def weight_shares(estimator: SpectrumEstimator, record: dict[str, float]) -> dict[str, float]:
    """Share of a record's similarity weight carried by each reference class."""
    frame = pd.DataFrame([record])[list(FEATURES)]
    counts = shared_leaf_counts(estimator.leaves(frame), estimator.reference_leaves)[0]
    classes = estimator.reference_classes
    return {c: float(counts[classes == c].sum() / counts.sum()) for c in REFERENCE_CLASSES}


def by_class(values: np.ndarray, classes: np.ndarray) -> dict[str, list[float]]:
    return {c: [float(v) for v in values[classes == c]] for c in REFERENCE_CLASSES}


def whole_cohort_range(estimator: SpectrumEstimator, cohort: pd.DataFrame) -> dict[str, Any]:
    """The final model's range and where the reference cases, cohort and anchors fall in it."""
    scale = estimator.scale
    classes = estimator.reference_classes
    reference_raw = estimator.raw_position(estimator.reference_features)
    cohort_raw = estimator.raw_position(cohort[list(FEATURES)])
    cohort_rescaled = scale(cohort_raw)
    base = float(np.mean([CLASS_TARGETS[c] for c in classes]))
    return {
        **scale.to_dict(),
        "span": scale.span,
        "n_grid_records": len(recordable_grid(cohort[list(FEATURES)])),
        "base_raw": base,
        "base_rescaled": float(scale(base)),
        "anchors_rescaled": {c: float(scale(t)) for c, t in CLASS_TARGETS.items()},
        "weight_shares_at_max": weight_shares(estimator, scale.argmax),
        "weight_shares_at_min": weight_shares(estimator, scale.argmin),
        "reference_positions_raw": by_class(reference_raw, classes),
        "reference_positions_rescaled": by_class(scale(reference_raw), classes),
        "cohort_raw_span": [float(np.nanmin(cohort_raw)), float(np.nanmax(cohort_raw))],
        "cohort_outside_range": int(((cohort_raw < scale.lo) | (cohort_raw > scale.hi)).sum()),
        "cohort_rescaled_span": [float(np.nanmin(cohort_rescaled)), float(np.nanmax(cohort_rescaled))],
        "cohort_at_floor": int((cohort_rescaled == 0.0).sum()),
        "cohort_at_ceiling": int((cohort_rescaled == 1.0).sum()),
    }


def fold_ranges(
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    folds: pd.DataFrame,
    configuration: Configuration,
    config: Config,
) -> list[dict[str, float]]:
    """Range each fold model reaches from its own four training folds."""
    ranges = []
    for k, train_rows, _ in iter_folds(cohort, folds):
        estimator = SpectrumEstimator(configuration, reference_cases, config).fit(cohort.iloc[train_rows])
        ranges.append({"fold": k, "lo": estimator.scale.lo, "hi": estimator.scale.hi})
    return ranges


def run() -> None:
    config = load_config()
    cohort = pd.read_parquet(paths.COHORT)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    folds = pd.read_csv(paths.FOLDS)
    configuration = load_selected_configuration()

    estimator = SpectrumEstimator(configuration, reference_cases, config).fit(cohort)
    result = whole_cohort_range(estimator, cohort)
    result["fold_ranges"] = fold_ranges(cohort, reference_cases, folds, configuration, config)
    paths.POSITION_RESCALING.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"reachable raw range [{result['lo']:.4f}, {result['hi']:.4f}] over {result['n_grid_records']} records; "
        f"base {result['base_raw']} -> {result['base_rescaled']:.3f}; "
        f"LL weight at the top {result['weight_shares_at_max']['LL']:.3f}"
    )
    print(f"wrote {paths.POSITION_RESCALING}")


if __name__ == "__main__":
    run()
