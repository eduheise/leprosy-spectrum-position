"""Direction sweeps: does the position move with each finding as the disease predicts?

One finding at a time is set, for every record, to each value it takes in the
cohort and to missing, with the record's other findings left as recorded; the
record is then re-positioned by the model that did not train on it. The mean
over the cohort at each value traces the response curve, carried with the
per-record spread (plus or minus one standard deviation).

A colour or morphology level is read differently: it is switched on as the
only level of its block in every record, and the shift from the record's own
position is reported, because a single indicator has no scale to sweep.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from leprosy_similarity.schema import CATEGORICAL_LEVELS, NUMERIC_FEATURES

Score = Callable[[pd.DataFrame], np.ndarray]


def sweep_values(records: pd.DataFrame, feature: str) -> list[float]:
    """Values swept for a finding: those it takes in the records, or 0/1 for an indicator."""
    if feature in NUMERIC_FEATURES:
        return sorted(np.unique(records[feature].dropna().to_numpy()).tolist())
    return [0.0, 1.0]


def direction(curve: np.ndarray, tolerance: float) -> dict[str, Any]:
    """Direction of travel of a mean curve, and whether it ever steps back.

    The direction compares the last value with the first; ``tolerance``
    absorbs rounding, so a curve that moves less than it is flat.
    """
    delta = float(curve[-1] - curve[0])
    travel = "up" if delta > tolerance else ("down" if delta < -tolerance else "flat")
    steps = np.diff(curve)
    rises, falls = bool(np.all(steps >= -tolerance)), bool(np.all(steps <= tolerance))
    if rises and falls:
        shape = "flat"
    elif rises:
        shape = "monotone_up"
    elif falls:
        shape = "monotone_down"
    else:
        shape = "non_monotone"
    return {"direction": travel, "delta": delta, "monotonicity": shape}


def sweep_finding(
    score: Score, records: pd.DataFrame, feature: str, expected_up: bool, tolerance: float
) -> dict[str, Any]:
    """Mean position and per-record spread at each value of one finding, then missing."""
    values = sweep_values(records, feature)
    mean, spread = [], []
    for value in [*values, np.nan]:
        edited = records.copy()
        edited[feature] = value
        position = score(edited)
        mean.append(float(np.nanmean(position)))
        spread.append(float(np.nanstd(position)))
    curve = np.array(mean[:-1])
    rank = float(spearmanr(values, curve).statistic) if len(set(values)) > 1 else float("nan")
    travel = direction(curve, tolerance)
    return {
        "feature": feature,
        "values": values,
        "mean": mean[:-1],
        "sd": spread[:-1],
        "missing_mean": mean[-1],
        "missing_sd": spread[-1],
        "spearman": rank,
        "expected_up": expected_up,
        "violation": bool(expected_up and travel["direction"] != "up"),
        **travel,
    }


def level_shifts(score: Score, records: pd.DataFrame, baseline: np.ndarray) -> list[dict[str, Any]]:
    """Per-record shift when each colour or morphology level is the only one present."""
    rows = []
    for variable, levels in CATEGORICAL_LEVELS.items():
        block = [f"{variable}__{level}" for level in levels]
        for level, column in zip(levels, block):
            edited = records.copy()
            edited[block] = 0.0
            edited[column] = 1.0
            position = score(edited)
            shift = position - baseline
            rows.append(
                {
                    "column": column,
                    "variable": variable,
                    "level": level,
                    "mean_position": float(np.nanmean(position)),
                    "mean_shift": float(np.nanmean(shift)),
                    "sd_shift": float(np.nanstd(shift)),
                }
            )
    return rows


def sweep_all(
    score: Score, records: pd.DataFrame, features: Iterable[str], expected_up: Iterable[str], tolerance: float
) -> list[dict[str, Any]]:
    """Sweep every finding in ``features``."""
    up = set(expected_up)
    return [sweep_finding(score, records, f, f in up, tolerance) for f in features]
