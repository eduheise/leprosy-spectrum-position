"""The variable screen: which candidate variables the schema admits.

Input:  data/processed/variable_screen.csv,
        data/processed/variable_screen_reaction_prevalence.csv,
        data/processed/variable_screen_shape.csv,
        data/processed/variable_screen_baseline.json
Output: reports/tables/feature_screen.csv  (tab:feature-selection)
        reports/numbers.json, keys ``screen.*``

Each entry is the rise in held-out multiclass log loss when the variable is
shuffled, averaged over the folds with its fold-to-fold standard deviation.
A variable is admitted when its rise clears that spread. The reaction tables
explain why the type-1 reaction ranks high although it has no pole: it marks
the centre of the spectrum rather than either end.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.reports.numbers import record_numbers

INPUTS = (
    paths.VARIABLE_SCREEN,
    paths.VARIABLE_SCREEN_REACTION_PREVALENCE,
    paths.VARIABLE_SCREEN_SHAPE,
    paths.VARIABLE_SCREEN_BASELINE,
)
OUTPUTS = (paths.TABLE_FEATURE_SCREEN,)


def numbers(
    screen: pd.DataFrame, prevalence: pd.DataFrame, shape: pd.DataFrame, baseline: dict[str, Any]
) -> dict[str, Any]:
    return {
        "screen.baseline_logloss": baseline["baseline_logloss"],
        "screen.n_records": baseline["n_records"],
        "screen.delta_logloss": dict(zip(screen["variable"], screen["delta_logloss"])),
        "screen.sd": dict(zip(screen["variable"], screen["sd"])),
        "screen.retained": screen.loc[screen["retained"], "variable"].tolist(),
        "screen.rejected": screen.loc[~screen["retained"], "variable"].tolist(),
        "screen.reaction_prevalence_pct": prevalence.set_index("class")[["type1_pct", "type2_pct"]].to_dict(),
        "screen.reaction_shape": shape.set_index("variable").to_dict(orient="index"),
    }


def run() -> None:
    screen = pd.read_csv(paths.VARIABLE_SCREEN)
    prevalence = pd.read_csv(paths.VARIABLE_SCREEN_REACTION_PREVALENCE)
    shape = pd.read_csv(paths.VARIABLE_SCREEN_SHAPE)
    baseline = json.loads(paths.VARIABLE_SCREEN_BASELINE.read_text(encoding="utf-8"))
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    screen.to_csv(paths.TABLE_FEATURE_SCREEN, index=False)
    record_numbers(numbers(screen, prevalence, shape, baseline))
    print(screen.to_string(index=False))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
