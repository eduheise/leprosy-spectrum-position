"""The reachable range of the selected configuration and the map onto [0, 1].

Input:  results/position_rescaling.json
Output: reports/numbers.json, keys ``rescaling.*``

The thesis quotes these numbers in the text; it shows no figure of them.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from leprosy_similarity import paths
from leprosy_similarity.reports.numbers import record_numbers

INPUTS = (paths.POSITION_RESCALING,)
OUTPUTS = ()


def rescaling_numbers(result: dict[str, Any]) -> dict[str, Any]:
    """Every number the thesis quotes about the range and the rescaling."""
    raw = result["reference_positions_raw"]
    rescaled = result["reference_positions_rescaled"]
    fold_lo = [f["lo"] for f in result["fold_ranges"]]
    fold_hi = [f["hi"] for f in result["fold_ranges"]]
    return {
        "rescaling.n_grid_records": result["n_grid_records"],
        "rescaling.reachable_min": result["lo"],
        "rescaling.reachable_max": result["hi"],
        "rescaling.span": result["span"],
        "rescaling.argmin_record": result["argmin"],
        "rescaling.argmax_record": result["argmax"],
        "rescaling.base_raw": result["base_raw"],
        "rescaling.base_rescaled": result["base_rescaled"],
        "rescaling.anchors_rescaled": result["anchors_rescaled"],
        "rescaling.weight_shares_at_max": result["weight_shares_at_max"],
        "rescaling.weight_shares_at_min": result["weight_shares_at_min"],
        "rescaling.bb_weight_at_max": result["weight_shares_at_max"]["BB"],
        "rescaling.reference_raw": {
            c: {"min": min(v), "max": max(v), "mean": float(np.mean(v))} for c, v in raw.items()
        },
        "rescaling.reference_rescaled_mean": {c: float(np.mean(v)) for c, v in rescaled.items()},
        "rescaling.cohort_raw_span": result["cohort_raw_span"],
        "rescaling.cohort_outside_range": result["cohort_outside_range"],
        "rescaling.cohort_rescaled_max": result["cohort_rescaled_span"][1],
        "rescaling.cohort_at_floor": result["cohort_at_floor"],
        "rescaling.cohort_at_ceiling": result["cohort_at_ceiling"],
        "rescaling.fold_ranges": result["fold_ranges"],
        "rescaling.fold_floor_range": [min(fold_lo), max(fold_lo)],
        "rescaling.fold_ceiling_range": [min(fold_hi), max(fold_hi)],
    }



def run() -> None:
    result = json.loads(paths.POSITION_RESCALING.read_text(encoding="utf-8"))
    numbers = rescaling_numbers(result)
    record_numbers(numbers)

    for key in ("rescaling.reachable_min", "rescaling.reachable_max", "rescaling.span", "rescaling.base_rescaled",
                "rescaling.bb_weight_at_max", "rescaling.reference_rescaled_mean", "rescaling.fold_floor_range",
                "rescaling.fold_ceiling_range"):
        print(f"{key}: {numbers[key]}")


if __name__ == "__main__":
    run()
