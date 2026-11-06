"""Derivation of the analytic cohort: the record-flow figure and its table.

Input:  data/processed/exclusion_attrition.csv
Output: reports/figures/exclusion_attrition.png  (fig:exclusion-attrition)
        reports/tables/exclusion_attrition.csv
        reports/numbers.json, keys ``cohort.attrition.*``

Records are removed in sequence: the eleven literature exclusion criteria
(clinical first, laboratory next), then the completeness criterion. Each step
is drawn with the number of records it removes.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from leprosy_similarity import paths
from leprosy_similarity.reports.numbers import record_numbers

INPUTS = (paths.EXCLUSION_ATTRITION,)
OUTPUTS = (paths.FIG_EXCLUSION_ATTRITION, paths.TABLE_EXCLUSION_ATTRITION)

GROUP_COLOURS = {"clinical": "#4C78A8", "laboratory": "#F58518", "completeness": "#54A24B"}
DPI = 150


def plot_attrition(attrition: pd.DataFrame, start: int) -> None:
    """Records remaining after each step, each step annotated with its removals."""
    labels = ["start", *attrition["step"]]
    counts = [start, *attrition["remaining"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(counts))
    ax.step(xs, counts, where="post", color="0.4", linewidth=1.2, zorder=1)
    ax.scatter(xs, counts, color="0.3", s=18, zorder=2)
    for k, row in attrition.reset_index(drop=True).iterrows():
        ax.annotate(f"-{int(row['removed'])}", (k + 1, counts[k + 1]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=GROUP_COLOURS[row["group"]])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("records remaining")
    ax.set_title("Derivation of the analytic cohort: sequential exclusion")
    ax.margins(x=0.02, y=0.10)
    ax.set_ylim(top=start * 1.06)
    fig.tight_layout()
    fig.savefig(paths.FIG_EXCLUSION_ATTRITION, dpi=DPI)
    plt.close(fig)


def numbers(attrition: pd.DataFrame, start: int) -> dict[str, Any]:
    criteria = attrition[attrition["group"] != "completeness"]
    return {
        "cohort.attrition.intake": start,
        "cohort.attrition.removed_by_step": dict(zip(attrition["step"], attrition["removed"].astype(int))),
        "cohort.attrition.removed_by_criteria": int(criteria["removed"].sum()),
        "cohort.attrition.after_criteria": int(criteria["remaining"].iloc[-1]),
        "cohort.attrition.removed_by_completeness": int(attrition["removed"].iloc[-1]),
        "cohort.attrition.analytic": int(attrition["remaining"].iloc[-1]),
    }


def run() -> None:
    attrition = pd.read_csv(paths.EXCLUSION_ATTRITION).sort_values("order")
    start = int(attrition["remaining"].iloc[0] + attrition["removed"].iloc[0])
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    plot_attrition(attrition, start)
    attrition.to_csv(paths.TABLE_EXCLUSION_ATTRITION, index=False)
    record_numbers(numbers(attrition, start))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
