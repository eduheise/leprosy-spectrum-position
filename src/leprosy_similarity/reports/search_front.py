"""The search's Pareto front, the elbow walk along it and the selected configuration.

Input:  results/search/study.db, results/selected_configuration.json
Output: reports/figures/search_front_by_backbone.png        (fig:val:pareto)
        reports/figures/search_front_by_missing_values.png  (fig:val:pareto-mv)
        reports/tables/search_by_backbone.csv          (tab:val:backbone)
        reports/tables/search_by_missing_values.csv    (tab:val:imputation)
        reports/tables/search_by_target_head.csv
        reports/tables/selected_hyperparameters.csv    (tab:val:selected-hp)
        reports/numbers.json, keys ``search.*``

Every figure and table is computed from the stored trials; nothing is refitted.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.trial import FrozenTrial

from leprosy_similarity import paths
from leprosy_similarity.config import Config, load_config
from leprosy_similarity.estimator import read_selection
from leprosy_similarity.reports.numbers import record_numbers
from leprosy_similarity.search import (
    BACKBONE_PARAM,
    MISSING_VALUES_PARAM,
    TARGET_HEAD_PARAM,
    Selection,
    feasible_trials,
    select_configuration,
)

INPUTS = (paths.STUDY_DB, paths.SELECTED_CONFIGURATION)
OUTPUTS = (
    paths.FIG_SEARCH_FRONT_BY_BACKBONE,
    paths.FIG_SEARCH_FRONT_BY_MISSING_VALUES,
    paths.TABLE_SEARCH_BY_BACKBONE,
    paths.TABLE_SEARCH_BY_MISSING_VALUES,
    paths.TABLE_SEARCH_BY_TARGET_HEAD,
    paths.TABLE_SELECTED_HYPERPARAMETERS,
)

BACKBONE_COLOURS = {"lightgbm": "#4C78A8", "xgboost": "#F58518", "catboost": "#54A24B", "rf": "#E45756"}
MISSING_VALUES_COLOURS = {
    "mean_median": "#4C78A8",
    "knn": "#F58518",
    "iterative_br": "#54A24B",
    "miceforest": "#E45756",
    "native_routing": "#72B7B2",
    "mia_indicators": "#B279A2",
}
SELECTED_COLOUR = "#F58518"


def group_summary(trials: Sequence[FrozenTrial], front: Sequence[FrozenTrial], param: str) -> pd.DataFrame:
    """Per level of one design axis: all trials, feasible trials, best of each objective, front points.

    Rows are ordered by best separability, as in the thesis tables.
    """
    feasible = feasible_trials(trials)
    on_front = {t.number for t in front}
    rows = []
    for level in sorted({t.params[param] for t in trials if param in t.params}):
        scored = [t for t in feasible if t.params.get(param) == level]
        rows.append({
            param: level,
            "trials": sum(t.params.get(param) == level for t in trials),
            "feasible_trials": len(scored),
            "best_separability": min(t.values[0] for t in scored),
            "best_robustness": min(t.values[1] for t in scored),
            "on_front": sum(t.number in on_front for t in scored),
        })
    return pd.DataFrame(rows).sort_values("best_separability", kind="stable").reset_index(drop=True)


def selected_table(selected: dict[str, Any], config: Config) -> pd.DataFrame:
    """The selected hyperparameters, the backbone seed and the two objective values."""
    rows = [{"item": name, "value": value} for name, value in selected["hyperparams"].items()]
    rows.append({"item": "random seed", "value": config.backbones.seed})
    rows.append({"item": "class-balanced label MAE (objective 1)", "value": selected["label_mae_macro"]})
    rows.append({"item": "masking-drift MAE (objective 2)", "value": selected["masking_mae"]})
    # Object dtype keeps whole-number hyperparameters as integers in the CSV.
    return pd.DataFrame(rows, dtype=object)


def draw_front(
    trials: Sequence[FrozenTrial],
    selection: Selection,
    param: str,
    colours: dict[str, str],
    coloured_by: str,
    path: str,
) -> None:
    """Every feasible trial in the objective plane, the front, and the selected trial."""
    feasible = feasible_trials(trials)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for level, colour in colours.items():
        points = [t.values for t in feasible if t.params.get(param) == level]
        if points:
            ax.scatter([p[0] for p in points], [p[1] for p in points], s=22, alpha=0.5, color=colour, label=level)
    front = selection.front
    ax.plot([t.values[0] for t in front], [t.values[1] for t in front], "-o", color="0.25", lw=1.4, ms=5,
            zorder=4, label="Pareto front")
    pick = selection.trial
    ax.scatter([pick.values[0]], [pick.values[1]], marker="*", s=460, color=SELECTED_COLOUR, edgecolor="black",
               linewidth=1.3, zorder=6, label=f"selected model ({pick.params.get(param)}, trial {pick.number})")
    ax.set_xlabel("separability — class-balanced label MAE")
    ax.set_ylabel("robustness — masking-drift MAE")
    ax.set_title(f"Multi-objective tuning: Pareto front (coloured by {coloured_by})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def elbow_numbers(selection: Selection, threshold: float) -> dict[str, Any]:
    """The walk along the front as the thesis narrates it."""
    front, steps, stop = selection.front, selection.steps, selection.stop_index
    extreme, pick = front[0], selection.trial
    later = steps[stop + 1:]
    last_from, last_to = front[-2], front[-1]
    return {
        "search.elbow.threshold": threshold,
        "search.elbow.separability_extreme_trial": extreme.number,
        "search.elbow.separability_extreme_values": list(extreme.values),
        "search.elbow.robustness_extreme_trial": front[-1].number,
        "search.elbow.steps": [
            {"from": s.from_trial, "to": s.to_trial, "separability_cost": s.separability_cost,
             "drift_gain": s.drift_gain, "exchange": s.exchange}
            for s in steps
        ],
        "search.elbow.steps_taken": stop,
        "search.elbow.first_declined_step": (
            {"to": steps[stop].to_trial, "exchange": steps[stop].exchange} if stop < len(steps) else None
        ),
        "search.elbow.separability_given_up_vs_extreme": pick.values[0] / extreme.values[0] - 1,
        "search.elbow.drift_recovered_vs_extreme": 1 - pick.values[1] / extreme.values[1],
        "search.elbow.later_steps_clearing_threshold": sum(s.exchange > threshold for s in later),
        "search.elbow.last_step_separability_given_up": last_to.values[0] / last_from.values[0] - 1,
        "search.elbow.last_step_drift_recovered": 1 - last_to.values[1] / last_from.values[1],
    }


def table_numbers(name: str, table: pd.DataFrame, param: str) -> dict[str, Any]:
    return {f"search.by_{name}": {row[param]: {k: v for k, v in row.items() if k != param}
                                  for row in table.to_dict("records")}}


def run() -> None:
    config = load_config()
    threshold = config.selection.elbow_threshold
    study = optuna.load_study(study_name=config.search.study_name, storage=paths.study_storage())
    trials = study.trials
    selection = select_configuration(trials, threshold)
    selected = read_selection()
    if selected["trial"] != selection.trial.number:
        raise SystemExit(
            f"selected_configuration.json holds trial {selected['trial']} but the study selects "
            f"{selection.trial.number}; rerun python main.py experiments --only select_configuration"
        )

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    draw_front(trials, selection, BACKBONE_PARAM, BACKBONE_COLOURS, "backbone", paths.FIG_SEARCH_FRONT_BY_BACKBONE)
    draw_front(trials, selection, MISSING_VALUES_PARAM, MISSING_VALUES_COLOURS, "missing-value strategy",
               paths.FIG_SEARCH_FRONT_BY_MISSING_VALUES)

    by_backbone = group_summary(trials, selection.front, BACKBONE_PARAM)
    by_missing_values = group_summary(trials, selection.front, MISSING_VALUES_PARAM)
    by_head = group_summary(trials, selection.front, TARGET_HEAD_PARAM)
    by_backbone.rename(columns={BACKBONE_PARAM: "backbone"}).to_csv(paths.TABLE_SEARCH_BY_BACKBONE, index=False)
    by_missing_values.rename(columns={MISSING_VALUES_PARAM: "missing_values"}).to_csv(
        paths.TABLE_SEARCH_BY_MISSING_VALUES, index=False)
    by_head.to_csv(paths.TABLE_SEARCH_BY_TARGET_HEAD, index=False)
    selected_table(selected, config).to_csv(paths.TABLE_SELECTED_HYPERPARAMETERS, index=False)

    feasible = feasible_trials(trials)
    pick = selection.trial
    numbers: dict[str, Any] = {
        "search.n_trials": len(trials),
        "search.n_feasible": len(feasible),
        "search.n_infeasible_by_design": sum("infeasible" in t.user_attrs for t in trials),
        "search.n_failed_fits": sum("error" in t.user_attrs for t in trials),
        "search.front.size": len(selection.front),
        "search.front.trials": [t.number for t in selection.front],
        "search.front.by_backbone": by_backbone.set_index(BACKBONE_PARAM)["on_front"].to_dict(),
        "search.selected.trial": pick.number,
        "search.selected.backbone": selected["backbone"],
        "search.selected.target_head": selected["target_head"],
        "search.selected.missing_values": selected["missing_values"],
        "search.selected.hyperparams": selected["hyperparams"],
        "search.selected.backbone_seed": config.backbones.seed,
        "search.selected.label_mae_macro": pick.values[0],
        "search.selected.masking_mae": pick.values[1],
        "search.selected.label_mae_micro": pick.user_attrs.get("label_mae_micro"),
        "search.selected.class_means": json.loads(pick.user_attrs.get("class_means", "{}")),
        **table_numbers("backbone", by_backbone, BACKBONE_PARAM),
        **table_numbers("missing_values", by_missing_values, MISSING_VALUES_PARAM),
        **table_numbers("target_head", by_head, TARGET_HEAD_PARAM),
        **elbow_numbers(selection, threshold),
    }
    record_numbers(numbers)

    print(f"{len(trials)} trials, {len(feasible)} feasible, front of {len(selection.front)}; "
          f"selected trial {pick.number}")
    print(by_backbone.to_string(index=False))
    print(by_missing_values.to_string(index=False))
    print(by_head.to_string(index=False))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
