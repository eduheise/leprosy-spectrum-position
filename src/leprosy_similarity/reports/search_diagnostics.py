"""How the search converged and which choices mattered to each objective.

Input:  results/search/study.db, results/selected_configuration.json
Output: reports/figures/search_hypervolume.png   (fig:val:hypervolume)
        reports/figures/search_importance.png    (fig:val:hp-importance)
        reports/numbers.json, keys ``search.hypervolume.*`` and ``search.importance.*``

The hypervolume trace is the area the front dominates after each trial,
against a reference point just beyond the worst feasible value of each
objective. The importances come in two groups: the three design axes over
every trial (optuna's fANOVA), and the selected backbone's hyperparameters
over that backbone's trials (a random forest's impurity decrease). Both
estimators are seeded from ``config.yaml``, so the figure is the same on
every run.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import FrozenTrial
from sklearn.ensemble import RandomForestRegressor

from leprosy_similarity import paths
from leprosy_similarity.config import load_config
from leprosy_similarity.estimator import read_selection
from leprosy_similarity.reports.numbers import record_numbers
from leprosy_similarity.search import (
    BACKBONE_PARAM,
    DESIGN_PARAMS,
    MISSING_VALUES_PARAM,
    TARGET_HEAD_PARAM,
    feasible_trials,
    hypervolume_trace,
)

INPUTS = (paths.STUDY_DB, paths.SELECTED_CONFIGURATION)
OUTPUTS = (paths.FIG_SEARCH_HYPERVOLUME, paths.FIG_SEARCH_IMPORTANCE)

SEPARABILITY_COLOUR = "#3B6BB0"
ROBUSTNESS_COLOUR = "#C25E2E"
OBJECTIVES = ("separability", "robustness")
# Display names for axis labels; the search keeps its own parameter names.
PARAM_LABELS = {
    BACKBONE_PARAM: "backbone",
    MISSING_VALUES_PARAM: "missing-value strategy",
    TARGET_HEAD_PARAM: "model type",
}


def hypervolume_summary(
    trials: Sequence[FrozenTrial], margin: float, visible_share: float, early_trials: int
) -> tuple[list[int], list[float], dict[str, Any]]:
    """The trace, its reference point, its final value and the later trials that visibly raised it."""
    feasible = sorted(feasible_trials(trials), key=lambda t: t.number)
    reference = (max(t.values[0] for t in feasible) * margin, max(t.values[1] for t in feasible) * margin)
    trace = hypervolume_trace(feasible, reference)
    numbers = [t.number for t in feasible]
    final = trace[-1]
    rises = np.diff([0.0, *trace])
    visible = [n for n, rise in zip(numbers, rises) if n >= early_trials and rise >= visible_share * final]
    early = max((h for n, h in zip(numbers, trace) if n < early_trials), default=0.0)
    return numbers, trace, {
        "search.hypervolume.reference": list(reference),
        "search.hypervolume.final": final,
        "search.hypervolume.early_trials": early_trials,
        "search.hypervolume.share_after_early_trials": early / final,
        "search.hypervolume.visible_rises_after_early_trials": visible,
        "search.hypervolume.last_rise_trial": max(n for n, rise in zip(numbers, rises) if rise > 0),
    }


def design_importances(study: optuna.Study, objective: int, seed: int) -> dict[str, float]:
    """fANOVA share of each design axis in one objective's variance."""
    importance = get_param_importances(
        study,
        evaluator=FanovaImportanceEvaluator(seed=seed),
        params=list(DESIGN_PARAMS),
        target=lambda t: t.values[objective],
    )
    return {name: float(value) for name, value in importance.items()}


def hyperparameter_importances(
    trials: Sequence[FrozenTrial], params: Sequence[str], objective: int, n_trees: int, seed: int
) -> dict[str, float]:
    """Impurity-decrease importance of one backbone's hyperparameters, largest first.

    The forest is fitted on that backbone's feasible trials only, where every
    one of its hyperparameters is defined; a categorical value is coded by
    its position among the observed choices.
    """
    rows = [t for t in trials if all(p in t.params for p in params)]
    choices = {p: sorted({t.params[p] for t in rows if isinstance(t.params[p], str)}) for p in params}
    features = np.array([
        [choices[p].index(t.params[p]) if isinstance(t.params[p], str) else float(t.params[p]) for p in params]
        for t in rows
    ])
    target = np.array([t.values[objective] for t in rows])
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=seed).fit(features, target)
    return dict(sorted(zip(params, map(float, forest.feature_importances_)), key=lambda kv: -kv[1]))


def draw_hypervolume(numbers: list[int], trace: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(numbers, trace, "-", color="0.25", lw=1.6)
    ax.set_xlabel("trial")
    ax.set_ylabel("dominated hypervolume")
    ax.set_title("Search convergence: dominated hypervolume vs trial")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.FIG_SEARCH_HYPERVOLUME, dpi=150)
    plt.close(fig)


def draw_importances(
    design: tuple[dict[str, float], dict[str, float]],
    hyperparams: tuple[dict[str, float], dict[str, float]],
    backbone: str,
    prefix: str,
) -> None:
    """Two panels of paired bars, one bar per objective, ordered by separability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        (axes[0], design, "Design-choice importance (all backbones)", ""),
        (axes[1], hyperparams, f"{backbone_label(backbone)} hyperparameter importance", prefix),
    ]
    for ax, (separability, robustness), title, strip in panels:
        keys = list(separability) or list(robustness)
        y = np.arange(len(keys))
        height = 0.38
        ax.barh(y + height / 2, [separability.get(k, 0) for k in keys], height, color=SEPARABILITY_COLOUR,
                label="separability (objective 1)")
        ax.barh(y - height / 2, [robustness.get(k, 0) for k in keys], height, color=ROBUSTNESS_COLOUR,
                label="robustness (objective 2)")
        ax.set_yticks(y)
        ax.set_yticklabels([PARAM_LABELS.get(k, k.removeprefix(strip)) for k in keys], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("relative importance")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.FIG_SEARCH_IMPORTANCE, dpi=150)
    plt.close(fig)


def backbone_label(backbone: str) -> str:
    return {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "rf": "Random Forest"}[backbone]


def run() -> None:
    config = load_config()
    settings = config.search_report
    seed = settings.importance_seed
    study = optuna.load_study(study_name=config.search.study_name, storage=paths.study_storage())
    trials = study.trials
    selected = read_selection()
    backbone = selected["backbone"]
    space = config.search.space[backbone]
    hyperparams = [f"{space['prefix']}{name}" for name in space["params"]]

    numbers, trace, hypervolume = hypervolume_summary(
        trials, settings.hypervolume_margin, settings.visible_rise_share, settings.early_trials)
    design = tuple(design_importances(study, k, seed) for k in range(2))
    own_trials = [t for t in feasible_trials(trials) if t.params.get(BACKBONE_PARAM) == backbone]
    per_hyperparam = tuple(
        hyperparameter_importances(own_trials, hyperparams, k, settings.hyperparameter_importance_trees, seed)
        for k in range(2)
    )

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    draw_hypervolume(numbers, trace)
    draw_importances(design, per_hyperparam, backbone, space["prefix"])

    importance_numbers = {
        f"search.importance.design.{objective}": values for objective, values in zip(OBJECTIVES, design)
    } | {
        f"search.importance.hyperparameters.{objective}": {k.removeprefix(space["prefix"]): v
                                                           for k, v in values.items()}
        for objective, values in zip(OBJECTIVES, per_hyperparam)
    } | {"search.importance.hyperparameters.backbone": backbone, "search.importance.seed": seed}
    record_numbers(hypervolume | importance_numbers)

    for key, value in (hypervolume | importance_numbers).items():
        print(f"{key}: {value}")
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
