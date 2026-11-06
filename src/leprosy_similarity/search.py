"""The configuration search and the rule that picks one configuration from it.

The search is a two-objective NSGA-II over the backbone, the target head, the
missing-value strategy and the backbone's hyperparameters. Each trial is
scored by ``objectives.evaluate_configuration`` on the shared 5-fold split:
objective 1 is the class-balanced label error, objective 2 the masking drift,
both minimised. No weight between the two is imposed; the search returns a
Pareto front, and ``select_configuration`` walks it with the elbow rule.

Optuna stores each trial's choices under fixed names (``classifier``,
``target_head``, ``mv_strategy`` and the backbone-prefixed hyperparameters),
so a stored study can be read back and replayed with
``Configuration.from_trial_params``.
"""
from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler
from optuna.trial import FrozenTrial, TrialState

from leprosy_similarity.backbones import suggest_hyperparams
from leprosy_similarity.config import Config
from leprosy_similarity.estimator import Configuration
from leprosy_similarity.objectives import evaluate_configuration

INFEASIBLE = (float("inf"), float("inf"))

# Optuna names of the three design axes, in the order they are drawn.
BACKBONE_PARAM = "classifier"
TARGET_HEAD_PARAM = "target_head"
MISSING_VALUES_PARAM = "mv_strategy"
DESIGN_PARAMS = (BACKBONE_PARAM, MISSING_VALUES_PARAM, TARGET_HEAD_PARAM)

# Diagnostics stored with each scored trial, besides the two objective values.
RECORDED_DIAGNOSTICS = ("label_mae_macro", "masking_mae", "label_mae_micro", "elapsed_s")


def build_objective(
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    folds: pd.DataFrame,
    config: Config,
) -> Callable[[optuna.Trial], tuple[float, float]]:
    """The function NSGA-II minimises: one configuration in, two errors out.

    The design axes are drawn first, in a fixed order, so the sampler's random
    stream is consumed the same way on every run. An excluded pairing returns
    infinite errors before any hyperparameter is drawn, which keeps it off the
    front without spending a fit. A configuration whose fit fails is scored the
    same way, with the error kept on the trial, rather than stopping a 500-trial
    search.
    """
    search = config.search
    infeasible = {tuple(pair) for pair in search.infeasible}

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        backbone = trial.suggest_categorical(BACKBONE_PARAM, list(search.backbones))
        target_head = trial.suggest_categorical(TARGET_HEAD_PARAM, list(search.target_heads))
        missing_values = trial.suggest_categorical(MISSING_VALUES_PARAM, list(search.missing_value_strategies))
        if (backbone, missing_values) in infeasible:
            trial.set_user_attr("infeasible", f"{backbone} + {missing_values}")
            return INFEASIBLE
        configuration = Configuration(
            backbone=backbone,
            target_head=target_head,
            missing_values=missing_values,
            hyperparams=suggest_hyperparams(trial, backbone, search.space),
        )
        try:
            result = evaluate_configuration(cohort, reference_cases, folds, configuration, config)
        except Exception as exc:  # noqa: BLE001 - any failed fit scores as infeasible
            trial.set_user_attr("error", repr(exc))
            return INFEASIBLE
        for key in RECORDED_DIAGNOSTICS:
            trial.set_user_attr(key, result[key])
        trial.set_user_attr("class_means", json.dumps(result["class_means"]))
        return result["label_mae_macro"], result["masking_mae"]

    return objective


def open_study(storage: str | None, config: Config, study_name: str | None = None) -> optuna.Study:
    """Create the two-objective study, or reopen it to continue a search (``None``: in memory)."""
    return optuna.create_study(
        study_name=study_name or config.search.study_name,
        storage=storage,
        sampler=NSGAIISampler(seed=config.search.sampler_seed),
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )


def remaining_trials(study: optuna.Study, n_trials: int) -> int:
    """Trials still to run for the study to hold ``n_trials``.

    Counting the stored trials makes the step safe to rerun: a finished study
    is left untouched. A search resumed after an interruption is valid but not
    bit-identical to an uninterrupted one, since the sampler is reseeded.
    """
    return max(0, n_trials - len(study.trials))


def feasible_trials(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    """Completed trials with two finite objective values."""
    return [
        t
        for t in trials
        if t.state == TrialState.COMPLETE and t.values and all(math.isfinite(v) for v in t.values)
    ]


def dominates(a: FrozenTrial, b: FrozenTrial) -> bool:
    """``a`` is no worse than ``b`` on both objectives and better on one."""
    return all(x <= y for x, y in zip(a.values, b.values)) and any(x < y for x, y in zip(a.values, b.values))


def pareto_front(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    """Non-dominated feasible trials, ordered by objective 1 (separability first)."""
    candidates = feasible_trials(trials)
    front = [t for t in candidates if not any(dominates(o, t) for o in candidates)]
    return sorted(front, key=lambda t: t.values[0])


@dataclass(frozen=True)
class ElbowStep:
    """One step along the front, from the separability end toward robustness."""

    from_trial: int
    to_trial: int
    separability_cost: float
    drift_gain: float

    @property
    def exchange(self) -> float:
        """Drift gained per unit of separability given up."""
        return self.drift_gain / self.separability_cost if self.separability_cost > 0 else math.inf


@dataclass(frozen=True)
class Selection:
    """The selected trial, the front it was walked on and every step along it."""

    trial: FrozenTrial
    front: list[FrozenTrial]
    steps: list[ElbowStep]
    # Position of the selected trial on the front (0 = separability extreme).
    stop_index: int


def front_steps(front: Sequence[FrozenTrial]) -> list[ElbowStep]:
    """Cost and gain of every step between neighbouring front points."""
    return [
        ElbowStep(
            from_trial=a.number,
            to_trial=b.number,
            separability_cost=b.values[0] - a.values[0],
            drift_gain=a.values[1] - b.values[1],
        )
        for a, b in itertools.pairwise(front)
    ]


def elbow_index(steps: Sequence[ElbowStep], threshold: float) -> int:
    """How many steps the walk takes before the first step not worth making.

    Starting at the separability extreme, take each step while the drift it
    buys exceeds ``threshold`` times the separability it costs. Both
    objectives are mean absolute errors on the same 0-1 position scale, so a
    threshold of one compares like with like. A step that costs nothing is
    always taken.
    """
    pick = 0
    for i, step in enumerate(steps):
        if step.separability_cost <= 0:
            pick = i + 1
            continue
        if step.drift_gain / step.separability_cost <= threshold:
            break
        pick = i + 1
    return pick


def select_configuration(trials: Sequence[FrozenTrial], threshold: float) -> Selection:
    """Elbow pick from the Pareto front of the feasible trials.

    The front's endpoints cannot be knees, so the rule does not take the most
    accurate configuration; it walks in from it while the exchange stays
    favourable, and stops at the first step that is not.
    """
    front = pareto_front(trials)
    if not front:
        raise ValueError("the study has no feasible trial")
    steps = front_steps(front)
    index = elbow_index(steps, threshold)
    return Selection(trial=front[index], front=front, steps=steps, stop_index=index)


def configuration_of(trial: FrozenTrial, config: Config) -> Configuration:
    """The configuration a stored trial evaluated."""
    return Configuration.from_trial_params(trial.params, config.search.space)


def hypervolume_trace(trials: Sequence[FrozenTrial], reference: tuple[float, float]) -> list[float]:
    """Area dominated by the trials seen so far, after each trial in number order.

    For two minimised objectives the dominated region below ``reference`` is
    a staircase: sweeping the points by objective 1, each point that lowers
    the running minimum of objective 2 adds the rectangle between it and the
    reference. Dominated points never lower that minimum, so they add nothing
    and need not be removed first.
    """
    seen: list[tuple[float, float]] = []
    trace = []
    for trial in sorted(trials, key=lambda t: t.number):
        seen.append((trial.values[0], trial.values[1]))
        inside = sorted(p for p in set(seen) if p[0] < reference[0] and p[1] < reference[1])
        area, floor = 0.0, reference[1]
        for x, y in inside:
            if y < floor:
                area += (reference[0] - x) * (floor - y)
                floor = y
        trace.append(area)
    return trace
