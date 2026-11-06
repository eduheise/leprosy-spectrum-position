"""The configuration search, the elbow rule and the shared numbers store."""
from __future__ import annotations

import copy
import itertools
import json
import math

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution
from optuna.trial import FixedTrial, TrialState, create_trial

from leprosy_similarity.config import parse_config
from leprosy_similarity.folds import assign_folds
from leprosy_similarity.reports.numbers import read_numbers, record_numbers
from leprosy_similarity.search import (
    build_objective,
    elbow_index,
    feasible_trials,
    front_steps,
    hypervolume_trace,
    open_study,
    pareto_front,
    remaining_trials,
    select_configuration,
)


def trial(number: int, values: tuple[float, float], state: TrialState = TrialState.COMPLETE, **params):
    """A stored trial with the given objective values and design choices."""
    distributions = {k: CategoricalDistribution([v]) for k, v in params.items()}
    frozen = create_trial(
        state=state,
        values=list(values) if state == TrialState.COMPLETE else None,
        params=params,
        distributions=distributions,
    )
    frozen.number = number
    return frozen


def brute_force_hypervolume(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Staircase area of the non-dominated points only, computed the long way."""
    front = [p for p in points if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points)]
    inside = sorted(p for p in set(front) if p[0] < reference[0] and p[1] < reference[1])
    area, floor = 0.0, reference[1]
    for x, y in inside:
        if y < floor:
            area += (reference[0] - x) * (floor - y)
            floor = y
    return area


@pytest.fixture
def small_config(config):
    """The project configuration with a search space small enough for a test."""
    raw = copy.deepcopy(config.raw)
    raw["search"]["backbones"] = ["xgboost", "rf"]
    raw["search"]["missing_value_strategies"] = ["native_routing", "mean_median"]
    raw["search"]["masking_sizes"] = [2, 4]
    space = raw["search"]["space"]
    space["xgboost"]["params"]["n_estimators"] = {"type": "int", "low": 5, "high": 8}
    space["xgboost"]["params"]["max_depth"] = {"type": "int", "low": 2, "high": 3}
    space["xgboost"]["params"]["min_child_weight"] = {"type": "int", "low": 1, "high": 2}
    space["rf"]["params"]["n_estimators"] = {"type": "int", "low": 5, "high": 8}
    raw["backbones"]["xgboost"]["fixed"]["n_jobs"] = 1
    raw["backbones"]["rf"]["fixed"]["n_jobs"] = 1
    return parse_config(raw)


@pytest.fixture
def small_cohort(cohort):
    """A cohort whose recordable grid is small, so each fold's rescaling is quick."""
    trimmed = cohort.copy()
    for feature in ("PGL1", "BACILOSCOP_N", "HISTOLOG"):
        trimmed[feature] = trimmed[feature].clip(upper=2)
    return trimmed


def test_front_keeps_only_non_dominated_feasible_trials():
    trials = [
        trial(0, (0.10, 0.30)),
        trial(1, (0.12, 0.20)),
        trial(2, (0.13, 0.25)),  # dominated by trial 1
        trial(3, (math.inf, math.inf)),  # infeasible
        trial(4, (0.20, 0.10)),
        trial(5, (0.0, 0.0), state=TrialState.FAIL),
    ]
    assert [t.number for t in feasible_trials(trials)] == [0, 1, 2, 4]
    assert [t.number for t in pareto_front(trials)] == [0, 1, 4]


def test_elbow_walks_in_while_each_step_is_worth_taking():
    # Exchanges along the front: 10, 5, then 0.125 -> the walk stops at index 2.
    front = [trial(n, v) for n, v in enumerate([(0.10, 0.30), (0.11, 0.20), (0.12, 0.15), (0.20, 0.14)])]
    steps = front_steps(front)
    assert [round(s.exchange, 6) for s in steps] == [10.0, 5.0, 0.125]
    assert elbow_index(steps, threshold=1.0) == 2
    selection = select_configuration(front, threshold=1.0)
    assert selection.trial.number == 2
    assert selection.stop_index == 2


def test_elbow_does_not_pass_a_declined_step_to_reach_a_later_good_one():
    # The second step is declined; the favourable third step is never reached.
    front = [trial(n, v) for n, v in enumerate([(0.10, 0.30), (0.11, 0.20), (0.20, 0.19), (0.21, 0.05)])]
    assert select_configuration(front, threshold=1.0).trial.number == 1


def test_elbow_on_a_single_point_front():
    assert select_configuration([trial(7, (0.1, 0.1))], threshold=1.0).trial.number == 7


def test_selection_without_feasible_trials_is_an_error():
    with pytest.raises(ValueError):
        select_configuration([trial(0, (math.inf, math.inf))], threshold=1.0)


def test_hypervolume_trace_equals_the_non_dominated_staircase():
    rng = np.random.default_rng(0)
    points = [(float(a), float(b)) for a, b in rng.random((60, 2))]
    points += [points[3], (points[5][0], points[5][1] + 0.01)]  # a duplicate and a tie on objective 1
    trials = [trial(n, p) for n, p in enumerate(points)]
    reference = (1.02, 1.02)
    trace = hypervolume_trace(trials, reference)
    expected = [brute_force_hypervolume(points[: k + 1], reference) for k in range(len(points))]
    assert trace == expected
    assert all(b >= a for a, b in itertools.pairwise(trace))


def test_excluded_pairing_is_infeasible_without_a_fit(small_cohort, reference_cases, small_config):
    folds = assign_folds(small_cohort, small_config.folds.n_folds, small_config.folds.seed)
    objective = build_objective(small_cohort, reference_cases, folds, small_config)
    fixed = FixedTrial({"classifier": "rf", "target_head": "anchored", "mv_strategy": "native_routing"})
    assert objective(fixed) == (math.inf, math.inf)
    assert fixed.user_attrs["infeasible"] == "rf + native_routing"


def test_objective_scores_a_configuration(small_cohort, reference_cases, small_config):
    folds = assign_folds(small_cohort, small_config.folds.n_folds, small_config.folds.seed)
    objective = build_objective(small_cohort, reference_cases, folds, small_config)
    fixed = FixedTrial({
        "classifier": "xgboost", "target_head": "multiclass", "mv_strategy": "mean_median",
        "xgb_n_estimators": 6, "xgb_max_depth": 2, "xgb_learning_rate": 0.1, "xgb_subsample": 0.9,
        "xgb_colsample_bytree": 0.9, "xgb_min_child_weight": 1,
    })
    separability, drift = objective(fixed)
    assert 0 <= separability <= 1 and 0 <= drift <= 1
    assert fixed.user_attrs["label_mae_macro"] == separability
    assert set(json.loads(fixed.user_attrs["class_means"])) == {"TT", "BT", "BB", "BL", "LL"}


def test_same_seed_same_search(small_cohort, reference_cases, small_config):
    """Two searches from the same sampler seed draw and score the same trials."""
    folds = assign_folds(small_cohort, small_config.folds.n_folds, small_config.folds.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    runs = []
    for name in ("first", "second"):
        study = open_study(None, small_config, study_name=name)
        study.optimize(build_objective(small_cohort, reference_cases, folds, small_config), n_trials=4)
        runs.append([(t.params, t.values) for t in study.trials])
    assert runs[0] == runs[1]
    assert sum(all(math.isfinite(v) for v in values) for _, values in runs[0]) >= 2


def test_a_finished_study_runs_no_more_trials(small_config):
    study = open_study(None, small_config, study_name="finished")
    study.add_trial(trial(0, (0.1, 0.2)))
    assert remaining_trials(study, 1) == 0
    assert remaining_trials(study, 3) == 2


def test_numbers_store_merges_keys(tmp_path):
    store = tmp_path / "numbers.json"
    record_numbers({"search.a": np.float64(0.5), "search.b": [np.int64(1), 2]}, path=store)
    record_numbers({"rescaling.c": {"TT": np.float32(0.25)}, "search.a": 0.75, "x": math.nan}, path=store)
    assert read_numbers(store) == {"search.a": 0.75, "search.b": [1, 2], "rescaling.c": {"TT": 0.25}, "x": None}
    assert list(json.loads(store.read_text())) == sorted(read_numbers(store))
