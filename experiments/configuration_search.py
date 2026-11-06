"""Search the configuration space with NSGA-II on the shared 5-fold split.

Input:  data/processed/cohort.parquet, reference_cases.parquet, folds.csv
Output: results/search/study.db  (Optuna study; one trial per configuration)

The full search runs ``search.n_trials`` trials (500, about 13 hours). The
step only runs the trials the stored study is missing, so rerunning it on a
finished study changes nothing.

    python -m experiments.configuration_search --smoke

runs a few trials into a temporary study instead, and compares each with
the trial of the same number in ``results/search/study.db`` when that study
exists: with the same sampler seed the first trials draw the same
configurations, so their objective values must agree.
"""
from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import optuna
import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.config import Config, load_config
from leprosy_similarity.search import build_objective, open_study, remaining_trials

INPUTS = (paths.COHORT, paths.REFERENCE_CASES, paths.FOLDS)
OUTPUTS = (paths.STUDY_DB,)

SMOKE_TRIALS = 4


def search(storage: str, n_trials: int, config: Config, study_name: str | None = None) -> optuna.Study:
    """Bring the study at ``storage`` up to ``n_trials`` trials."""
    cohort = pd.read_parquet(paths.COHORT)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    folds = pd.read_csv(paths.FOLDS)
    study = open_study(storage, config, study_name)
    todo = remaining_trials(study, n_trials)
    print(f"study '{study.study_name}': {len(study.trials)} trials stored, {todo} to run")
    if todo:
        objective = build_objective(cohort, reference_cases, folds, config)
        study.optimize(objective, n_trials=todo)
    return study


def run() -> None:
    config = load_config()
    paths.SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = search(paths.study_storage(), config.search.n_trials, config)
    print(f"{len(study.trials)} trials; {len(study.best_trials)} on the Pareto front")


def compare_with_stored(smoke: optuna.Study, config: Config) -> None:
    """Print each smoke trial next to the stored trial of the same number."""
    if not paths.STUDY_DB.exists():
        print("no stored study to compare with")
        return
    stored = {t.number: t for t in optuna.load_study(
        study_name=config.search.study_name, storage=paths.study_storage()).trials}
    for trial in smoke.trials:
        reference = stored.get(trial.number)
        if reference is None:
            continue
        same_params = trial.params == reference.params
        both = list(zip(trial.values, reference.values))
        difference = max(0.0 if math.isinf(a) and math.isinf(b) else abs(a - b) for a, b in both)
        print(
            f"trial {trial.number}: {trial.params.get('classifier')}/{trial.params.get('mv_strategy')}/"
            f"{trial.params.get('target_head')}  values {trial.values}  stored {reference.values}  "
            f"same params {same_params}  max |difference| {difference}"
        )


def smoke(n_trials: int) -> None:
    """A few trials into a throwaway study, checked against the stored search."""
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        storage = f"sqlite:///{Path(folder) / 'smoke.db'}"
        study = search(storage, n_trials, config)
        compare_with_stored(study, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="run a few trials into a temporary study")
    parser.add_argument("--trials", type=int, default=SMOKE_TRIALS, help="trials for --smoke")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.trials)
    else:
        run()
