"""Select one configuration from the search's Pareto front with the elbow rule.

Input:  results/search/study.db
Output: results/selected_configuration.json

The file holds the configuration (backbone, target head, missing-value
strategy, hyperparameters) that every later experiment refits, the trial it
came from and that trial's two objective values. Nothing downstream searches
again.
"""
from __future__ import annotations

import json

import optuna

from leprosy_similarity import paths
from leprosy_similarity.config import load_config
from leprosy_similarity.search import configuration_of, select_configuration

INPUTS = (paths.STUDY_DB,)
OUTPUTS = (paths.SELECTED_CONFIGURATION,)


def run() -> None:
    config = load_config()
    study = optuna.load_study(study_name=config.search.study_name, storage=paths.study_storage())
    selection = select_configuration(study.trials, config.selection.elbow_threshold)
    trial = selection.trial
    selected = {
        **configuration_of(trial, config).to_dict(),
        "trial": trial.number,
        "label_mae_macro": trial.values[0],
        "masking_mae": trial.values[1],
    }
    paths.SELECTED_CONFIGURATION.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(
        f"selected trial {trial.number} ({selected['backbone']} / {selected['missing_values']} / "
        f"{selected['target_head']}), step {selection.stop_index} of a {len(selection.front)}-point front: "
        f"class-balanced MAE {trial.values[0]:.4f}, masking drift {trial.values[1]:.4f}"
    )
    print(f"wrote {paths.SELECTED_CONFIGURATION}")


if __name__ == "__main__":
    run()
