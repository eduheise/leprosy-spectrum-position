"""Out-of-fold positions, recovery metrics and direction sweeps on synthetic records."""
from __future__ import annotations

import numpy as np
import pandas as pd

from leprosy_similarity.cross_validation import fit_fold_models, fold_of_rows, held_out_position
from leprosy_similarity.estimator import Configuration
from leprosy_similarity.folds import assign_folds
from leprosy_similarity.recovery import class_ordering, nearest_class, who_recovery, within_class_contrast
from leprosy_similarity.schema import CLASS_TARGETS, FEATURES
from leprosy_similarity.sweeps import direction, level_shifts, sweep_finding

SMALL_FOREST = Configuration(
    "rf", "multiclass", "knn", {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"}
)


def test_each_record_is_positioned_by_the_model_that_held_it_out(cohort, reference_cases, config):
    folds = assign_folds(cohort, n_folds=5, seed=42)
    models = fit_fold_models(cohort, reference_cases, folds, SMALL_FOREST, config)
    records = cohort[list(FEATURES)]
    position = held_out_position(models, records)
    for model in models:
        assert not set(model.train_rows) & set(model.test_rows)
        own = model.estimator.position(records.iloc[model.test_rows])
        np.testing.assert_array_equal(position[model.test_rows], own)
    assert np.all((position >= 0) & (position <= 1))
    np.testing.assert_array_equal(fold_of_rows(models, len(cohort)), folds["fold"].to_numpy())


def test_recovery_of_a_perfect_position():
    classes = np.array(["TT", "BT", "BB", "BL", "LL"] * 4)
    position = np.array([CLASS_TARGETS[c] for c in classes])
    order = class_ordering(position, classes)
    assert order["spearman"] == 1.0 and order["quadratic_kappa"] == 1.0
    assert order["mae"] == 0.0 and order["n_exact"] == len(classes)
    assert list(nearest_class(np.array([0.1, 0.4, 0.9]))) == ["TT", "BB", "LL"]

    is_mb = position >= 0.5
    who = who_recovery(position, is_mb)
    assert who["auc"] == 1.0 and who["agreement"] == 1.0 and who["youden_threshold"] == 0.5
    contrast = within_class_contrast(position, is_mb)
    assert contrast["rank_biserial"] == 1.0


def test_direction_reads_rises_falls_and_flats():
    assert direction(np.array([0.1, 0.2, 0.3]), 1e-4)["monotonicity"] == "monotone_up"
    assert direction(np.array([0.3, 0.2, 0.1]), 1e-4)["direction"] == "down"
    expected = {"direction": "up", "delta": 0.1, "monotonicity": "non_monotone"}
    assert direction(np.array([0.1, 0.3, 0.2]), 1e-4) == expected
    assert direction(np.array([0.5, 0.50001]), 1e-4)["direction"] == "flat"


def test_sweep_flags_a_violation_only_against_the_expectation(cohort):
    records = cohort[list(FEATURES)].reset_index(drop=True)

    def falls_with_smear(edited: pd.DataFrame) -> np.ndarray:
        return 1.0 - edited["BACILOSCOP_N"].fillna(3.0).to_numpy() / 6.0

    expected = sweep_finding(falls_with_smear, records, "BACILOSCOP_N", expected_up=True, tolerance=1e-4)
    assert expected["direction"] == "down" and expected["violation"]
    assert expected["missing_mean"] == 0.5
    free = sweep_finding(falls_with_smear, records, "BACILOSCOP_N", expected_up=False, tolerance=1e-4)
    assert not free["violation"]


def test_level_shift_is_zero_for_a_score_blind_to_the_level(cohort):
    records = cohort[list(FEATURES)].reset_index(drop=True)

    def smear_only(edited: pd.DataFrame) -> np.ndarray:
        return edited["BACILOSCOP_N"].fillna(0.0).to_numpy()

    rows = level_shifts(smear_only, records, smear_only(records))
    assert len(rows) == 9
    assert all(r["mean_shift"] == 0.0 and r["sd_shift"] == 0.0 for r in rows)
