import numpy as np

from leprosy_similarity.estimator import Configuration, SpectrumEstimator
from leprosy_similarity.folds import assign_folds, iter_folds
from leprosy_similarity.objectives import evaluate_configuration

SMALL_FOREST = Configuration(
    "rf", "multiclass", "knn", {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"}
)


def test_no_record_is_on_both_sides_of_a_fold(cohort):
    folds = assign_folds(cohort, n_folds=5, seed=42)
    seen = []
    for _, train, test in iter_folds(cohort, folds):
        assert not set(train) & set(test)
        assert len(train) + len(test) == len(cohort)
        seen.extend(test)
    assert sorted(seen) == list(range(len(cohort)))
    assert assign_folds(cohort, 5, 42).equals(folds)


def test_positions_lie_in_unit_interval_and_repeat(cohort, reference_cases, config):
    first = SpectrumEstimator(SMALL_FOREST, reference_cases, config).fit(cohort).position(cohort)
    second = SpectrumEstimator(SMALL_FOREST, reference_cases, config).fit(cohort).position(cohort)
    assert np.all((first >= 0) & (first <= 1))
    np.testing.assert_array_equal(first, second)


def test_objectives_repeat_with_the_seed(cohort, reference_cases, config):
    folds = assign_folds(cohort, n_folds=5, seed=42)
    a = evaluate_configuration(cohort, reference_cases, folds, SMALL_FOREST, config)
    b = evaluate_configuration(cohort, reference_cases, folds, SMALL_FOREST, config)
    assert a["label_mae_macro"] == b["label_mae_macro"]
    assert a["masking_mae"] == b["masking_mae"]
    assert 0 <= a["label_mae_macro"] <= 1 and 0 <= a["masking_mae"] <= 1
