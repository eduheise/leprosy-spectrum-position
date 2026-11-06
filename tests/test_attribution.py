"""Tree Similarity SHAP: exact efficiency on the rescaled position, with fold models."""
from __future__ import annotations

import numpy as np
import pytest

from leprosy_similarity.attribution import (
    base_position,
    ensemble_trees,
    feature_contributions,
    model_inputs,
    pair_attributions,
)
from leprosy_similarity.cross_validation import fit_fold_models
from leprosy_similarity.estimator import Configuration
from leprosy_similarity.folds import assign_folds
from leprosy_similarity.schema import FEATURES

# XGBoost's cover stands in for a leaf's sample count; a leaf whose cover
# rounds below two has no pair to share it and breaks the identity, so the
# small model keeps every leaf at a cover of at least three, as the selected
# configuration (minimum child weight 7) does.
SMALL = {
    "xgboost": Configuration(
        "xgboost", "multiclass", "knn",
        {"n_estimators": 8, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9,
         "colsample_bytree": 0.8, "min_child_weight": 3},
    ),
    "lightgbm": Configuration(
        "lightgbm", "anchored", "knn",
        {"n_estimators": 8, "num_leaves": 7, "max_depth": 3, "learning_rate": 0.1, "min_child_samples": 5,
         "feature_fraction": 0.9, "bagging_fraction": 0.9},
    ),
}


@pytest.mark.parametrize("backbone", sorted(SMALL))
def test_contributions_sum_to_rescaled_position_minus_base(backbone, cohort, reference_cases, config):
    folds = assign_folds(cohort, n_folds=5, seed=42)
    model = fit_fold_models(cohort, reference_cases, folds, SMALL[backbone], config)[0]
    estimator = model.estimator
    held_out = cohort.iloc[model.test_rows][list(FEATURES)]
    filled = estimator.strategy.transform(held_out).to_numpy()
    references = estimator.strategy.transform(reference_cases[list(FEATURES)]).to_numpy()
    trees = ensemble_trees(backbone, estimator.model)
    assert len(trees) == estimator.reference_leaves.shape[1]

    attributions, similarity_base = pair_attributions(
        trees, model_inputs(backbone, filled), model_inputs(backbone, references), n_jobs=1
    )
    contributions = feature_contributions(attributions, estimator.reference_classes, similarity_base)
    raw = estimator.raw_position(held_out)
    base = base_position(estimator.reference_classes)
    assert np.max(np.abs(contributions.sum(axis=1) - (raw - base))) <= 1e-12

    scale = estimator.scale
    rescaled = contributions / scale.span
    linear = (raw - scale.lo) / scale.span
    assert np.max(np.abs(rescaled.sum(axis=1) - (linear - (base - scale.lo) / scale.span))) <= 1e-12


def test_parallel_split_does_not_change_attributions(cohort, reference_cases, config):
    folds = assign_folds(cohort, n_folds=5, seed=42)
    estimator = fit_fold_models(cohort, reference_cases, folds, SMALL["xgboost"], config)[0].estimator
    records = model_inputs("xgboost", estimator.strategy.transform(cohort[list(FEATURES)].iloc[:12]).to_numpy())
    references = model_inputs("xgboost", estimator.strategy.transform(reference_cases[list(FEATURES)]).to_numpy())
    trees = ensemble_trees("xgboost", estimator.model)
    serial, base_serial = pair_attributions(trees, records, references, n_jobs=1)
    parallel, base_parallel = pair_attributions(trees, records, references, n_jobs=2)
    np.testing.assert_array_equal(serial, parallel)
    assert base_serial == base_parallel


def test_base_position_is_mean_anchor():
    assert base_position(np.array(["TT", "LL"])) == 0.5
    assert base_position(np.array(["TT", "TT", "BB", "LL"])) == pytest.approx(0.375)
