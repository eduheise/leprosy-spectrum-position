"""The confidence sweep and its calibration, on synthetic records."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from leprosy_similarity.confidence import (
    band_at_level,
    case_split,
    coverage_by_size,
    dispersion,
    exact_raw_position,
    make_completions,
    multipliers,
    select_arm,
)
from leprosy_similarity.confidence_sweep import run_sweep, summarise
from leprosy_similarity.estimator import Configuration, SpectrumEstimator
from leprosy_similarity.folds import assign_folds, fold_of
from leprosy_similarity.masking_sweep import build_masking_batch, sample_cases
from leprosy_similarity.platform_bundle import build_bundle, write_bundle
from leprosy_similarity.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

SMALL_FOREST = Configuration(
    "rf", "multiclass", "knn", {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"}
)
SMALL_BOOSTER = Configuration(
    "xgboost",
    "multiclass",
    "knn",
    {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
     "min_child_weight": 1},
)
FAST_ARMS = ("knn_donor", "bayesian_ridge")


@pytest.fixture
def small_config(config):
    """The project configuration with a sweep small enough for a unit test."""
    confidence = dataclasses.replace(config.confidence, records_per_class=4, masks_per_record=3, donor_draws=4)
    return dataclasses.replace(config, confidence=confidence)


def test_masks_hide_only_observed_findings_and_repeat(cohort):
    cases = sample_cases(cohort, list(FEATURES), 3, len(FEATURES) - 1, seed=42)
    batch = build_masking_batch(cases, list(FEATURES), 4, (2, 3), seed=42)
    again = build_masking_batch(cases, list(FEATURES), 4, (2, 3), seed=42)
    assert len(batch.registry) == 4 * len(cases)
    pd.testing.assert_frame_equal(batch.masked, again.masked)
    for i, row in batch.registry.iterrows():
        hidden = row["masked_features"].split(",")
        assert row["mask_size"] == len(hidden) and row["mask_size"] in (2, 3)
        assert batch.full.loc[i, hidden].notna().all()
        assert batch.masked.loc[i, hidden].isna().all()
        kept = [c for c in FEATURES if c not in hidden]
        pd.testing.assert_series_equal(batch.full.loc[i, kept], batch.masked.loc[i, kept])


def test_sample_draws_per_class_among_near_complete_records(cohort):
    cases = sample_cases(cohort, list(FEATURES), 3, len(FEATURES) - 1, seed=42)
    assert (cases["n_present"] >= len(FEATURES) - 1).all()
    assert cases["class"].value_counts().max() <= 3
    assert cases["record_id"].is_unique


def test_case_split_keeps_a_record_on_one_side():
    cases = np.repeat(np.arange(10), 5)
    held = case_split(cases, seed=7)
    assert held.sum() == 25
    for case in range(10):
        assert len(set(held[cases == case])) == 1
    np.testing.assert_array_equal(held, case_split(cases, seed=7))


def test_band_calibrated_on_one_half_covers_the_other():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.05, 0.2, 4000)
    drift = sigma * np.abs(rng.normal(0, 1.5, 4000))
    held = np.arange(4000) % 2 == 1
    coverage, width, n = band_at_level(sigma, drift, ~held, held, 0.9)
    assert n == 2000
    assert abs(coverage - 0.9) < 0.03
    assert width > 0


def test_zero_sigma_pairs_inform_no_multiplier_but_count_when_held_out():
    sigma = np.array([0.0, 0.1, 0.1, 0.1, 0.0, 0.0])
    drift = np.array([0.3, 0.05, 0.1, 0.05, 0.2, 0.0])
    held = np.array([False, False, False, True, True, True])
    coverage, _, n = band_at_level(sigma, drift, ~held, held, 1.0)
    # k = 1.0 from the two calibration pairs with a band; the drifting
    # held-out pair with no band is a miss, the one that does not drift is covered.
    assert n == 3
    assert coverage == pytest.approx(2 / 3)


def test_selected_arm_is_narrowest_at_most_levels():
    table = pd.DataFrame(
        {
            "level": [0.8, 0.8, 0.9, 0.9, 0.95, 0.95],
            "arm": ["a", "b"] * 3,
            "coverage": [0.8] * 6,
            "mean_width": [0.20, 0.21, 0.40, 0.30, 0.50, 0.45],
        }
    )
    assert select_arm(table) == "b"


def test_donor_draws_copy_real_values_and_keep_observed_ones(cohort, config):
    pool = cohort[list(FEATURES)]
    query = cohort[list(FEATURES)].iloc[:6].copy()
    query.iloc[:, [0, 3]] = np.nan
    completions = make_completions("knn_donor", pool, config.confidence, config.imputation, seed=42)(query)
    assert len(completions) == config.confidence.donor_draws
    for completed in completions:
        assert not completed.isna().any().any()
        observed = query.notna()
        pd.testing.assert_frame_equal(completed[observed], query[observed])
        for column in ("NU_LESOES", "PGL1"):
            assert set(completed[column]) <= set(pool[column].dropna())
    again = make_completions("knn_donor", pool, config.confidence, config.imputation, seed=42)(query)
    for a, b in zip(completions, again):
        pd.testing.assert_frame_equal(a, b)


def test_bayesian_draws_stay_in_range_with_binary_levels(cohort, config):
    pool = cohort[list(FEATURES)]
    query = pool.iloc[:10].copy()
    query.iloc[:, [0, 4, 6]] = np.nan
    completions = make_completions("bayesian_ridge", pool, config.confidence, config.imputation, seed=42)(query)
    for completed in completions:
        numeric = completed[list(NUMERIC_FEATURES)]
        assert (numeric >= pool[list(NUMERIC_FEATURES)].min()).all().all()
        assert (numeric <= pool[list(NUMERIC_FEATURES)].max()).all().all()
        assert set(np.unique(completed[list(CATEGORICAL_FEATURES)])) <= {0.0, 1.0}


def test_every_pair_is_scored_by_a_model_that_held_its_record_out(cohort, reference_cases, small_config):
    folds = assign_folds(cohort, n_folds=5, seed=42)
    pairs, models = run_sweep(cohort, reference_cases, folds, SMALL_FOREST, small_config, FAST_ARMS)
    record_fold = dict(zip(cohort["record_id"], fold_of(cohort, folds)))
    assert (pairs["fold"] == pairs["record_id"].map(record_fold)).all()
    for model in models:
        trained_on = set(cohort.iloc[model.train_rows]["record_id"])
        assert not trained_on & set(pairs.loc[pairs["fold"] == model.fold, "record_id"])
    assert (pairs["drift"] >= 0).all() and (pairs["drift"] <= 1).all()
    for arm in FAST_ARMS:
        assert (pairs[f"sigma_{arm}"] >= 0).all()

    again, _ = run_sweep(cohort, reference_cases, folds, SMALL_FOREST, small_config, FAST_ARMS)
    pd.testing.assert_frame_equal(pairs, again)

    summary = summarise(pairs, models, SMALL_FOREST, FAST_ARMS, small_config)
    assert summary["selected_arm"] in FAST_ARMS
    assert summary["n_pairs"] == len(pairs)
    json.dumps(summary)


def test_multiplier_grows_with_the_requested_coverage(config):
    rng = np.random.default_rng(1)
    n = 400
    pairs = pd.DataFrame(
        {
            "case": np.repeat(np.arange(80), 5),
            "mask_size": np.tile([2, 3, 4, 5, 5], 80),
            "sigma_knn_donor": rng.uniform(0.02, 0.2, n),
        }
    )
    pairs["drift"] = pairs["sigma_knn_donor"] * np.abs(rng.normal(0, 1.5, n))
    held = case_split(pairs["case"].to_numpy(), seed=7)
    calibrated = multipliers(pairs, "knn_donor", held, (0.8, 0.9, 0.95), config.confidence.curve_grid)
    ks = [calibrated[level][0] for level in (0.8, 0.9, 0.95)]
    assert ks == sorted(ks)
    sizes, coverage = coverage_by_size(pairs, "knn_donor", held, np.array([0.5, 0.9]))
    assert list(sizes) == [2, 3, 4, 5]
    assert coverage.shape == (2, 4)
    assert ((coverage >= 0) & (coverage <= 100)).all()


def test_platform_bundle_and_manifest(cohort, reference_cases, config, tmp_path):
    estimator = SpectrumEstimator(SMALL_BOOSTER, reference_cases, config).fit(cohort)
    bundle = build_bundle(estimator, cohort, reference_cases, {0.9: 2.0, 0.95: 3.0}, 10, 20, 5, 42)
    assert bundle["band_k"] == {"0.90": 2.0, "0.95": 3.0}
    assert bundle["prototypes"].shape == (len(reference_cases), len(FEATURES))
    assert not np.isnan(bundle["prototypes"]).any()
    assert bundle["donor_pool"].shape == (len(cohort), len(FEATURES))
    trees = bundle["trees"]
    assert len(set(trees["tree"])) == estimator.model.get_booster().num_boosted_rounds() * 5
    assert (trees["yes"][trees["feature"] < 0] == -1).all()

    bundle_path, manifest_path = tmp_path / "model.joblib", tmp_path / "manifest.json"
    manifest = write_bundle(bundle, bundle_path, manifest_path, {"fitted_on": "synthetic"})
    assert manifest["sha256"] == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert json.loads(manifest_path.read_text())["files"] == {"model.joblib": manifest["sha256"]}
    loaded = joblib.load(bundle_path)
    np.testing.assert_array_equal(
        loaded["model"].apply(bundle["prototypes"]), estimator.model.apply(bundle["prototypes"])
    )


def test_bundle_refuses_a_backbone_the_platform_cannot_explain(cohort, reference_cases, config):
    estimator = SpectrumEstimator(SMALL_FOREST, reference_cases, config).fit(cohort)
    with pytest.raises(NotImplementedError):
        build_bundle(estimator, cohort, reference_cases, {0.9: 2.0}, 10, 20, 5, 42)


def test_exact_position_agrees_and_identical_completions_have_no_spread(cohort, reference_cases, config):
    estimator = SpectrumEstimator(SMALL_FOREST, reference_cases, config).fit(cohort)
    records = cohort[list(FEATURES)]
    np.testing.assert_allclose(exact_raw_position(estimator, records), estimator.raw_position(records), atol=1e-12)
    complete = estimator.strategy.transform(records.iloc[:5])
    assert (dispersion(estimator, [complete, complete.copy(), complete.copy()]) == 0).all()
