"""The confidence sweep, scored on held-out records of the shared split.

Each (record, mask) pair of the masking sweep (``masking_sweep.py``) is
scored by the fold model that did not train on its record, with the
completion generators of ``confidence.py`` fitted on that fold's training
records. ``summarise`` turns the scored pairs into every quantity the
results report: the imputer comparison, the selected imputer, its
multiplier per requested coverage, the fixed band and the zero-drift share.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from leprosy_similarity.confidence import (
    ARMS,
    POLE_CLASSES,
    case_split,
    dispersion,
    fixed_band_by_size,
    make_completions,
    multipliers,
    select_arm,
    sharpness_table,
)
from leprosy_similarity.config import Config
from leprosy_similarity.cross_validation import FoldModel, fit_fold_models
from leprosy_similarity.estimator import Configuration
from leprosy_similarity.folds import fold_of
from leprosy_similarity.masking_sweep import (
    MaskingBatch,
    build_masking_batch,
    build_single_finding_batch,
    sample_cases,
)
from leprosy_similarity.schema import FEATURES


def sweep_cases(cohort: pd.DataFrame, config: Config) -> pd.DataFrame:
    """The sampled records: near-complete records, a fixed number per class.

    A record is near-complete when it lacks at most one finding, a threshold
    derived from the schema width so it cannot exceed it.
    """
    settings = config.confidence
    return sample_cases(cohort, list(FEATURES), settings.records_per_class, len(FEATURES) - 1, config.seed)


def masking_batch(cohort: pd.DataFrame, config: Config) -> MaskingBatch:
    """The sweep: the sampled records, several random masks each."""
    settings = config.confidence
    cases = sweep_cases(cohort, config)
    return build_masking_batch(cases, list(FEATURES), settings.masks_per_record, settings.mask_sizes, config.seed)


def single_finding_pairs(
    cohort: pd.DataFrame, folds: pd.DataFrame, fold_models: list[FoldModel], arm: str, config: Config
) -> pd.DataFrame:
    """The same sampled records with one finding hidden at a time.

    Scored by the same fold models with the selected arm, for the by-size
    figure only: these pairs take no part in calibrating ``k`` or in any
    coverage the sweep reports.
    """
    batch = build_single_finding_batch(sweep_cases(cohort, config), list(FEATURES))
    record_fold = pd.Series(fold_of(cohort, folds), index=cohort["record_id"].to_numpy())
    registry = batch.registry.assign(fold=record_fold.loc[batch.registry["record_id"]].to_numpy())
    scored = []
    for model in fold_models:
        rows = registry.index[registry["fold"] == model.fold]
        pool = cohort.iloc[model.train_rows][list(FEATURES)]
        scored.append(score_pairs(model, pool, batch.full.loc[rows], batch.masked.loc[rows], config, (arm,)))
    return registry.join(pd.concat(scored))


def score_pairs(
    model: FoldModel,
    pool: pd.DataFrame,
    full: pd.DataFrame,
    masked: pd.DataFrame,
    config: Config,
    arms: tuple[str, ...] = ARMS,
) -> pd.DataFrame:
    """Drift and per-arm ``sigma`` of pairs scored by one fold model.

    ``pool`` is the fold's training records: the imputers that generate the
    completions are fitted on them, never on the held-out record.
    """
    estimator = model.estimator
    position_full = estimator.position(full)
    position_masked = estimator.position(masked)
    scored = pd.DataFrame(
        {
            "position_full": position_full,
            "position_masked": position_masked,
            "drift": np.abs(position_full - position_masked),
        },
        index=full.index,
    )
    for arm in arms:
        completions = make_completions(arm, pool, config.confidence, config.imputation, config.seed)
        scored[f"sigma_{arm}"] = dispersion(estimator, completions(masked))
        # The same record with nothing hidden, the zero-test end of the sweep:
        # only its own gaps are completed. Drawn after the masked completions,
        # so it does not change them.
        scored[f"sigma_full_{arm}"] = dispersion(estimator, completions(full))
    return scored


def run_sweep(
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    folds: pd.DataFrame,
    configuration: Configuration,
    config: Config,
    arms: tuple[str, ...] = ARMS,
) -> tuple[pd.DataFrame, list[FoldModel]]:
    """Every pair of the sweep, scored on the held-out fold of its record."""
    batch = masking_batch(cohort, config)
    record_fold = pd.Series(fold_of(cohort, folds), index=cohort["record_id"].to_numpy())
    registry = batch.registry.assign(fold=record_fold.loc[batch.registry["record_id"]].to_numpy())
    fold_models = fit_fold_models(cohort, reference_cases, folds, configuration, config)
    scored = []
    for model in fold_models:
        rows = registry.index[registry["fold"] == model.fold]
        pool = cohort.iloc[model.train_rows][list(FEATURES)]
        scored.append(score_pairs(model, pool, batch.full.loc[rows], batch.masked.loc[rows], config, arms))
    pairs = registry.join(pd.concat(scored))
    return pairs, fold_models


def calibration_split(pairs: pd.DataFrame, config: Config) -> np.ndarray:
    """Held-out half of the pairs, by record."""
    return case_split(pairs["case"].to_numpy(), config.confidence.calibration_split_seed)


def rank_agreement(pairs: pd.DataFrame, arm: str, multiplier: float) -> float:
    """Spearman correlation between the confidence ``1 - k sigma`` and the drift.

    A useful confidence orders records by how far they actually move, so the
    correlation should be negative.
    """
    confidence = np.clip(1.0 - multiplier * pairs[f"sigma_{arm}"], 0.0, 1.0)
    return float(spearmanr(confidence, pairs["drift"]).statistic)


def summarise(
    pairs: pd.DataFrame,
    fold_models: list[FoldModel],
    configuration: Configuration,
    arms: tuple[str, ...],
    config: Config,
) -> dict[str, Any]:
    """Every quantity the confidence results report, from the scored pairs."""
    settings = config.confidence
    held = calibration_split(pairs, config)
    table = sharpness_table(pairs, arms, settings.selection_levels, held)
    arm = select_arm(table)
    sigma = pairs[f"sigma_{arm}"]
    levels = sorted(set(settings.levels) | set(settings.curve_levels))
    calibrated = multipliers(pairs, arm, held, levels, settings.curve_grid)
    fixed = settings.fixed_band_multiplier
    by_size = fixed_band_by_size(pairs, arm, fixed)
    confidence = np.clip(1.0 - fixed * sigma, 0.0, 1.0)
    pole = pairs["class"].isin(POLE_CLASSES)
    return {
        "configuration": configuration.to_dict(),
        "fold_scales": {str(m.fold): {"lo": m.estimator.scale.lo, "hi": m.estimator.scale.hi} for m in fold_models},
        "n_records": int(pairs["case"].nunique()),
        "n_pairs": len(pairs),
        "pairs_by_fold": {str(k): int(v) for k, v in pairs["fold"].value_counts().sort_index().items()},
        "split": {
            "held_out_records": int(pairs.loc[held, "case"].nunique()),
            "held_out_pairs": int(held.sum()),
            "calibration_pairs": int((~held).sum()),
            "calibration_pairs_nonzero_sigma": int(((~held) & (sigma > 0)).sum()),
        },
        "sharpness": table.to_dict(orient="records"),
        "selected_arm": arm,
        "rank_agreement": {a: rank_agreement(pairs, a, fixed) for a in arms},
        "multipliers": {f"{level:g}": {"k": k, "held_out_coverage": c} for level, (k, c) in calibrated.items()},
        "fixed_band": {
            "multiplier": fixed,
            "coverage": float((pairs["drift"] <= fixed * sigma).mean()),
            "by_size": by_size.to_dict(orient="records"),
        },
        "zero_drift": {"n": int((pairs["drift"] == 0).sum()), "share": float((pairs["drift"] == 0).mean())},
        "mean_confidence": {
            "by_class": {str(c): float(v) for c, v in confidence.groupby(pairs["class"]).mean().items()},
            "poles": float(confidence[pole].mean()),
            "middle": float(confidence[~pole].mean()),
        },
    }

