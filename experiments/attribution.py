"""Per-feature attribution of every record's held-out position.

Inputs:  data/processed/{cohort.parquet, reference_cases.parquet, folds.csv},
         results/selected_configuration.json
Outputs: results/attribution.npz             contributions per record and feature
         results/imputation_reconstruction.csv  how well the imputer rebuilds a hidden value

Each record is explained by the fold model that did not train on it, the same
model that gives its out-of-fold position, so the explanation describes the
position the results report. Tree Similarity SHAP decomposes the record's
similarity to each of the 26 reference cases into per-feature shares, which
project onto the position (``leprosy_similarity.attribution``). The
contributions are reported on the fold model's rescaled scale, that is
divided by its span ``hi - lo``; they then sum to the record's rescaled
position minus the rescaled base. That identity is checked for every record.

Because a missing finding is filled by the imputer before the trees see it, a
contribution may belong to an imputed value. Two measurements address this:
the contributions are kept with the record's observed-or-imputed mask, and
the imputer's reconstruction of observed values, hidden one fold of the shared
split at a time, is scored by MAE and R^2.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.attribution import (
    base_position,
    ensemble_trees,
    feature_contributions,
    model_inputs,
    pair_attributions,
)
from leprosy_similarity.config import Config, load_config
from leprosy_similarity.cross_validation import FoldModel, fit_fold_models
from leprosy_similarity.estimator import Configuration, load_selected_configuration
from leprosy_similarity.folds import fold_of
from leprosy_similarity.imputation import make_strategy
from leprosy_similarity.schema import FEATURES, NUMERIC_FEATURES

INPUTS = (paths.COHORT, paths.REFERENCE_CASES, paths.FOLDS, paths.SELECTED_CONFIGURATION)
OUTPUTS = (paths.ATTRIBUTION, paths.IMPUTATION_RECONSTRUCTION)


def explain_fold(
    model: FoldModel, cohort: pd.DataFrame, reference_cases: pd.DataFrame, n_jobs: int
) -> dict[str, np.ndarray]:
    """Attributions and contributions of the records held out of one fold."""
    estimator = model.estimator
    backbone = estimator.configuration.backbone
    held_out = cohort.iloc[model.test_rows][list(FEATURES)]
    filled = estimator.strategy.transform(held_out)
    references = estimator.strategy.transform(reference_cases[list(FEATURES)])
    trees = ensemble_trees(backbone, estimator.model)
    attributions, similarity_base = pair_attributions(
        trees, model_inputs(backbone, filled.to_numpy()), model_inputs(backbone, references.to_numpy()), n_jobs
    )
    raw_contributions = feature_contributions(attributions, estimator.reference_classes, similarity_base)
    scale = estimator.scale
    n = len(model.test_rows)
    return {
        "rows": model.test_rows,
        "fold": np.full(n, model.fold),
        "attributions": attributions,
        "contributions_raw": raw_contributions,
        "contributions": raw_contributions / scale.span,
        "raw_position": estimator.raw_position(held_out),
        "position": estimator.position(held_out),
        "scale_lo": np.full(n, scale.lo),
        "scale_hi": np.full(n, scale.hi),
        "filled": filled[list(FEATURES)].to_numpy(float),
    }


def efficiency_gaps(parts: dict[str, np.ndarray], base_raw: float) -> tuple[np.ndarray, np.ndarray]:
    """Gap between summed contributions and position minus base, raw and rescaled.

    The rescaled identity is checked against the linear map before clipping:
    a record beyond the fold model's reach is clipped to 0 or 1, but its
    contributions still describe its unclipped position.
    """
    span = parts["scale_hi"] - parts["scale_lo"]
    raw_gap = np.abs(parts["contributions_raw"].sum(axis=1) - (parts["raw_position"] - base_raw))
    linear = (parts["raw_position"] - parts["scale_lo"]) / span
    base = (base_raw - parts["scale_lo"]) / span
    gap = np.abs(parts["contributions"].sum(axis=1) - (linear - base))
    return raw_gap, gap


def explain_cohort(
    cohort: pd.DataFrame, reference_cases: pd.DataFrame, fold_models: list[FoldModel], n_jobs: int
) -> dict[str, np.ndarray]:
    """Every record explained by its own fold model, back in cohort order."""
    per_fold = []
    for model in fold_models:
        start = time.time()
        per_fold.append(explain_fold(model, cohort, reference_cases, n_jobs))
        print(f"  fold {model.fold}: {len(model.test_rows)} records in {time.time() - start:.1f} s", flush=True)
    order = np.argsort(np.concatenate([p["rows"] for p in per_fold]))
    return {key: np.concatenate([p[key] for p in per_fold])[order] for key in per_fold[0] if key != "rows"}


def reconstruction(
    cohort: pd.DataFrame, folds: pd.DataFrame, configuration: Configuration, config: Config
) -> pd.DataFrame:
    """Hide observed values one fold at a time, refill them, score against the truth.

    For each numeric finding, the observed values of one fold of the shared
    split are hidden, the selected missing-value strategy is refitted on the
    cohort with those values hidden, and the hidden values are filled.
    """
    records = cohort[list(FEATURES)].reset_index(drop=True)
    fold = fold_of(cohort, folds)
    rows = []
    for feature in NUMERIC_FEATURES:
        observed = records[feature].notna().to_numpy()
        truth, filled = [], []
        for k in np.unique(fold):
            hidden = observed & (fold == k)
            masked = records.copy()
            masked.loc[hidden, feature] = np.nan
            strategy = make_strategy(configuration.missing_values, config.imputation).fit(masked)
            truth.append(records.loc[hidden, feature].to_numpy(float))
            filled.append(strategy.transform(masked).loc[hidden, feature].to_numpy(float))
        truth_all, filled_all = np.concatenate(truth), np.concatenate(filled)
        spread = ((truth_all - truth_all.mean()) ** 2).sum()
        rows.append(
            {
                "feature": feature,
                "n_observed": int(observed.sum()),
                "observed_share": float(observed.mean()),
                "mae": float(np.abs(truth_all - filled_all).mean()),
                "r2": float(1 - ((truth_all - filled_all) ** 2).sum() / spread) if spread > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    config = load_config()
    settings = config.attribution
    cohort = pd.read_parquet(paths.COHORT).reset_index(drop=True)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    folds = pd.read_csv(paths.FOLDS)
    configuration = load_selected_configuration()
    reference_classes = reference_cases["class"].astype(str).to_numpy()

    fold_models = fit_fold_models(cohort, reference_cases, folds, configuration, config)
    start = time.time()
    parts = explain_cohort(cohort, reference_cases, fold_models, settings.n_jobs)
    seconds_per_record = (time.time() - start) / len(cohort)

    base_raw = base_position(reference_classes)
    raw_gap, gap = efficiency_gaps(parts, base_raw)
    print(f"efficiency: max gap {gap.max():.2e} rescaled, {raw_gap.max():.2e} raw; "
          f"{seconds_per_record:.2f} s per record")
    if gap.max() > settings.efficiency_tolerance:
        raise RuntimeError(f"contributions do not sum to the position: largest gap {gap.max():.3e}")

    span = parts["scale_hi"] - parts["scale_lo"]
    np.savez_compressed(
        paths.ATTRIBUTION,
        record_id=cohort["record_id"].to_numpy(),
        classes=cohort["class"].to_numpy(dtype=str),
        features=np.array(FEATURES, dtype=str),
        reference_classes=reference_classes.astype(str),
        observed=cohort[list(FEATURES)].notna().to_numpy(),
        base_raw=base_raw,
        base=(base_raw - parts["scale_lo"]) / span,
        efficiency_gap=gap,
        efficiency_gap_raw=raw_gap,
        seconds_per_record=seconds_per_record,
        **parts,
    )
    table = reconstruction(cohort, folds, configuration, config)
    table.to_csv(paths.IMPUTATION_RECONSTRUCTION, index=False)
    print(table.to_string(index=False))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
