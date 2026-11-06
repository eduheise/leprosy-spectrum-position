"""Out-of-fold positions of the selected configuration, and what they recover.

Inputs:  data/processed/{cohort.parquet, reference_cases.parquet, folds.csv},
         results/selected_configuration.json
Outputs: results/oof_positions.parquet  one held-out position per record
         results/spectrum_position.json  recovery, ordering and sweeps

For each fold of the shared split, the selected configuration is fitted on the
other four folds, with its own rescaling range, and positions the held-out
fold. Pooled over the five folds, every record has one position from a model
that never saw it. On those positions:

- WHO recovery: ROC-AUC for multibacillary treatment, the Youden cut, and
  the increment of the position over bacilloscopy alone (logistic models
  scored on the same five folds);
- Ridley-Jopling ordering: Spearman, quadratic kappa against the nearest
  class target, error per class, silhouette, and the multibacillary against
  paucibacillary contrast within the borderline-tuberculoid class;
- direction sweeps: each finding set to each of its values, and to missing,
  for every record, with the record re-positioned by its own fold model.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.config import Config, load_config
from leprosy_similarity.cross_validation import FoldModel, fit_fold_models, fold_of_rows, held_out_position
from leprosy_similarity.estimator import load_selected_configuration
from leprosy_similarity.folds import iter_folds
from leprosy_similarity.recovery import bacilloscopy_increment, class_ordering, who_recovery, within_class_contrast
from leprosy_similarity.schema import FEATURES
from leprosy_similarity.sweeps import level_shifts, sweep_all

INPUTS = (paths.COHORT, paths.REFERENCE_CASES, paths.FOLDS, paths.SELECTED_CONFIGURATION)
OUTPUTS = (paths.OOF_POSITIONS, paths.SPECTRUM_POSITION)

# The class whose multibacillary and paucibacillary members are contrasted:
# the one class the WHO rule splits in this cohort.
CONTRAST_CLASS = "BT"


def oof_table(cohort: pd.DataFrame, fold_models: list[FoldModel]) -> pd.DataFrame:
    """One row per record: its fold, labels, and held-out position (raw and rescaled)."""
    records = cohort[list(FEATURES)]
    return pd.DataFrame(
        {
            "record_id": cohort["record_id"].to_numpy(),
            "fold": fold_of_rows(fold_models, len(cohort)),
            "class": cohort["class"].astype(str).to_numpy(),
            "target": cohort["target"].to_numpy(float),
            "who_class": cohort["who_class"].to_numpy(),
            "raw_position": held_out_position(fold_models, records, rescaled=False),
            "position": held_out_position(fold_models, records),
            "has_bacilloscopy": cohort["BACILOSCOP_N"].notna().to_numpy(),
            "has_histology": cohort["HISTOLOG"].notna().to_numpy(),
        }
    )


def recovery(cohort: pd.DataFrame, oof: pd.DataFrame, folds: pd.DataFrame, config: Config) -> dict[str, Any]:
    """WHO recovery, the within-class contrast and the increment over bacilloscopy."""
    has_who = oof["who_class"].notna().to_numpy()
    position = oof["position"].to_numpy()[has_who]
    is_mb = (oof["who_class"].to_numpy()[has_who] == "MB")
    classes = oof["class"].to_numpy()[has_who]
    in_class = classes == CONTRAST_CLASS
    rated = cohort.loc[has_who].reset_index(drop=True)
    splits = [(train, test) for _, train, test in iter_folds(rated, folds)]
    return {
        "who": who_recovery(position, is_mb),
        "within_class": {
            "class": CONTRAST_CLASS,
            **within_class_contrast(position[in_class], is_mb[in_class]),
        },
        "bacilloscopy_increment": bacilloscopy_increment(
            position,
            rated["BACILOSCOP_N"].to_numpy(float),
            is_mb,
            splits,
            config.spectrum_position.logistic_max_iter,
        ),
    }


def direction_sweeps(cohort: pd.DataFrame, fold_models: list[FoldModel], config: Config) -> dict[str, Any]:
    """Response of the held-out position to each finding and each categorical level."""
    settings = config.spectrum_position
    records = cohort[list(FEATURES)].reset_index(drop=True)

    def score(edited: pd.DataFrame) -> np.ndarray:
        return held_out_position(fold_models, edited)

    baseline = score(records)
    sweeps = sweep_all(score, records, FEATURES, settings.expected_lepromatous, settings.direction_tolerance)
    return {
        "baseline_mean_position": float(np.nanmean(baseline)),
        "findings": sweeps,
        "violations": [s["feature"] for s in sweeps if s["violation"]],
        "levels": level_shifts(score, records, baseline),
    }


def print_summary(summary: dict[str, Any]) -> None:
    who, order = summary["recovery"]["who"], summary["ordering"]
    increment = summary["recovery"]["bacilloscopy_increment"]
    print(f"WHO: AUC {who['auc']:.4f}, Youden cut {who['youden_threshold']:.4f}, "
          f"agreement {who['agreement']:.4f}, sensitivity {who['sensitivity']:.4f}, "
          f"specificity {who['specificity']:.4f}")
    for fill in ("zero", "median"):
        r = increment[fill]
        print(f"bacilloscopy ({fill} fill): alone {r['auc_bacilloscopy']:.4f}, "
              f"with position {r['auc_bacilloscopy_and_position']:.4f}, increment {r['increment']:+.4f}")
    print(f"ordering: Spearman {order['spearman']:.4f}, kappa {order['quadratic_kappa']:.4f}, "
          f"MAE {order['mae']:.4f} (class-balanced {order['mae_class_balanced']:.4f}), "
          f"silhouette {order['silhouette']:.4f}")
    print("class means: " + ", ".join(f"{c} {v:.3f}" for c, v in order["per_class_mean"].items()))
    for s in summary["sweeps"]["findings"]:
        flag = "  VIOLATION" if s["violation"] else ""
        print(f"  {s['feature']:36s} {s['direction']:4s} {s['monotonicity']:13s} delta {s['delta']:+.4f}{flag}")


def run() -> None:
    config = load_config()
    cohort = pd.read_parquet(paths.COHORT).reset_index(drop=True)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    folds = pd.read_csv(paths.FOLDS)
    configuration = load_selected_configuration()

    fold_models = fit_fold_models(cohort, reference_cases, folds, configuration, config)
    oof = oof_table(cohort, fold_models)
    oof.to_parquet(paths.OOF_POSITIONS, index=False)

    summary = {
        "configuration": configuration.to_dict(),
        "n_folds": len(fold_models),
        "fold_scales": [m.estimator.scale.to_dict() for m in fold_models],
        "recovery": recovery(cohort, oof, folds, config),
        "ordering": class_ordering(oof["position"].to_numpy(), oof["class"].to_numpy()),
        "sweeps": direction_sweeps(cohort, fold_models, config),
    }
    paths.SPECTRUM_POSITION.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
