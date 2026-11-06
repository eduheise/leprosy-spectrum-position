"""The final model: the selected configuration fitted on the whole analytic cohort.

Input:  data/processed/cohort.parquet, reference_cases.parquet,
        results/selected_configuration.json, results/confidence_summary.json
Output: results/final_model/model.joblib, results/final_model/manifest.json

This is the model the clinical platform serves. It is never used to evaluate
the method: every reported result reads a record through a fold model that
did not train on it. The band multipliers it ships are the ones calibrated on
held-out records by the confidence experiment, for the arm that experiment
selected.
"""
from __future__ import annotations

import json
from importlib.metadata import version

import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.config import load_config
from leprosy_similarity.estimator import SpectrumEstimator, load_selected_configuration
from leprosy_similarity.platform_bundle import build_bundle, git_head, sha256, write_bundle
from leprosy_similarity.rescaling import recordable_grid
from leprosy_similarity.schema import FEATURES

INPUTS = (paths.COHORT, paths.REFERENCE_CASES, paths.SELECTED_CONFIGURATION, paths.CONFIDENCE_SUMMARY)
OUTPUTS = (paths.FINAL_MODEL_BUNDLE, paths.FINAL_MODEL_MANIFEST)

# Libraries whose versions decide the fitted model and its positions.
NUMERICAL_LIBRARIES = ("xgboost", "scikit-learn", "numpy", "pandas")


def band_multipliers(summary: dict, levels: tuple[float, ...]) -> dict[float, float]:
    """``k`` per coverage the platform offers, from the held-out calibration."""
    calibrated = summary["multipliers"]
    return {level: calibrated[f"{level:g}"]["k"] for level in levels}


def run() -> None:
    config = load_config()
    cohort = pd.read_parquet(paths.COHORT)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    configuration = load_selected_configuration()
    summary = json.loads(paths.CONFIDENCE_SUMMARY.read_text(encoding="utf-8"))
    settings = config.confidence
    if summary["selected_arm"] != "knn_donor":
        raise NotImplementedError(
            f"the platform draws its band from kNN donors, but the confidence experiment selected "
            f"{summary['selected_arm']!r}"
        )

    estimator = SpectrumEstimator(configuration, reference_cases, config).fit(cohort)
    bundle = build_bundle(
        estimator,
        cohort,
        reference_cases,
        band_multipliers(summary, settings.levels),
        n_grid_records=len(recordable_grid(cohort[list(FEATURES)])),
        donor_draws=settings.donor_draws,
        donor_neighbours=settings.donor_neighbours,
        donor_seed=config.seed,
    )
    provenance = {
        "configuration": configuration.to_dict(),
        "selected_configuration_sha256": sha256(paths.SELECTED_CONFIGURATION),
        "confidence_summary_sha256": sha256(paths.CONFIDENCE_SUMMARY),
        "fitted_on": f"analytic cohort, n={len(cohort)}",
        "class_counts": cohort["class"].value_counts().to_dict(),
        "scale": bundle["scale"],
        "band_arm": summary["selected_arm"],
        "band_k": bundle["band_k"],
        "source_commit": git_head(paths.ROOT),
        "versions": {name: version(name) for name in NUMERICAL_LIBRARIES},
    }
    manifest = write_bundle(bundle, paths.FINAL_MODEL_BUNDLE, paths.FINAL_MODEL_MANIFEST, provenance)
    print(
        f"final model on {len(cohort)} records: range [{bundle['scale']['lo']:.4f}, {bundle['scale']['hi']:.4f}], "
        f"band k {', '.join(f'{p}: {k:.2f}' for p, k in bundle['band_k'].items())}; sha256 {manifest['sha256'][:12]}"
    )
    print(f"wrote {paths.FINAL_MODEL_BUNDLE} and {paths.FINAL_MODEL_MANIFEST}")


if __name__ == "__main__":
    run()
