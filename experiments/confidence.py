"""Confidence under missingness: does the band cover the drift of a held-out record?

Input:  data/processed/cohort.parquet, reference_cases.parquet, folds.csv,
        results/selected_configuration.json
Output: results/confidence.parquet (one row per record and mask),
        results/confidence_summary.json,
        results/confidence_single_finding.parquet (one finding hidden at a time)

A masking sweep is drawn from the analytic cohort: records that observe every
finding but one, a fixed number per class, and from each several random
subsets of observed findings to hide. Each (record, mask) pair is scored by
the fold model of the shared split that did not train on the record. A record
the model trained on is partly memorised, so its drift when findings are
hidden would understate what happens to a new patient.

For each pair, the drift is the distance between the rescaled position of
the masked record and of the record as observed, both filled by the selected
missing-value strategy. Each masked record is then completed ``M`` times by
each candidate imputer, fitted on the fold's training records, and ``sigma``
is the spread of the ``M`` rescaled positions.

The sampled records are split in half. On one half the multiplier ``k`` is
calibrated as a quantile of ``drift / sigma``; on the other the coverage it
delivers is measured. The imputer whose band is narrowest at the requested
coverages is selected, and its ``k`` per requested coverage is what the final
model ships.

The same records are also scored with each observed finding hidden on its
own, the one-test end of the by-size figure. Those pairs are kept apart from
the sweep and change none of its numbers.
"""
from __future__ import annotations

import json

import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.confidence import ARMS
from leprosy_similarity.confidence_sweep import run_sweep, single_finding_pairs, summarise
from leprosy_similarity.config import load_config
from leprosy_similarity.estimator import load_selected_configuration

INPUTS = (paths.COHORT, paths.REFERENCE_CASES, paths.FOLDS, paths.SELECTED_CONFIGURATION)
OUTPUTS = (paths.CONFIDENCE, paths.CONFIDENCE_SUMMARY, paths.CONFIDENCE_SINGLE_FINDING)


def run() -> None:
    config = load_config()
    cohort = pd.read_parquet(paths.COHORT)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    folds = pd.read_csv(paths.FOLDS)
    configuration = load_selected_configuration()
    pairs, fold_models = run_sweep(cohort, reference_cases, folds, configuration, config)
    summary = summarise(pairs, fold_models, configuration, ARMS, config)
    paths.CONFIDENCE.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(paths.CONFIDENCE, index=False)
    paths.CONFIDENCE_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    single = single_finding_pairs(cohort, folds, fold_models, summary["selected_arm"], config)
    single.to_parquet(paths.CONFIDENCE_SINGLE_FINDING, index=False)
    print(
        f"{summary['n_pairs']} pairs on {summary['n_records']} held-out records; "
        f"selected arm {summary['selected_arm']}; "
        + ", ".join(f"k({p}) = {m['k']:.2f}" for p, m in summary["multipliers"].items())
    )


if __name__ == "__main__":
    run()
