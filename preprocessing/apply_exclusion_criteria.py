"""Apply the exclusion criteria and the completeness criterion (1,030 -> 916).

Input:  data/processed/raw_cohort.parquet
Output: data/processed/cohort.parquet           the analytic cohort
        data/processed/exclusion_audit.csv      one row per raw record
        data/processed/exclusion_attrition.csv  records removed at each step
        data/processed/schema.json              the schema the cohort is written in
"""
from __future__ import annotations

import json

import pandas as pd

from leprosy_similarity import paths, schema
from leprosy_similarity.exclusion_criteria import apply_exclusion_criteria

INPUTS = (paths.RAW_COHORT,)
OUTPUTS = (paths.COHORT, paths.EXCLUSION_AUDIT, paths.EXCLUSION_ATTRITION, paths.SCHEMA)
COHORT_COLUMNS = ["record_id", *schema.FEATURES, "class", "target", "site", "who_class"]


def run() -> None:
    raw = pd.read_parquet(paths.RAW_COHORT)
    result = apply_exclusion_criteria(raw)
    cohort = result.cohort[COHORT_COLUMNS]

    cohort.to_parquet(paths.COHORT, index=False)
    result.audit.to_csv(paths.EXCLUSION_AUDIT, index=False)
    result.attrition.to_csv(paths.EXCLUSION_ATTRITION, index=False)
    paths.SCHEMA.write_text(json.dumps(schema.describe(), indent=2))

    print(result.attrition.to_string(index=False))
    counts = cohort["class"].value_counts().reindex(schema.CLASSES, fill_value=0)
    print(f"analytic cohort: {len(cohort)} records")
    print("  " + "  ".join(f"{c} {n}" for c, n in counts.items()))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
