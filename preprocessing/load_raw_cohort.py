"""Read the raw cohort spreadsheet into the schema (1,030 labelled records).

Input:  data/raw/rafa_total.xlsx
Output: data/processed/raw_cohort.parquet
"""
from __future__ import annotations

from leprosy_similarity import paths
from leprosy_similarity.cohort import load_raw_cohort
from leprosy_similarity.schema import CLASSES, FEATURES

INPUTS = (paths.RAW_COHORT_XLSX,)
OUTPUTS = (paths.RAW_COHORT,)


def run() -> None:
    raw = load_raw_cohort([paths.RAW_COHORT_XLSX])
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(paths.RAW_COHORT, index=False)

    counts = raw["class"].value_counts().reindex(CLASSES, fill_value=0)
    print(f"raw cohort: {len(raw)} labelled records")
    print("  " + "  ".join(f"{c} {n}" for c, n in counts.items()))
    missing = raw[list(FEATURES)].isna().mean().sort_values(ascending=False)
    print("missing share per finding:")
    for feature, share in missing.items():
        print(f"  {feature:36s} {share:6.1%}")
    print(f"wrote {paths.RAW_COHORT}")


if __name__ == "__main__":
    run()
