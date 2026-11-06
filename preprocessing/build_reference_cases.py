"""Write the 26 specialist reference cases in the schema, and check the criteria on them.

Input:  data/reference_cases/prototypes.xlsx
Output: data/processed/reference_cases.parquet
        data/processed/reference_case_criteria_check.csv  (empty when every
        exclusion criterion admits every reference case's own class)
"""
from __future__ import annotations

from leprosy_similarity import paths
from leprosy_similarity.exclusion_criteria import check_reference_cases
from leprosy_similarity.reference_cases import load_reference_cases
from leprosy_similarity.schema import REFERENCE_CLASSES

INPUTS = (paths.REFERENCE_CASES_XLSX,)
OUTPUTS = (paths.REFERENCE_CASES, paths.REFERENCE_CASE_CRITERIA_CHECK)


def run() -> None:
    reference_cases = load_reference_cases(paths.REFERENCE_CASES_XLSX)
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    reference_cases.to_parquet(paths.REFERENCE_CASES, index=False)

    conflicts = check_reference_cases(reference_cases)
    conflicts.to_csv(paths.REFERENCE_CASE_CRITERIA_CHECK, index=False)

    counts = reference_cases["class"].value_counts().reindex(REFERENCE_CLASSES, fill_value=0)
    print(f"reference cases: {len(reference_cases)}")
    print("  " + "  ".join(f"{c} {n}" for c, n in counts.items()))
    print(f"criteria ruling out a reference case's own class: {len(conflicts)}")
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
