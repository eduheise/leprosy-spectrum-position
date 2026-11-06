"""Fix the one stratified 5-fold split shared by the search and every result.

Input:  data/processed/cohort.parquet
Output: data/processed/folds.csv  (record_id, fold)
"""
from __future__ import annotations

import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.config import load_config
from leprosy_similarity.folds import assign_folds

INPUTS = (paths.COHORT,)
OUTPUTS = (paths.FOLDS,)


def run() -> None:
    config = load_config()
    cohort = pd.read_parquet(paths.COHORT)
    folds = assign_folds(cohort, config.folds.n_folds, config.folds.seed)
    folds.to_csv(paths.FOLDS, index=False)

    table = pd.crosstab(folds["fold"], cohort["class"].to_numpy())
    print(f"{config.folds.n_folds} folds over {len(folds)} records (seed {config.folds.seed})")
    print(table.to_string())
    print(f"wrote {paths.FOLDS}")


if __name__ == "__main__":
    run()
