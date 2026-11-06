"""The masking sweep behind the confidence experiment.

Near-complete records are sampled per class, and from each record several
random subsets of its observed findings are hidden. Every (record, mask) pair
gives two versions of one patient: the record as observed, and the record
with the subset hidden. How far the position moves between the two is the
drift the confidence band has to cover.

The sweep is drawn once from the whole analytic cohort; each pair is then
scored by the fold model that did not train on its record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaskingBatch:
    """One row per (record, mask) pair.

    ``registry`` holds ``case`` (the sampled record's number, 0-based),
    ``record_id``, ``class``, ``subset`` (the mask's number within its
    record), ``mask_size``, ``masked_features`` (comma-separated) and
    ``n_present_full``; ``full`` and ``masked`` hold the findings, aligned
    row by row with the registry.
    """

    registry: pd.DataFrame
    full: pd.DataFrame
    masked: pd.DataFrame


def sample_cases(
    records: pd.DataFrame, features: list[str], n_per_class: int, min_present: int, seed: int
) -> pd.DataFrame:
    """Draw ``n_per_class`` records per class among those observing ``min_present`` findings.

    Classes are visited in alphabetical order and each draws without
    replacement from its eligible records, with one generator for the whole
    sample. The result keeps every column of ``records`` plus ``n_present``,
    in drawing order, with a fresh 0-based index.
    """
    n_present = records[features].notna().sum(axis=1)
    eligible = records[n_present >= min_present].copy()
    eligible["n_present"] = n_present[eligible.index]
    rng = np.random.default_rng(seed)
    chunks = []
    for label in sorted(eligible["class"].unique()):
        pool = eligible[eligible["class"] == label]
        take = min(n_per_class, len(pool))
        chunks.append(eligible.loc[rng.choice(pool.index.to_numpy(), size=take, replace=False)])
    return pd.concat(chunks, axis=0).reset_index(drop=True)


def build_masking_batch(
    cases: pd.DataFrame, features: list[str], masks_per_record: int, mask_sizes: tuple[int, ...], seed: int
) -> MaskingBatch:
    """Hide ``masks_per_record`` random subsets of each case's observed findings.

    Each subset's size is drawn uniformly from ``mask_sizes`` and capped at
    the number of findings the case observes; the hidden findings are then
    drawn without replacement among the observed ones. One generator serves
    the whole batch, so the batch depends only on the cases and the seed.
    """
    rng = np.random.default_rng(seed)
    columns = np.array(features)
    registry: list[dict[str, object]] = []
    full_rows: list[pd.Series] = []
    masked_rows: list[pd.Series] = []
    for case, row in cases.iterrows():
        findings = row[features]
        present = findings.notna().to_numpy()
        observed = columns[present]
        for subset in range(masks_per_record):
            size = min(int(rng.choice(mask_sizes)), len(observed))
            hidden = [] if size == 0 else list(rng.choice(observed, size=size, replace=False))
            masked = findings.copy()
            masked[hidden] = np.nan
            full_rows.append(findings)
            masked_rows.append(masked)
            registry.append(
                {
                    "case": int(case),
                    "record_id": row["record_id"],
                    "class": row["class"],
                    "subset": subset,
                    "mask_size": len(hidden),
                    "masked_features": ",".join(hidden),
                    "n_present_full": int(present.sum()),
                }
            )
    return MaskingBatch(
        registry=pd.DataFrame(registry),
        full=pd.DataFrame(full_rows, columns=features).reset_index(drop=True).astype(float),
        masked=pd.DataFrame(masked_rows, columns=features).reset_index(drop=True).astype(float),
    )


def build_single_finding_batch(cases: pd.DataFrame, features: list[str]) -> MaskingBatch:
    """Hide each observed finding of each case on its own, one pair per finding.

    Every single-finding mask is enumerated, so the batch needs no random draw.
    """
    registry: list[dict[str, object]] = []
    full_rows: list[pd.Series] = []
    masked_rows: list[pd.Series] = []
    for case, row in cases.iterrows():
        findings = row[features]
        present = findings.notna().to_numpy()
        for subset, hidden in enumerate(np.array(features)[present]):
            masked = findings.copy()
            masked[hidden] = np.nan
            full_rows.append(findings)
            masked_rows.append(masked)
            registry.append(
                {
                    "case": int(case),
                    "record_id": row["record_id"],
                    "class": row["class"],
                    "subset": subset,
                    "mask_size": 1,
                    "masked_features": hidden,
                    "n_present_full": int(present.sum()),
                }
            )
    return MaskingBatch(
        registry=pd.DataFrame(registry),
        full=pd.DataFrame(full_rows, columns=features).reset_index(drop=True).astype(float),
        masked=pd.DataFrame(masked_rows, columns=features).reset_index(drop=True).astype(float),
    )
