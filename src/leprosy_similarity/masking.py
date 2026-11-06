"""Hiding observed findings, to measure how far a position moves without them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from leprosy_similarity.schema import FEATURES


def mask_rows(records: pd.DataFrame, sizes: np.ndarray, seed: int) -> tuple[pd.DataFrame, list[tuple[str, ...]]]:
    """Hide ``sizes[i]`` observed findings of row i, chosen at random.

    At least one finding always stays observed: the drift is the distance
    between a position and its degraded twin, and with nothing observed there
    is no twin, only the model's answer to an empty record. Returns the masked
    frame (same index) and, per row, the names of the hidden findings.
    """
    rng = np.random.default_rng(seed)
    out = records[list(FEATURES)].copy()
    hidden_per_row: list[tuple[str, ...]] = []
    for position, (index, row) in enumerate(out.iterrows()):
        observed = [c for c in FEATURES if not pd.isna(row[c])]
        k = min(int(sizes[position]), max(len(observed) - 1, 0))
        if k == 0:
            hidden_per_row.append(())
            continue
        hidden = tuple(rng.choice(observed, size=k, replace=False).tolist())
        for column in hidden:
            out.at[index, column] = np.nan
        hidden_per_row.append(hidden)
    return out, hidden_per_row
