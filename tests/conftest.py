"""Tiny synthetic records, generated in code, so the tests need no patient data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leprosy_similarity.config import load_config
from leprosy_similarity.schema import CATEGORICAL_LEVELS, CLASS_TARGETS, CLASSES, FEATURES

NUMERIC_RANGES = {
    "NU_LESOES": (1, 3),
    "REAC_TIPO_1": (0, 1),
    "REAC_TIPO_2": (0, 1),
    "PGL1": (0, 4),
    "BACILOSCOP_N": (0, 6),
    "HISTOLOG": (0, 6),
}


def synthetic_records(n: int, classes: tuple[str, ...], seed: int, missing: float = 0.15) -> pd.DataFrame:
    """Records whose findings drift with the class, with some findings missing."""
    rng = np.random.default_rng(seed)
    labels = np.array([classes[i % len(classes)] for i in range(n)])
    pole = np.array([CLASS_TARGETS[c] for c in labels])
    frame = pd.DataFrame(index=range(n), columns=list(FEATURES), dtype=float)
    for feature, (low, high) in NUMERIC_RANGES.items():
        centre = low + pole * (high - low)
        frame[feature] = np.clip(np.round(centre + rng.normal(0, 0.8, n)), low, high)
    for variable, levels in CATEGORICAL_LEVELS.items():
        pick = np.clip((pole * len(levels) + rng.integers(-1, 2, n)).astype(int), 0, len(levels) - 1)
        for i, level in enumerate(levels):
            frame[f"{variable}__{level}"] = (pick == i).astype(float)
    holes = rng.random((n, 6)) < missing
    for j, feature in enumerate(NUMERIC_RANGES):
        if feature.startswith("REAC"):
            continue
        frame.loc[holes[:, j], feature] = np.nan
    block = [c for c in FEATURES if c.startswith("lesion_element__")]
    frame.loc[rng.random(n) < missing, block] = np.nan
    frame.insert(0, "record_id", range(n))
    frame["class"] = labels
    frame["target"] = pole
    return frame


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def cohort() -> pd.DataFrame:
    return synthetic_records(100, CLASSES, seed=1)


@pytest.fixture
def reference_cases() -> pd.DataFrame:
    refs = synthetic_records(9, ("TT", "BB", "LL"), seed=2, missing=0.0)
    return refs.rename(columns={"record_id": "reference_id"}).drop(columns="target")
