"""The 26 specialist reference cases, written in the 15-column schema.

Each row of the reference-case spreadsheet is one specialist's description of
a typical TT, BB or LL patient. Specialists often allowed several values for a
finding ("1 ou 2", "0 a 4", "eritematosa ou normocromica"); each cell is
collapsed to one value so a reference case is a single record:

- a numeric finding takes the mean of the values allowed;
- the lesion count is first bucketed into the cohort's three tiers, so both
  sources sit on the same axis, and the tiers are averaged;
- a categorical finding spreads one unit evenly over the levels allowed;
- a blank or unknown cell is NaN (across the whole block for a categorical).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leprosy_similarity.cohort import categorical_level, lesion_count_tier
from leprosy_similarity.schema import CATEGORICAL_LEVELS, FEATURES

CLASS_COLUMN = "R&J"
# Source column of each numeric finding, and the top of its scale.
NUMERIC_SOURCE: dict[str, tuple[str, int]] = {
    "NU_LESOES": ("Lesões", 15),
    "REAC_TIPO_1": ("Reação T1", 1),
    "REAC_TIPO_2": ("Reação T2", 1),
    "PGL1": ("antipgl1", 4),
    "BACILOSCOP_N": ("índice baciloscópico", 6),
    "HISTOLOG": ("índice histológico", 6),
}
CATEGORICAL_SOURCE: dict[str, str] = {
    "lesion_color": "cor da lesão",
    "lesion_element": "lesão elementar",
}
RANGE_WORDS = (" a ", " ate ", " até ")


def parse_values(raw: Any, *, cap: int) -> list[int]:
    """Integer values a specialist allowed in one cell, at most ``cap``.

    Reads "1", "0 ou 1", "1 a 5", "6+" (6 up to the cap), and cells the
    spreadsheet turned into dates ("1 a 5" stored as 2026-05-01), whose month
    and day are the two ends of the range.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    text = text.lower().replace("grau", "").strip()
    if text in {"desconhecido", "unknown"}:
        return []
    date = re.match(r"^\d{4}-(\d{2})-(\d{2})", text)
    if date:
        low, high = sorted((int(date.group(1)), int(date.group(2))))
        return list(range(max(0, low), min(cap, high) + 1))
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if "+" in text and numbers:
        return list(range(max(0, numbers[0]), cap + 1))
    is_range = any(word in text for word in RANGE_WORDS) or (
        "-" in text and any(ch.isdigit() for ch in text)
    )
    if is_range and len(numbers) >= 2:
        low, high = sorted(numbers[:2])
        return list(range(max(0, low), min(cap, high) + 1))
    return [n for n in numbers if n <= cap]


def parse_levels(variable: str, raw: Any) -> list[str]:
    """Distinct schema levels named in a cell, in the order written."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    levels: list[str] = []
    for part in re.split(r"[,;]| ou ", str(raw)):
        level = categorical_level(variable, part)
        if level is not None and level not in levels:
            levels.append(level)
    return levels


def numeric_value(feature: str, raw: Any) -> float:
    """Mean of the values allowed; lesion counts are bucketed into tiers first."""
    _, cap = NUMERIC_SOURCE[feature]
    values = parse_values(raw, cap=cap)
    if not values:
        return float("nan")
    if feature == "NU_LESOES":
        return float(np.mean([lesion_count_tier(v) for v in values]))
    return float(np.mean(values))


def categorical_share(variable: str, raw: Any) -> dict[str, float]:
    """One unit spread evenly over the levels allowed; NaN if none."""
    levels = CATEGORICAL_LEVELS[variable]
    allowed = parse_levels(variable, raw)
    if not allowed:
        return {f"{variable}__{level}": float("nan") for level in levels}
    share = 1.0 / len(allowed)
    return {f"{variable}__{level}": share if level in allowed else 0.0 for level in levels}


def load_reference_cases(xlsx_path: Path) -> pd.DataFrame:
    """One record per reference case.

    Columns: ``reference_id`` (1-based row of the spreadsheet), ``class``
    (TT, BB or LL), then the 15 schema columns.
    """
    sheet = pd.read_excel(xlsx_path)
    records = []
    for i, row in sheet.iterrows():
        record: dict[str, Any] = {"reference_id": int(i) + 1, "class": str(row[CLASS_COLUMN]).strip()}
        for feature, (source, _) in NUMERIC_SOURCE.items():
            record[feature] = numeric_value(feature, row.get(source))
        for variable, source in CATEGORICAL_SOURCE.items():
            record.update(categorical_share(variable, row.get(source)))
        records.append(record)
    return pd.DataFrame.from_records(records, columns=["reference_id", "class", *FEATURES])
