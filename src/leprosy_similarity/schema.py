"""The 15-column schema every record is written in.

The schema holds the findings present both on the cohort's intake form and on
the specialists' reference-case form, and retained by the variable screen
(``preprocessing/screen_variables.py``): a variable stays only if a forest
trained on the Ridley-Jopling label reads measurable signal from it.

Numeric (6):
    NU_LESOES      lesion count tier: 1 = 1-5 lesions, 2 = 6-11, 3 = diffuse or 12+
    REAC_TIPO_1    type-1 reaction, 0/1
    REAC_TIPO_2    type-2 reaction (ENL), 0/1
    PGL1           anti-PGL-1 serology, 0..4
    BACILOSCOP_N   bacilloscopy index (Ridley logarithmic scale), 0..6
    HISTOLOG       histological index, 0..6

Categorical (2), one-hot encoded, one column per level:
    lesion_color    eritematosa, hipocromica, hipercromica, normocromica
    lesion_element  placa, papula, nodulo, macula, infiltracao_difusa

A missing finding is NaN; a missing categorical is NaN across its whole block.

Sensibility was screened out: altered sensation is a cardinal sign of leprosy,
so it is near-constant in a confirmed cohort and separates leprosy from
non-leprosy rather than one pole of the spectrum from the other. Its levels
are kept here only so the screen can rebuild the column and repeat the
measurement that removed it.
"""
from __future__ import annotations

NUMERIC_FEATURES: tuple[str, ...] = (
    "NU_LESOES",
    "REAC_TIPO_1",
    "REAC_TIPO_2",
    "PGL1",
    "BACILOSCOP_N",
    "HISTOLOG",
)

CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "lesion_color": ("eritematosa", "hipocromica", "hipercromica", "normocromica"),
    "lesion_element": ("placa", "papula", "nodulo", "macula", "infiltracao_difusa"),
}

SCREENED_OUT_LEVELS: dict[str, tuple[str, ...]] = {
    "sensibility": ("alterada", "duvidosa", "normal"),
}


def one_hot_columns(levels: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Column names ``<variable>__<level>`` for every level, in declared order."""
    return tuple(f"{name}__{level}" for name, values in levels.items() for level in values)


CATEGORICAL_FEATURES: tuple[str, ...] = one_hot_columns(CATEGORICAL_LEVELS)
FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SCREENED_OUT_FEATURES: tuple[str, ...] = one_hot_columns(SCREENED_OUT_LEVELS)

# Supervision targets of the Ridley-Jopling classes. They are equally spaced by
# assumption; the reference cases sit at TT 0, BB 0.5 and LL 1.
CLASS_TARGETS: dict[str, float] = {
    "TT": 0.0,
    "BT": 0.25,
    "BB": 0.5,
    "BL": 0.75,
    "LL": 1.0,
}
CLASSES: tuple[str, ...] = tuple(sorted(CLASS_TARGETS, key=CLASS_TARGETS.__getitem__))
REFERENCE_CLASSES: tuple[str, ...] = ("TT", "BB", "LL")


def categorical_blocks(columns: tuple[str, ...] = CATEGORICAL_FEATURES) -> dict[str, list[str]]:
    """Group one-hot columns by the variable they encode."""
    blocks: dict[str, list[str]] = {}
    for column in columns:
        blocks.setdefault(column.split("__")[0], []).append(column)
    return blocks


def describe() -> dict:
    """Machine-readable description of the schema, written to ``schema.json``."""
    return {
        "numeric": list(NUMERIC_FEATURES),
        "categorical": {name: list(levels) for name, levels in CATEGORICAL_LEVELS.items()},
        "features": list(FEATURES),
        "class_targets": dict(CLASS_TARGETS),
        "reference_classes": list(REFERENCE_CLASSES),
    }
