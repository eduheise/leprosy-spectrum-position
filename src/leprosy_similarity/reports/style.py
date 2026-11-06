"""One look for the results figures, and clinical names for every column.

The figures share one seaborn theme (white grid, paper context, colour-blind
palette, larger text), one palette per grouping (the WHO classes and the five
Ridley-Jopling classes), and readable names, so no figure shows an internal
column name such as ``NU_LESOES`` or an axis value no clinician would write.

The theme is applied inside ``figure_style()`` only, so report steps that run
in the same process do not change each other's figures.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

from leprosy_similarity.schema import CLASSES, NUMERIC_FEATURES

DPI = 220

_COLOURBLIND = sns.color_palette("colorblind")
WHO_PALETTE = {"PB": _COLOURBLIND[0], "MB": _COLOURBLIND[3]}
RJ_PALETTE = dict(zip(CLASSES, sns.color_palette("viridis", len(CLASSES))))

FINDING_NAMES = {
    "NU_LESOES": "Number of skin lesions",
    "BACILOSCOP_N": "Bacilloscopy index",
    "PGL1": "Anti-PGL-1 titre",
    "HISTOLOG": "Histological index",
    "REAC_TIPO_1": "Type-1 reaction (absent/present)",
    "REAC_TIPO_2": "Type-2 reaction (absent/present)",
    "lesion_color": "Lesion colour",
    "lesion_element": "Lesion type",
}

LEVEL_NAMES = {
    "lesion_color__eritematosa": "erythematous",
    "lesion_color__hipocromica": "hypochromic",
    "lesion_color__hipercromica": "hyperchromic",
    "lesion_color__normocromica": "normochromic",
    "lesion_element__placa": "plaque",
    "lesion_element__papula": "papule",
    "lesion_element__nodulo": "nodule",
    "lesion_element__macula": "macule",
    "lesion_element__infiltracao_difusa": "diffuse infiltration",
}

GROUP_NAMES = {"lesion_color": "colour", "lesion_element": "type"}

LESION_COUNT_TIERS = {1: "1–5", 2: "6–11", 3: "diffuse\ninfiltration"}


@contextmanager
def figure_style() -> Iterator[None]:
    """The shared theme, active only inside the ``with`` block."""
    rc = {**sns.axes_style("whitegrid"), **sns.plotting_context("paper")}
    rc.update(
        {
            "axes.prop_cycle": cycler(color=_COLOURBLIND),
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "font.size": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    with plt.rc_context(rc):
        yield


def finding_name(feature: str) -> str:
    """Readable name of a numeric finding or of a whole categorical variable."""
    return FINDING_NAMES.get(feature, feature)


def level_name(column: str) -> str:
    """``lesion_element__nodulo`` -> ``type: nodule``."""
    group = column.split("__", 1)[0]
    return f"{GROUP_NAMES.get(group, group)}: {LEVEL_NAMES.get(column, column)}"


def feature_name(feature: str) -> str:
    """Readable name of any schema column."""
    return finding_name(feature) if feature in NUMERIC_FEATURES else level_name(feature)


def value_labels(feature: str, values: list[float]) -> list[str]:
    """Clinical tick text for the values a numeric finding is swept over."""
    labels = []
    for value in values:
        grade = round(value)
        if feature in ("BACILOSCOP_N", "HISTOLOG"):
            labels.append("0 (neg)" if grade == 0 else f"{grade}+")
        elif feature in ("REAC_TIPO_1", "REAC_TIPO_2"):
            labels.append("absent" if grade == 0 else "present")
        elif feature == "NU_LESOES":
            labels.append(LESION_COUNT_TIERS.get(grade, str(grade)))
        else:
            labels.append(str(grade))
    return labels
