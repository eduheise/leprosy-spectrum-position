"""Reading the raw cohort spreadsheet into the 15-column schema.

The spreadsheet holds one row per patient from the reference centres, with the
intake form's findings as free text ("1 a 5", "++", "Eritematosa"). Each
finding is mapped onto the schema of ``schema.py``; a value that cannot be read
becomes NaN rather than a guess.

Two kinds of row are dropped while reading, before any exclusion criterion:
rows without a valid Ridley-Jopling label (blank or "Indeterminada"), and the
Goiania sample, which is not part of this study. Its rows carry only the
reaction flags and the class, so each would be removed later by the
completeness criterion anyway; keeping them would only inflate the reported
intake size.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from leprosy_similarity.schema import (
    CATEGORICAL_LEVELS,
    CLASS_TARGETS,
    FEATURES,
    SCREENED_OUT_FEATURES,
    SCREENED_OUT_LEVELS,
)

LABEL_COLUMN = "Forma Clínica"
TREATMENT_COLUMN = "Forma de Tratamento"
# The spreadsheet stacks the centres' databases one after another, in this
# order and with these row counts, so a row's position gives its centre. The
# genotype panels cannot be used for this: Goiania typed some of the same
# genes as Bauru, and some of its rows lack its own marker genotype.
SITE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("bauru", 372),
    ("fortaleza_manaus", 670),
    ("goiania", 409),
)
EXCLUDED_SITES = ("goiania",)

# Raw column of each categorical variable.
CATEGORICAL_SOURCE: dict[str, str] = {
    "lesion_color": "Cor da Lesão",
    "lesion_element": "Tipo de Lesão",
}


def normalise_text(value: Any) -> str | None:
    """Lower-case, accent-free, stripped text; None for a blank cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def categorical_level(variable: str, raw: Any) -> str | None:
    """Map a raw categorical value onto its schema level, or None.

    Exact level names match first; otherwise word stems tolerate the spelling
    variants found on the forms ("infiltrac difusa", "hipo;hiper").
    """
    normalised = normalise_text(raw)
    if normalised is None:
        return None
    text = normalised.replace(" ", "_")
    if text in CATEGORICAL_LEVELS[variable]:
        return text
    if variable == "lesion_element":
        for stem, level in (
            ("infiltrac", "infiltracao_difusa"),
            ("papul", "papula"),
            ("nodul", "nodulo"),
            ("macul", "macula"),
            ("placa", "placa"),
        ):
            if stem in text:
                return level
    if variable == "lesion_color":
        if "eritemat" in text:
            return "eritematosa"
        if "normocrom" in text:
            return "normocromica"
        # "hiper" is tested before "hipo" so the abbreviated form "hipo;hiper"
        # resolves the same way on every form.
        if "hipercrom" in text or "hiper" in text:
            return "hipercromica"
        if "hipocrom" in text or "hipo" in text:
            return "hipocromica"
    return None


def one_hot(
    variable: str, level: str | None, levels: dict[str, tuple[str, ...]] = CATEGORICAL_LEVELS
) -> dict[str, float]:
    """One-hot block of a categorical; NaN across the block when unknown."""
    return {
        f"{variable}__{candidate}": float("nan") if level is None else float(candidate == level)
        for candidate in levels[variable]
    }


def plus_scale(value: Any) -> float:
    """Read a graded laboratory result: "Negativo" -> 0, "+++" -> 3, "4+" -> 4."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    if (normalise_text(text) or "") in {"negativo", "negative", "neg", "0"}:
        return 0.0
    if set(text) <= {"+"}:
        return float(len(text))
    try:
        return float(text.replace("+", "").strip())
    except ValueError:
        return float("nan")


def lesion_count_tier(count: float) -> float:
    """Bucket a lesion count into the form's three tiers (1-5, 6-11, 12+)."""
    if pd.isna(count):
        return float("nan")
    if count <= 5:
        return 1.0
    if count <= 11:
        return 2.0
    return 3.0


def lesion_tier(value: Any) -> float:
    """The intake form records lesions in three bins; map each to its tier."""
    text = normalise_text(value)
    if text is None:
        return float("nan")
    if text == "1 a 5":
        return 1.0
    if text == "6 a 11":
        return 2.0
    if "infiltrac" in text:
        return 3.0
    try:
        return lesion_count_tier(float(text))
    except ValueError:
        return float("nan")


def reaction_flag(kind: str) -> Callable[[Any], float]:
    """Reader of the single reaction field for one reaction type.

    The form has one field naming the reaction ("Tipo 1", "Tipo 2" or "Sem
    Reação"); a row records type 1 exactly when that field names it, and so on.
    """
    other = "Tipo 2" if kind == "Tipo 1" else "Tipo 1"

    def read(value: Any) -> float:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return float("nan")
        text = str(value).strip()
        if kind in text:
            return 1.0
        if other in text or "Sem Re" in text or "Sem re" in text:
            return 0.0
        return float("nan")

    return read


def _column(raw: pd.DataFrame, name: str) -> pd.Series:
    return raw.get(name, pd.Series([None] * len(raw)))


def encode_findings(raw: pd.DataFrame) -> pd.DataFrame:
    """Encode raw intake-form rows into the 15 schema columns."""
    out = pd.DataFrame(index=raw.index, columns=FEATURES, dtype=float)
    out["NU_LESOES"] = _column(raw, "Número de Lesões").map(lesion_tier).astype(float)
    out["REAC_TIPO_1"] = _column(raw, "Tipo de Reação").map(reaction_flag("Tipo 1")).astype(float)
    out["REAC_TIPO_2"] = _column(raw, "Tipo de Reação").map(reaction_flag("Tipo 2")).astype(float)
    out["PGL1"] = _column(raw, "Intensidade do PGL-1+").map(plus_scale).astype(float)
    out["BACILOSCOP_N"] = _column(raw, "Indice Baciloscópico").map(plus_scale).astype(float)
    out["HISTOLOG"] = _column(raw, "Indice Histológico").map(plus_scale).astype(float)
    for variable, source in CATEGORICAL_SOURCE.items():
        values = _column(raw, source)
        for i in raw.index:
            for column, value in one_hot(variable, categorical_level(variable, values.loc[i])).items():
                out.at[i, column] = value
    return out


def sensibility_level(raw: Any) -> str | None:
    """Level of the screened-out sensibility finding, or None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip().lower()
    for stem, level in (("altera", "alterada"), ("duvid", "duvidosa"), ("normal", "normal")):
        if stem in text:
            return level
    return None


def encode_screened_out(raw: pd.DataFrame) -> pd.DataFrame:
    """One-hot the variables the screen removed, for the screen alone."""
    out = pd.DataFrame(index=raw.index, columns=SCREENED_OUT_FEATURES, dtype=float)
    values = _column(raw, "Sensibilidade")
    for i in raw.index:
        level = sensibility_level(values.loc[i])
        for column, value in one_hot("sensibility", level, SCREENED_OUT_LEVELS).items():
            out.at[i, column] = value
    return out


def who_class(treatment: Any) -> str | float:
    """WHO operational class from the regimen: 6 months PB, 12 months MB.

    Held out as an external validation target, never used as a finding.
    """
    if treatment is None or (isinstance(treatment, float) and pd.isna(treatment)):
        return float("nan")
    text = str(treatment).strip().lower()
    if "6" in text:
        return "PB"
    if "12" in text:
        return "MB"
    return float("nan")


def collection_site(raw: pd.DataFrame, blocks: Iterable[tuple[str, int]] = SITE_BLOCKS) -> pd.Series:
    """Collection site of each row, read off the block of rows it sits in."""
    blocks = tuple(blocks)
    expected = sum(size for _, size in blocks)
    if len(raw) != expected:
        raise ValueError(
            f"the spreadsheet has {len(raw)} rows, but its centre blocks add up to "
            f"{expected}; update SITE_BLOCKS in cohort.py to the new layout"
        )
    return pd.Series([site for site, size in blocks for _ in range(size)], index=raw.index, dtype=object)


def read_sheets(path: Path) -> pd.DataFrame:
    """Every non-empty sheet of a workbook, stacked."""
    workbook = pd.ExcelFile(path)
    frames = [workbook.parse(sheet) for sheet in workbook.sheet_names]
    frames = [frame for frame in frames if frame.shape[1] and len(frame)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_raw_cohort(paths: Iterable[Path], site_blocks: Iterable[tuple[str, int]] = SITE_BLOCKS) -> pd.DataFrame:
    """Read the raw spreadsheets into one encoded frame.

    Columns: ``record_id`` (row position in the read cohort), the 15 schema
    columns, the screened-out sensibility columns, ``class`` (Ridley-Jopling
    label), ``target`` (its supervision target), ``site`` and ``who_class``.
    """
    raws = [raw for raw in (read_sheets(Path(p)) for p in paths) if LABEL_COLUMN in raw]
    if not raws:
        raise ValueError(f"no labelled rows found in {list(paths)}")
    raw = pd.concat(raws, ignore_index=True)

    labels = raw[LABEL_COLUMN].astype("string").str.strip()
    site = collection_site(raw, site_blocks)
    keep = labels.isin(CLASS_TARGETS.keys()) & ~site.isin(EXCLUDED_SITES)

    def kept(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return frame.loc[keep].reset_index(drop=True)

    out = kept(encode_findings(raw)[list(FEATURES)]).copy()
    screened = kept(encode_screened_out(raw))
    for column in SCREENED_OUT_FEATURES:
        out[column] = screened[column]
    out["class"] = kept(labels).astype(str)
    out["target"] = out["class"].map(CLASS_TARGETS).astype(float)
    out["site"] = kept(site)
    out["who_class"] = kept(_column(raw, TREATMENT_COLUMN)).map(who_class)
    out.insert(0, "record_id", range(len(out)))
    return out
