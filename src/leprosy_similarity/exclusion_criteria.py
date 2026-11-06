"""Literature-grounded exclusion criteria that derive the analytic cohort.

Each criterion is a finding that rules some Ridley-Jopling classes out: when
the finding is present, the patient cannot belong to a class outside the
criterion's admissible set. A record whose recorded label is ruled out by any
criterion it fires contradicts its own findings and is removed; it is not
relabelled. A final completeness criterion removes records with neither a
bacilloscopy nor a histological index, since without either the position has
no laboratory evidence to rest on.

Two criteria are weaker than their textbook statements, because the
specialists' own reference cases contradict the literal version: ENL excludes
only TT and BT (BB reference cases record ENL), and the type-1 reaction
excludes only LL (TT reference cases record it). ``check_reference_cases``
repeats that test: no criterion may rule out a reference case's own class.

Removals are attributed for the attrition table by walking the criteria in
presentation order (clinical first, then laboratory) and charging each
removed record to the first criterion that rules its label out. The order
changes the table, never the set of records removed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

ALL_CLASSES: frozenset[str] = frozenset({"TT", "BT", "BB", "BL", "LL"})


@dataclass(frozen=True)
class Criterion:
    name: str
    display: str
    group: str
    fires: Callable[[pd.Series], bool]
    admissible: frozenset[str]
    citation: str


def _value(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def equals(column: str, target: float) -> Callable[[pd.Series], bool]:
    """Fires when ``column`` is recorded and equal to ``target``."""

    def fires(row: pd.Series) -> bool:
        value = _value(row, column)
        return value is not None and value == target

    return fires


def at_least(column: str, threshold: float) -> Callable[[pd.Series], bool]:
    """Fires when ``column`` is recorded and at least ``threshold``."""

    def fires(row: pd.Series) -> bool:
        value = _value(row, column)
        return value is not None and value >= threshold

    return fires


# Presentation order: clinical criteria first, laboratory criteria last.
CRITERIA: tuple[Criterion, ...] = (
    # Tier 1 is the form's '1 a 5' bin. BL lesions are numerous and LL lesions multiple and symmetric, so 1-5
    # lesions is inconsistent with a recorded BL or LL (single-lesion BL/LL is reported but rare and treated as a
    # probable recording error).
    Criterion(
        name="few_lesions",
        display="1-5 lesions",
        group="clinical",
        fires=equals("NU_LESOES", 1),
        admissible=frozenset({"TT", "BT", "BB"}),
        citation="Ridley & Jopling 1966, pp. 258-259; WHO TRS 874 (1998), p. 7",
    ),
    # TT has one to a few lesions; tier 3 (diffuse or 12+) excludes TT.
    Criterion(
        name="many_lesions",
        display="diffuse infiltration or 12+ lesions",
        group="clinical",
        fires=equals("NU_LESOES", 3),
        admissible=frozenset({"BT", "BB", "BL", "LL"}),
        citation="Ridley & Jopling 1966",
    ),
    # Diffuse infiltration without discrete edges is the lepromatous-side phenotype.
    Criterion(
        name="diffuse_infiltration",
        display="diffuse infiltration",
        group="clinical",
        fires=equals("lesion_element__infiltracao_difusa", 1),
        admissible=frozenset({"BL", "LL"}),
        citation="Ridley & Jopling 1966",
    ),
    # Nodules need sustained bacillary proliferation, absent in tuberculoid disease.
    Criterion(
        name="nodule",
        display="nodule",
        group="clinical",
        fires=equals("lesion_element__nodulo", 1),
        admissible=frozenset({"BB", "BL", "LL"}),
        citation="Ridley & Jopling 1966",
    ),
    # ENL is an immune-complex syndrome of multibacillary disease; it excludes TT and BT.
    Criterion(
        name="enl",
        display="ENL (type-2 reaction)",
        group="clinical",
        fires=equals("REAC_TIPO_2", 1),
        admissible=frozenset({"BB", "BL", "LL"}),
        citation="Cuevas et al., Semin Cutan Med Surg 2007",
    ),
    # Reversal reactions need unstable cell-mediated immunity, which is absent at LL.
    Criterion(
        name="type1_reaction",
        display="type-1 reaction",
        group="clinical",
        fires=equals("REAC_TIPO_1", 1),
        admissible=frozenset({"TT", "BT", "BB", "BL"}),
        citation="Walker & Lockwood, Lepr Rev 2008",
    ),
    # TT lesions are bacilli-negative by definition.
    Criterion(
        name="smear_positive",
        display="positive skin smear",
        group="laboratory",
        fires=at_least("BACILOSCOP_N", 1),
        admissible=frozenset({"BT", "BB", "BL", "LL"}),
        citation="Ridley & Jopling 1966; WHO TRS 874 (1998)",
    ),
    # BT's bacilloscopy index stays at or below 2+.
    Criterion(
        name="bacilloscopy_4",
        display="bacilloscopy index >= 4",
        group="laboratory",
        fires=at_least("BACILOSCOP_N", 4),
        admissible=frozenset({"BB", "BL", "LL"}),
        citation="Ridley & Jopling 1966",
    ),
    # BB's bacilloscopy index stays at or below 3+.
    Criterion(
        name="bacilloscopy_5",
        display="bacilloscopy index >= 5",
        group="laboratory",
        fires=at_least("BACILOSCOP_N", 5),
        admissible=frozenset({"BL", "LL"}),
        citation="Ridley & Jopling 1966",
    ),
    # LL is index 5+ at diagnosis; a negative smear cannot be LL under adequate sampling.
    Criterion(
        name="smear_negative",
        display="negative skin smear",
        group="laboratory",
        fires=equals("BACILOSCOP_N", 0),
        admissible=frozenset({"TT", "BT", "BB", "BL"}),
        citation="Ridley & Jopling 1966",
    ),
    # The anti-PGL-1 titre follows antigen load; TT cannot reach the maximum titre.
    Criterion(
        name="pgl1_maximum",
        display="maximum anti-PGL-1 titre",
        group="laboratory",
        fires=equals("PGL1", 4),
        admissible=frozenset({"BT", "BB", "BL", "LL"}),
        citation="Buhrer-Sekula et al., Lepr Rev 2007; Cho et al., Clin Diagn Lab Immunol 2001",
    ),
)

COMPLETENESS_STEP = "completeness (bacilloscopy or histology)"
COMPLETENESS_REASON = "no bacilloscopy and no histology"


def ruling_out(row: pd.Series, label: str) -> list[Criterion]:
    """Criteria that fire on ``row`` and rule ``label`` out, in presentation order."""
    return [c for c in CRITERIA if c.fires(row) and label not in c.admissible]


def has_laboratory_evidence(cohort: pd.DataFrame) -> pd.Series:
    """True where a bacilloscopy or a histological index is recorded."""
    return cohort["BACILOSCOP_N"].notna() | cohort["HISTOLOG"].notna()


@dataclass(frozen=True)
class ExclusionResult:
    cohort: pd.DataFrame
    audit: pd.DataFrame
    attrition: pd.DataFrame


def apply_exclusion_criteria(raw: pd.DataFrame) -> ExclusionResult:
    """Remove label-contradicting and evidence-free records, with the audit.

    Returns the analytic cohort (index reset), a per-record audit
    (``record_id``, ``deleted``, ``step``, ``excluding_criteria``,
    ``original_class``) and the attrition table (``order``, ``step``,
    ``group``, ``removed``, ``remaining``).
    """
    ruled_out = {i: ruling_out(raw.loc[i], str(raw.at[i, "class"])) for i in raw.index}
    remaining = pd.Series(True, index=raw.index)
    step = pd.Series("", index=raw.index, dtype=object)
    steps = []
    for criterion in CRITERIA:
        removed = [i for i in raw.index[remaining] if ruled_out[i] and ruled_out[i][0] is criterion]
        remaining[removed] = False
        step[removed] = criterion.display
        steps.append((criterion.display, criterion.group, len(removed), int(remaining.sum())))

    incomplete = raw.index[remaining & ~has_laboratory_evidence(raw)]
    remaining[incomplete] = False
    step[incomplete] = COMPLETENESS_REASON
    steps.append((COMPLETENESS_STEP, "completeness", len(incomplete), int(remaining.sum())))

    attrition = pd.DataFrame(
        [
            {"order": k + 1, "step": name, "group": group, "removed": removed, "remaining": left}
            for k, (name, group, removed, left) in enumerate(steps)
        ]
    )
    audit = pd.DataFrame(
        {
            "record_id": raw["record_id"].to_numpy(),
            "deleted": (~remaining).to_numpy(),
            "step": step.to_numpy(),
            "excluding_criteria": [";".join(c.name for c in ruled_out[i]) for i in raw.index],
            "original_class": raw["class"].astype(str).to_numpy(),
        }
    )
    return ExclusionResult(raw[remaining].reset_index(drop=True), audit, attrition)


def check_reference_cases(reference_cases: pd.DataFrame) -> pd.DataFrame:
    """Every (reference case, criterion) pair where the criterion rules out the case's class.

    An empty frame means the criteria agree with every specialist reference
    case; a row means the criterion is overstated for this cohort.
    """
    rows = [
        {
            "reference_id": int(row["reference_id"]),
            "class": row["class"],
            "criterion": criterion.name,
            "admissible": ",".join(sorted(criterion.admissible)),
            "citation": criterion.citation,
        }
        for _, row in reference_cases.iterrows()
        for criterion in ruling_out(row, row["class"])
    ]
    return pd.DataFrame(rows, columns=["reference_id", "class", "criterion", "admissible", "citation"])
