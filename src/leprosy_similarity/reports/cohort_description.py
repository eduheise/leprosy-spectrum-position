"""Description of the analytic cohort and of the reference cases.

Input:  data/processed/raw_cohort.parquet, data/processed/cohort.parquet,
        data/processed/exclusion_audit.csv, data/processed/reference_cases.parquet,
        data/processed/reference_case_criteria_check.csv
Output: reports/tables/cohort_by_class.csv  (sec:val:cohort)
        reports/tables/reference_cases.csv  (tab:prototype-coverage, anx:reference-cases)
        reports/numbers.json, keys ``cohort.description.*`` and ``reference_cases.*``

Everything here is counted on the records as recorded, before any model sees
them: how the two exclusion steps move the class mix, what the analytic cohort
carries per class, where its missing values sit, how the recorded WHO class
follows the smear, and what the 26 reference cases hold. The numbers are
descriptive, so they come from the preprocessing files alone.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from leprosy_similarity import paths
from leprosy_similarity.exclusion_criteria import ALL_CLASSES, COMPLETENESS_REASON, CRITERIA
from leprosy_similarity.reports.numbers import record_numbers
from leprosy_similarity.schema import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURES,
    REFERENCE_CLASSES,
)

INPUTS = (
    paths.RAW_COHORT,
    paths.COHORT,
    paths.EXCLUSION_AUDIT,
    paths.REFERENCE_CASES,
    paths.REFERENCE_CASE_CRITERIA_CHECK,
)
OUTPUTS = (paths.TABLE_COHORT_BY_CLASS, paths.TABLE_REFERENCE_CASES)

# The findings of the clinical examination. One centre records none of them,
# nor the anti-PGL-1 serology, so they go missing together.
CLINICAL_EXAMINATION: tuple[str, ...] = ("NU_LESOES", *CATEGORICAL_FEATURES)
CLINICAL_BLOCK: tuple[str, ...] = (*CLINICAL_EXAMINATION, "PGL1")
# The one centre with its own pathology service.
PATHOLOGY_CENTRE = "bauru"
# The screened-out sensibility, as the share of records with altered sensation.
ALTERED_SENSIBILITY = "sensibility__alterada"
# Lesion-count tier 1 is one to five lesions; the WHO rule calls more than
# five multibacillary.
FEW_LESIONS_TIER = 1.0


def per_class(series: pd.Series) -> dict[str, Any]:
    """A per-class series as a dict in spectrum order, TT to LL."""
    return series.reindex(CLASSES).to_dict()


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return per_class(frame["class"].value_counts().reindex(CLASSES, fill_value=0))


def class_shares_pct(frame: pd.DataFrame) -> dict[str, float]:
    return per_class(frame["class"].value_counts(normalize=True) * 100)


def attrition_by_class(raw: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Any]:
    """How the exclusion criteria and the completeness criterion move the class mix."""
    step = raw["record_id"].map(audit.set_index("record_id")["step"])
    by_completeness = step == COMPLETENESS_REASON
    by_criteria = step.notna() & ~by_completeness
    after_criteria = raw[~by_criteria]
    analytic = raw[step.isna()]
    incomplete = raw[by_completeness]
    stages = {"intake": raw, "after_criteria": after_criteria, "analytic": analytic}
    return {
        "cohort.description.removed_by_criteria_by_class": class_counts(raw[by_criteria]),
        "cohort.description.removed_by_completeness_by_class": class_counts(incomplete),
        "cohort.description.after_criteria_by_class": class_counts(after_criteria),
        "cohort.description.removed_by_completeness_by_site": incomplete["site"].value_counts().to_dict(),
        "cohort.description.removed_by_completeness_multibacillary": int((incomplete["who_class"] == "MB").sum()),
        "cohort.description.class_share_pct": {name: class_shares_pct(frame) for name, frame in stages.items()},
        "cohort.description.paucibacillary": {
            name: {"n": int((frame["who_class"] == "PB").sum()), "of": len(frame)} for name, frame in stages.items()
        },
    }


def composition(cohort: pd.DataFrame) -> dict[str, Any]:
    """Class and centre counts, and which laboratory index each record carries."""
    smear = cohort["BACILOSCOP_N"].notna()
    return {
        "cohort.description.n_records": len(cohort),
        "cohort.description.by_class": class_counts(cohort),
        "cohort.description.by_site": cohort["site"].value_counts().to_dict(),
        "cohort.description.n_with_bacilloscopy": int(smear.sum()),
        "cohort.description.n_histology_only": int((~smear & cohort["HISTOLOG"].notna()).sum()),
    }


def marginals(cohort: pd.DataFrame) -> dict[str, Any]:
    """Per-class distribution of the graded findings and of the lesion levels.

    Means and level shares are over the records that carry the finding.
    """
    by_class = cohort.groupby("class")
    tiers = pd.crosstab(cohort["class"], cohort["NU_LESOES"]).reindex(CLASSES, fill_value=0)
    return {
        "cohort.description.lesion_tier_by_class": {
            label: {str(int(tier)): int(n) for tier, n in row.items()} for label, row in tiers.iterrows()
        },
        "cohort.description.mean_by_class": {
            column: per_class(by_class[column].mean()) for column in ("BACILOSCOP_N", "PGL1", "HISTOLOG")
        },
        "cohort.description.level_pct_by_class": {
            column: per_class(by_class[column].mean() * 100) for column in CATEGORICAL_FEATURES
        },
    }


def histology(cohort: pd.DataFrame) -> dict[str, Any]:
    """Where the histological index is recorded, and whether it tracks the class within one centre."""
    recorded = cohort[cohort["HISTOLOG"].notna()]
    centre = recorded[recorded["site"] == PATHOLOGY_CENTRE]
    rho, p = spearmanr(centre["HISTOLOG"], centre["target"])
    return {
        "cohort.description.histology_recorded": len(recorded),
        "cohort.description.histology_recorded_share": len(recorded) / len(cohort),
        "cohort.description.bacilloscopy_recorded_share": float(cohort["BACILOSCOP_N"].notna().mean()),
        "cohort.description.histology_by_site": recorded["site"].value_counts().to_dict(),
        "cohort.description.histology_spearman_within_centre": {"rho": rho, "p": p, "n": len(centre)},
    }


def missingness(cohort: pd.DataFrame) -> dict[str, Any]:
    """The block of records without a clinical examination, and what they carry instead."""
    others = [c for c in FEATURES if c != "HISTOLOG"]
    no_examination = cohort[list(CLINICAL_BLOCK)].isna().all(axis=1)
    block = cohort[no_examination]
    smear, biopsy = block["BACILOSCOP_N"].notna(), block["HISTOLOG"].notna()
    return {
        "cohort.description.missing_by_column": cohort[list(FEATURES)].isna().sum().to_dict(),
        "cohort.description.only_histology_missing": int(cohort[others].notna().all(axis=1).sum()),
        "cohort.description.without_clinical_examination": int(no_examination.sum()),
        "cohort.description.without_clinical_examination_carries": {
            "both_indices": int((smear & biopsy).sum()),
            "bacilloscopy_only": int((smear & ~biopsy).sum()),
            "histology_only": int((~smear & biopsy).sum()),
        },
        "cohort.description.without_clinical_examination_by_site": block["site"].value_counts().to_dict(),
        "cohort.description.without_clinical_examination_by_class": class_counts(block),
    }


def operational_class(cohort: pd.DataFrame) -> dict[str, Any]:
    """How the recorded WHO class follows the smear and the lesion count.

    The strict WHO rule calls a patient multibacillary when the smear is
    positive or there are more than five lesions; it can be evaluated only
    where both are recorded.
    """
    smeared = cohort[cohort["BACILOSCOP_N"].notna() & cohort["who_class"].notna()]
    positive = smeared["BACILOSCOP_N"] > 0
    negative = smeared[~positive & smeared["NU_LESOES"].notna()]
    few = negative["NU_LESOES"] == FEW_LESIONS_TIER
    rule_frame = smeared[smeared["NU_LESOES"].notna()]
    rule = ((rule_frame["BACILOSCOP_N"] > 0) | (rule_frame["NU_LESOES"] > FEW_LESIONS_TIER)).map(
        {True: "MB", False: "PB"}
    )
    rho, _ = spearmanr(smeared["BACILOSCOP_N"], smeared["target"])
    return {
        "cohort.description.smear_positive": {
            "n": int(positive.sum()), "multibacillary_share": float((smeared[positive]["who_class"] == "MB").mean())
        },
        "cohort.description.smear_negative": {
            "n": int((~positive).sum()), "paucibacillary_share": float((smeared[~positive]["who_class"] == "PB").mean())
        },
        "cohort.description.smear_negative_few_lesions": {
            "n": int(few.sum()), "paucibacillary": int((negative[few]["who_class"] == "PB").sum())
        },
        "cohort.description.smear_negative_more_lesions": {
            "n": int((~few).sum()), "multibacillary": int((negative[~few]["who_class"] == "MB").sum())
        },
        "cohort.description.who_rule_agreement": {
            "n": len(rule_frame), "agree": int((rule == rule_frame["who_class"]).sum()),
            "share": float((rule == rule_frame["who_class"]).mean()),
        },
        "cohort.description.bacilloscopy_class_spearman": {"rho": rho, "n": len(smeared)},
    }


def sensibility(raw: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, Any]:
    """The screened-out sensibility on the analytic records: recorded, altered, by class."""
    altered = ALTERED_SENSIBILITY
    records = raw[raw["record_id"].isin(cohort["record_id"])]
    recorded = records[records[altered].notna()]
    altered_pct = recorded.groupby("class")[altered].mean() * 100
    return {
        "cohort.description.sensibility_recorded": len(recorded),
        "cohort.description.sensibility_altered_share": float(recorded[altered].mean()),
        "cohort.description.sensibility_altered_pct_by_class": per_class(altered_pct),
    }


def excluded_classes(row: pd.Series) -> frozenset[str]:
    """Every class ruled out by some criterion the record fires."""
    ruled_out: set[str] = set()
    for criterion in CRITERIA:
        if criterion.fires(row):
            ruled_out |= ALL_CLASSES - criterion.admissible
    return frozenset(ruled_out)


def criteria_consistency(raw: pd.DataFrame, conflicts: pd.DataFrame) -> dict[str, Any]:
    """The two consistency checks of the criteria: on the reference cases and on the intake records."""
    return {
        "cohort.description.reference_case_conflicts": len(conflicts),
        "cohort.description.records_with_every_class_excluded": int(
            sum(excluded_classes(row) == ALL_CLASSES for _, row in raw.iterrows())
        ),
    }


def reference_case_numbers(reference_cases: pd.DataFrame) -> dict[str, Any]:
    by_class = reference_cases.groupby("class")
    return {
        "reference_cases.n": len(reference_cases),
        "reference_cases.by_class": by_class.size().reindex(REFERENCE_CLASSES).to_dict(),
        "reference_cases.type1_reaction_by_class": (
            by_class["REAC_TIPO_1"].apply(lambda v: int((v > 0).sum())).reindex(REFERENCE_CLASSES).to_dict()
        ),
        "reference_cases.n_missing_values": int(reference_cases[list(FEATURES)].isna().sum().sum()),
    }


def cohort_by_class_table(raw: pd.DataFrame, cohort: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts through the derivation and per-class means of the graded findings."""
    step = raw["record_id"].map(audit.set_index("record_id")["step"])
    after_criteria = raw[step.isna() | (step == COMPLETENESS_REASON)]
    table = pd.DataFrame(
        {
            "intake": raw["class"].value_counts(),
            "after_criteria": after_criteria["class"].value_counts(),
            "analytic": cohort["class"].value_counts(),
        }
    ).reindex(CLASSES).fillna(0).astype(int)
    for column in ("BACILOSCOP_N", "PGL1", "HISTOLOG"):
        table[f"mean_{column}"] = cohort.groupby("class")[column].mean().reindex(CLASSES)
    table.index.name = "class"
    return table.reset_index()


def run() -> None:
    raw = pd.read_parquet(paths.RAW_COHORT)
    cohort = pd.read_parquet(paths.COHORT)
    audit = pd.read_csv(paths.EXCLUSION_AUDIT)
    reference_cases = pd.read_parquet(paths.REFERENCE_CASES)
    conflicts = pd.read_csv(paths.REFERENCE_CASE_CRITERIA_CHECK)

    paths.TABLES.mkdir(parents=True, exist_ok=True)
    cohort_by_class_table(raw, cohort, audit).to_csv(paths.TABLE_COHORT_BY_CLASS, index=False)
    reference_cases[["reference_id", "class", *FEATURES]].to_csv(paths.TABLE_REFERENCE_CASES, index=False)
    record_numbers(
        {
            **attrition_by_class(raw, audit),
            **composition(cohort),
            **marginals(cohort),
            **histology(cohort),
            **missingness(cohort),
            **operational_class(cohort),
            **sensibility(raw, cohort),
            **criteria_consistency(raw, conflicts),
            **reference_case_numbers(reference_cases),
        }
    )
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
