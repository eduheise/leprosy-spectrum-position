"""Every file location the pipeline reads or writes, defined once.

The pipeline has three stages that hand over through files on disk:
preprocessing writes ``data/processed``, experiments read it and write
``results``, and reports read ``results`` and write ``reports``. Patient data
lives only in ``data/raw``, ``data/processed`` and ``results``, all three
excluded from version control.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.yaml"

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_COHORT_XLSX = RAW / "rafa_total.xlsx"

REFERENCE_CASES_DIR = DATA / "reference_cases"
REFERENCE_CASES_XLSX = REFERENCE_CASES_DIR / "prototypes.xlsx"

PROCESSED = DATA / "processed"
RAW_COHORT = PROCESSED / "raw_cohort.parquet"
COHORT = PROCESSED / "cohort.parquet"
REFERENCE_CASES = PROCESSED / "reference_cases.parquet"
SCHEMA = PROCESSED / "schema.json"
EXCLUSION_AUDIT = PROCESSED / "exclusion_audit.csv"
EXCLUSION_ATTRITION = PROCESSED / "exclusion_attrition.csv"
REFERENCE_CASE_CRITERIA_CHECK = PROCESSED / "reference_case_criteria_check.csv"
VARIABLE_SCREEN = PROCESSED / "variable_screen.csv"
VARIABLE_SCREEN_REACTION_PREVALENCE = PROCESSED / "variable_screen_reaction_prevalence.csv"
VARIABLE_SCREEN_SHAPE = PROCESSED / "variable_screen_shape.csv"
VARIABLE_SCREEN_BASELINE = PROCESSED / "variable_screen_baseline.json"
FOLDS = PROCESSED / "folds.csv"

RESULTS = ROOT / "results"
SEARCH_DIR = RESULTS / "search"
STUDY_DB = SEARCH_DIR / "study.db"
SELECTED_CONFIGURATION = RESULTS / "selected_configuration.json"
POSITION_RESCALING = RESULTS / "position_rescaling.json"
OOF_POSITIONS = RESULTS / "oof_positions.parquet"
SPECTRUM_POSITION = RESULTS / "spectrum_position.json"
ATTRIBUTION = RESULTS / "attribution.npz"
IMPUTATION_RECONSTRUCTION = RESULTS / "imputation_reconstruction.csv"
CONFIDENCE = RESULTS / "confidence.parquet"
CONFIDENCE_SUMMARY = RESULTS / "confidence_summary.json"
CONFIDENCE_SINGLE_FINDING = RESULTS / "confidence_single_finding.parquet"
FINAL_MODEL_DIR = RESULTS / "final_model"
FINAL_MODEL_BUNDLE = FINAL_MODEL_DIR / "model.joblib"
FINAL_MODEL_MANIFEST = FINAL_MODEL_DIR / "manifest.json"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
NUMBERS = REPORTS / "numbers.json"

# Every figure and table the reports stage writes. A figure and the table
# behind it share a name.

# Cohort derivation, cohort description and variable screen.
FIG_EXCLUSION_ATTRITION = FIGURES / "exclusion_attrition.png"
TABLE_EXCLUSION_ATTRITION = TABLES / "exclusion_attrition.csv"
TABLE_COHORT_BY_CLASS = TABLES / "cohort_by_class.csv"
TABLE_REFERENCE_CASES = TABLES / "reference_cases.csv"
TABLE_FEATURE_SCREEN = TABLES / "feature_screen.csv"

# Configuration search and selection.
FIG_SEARCH_FRONT_BY_BACKBONE = FIGURES / "search_front_by_backbone.png"
FIG_SEARCH_FRONT_BY_MISSING_VALUES = FIGURES / "search_front_by_missing_values.png"
FIG_SEARCH_HYPERVOLUME = FIGURES / "search_hypervolume.png"
FIG_SEARCH_IMPORTANCE = FIGURES / "search_importance.png"
TABLE_SEARCH_BY_BACKBONE = TABLES / "search_by_backbone.csv"
TABLE_SEARCH_BY_MISSING_VALUES = TABLES / "search_by_missing_values.csv"
TABLE_SEARCH_BY_TARGET_HEAD = TABLES / "search_by_target_head.csv"
TABLE_SELECTED_HYPERPARAMETERS = TABLES / "selected_hyperparameters.csv"

# Spectrum position: response to each finding and recovery of WHO and Ridley-Jopling.
FIG_DIRECTION_SWEEPS = FIGURES / "direction_sweeps.png"
FIG_LEVEL_SHIFTS = FIGURES / "level_shifts.png"
FIG_WHO_RECOVERY = FIGURES / "who_recovery.png"
FIG_POSITION_BY_CLASS = FIGURES / "position_by_class.png"
TABLE_RECOVERY = TABLES / "recovery.csv"
TABLE_CLASS_CONFUSION = TABLES / "class_confusion.csv"
TABLE_DIRECTION_SWEEPS = TABLES / "direction_sweeps.csv"
TABLE_LEVEL_SHIFTS = TABLES / "level_shifts.csv"

# Per-finding attribution.
FIG_ATTRIBUTION = FIGURES / "attribution.png"
TABLE_ATTRIBUTION = TABLES / "attribution.csv"
TABLE_OBSERVED_IMPUTED = TABLES / "observed_imputed.csv"

# Confidence under missing tests.
FIG_CONFIDENCE_BAND = FIGURES / "confidence_band.png"
FIG_CONFIDENCE_CALIBRATION_CURVE = FIGURES / "confidence_calibration_curve.png"
FIG_CONFIDENCE_COVERAGE_HEATMAP = FIGURES / "confidence_coverage_heatmap.png"
TABLE_CONFIDENCE_CALIBRATION = TABLES / "confidence_calibration.csv"


def study_storage() -> str:
    """Optuna storage URL of the configuration search."""
    return f"sqlite:///{STUDY_DB}"
