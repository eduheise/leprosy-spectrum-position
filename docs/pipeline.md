# Pipeline

`main.py` runs three stages. Each stage reads files that earlier stages wrote
and writes its own, so a stage can be rerun alone from what is on disk:

```
python main.py preprocess     # data/raw, data/reference_cases -> data/processed
python main.py experiments    # data/processed -> results
python main.py reports        # data/processed, results -> reports
python main.py all            # the three stages in order
```

Prefix each command with `uv run` unless the project environment is active.

## Steps

A step is a module with `INPUTS`, `OUTPUTS` (paths from
`src/leprosy_similarity/paths.py`) and `run()`. Steps run in the order below.
Before a step starts, `main.py` checks that its inputs exist; if one is
missing it stops and prints the command that writes it.

### preprocess

| Step | Reads | Writes | Time |
|---|---|---|---|
| `load_raw_cohort` | `data/raw/rafa_total.xlsx` | `data/processed/raw_cohort.parquet` | < 1 s |
| `apply_exclusion_criteria` | `raw_cohort.parquet` | `cohort.parquet`, `exclusion_audit.csv`, `exclusion_attrition.csv`, `schema.json` | < 1 s |
| `build_reference_cases` | `data/reference_cases/prototypes.xlsx` | `reference_cases.parquet`, `reference_case_criteria_check.csv` | < 1 s |
| `make_folds` | `cohort.parquet` | `folds.csv` | < 1 s |
| `screen_variables` | `raw_cohort.parquet`, `exclusion_audit.csv`, `cohort.parquet` | `variable_screen.csv`, `variable_screen_reaction_prevalence.csv`, `variable_screen_shape.csv`, `variable_screen_baseline.json` | 20 s |

All files are in `data/processed/` unless a folder is given.

### experiments

Every step except `select_configuration` reads `cohort.parquet` and
`reference_cases.parquet` from `data/processed/`, and every step except
`select_configuration` and `final_model` also reads `folds.csv`; the table
lists what else it reads.

| Step | Reads | Writes (in `results/`) | Time |
|---|---|---|---|
| `configuration_search` | | `search/study.db` | about 13 h for 500 trials |
| `select_configuration` | `search/study.db` | `selected_configuration.json` | < 1 s |
| `position_rescaling` | `selected_configuration.json` | `position_rescaling.json` | 40 s |
| `spectrum_position` | `selected_configuration.json` | `oof_positions.parquet`, `spectrum_position.json` | 50 s |
| `attribution` | `selected_configuration.json` | `attribution.npz`, `imputation_reconstruction.csv` | 2.5 min |
| `confidence` | `selected_configuration.json` | `confidence.parquet`, `confidence_summary.json` | 11 min |
| `final_model` | `selected_configuration.json`, `confidence_summary.json` | `final_model/model.joblib`, `final_model/manifest.json` | 7 s |

`configuration_search` runs only the trials the stored study lacks, so on a
finished study it does nothing. An interrupted search resumes where it
stopped; the result is valid but not bit-identical to an uninterrupted run,
because the sampler is reseeded on resumption.
To check the search code without the 13-hour run:

```
python -m experiments.configuration_search --smoke [--trials N]
```

runs a few trials into a temporary study and compares each with the stored
trial of the same number; with the same sampler seed they must agree exactly.

`final_model` writes the bundle the clinical platform serves: the selected
configuration fitted on all 916 records, its rescaling range, the reference
cases, the donor pool and the multiplier `k` at each coverage the platform
offers, with a manifest holding the SHA-256 of the bundle, of the selected
configuration and of the confidence summary.

### reports

| Step | Reads | Writes (in `reports/`) |
|---|---|---|
| `exclusion_attrition` | `data/processed/exclusion_attrition.csv` | `figures/exclusion_attrition.png`, `tables/exclusion_attrition.csv` |
| `cohort_description` | `data/processed/raw_cohort.parquet`, `cohort.parquet`, `exclusion_audit.csv`, `reference_cases.parquet`, `reference_case_criteria_check.csv` | `tables/cohort_by_class.csv`, `tables/reference_cases.csv` |
| `feature_screen` | `data/processed/variable_screen*.csv`, `variable_screen_baseline.json` | `tables/feature_screen.csv` |
| `search_front` | `results/search/study.db`, `results/selected_configuration.json` | `figures/search_front_by_backbone.png`, `figures/search_front_by_missing_values.png`, `tables/search_by_backbone.csv`, `tables/search_by_missing_values.csv`, `tables/search_by_target_head.csv`, `tables/selected_hyperparameters.csv` |
| `search_diagnostics` | `results/search/study.db`, `results/selected_configuration.json` | `figures/search_hypervolume.png`, `figures/search_importance.png` |
| `position_rescaling` | `results/position_rescaling.json` | — (numbers only, see below) |
| `spectrum_position` | `results/oof_positions.parquet`, `results/spectrum_position.json` | `figures/direction_sweeps.png`, `figures/level_shifts.png`, `figures/who_recovery.png`, `figures/position_by_class.png`, `tables/recovery.csv`, `tables/class_confusion.csv`, `tables/direction_sweeps.csv`, `tables/level_shifts.csv` |
| `attribution` | `results/attribution.npz`, `results/imputation_reconstruction.csv` | `figures/attribution.png`, `tables/attribution.csv`, `tables/observed_imputed.csv` |
| `confidence` | `results/confidence.parquet`, `results/confidence_summary.json` | `tables/confidence_calibration.csv`, `figures/confidence_band.png`, `figures/confidence_calibration_curve.png`, `figures/confidence_coverage_heatmap.png` |

Every report step also merges its numbers into `reports/numbers.json` under
its own prefix (`cohort.`, `reference_cases.`, `screen.`, `search.`, `rescaling.`, `position.`,
`attribution.`, `confidence.`). The whole stage takes about 5 seconds and
refits nothing. [reports/README.md](../reports/README.md) maps every output
to the thesis.

## Rerunning one step

```
python main.py experiments --only confidence
python main.py reports --only confidence
```

`--only` takes the step name (the module name). Steps downstream of the one
rerun are not rerun automatically; rerun them, or the whole later stage, when
their inputs change. For example, after a new search:

```
python main.py experiments --only select_configuration
python main.py experiments --only position_rescaling   # and every later step
python main.py reports
```

Each step can also be run as a module, for example
`python -m experiments.confidence`, without the input check.

## Adding a step

Write a module in `preprocessing/`, `experiments/` or
`src/leprosy_similarity/reports/` with `INPUTS`, `OUTPUTS` and `run()`; add any
new path to `paths.py` and any new constant to `config.yaml` with a typed field
in `config.py`; then add the module's dotted name to `STAGES` in `main.py`, in
run order.

## Library map

| Module | Role |
|---|---|
| `paths`, `config` | every path and every constant |
| `schema` | the 15 columns, the classes and their targets |
| `cohort`, `exclusion_criteria`, `reference_cases` | reading the workbooks and deriving the analytic cohort |
| `folds` | the shared split |
| `imputation` | the missing-value strategies, the deterministic kNN imputer and the whole-record donor draws |
| `backbones` | building the four tree libraries, reading leaf indices, the hyperparameter space |
| `estimator` | `Configuration` and `SpectrumEstimator` (fit, raw position, proximity, rescaled position) |
| `rescaling` | the recordable grid and the position range |
| `masking`, `objectives`, `search` | the two search objectives, NSGA-II, the Pareto front and the elbow rule |
| `cross_validation` | the selected configuration fitted once per fold, and held-out positions |
| `recovery`, `sweeps` | WHO and Ridley-Jopling recovery, direction sweeps |
| `attribution` | Tree Similarity SHAP projected onto the position |
| `masking_sweep`, `confidence`, `confidence_sweep` | the masking sweep, the completion generators, `sigma`, `k(p)` |
| `platform_bundle` | the final-model bundle and its manifest |
| `reports/` | one module per figure or table group, the shared figure style, and `numbers.json` |
