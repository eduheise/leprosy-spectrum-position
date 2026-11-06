# Reports

Everything in this folder is written by `python main.py reports` from the
files in `results/` and `data/processed/`, and is committed. Nothing here is
patient-level: the figures, tables and numbers are aggregates over the cohort.

- `figures/`: every figure the thesis shows from the analysis;
- `tables/`: every table, as CSV;
- `numbers.json`: every number the thesis quotes, keyed by name.

The tables below say which thesis figure, table or section uses each item.
Labels are the LaTeX labels of the thesis (Chapter 5, Results and Discussion,
unless noted).

## Figures

| File | Thesis label | Content | Written by |
|---|---|---|---|
| `exclusion_attrition.png` | `fig:exclusion-attrition` | records removed and remaining at each exclusion step | `reports/exclusion_attrition.py` |
| `search_hypervolume.png` | `fig:val:hypervolume` | hypervolume of the Pareto front over the 500 trials | `reports/search_diagnostics.py` |
| `search_importance.png` | `fig:val:hp-importance` | importance of the design axes (left; `target_head` is labelled "model type") and of the selected backbone's hyperparameters (right), per objective | `reports/search_diagnostics.py` |
| `search_front_by_backbone.png` | `fig:val:pareto` | all trials and the Pareto front, coloured by backbone, with the elbow walk and the selected trial | `reports/search_front.py` |
| `search_front_by_missing_values.png` | `fig:val:pareto-mv` | the same, coloured by missing-value strategy | `reports/search_front.py` |
| `direction_sweeps.png` | `fig:val:variable-behaviour` | mean held-out position as each numeric finding is swept over its values | `reports/spectrum_position.py` |
| `level_shifts.png` | `fig:val:categorical-behaviour` | shift of the mean position when every record is given one colour or type level | `reports/spectrum_position.py` |
| `attribution.png` | `fig:val:attribution` | beeswarm of per-finding contributions to the held-out position | `reports/attribution.py` |
| `who_recovery.png` | `fig:val:who` | ROC curve of the position for MB against PB, and the position density by WHO class | `reports/spectrum_position.py` |
| `position_by_class.png` | `fig:val:who-kde` | density of the held-out position per Ridley-Jopling class | `reports/spectrum_position.py` |
| `confidence_band.png` | `fig:val:confidence-coverage` | drift against the fixed ±2σ band, and its coverage, by number of hidden tests | `reports/confidence.py` |
| `confidence_calibration_curve.png` | `fig:val:coverage-target` | the calibrated multiplier k(p), with held-out quantiles | `reports/confidence.py` |
| `confidence_coverage_heatmap.png` | `fig:val:missingness` | held-out coverage of k(p) by requested level and number of hidden tests | `reports/confidence.py` |

All report modules are in `src/leprosy_similarity/reports/`.

## Tables

| File | Thesis label | Content |
|---|---|---|
| `exclusion_attrition.csv` | `fig:exclusion-attrition`, `sec:data:exclusion-criteria:attrition` | step, group, records removed and remaining |
| `cohort_by_class.csv` | `sec:val:cohort` | records per class at intake, after the criteria and in the analytic cohort; per-class mean bacilloscopy, anti-PGL-1 and histological index |
| `reference_cases.csv` | `tab:prototype-coverage` (Chapter 4), `anx:reference-cases` (Annex A) | the 26 reference cases in the 15-column schema, as the estimator reads them |
| `feature_screen.csv` | `tab:feature-selection` | rise in held-out log loss per candidate variable, its SD, and whether it is retained |
| `search_by_backbone.csv` | `tab:val:backbone` | trials, best separability, best robustness and front points per backbone |
| `search_by_missing_values.csv` | `tab:val:imputation` | the same per missing-value strategy |
| `search_by_target_head.csv` | `sec:val:config-search` (text) | the same per model type (regression or multiclass; `target_head` in the code, where regression is `anchored`) |
| `selected_hyperparameters.csv` | `tab:val:selected-hp` | the selected configuration's hyperparameters and objective values |
| `recovery.csv` | `sec:val:foundations` (text) | WHO AUC, Youden cut, agreement, sensitivity, specificity, the BT contrast and the bacilloscopy increment, one metric per row |
| `class_confusion.csv` | `sec:val:foundations` (text) | recorded class against the nearest class target of the held-out position |
| `direction_sweeps.csv` | `fig:val:variable-behaviour`, `sec:val:variable-behaviour` | per finding and value: mean held-out position and its SD (the curves of the figure) |
| `level_shifts.csv` | `fig:val:categorical-behaviour` | mean position and shift (mean and SD) per colour and type level |
| `attribution.csv` | `tab:val:attribution` | mean absolute contribution and value-contribution correlation per finding |
| `observed_imputed.csv` | `tab:val:observed-imputed` | mean absolute contribution when the finding was observed against imputed, with the imputer's reconstruction MAE and R² |
| `confidence_calibration.csv` | `tab:val:confidence-calibration` | held-out coverage and mean band width per completion generator at 80, 90 and 95% |

## numbers.json

Keys are grouped by prefix; each group is written by one report module
(`cohort.attrition.*` by `exclusion_attrition`, `cohort.description.*` and
`reference_cases.*` by `cohort_description`).

| Key prefix | Thesis section | Content |
|---|---|---|
| `cohort.attrition.*` | `sec:data:exclusion-criteria:attrition` | intake 1,030, removed by the criteria, by completeness, the analytic 916, removals per step |
| `cohort.description.*` | `sec:val:cohort` (`sec:data:exclusion-criteria:attrition`, `sec:data:analytics`), `sec:val:variable-selection` | class mix through the two exclusion steps, class and centre counts, lesion tiers, per-class means and level shares, histology coverage and its within-centre Spearman, the block without a clinical examination, the WHO class against the smear and the strict rule, sensibility, the two consistency checks of the criteria |
| `reference_cases.*` | `tab:prototype-coverage` (Chapter 4), `sec:val:attribution:discussion` | cases per class, type-1 reactions per class, missing values |
| `screen.*` | `sec:val:variable-selection`, `tab:feature-selection` | log-loss rises and SDs, retained and rejected variables, baseline log loss, reaction prevalence by class and the reaction-shape fits |
| `search.n_trials`, `search.n_feasible`, `search.n_infeasible_by_design`, `search.n_failed_fits` | `sec:val:config-search` | trial counts |
| `search.by_backbone`, `search.by_missing_values`, `search.by_target_head` | `tab:val:backbone`, `tab:val:imputation`, `sec:val:config-search` | the per-group summaries of the tables |
| `search.front.*` | `fig:val:pareto`, `sec:val:config-search` | front size, trials and backbone counts |
| `search.elbow.*` | `fig:val:pareto`, `sec:val:config-search` | every step of the elbow walk, the first declined step, separability given up and drift recovered |
| `search.selected.*` | `tab:val:selected-hp`, `sec:val:config-search` | the selected trial, its configuration, hyperparameters, backbone seed, objectives and class means |
| `search.hypervolume.*` | `fig:val:hypervolume` | final hypervolume, reference point, share gained by trial 100, visible rises and the last rise |
| `search.importance.*` | `fig:val:hp-importance` | design-axis and hyperparameter importances per objective, and their seed |
| `rescaling.*` | `sec:val:position-range` | reachable range and span over the 58,800-record grid, the extreme records and their reference weights, base and anchors on both scales, the reference cases' and the cohort's positions, the five fold ranges |
| `position.order.*` | `sec:val:foundations`, `fig:val:who-kde` | Spearman, quadratic kappa, MAE (plain and class-balanced, poles and middle, per class), class means and medians, steps between means, exact and within-one-step counts, largest confusion, silhouette |
| `position.who`, `position.who.*` | `fig:val:who`, `sec:val:foundations` | WHO AUC, cut, agreement, sensitivity, specificity; the BT within-class contrast; the bacilloscopy increment with both fills |
| `position.sweep.*` | `fig:val:variable-behaviour`, `fig:val:categorical-behaviour`, `sec:val:variable-behaviour` | per-finding sweep curves, ranges and directions, level shifts, baseline, violations |
| `position.n_*` | `sec:val:foundations` | records, folds, and positions clipped at 0 or 1 |
| `attribution.*` | `tab:val:attribution`, `tab:val:observed-imputed`, `fig:val:attribution` | mean absolute contribution and correlation per finding, observed against imputed, base per fold, efficiency gaps |
| `confidence.sharpness.*`, `confidence.runner_up*`, `confidence.selected_arm` | `tab:val:confidence-calibration`, `sec:val:confidence-selection` | coverage and width per generator and level; the selected generator and the runner-up's extra width |
| `confidence.k.*`, `confidence.k_held_out_coverage.*` | `fig:val:coverage-target`, `sec:val:platform` | k(p) at 0.80, 0.90, 0.95, 0.99, 0.999 and its held-out coverage |
| `confidence.fixed_band*` | `fig:val:confidence-coverage` | ±2σ coverage overall and per number hidden; drift median, 90th percentile, band half-width and zero-drift share per number hidden |
| `confidence.zero_drift_*`, `confidence.mean_confidence_*`, `confidence.pairs_by_size.*`, `confidence.n_*`, `confidence.*_pairs*`, `confidence.held_out_records`, `confidence.rank_agreement.*` | `sec:val:stability` | sweep sizes, zero-drift pairs, mean confidence at the poles and in the middle, rank agreement of σ with drift |
