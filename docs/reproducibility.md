# Reproducibility

With the same cohort file, `python main.py all` gives the same
`reports/numbers.json`, tables and figures on every run, apart from the
LightGBM thread-order effect on the search described below.
This page lists what that rests on.

## Seeds

Every random choice is seeded from `config.yaml`; no seed is written in code.

| Seed | Value | Used by |
|---|---|---|
| `folds.seed` | 42 | the shared 5-fold stratified split (`StratifiedKFold`, shuffled) |
| `search.sampler_seed` | 42 | NSGA-II sampler of the configuration search |
| `search.masking_seed` | 43 | masking sizes and hidden columns of the drift objective (its own generator, so it never shares a stream with the split) |
| `backbones.seed` | 7 | every tree ensemble (LightGBM, XGBoost, CatBoost, random forest) |
| `imputation.iterative.seed` | 0 | iterative Bayesian-ridge strategy |
| `imputation.miceforest.seed` | 0 | MICE-Forest strategy |
| `variable_screen.seed` | 7 | folds, forest and permutations of the variable screen |
| `seed` | 42 | the confidence sweep (records sampled, columns hidden) and its completions (draw `m` uses seed `42 + m`); the platform's donor draws |
| `confidence.calibration_split_seed` | 7 | the split of the sweep's records into calibration and held-out halves |
| `search_report.importance_seed` | 0 | fANOVA design-axis importances and the hyperparameter-importance forest |

The backbone seed (7) differs from the split seed (42) because the stored
500-trial search was run with it; changing it would change every trial.

## Versions

The numerical libraries are pinned in `pyproject.toml` and locked in
`uv.lock`; `uv sync` installs exactly these:

| Package | Version |
|---|---|
| Python | 3.12 |
| numpy | 2.4.3 |
| pandas | 3.0.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| optuna | 4.9.0 |
| miceforest | 6.0.5 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

Tree Similarity SHAP comes from the `shapley-values` package, a uv path
dependency on `../shapley_values` (`[tool.uv.sources]` in `pyproject.toml`).
When that package is published, the path source will be replaced by a pinned
git reference.

## Determinism

- **kNN imputation** computes the nan-Euclidean distance exactly and breaks
  ties between equidistant donors by their order in the training data, so the
  fill is the same on every machine and architecture.
- **MICE-Forest** draws from one random stream across calls, so the fill a
  record gets depends on what was filled before it. The search objective
  keeps a fixed call order per fold (fit, reference cases, held-out fold,
  masked fold, rescaling grid). The selected configuration uses kNN, which
  has no such dependence.
- **The variable screen** fits its random forests on every core but predicts
  on one thread: a parallel prediction sums the trees' probabilities in the
  order the threads finish, which changes the last bits of the log loss.
- **Rescaling** positions the full recordable grid (58,800 records) in
  batches; the batch size changes memory use only.
- **Attribution** spreads records over worker processes; each record's sums
  run in the same order whatever the number of workers, so the result does
  not depend on it. Only its running time does, which is printed and kept
  in `results/attribution.npz` but left out of `reports/numbers.json`, so
  that file is byte-identical between runs.
- **Confidence** computes `sigma` from whole shared-leaf counts rather than
  from floating-point similarities, so two completions that reach the same
  position through different leaves give a spread of exactly zero.
- **The search** is reproducible from its seed when run without
  interruption. A resumed search is valid but not bit-identical. A full
  rerun from scratch reproduced 996 of the 1,000 stored objective values
  exactly and selected the same configuration. The other four belong to two
  LightGBM trials. LightGBM trains on every core and sums its histograms in
  thread order, so those values moved in the fifth to sixth decimal (at most
  3e-5). The only output this changes is the design-axis importance (fANOVA) in the
  importance figure, by less than 0.001.
  `python -m experiments.configuration_search --smoke` reruns the first trials
  and checks them against the stored study.
- **Figures** are written by matplotlib's Agg backend with a fixed style;
  reruns give byte-identical PNG files on the same machine.

## Checks

- `uv run pytest` runs on small synthetic fixtures generated in code and needs
  no patient data. Among other things it checks the exact efficiency of the
  attribution, deterministic kNN, rescaling bounds, that no record is on both
  sides of any fold, and that the same seed gives the same output.
- The pipeline refuses to start a step whose inputs are missing.
- `experiments/attribution.py` stops if any record's contributions miss its
  position by more than `attribution.efficiency_tolerance` (1e-12).
- `final_model` writes a manifest with the SHA-256 of the bundle, of the
  selected configuration and of the confidence summary, so the platform can
  show which model it serves.

## Runtimes

On an Apple-silicon laptop: `preprocess` about 25 s; the configuration
search about 13 h; the other experiment steps about 15 min in total;
`reports` about 5 s. See [pipeline.md](pipeline.md) for each step.
