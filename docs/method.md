# Method

This page states the protocol the code implements, with the equations behind
each step. It follows the Methods chapter of the thesis; where a constant is
named, its value lives in `config.yaml`.

## 1. Cohort

The raw workbook yields 1,030 records labelled with a Ridley-Jopling class,
from Brazilian reference centres (the sites are described in
[data.md](data.md)).
Exclusion criteria from the literature then remove two kinds of record, in a
fixed order:

- records whose label contradicts their own findings, for example one to five
  lesions recorded for a BL or LL patient (`exclusion_criteria.py`);
- records with neither a bacilloscopy nor a histological index (the
  completeness criterion).

916 records remain: the **analytic cohort** (TT 62, BT 268, BB 70, BL 289,
LL 227). Every record's fate is written to `data/processed/exclusion_audit.csv`,
and the counts per step to `exclusion_attrition.csv`. The criteria are also
checked against the 26 reference cases; none rules out a reference case's
class.

## 2. Variables

A variable screen (`preprocessing/screen_variables.py`) decides which
candidate findings enter the schema. A random forest is trained on the
Ridley-Jopling label in 5-fold cross-validation; each candidate (the one-hot
levels of a categorical permuted together) is shuffled in the held-out fold,
and its importance is the rise in held-out multiclass log loss. A variable is
kept when its mean rise exceeds its fold-to-fold standard deviation. The
screen runs before the configuration search and does not depend on it.

The resulting schema has 15 columns:

| Column | Values |
|---|---|
| `NU_LESOES` | lesion count tier: 1 (1-5), 2 (6-11), 3 (diffuse infiltration) |
| `REAC_TIPO_1`, `REAC_TIPO_2` | type-1 and type-2 reaction history, 0/1 |
| `PGL1` | anti-PGL-1 titre, 0-4 |
| `BACILOSCOP_N` | bacilloscopy index, 0-6 |
| `HISTOLOG` | histological index, 0-6 |
| `lesion_color__*` | one-hot, 4 levels: erythematous, hypochromic, hyperchromic, normochromic |
| `lesion_element__*` | one-hot, 5 levels: plaque, papule, nodule, macule, diffuse infiltration |

The WHO operational class (paucibacillary or multibacillary) is never a
feature: it is defined by a rule on lesion count and bacilloscopy, which are
features already. It is kept as a validation target.

## 3. Reference cases and anchors

26 reference cases were elicited from leprosy specialists: 9 TT, 8 BB and
9 LL (`data/reference_cases/`, listed in [data.md](data.md)). Each reference
case $p$ carries the anchor of its class,

$$
\mathrm{anchor}(\mathrm{TT}) = 0,\qquad
\mathrm{anchor}(\mathrm{BB}) = 0.5,\qquad
\mathrm{anchor}(\mathrm{LL}) = 1 .
$$

For supervision and scoring, the five classes have the targets TT 0, BT 0.25,
BB 0.5, BL 0.75, LL 1. Equal spacing is an assumption, adopted as the
reference scale; the position itself is continuous and the classes are free
to settle at unequal distances.

## 4. Estimator

A tree ensemble (the **backbone**) is trained on the cohort's
Ridley-Jopling labels. The similarity between a record $x$ and a reference
case $p$ is the share of the $T$ trees in which they reach the same leaf,

$$
\pi(x, p) \;=\; \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\big[\ell_t(x) = \ell_t(p)\big],
$$

and the **raw position** is the Nadaraya-Watson estimate over the reference
cases,

$$
\hat y(x) \;=\; \frac{\sum_{p} \pi(x, p)\,\mathrm{anchor}(c(p))}{\sum_{p} \pi(x, p)} .
$$

A record with missing findings reaches a leaf in every tree either because
the missing-value strategy completed it first, or because the trees route the
missing value themselves (native routing). The code is
`estimator.SpectrumEstimator`.

### Rescaling

The raw position never reaches 0 or 1, because every record keeps some
similarity to the reference cases of the other classes. The fitted model
therefore positions every **recordable complete record**: every combination
of the values each column takes in the training records, with exactly one
level in each of the colour and morphology blocks (58,800 records for the
analytic cohort). The lowest and highest raw positions over that grid,
$\ell$ and $h$, define the reported position

$$
\tilde y(x) \;=\; \min\!\Big(1,\ \max\!\Big(0,\ \frac{\hat y(x) - \ell}{h - \ell}\Big)\Big).
$$

Every fitted model computes its own $\ell$ and $h$ from its own training
records, so in cross-validation a held-out record plays no part in the scale
it is read on. The search objectives, every result and the platform read
$\tilde y$. The code is `rescaling.py`.

## 5. The shared split

The 916 records are split once into 5 folds, stratified by class, with seed
42 (`data/processed/folds.csv`). The configuration search and every
cross-validated result use this one split.

## 6. Configuration search

Three design axes and the hyperparameters are searched together:

- **backbone**: LightGBM, XGBoost, CatBoost or random forest;
- **model type** (regression or multiclass): *multiclass* (the five classes)
  or *regression* on the class targets. The code calls this axis
  `target_head`, with the values `multiclass` and `anchored`;
- **missing-value strategy**: native routing, mean/median, distance-weighted
  kNN ($k = 5$, exact nan-Euclidean distance, ties broken by training order),
  iterative Bayesian ridge, MICE-Forest, or MIA indicators. A random forest
  cannot route a missing value, so (random forest, native routing) is
  infeasible.

The hyperparameter ranges are in `config.yaml` under `search.space`. Every
backbone is seeded with 7.

For each configuration and each fold, the strategy and backbone are fitted on
the other four folds, and the held-out fold is positioned twice: as recorded,
and after hiding a random number of its observed columns (drawn from
$\{2, 4, 6, 8, 10, 12\}$, capped so at least one stays observed) and refilling
them with the configuration's own strategy. Each fold's positions are
rescaled by that fold's own range. Pooled over the five folds, the two
objectives are

$$
\text{separability} \;=\; \frac{1}{5}\sum_{c} \frac{1}{n_c}\sum_{i \in c} \big|\tilde y_i - \mathrm{target}(c)\big|,
\qquad
\text{drift} \;=\; \frac{1}{n}\sum_i \big|\tilde y_i - \tilde y_i^{\text{masked}}\big| .
$$

Separability is class-balanced, so the small TT and BB classes count as much
as the large ones. NSGA-II (Optuna, seed 42) minimises both over 500 trials.

**Selection.** The Pareto front is ordered by separability, from the lowest
error outward. A step along the front costs $\Delta_{\text{sep}}$ of
separability and buys $\Delta_{\text{drift}}$ of drift; the walk continues
while

$$
\frac{\Delta_{\text{drift}}}{\Delta_{\text{sep}}} > 1
$$

and stops at the last step that clears it. Both objectives are errors on the
same 0-1 scale, so a threshold of one compares like with like. The rule
selects one configuration (`results/selected_configuration.json`); nothing
after it searches again.

## 7. Results on held-out folds

The selected configuration is refitted once per fold of the same split. Each
record is read by the fold model that did not train on it, on that model's
own scale, and the five held-out folds are pooled. On these positions:

- **WHO recovery**: ROC AUC of the position for MB against PB, the Youden
  cut, agreement, sensitivity and specificity; the AUC of a logistic model on
  bacilloscopy alone against bacilloscopy plus position, on the same folds.
- **Ridley-Jopling ordering**: Spearman correlation and quadratic-weighted
  kappa between position and class, class means and medians, the error at the
  poles and in the middle, the confusion against the nearest class target, and
  silhouette.
- **Direction sweeps**: every record has one finding set to each of its
  values in turn, and the mean held-out position is traced. Findings that the
  disease moves toward the lepromatous pole (bacilloscopy, histology, titre,
  lesion count, type-2 reaction) are expected to rise; a curve that does not
  is a violation. Colour and morphology levels are reported as shifts.
- **Attribution** (Tree Similarity SHAP): for every record and reference case,
  the similarity is split into one additive share per feature,
  $\pi(x, p) = \pi_0 + \sum_f \phi(x, p, f)$. Because the position is a
  similarity-weighted mean, the shares project onto it exactly:

  $$
  \mathrm{contribution}(x, f) \;=\; \frac{\sum_p \phi(x, p, f)\,\big(\mathrm{anchor}(c(p)) - b\big)}{\sum_p \pi(x, p)},
  $$

  where $b$ is the mean reference anchor. The contributions of a record sum to
  its position minus $b$; divided by the fold model's span $h - \ell$, they
  sum to the rescaled position minus the rescaled base. The identity is
  checked for every record (largest gap about $2 \times 10^{-15}$).

## 8. Confidence under missing tests

A record's missing values are completed $M = 20$ times, the record is
positioned on each completion, and $\sigma$ is the standard deviation of the
20 rescaled positions. The band is $\pm k\sigma$ and the confidence is
$1 - k\sigma$.

Three generators of completions are compared: whole-record kNN donors (in each
draw, all of a record's missing values are copied from one donor drawn among
the five nearest training records that observed them), Bayesian-ridge
posterior draws, and MICE-Forest.

The multiplier $k$ is measured, not assumed:

1. **Sweep.** 30 records per class that lack at most one column are drawn
   (seed 42), 150 in all. From each, 5 random sets of 2 to 5 observed columns
   are hidden, giving 750 (record, mask) pairs.
2. **Held-out scoring.** Each pair is scored by the fold model that did not
   train on its record, with the generators fitted on that fold's training
   records. The drift is the distance between the masked and the full
   position.
3. **Split.** The 150 records are split in half (seed 7), all masks of a
   record on the same side.
4. **Calibration.** On the calibration half, $k(p)$ is the $p$-quantile of
   $\text{drift}/\sigma$ (a monotone spline across $p$). Its coverage is
   reported on the other half.
5. **Choice of generator.** At requested coverages 0.80, 0.90 and 0.95, the
   generator with the narrowest band at the most levels is selected
   (sharpness subject to calibration).
6. The coverage of a fixed $\pm 2\sigma$ band, overall and by number of hidden
   columns, and the coverage of $k(p)$ by requested level and number of hidden
   columns are reported.

## 9. Final model

The selected configuration fitted on all 916 records, with its own rescaling
range, the reference cases, the donor pool and $k$ at the coverages the
platform offers, is written to `results/final_model/` with a SHA-256
manifest. It is the model the clinical platform serves and is not used for
any evaluation.
