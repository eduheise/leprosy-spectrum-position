# Data

Two inputs feed the pipeline: the patient cohort, which is private and stays
out of the repository, and the 26 specialist reference cases, which are
public and tracked.

## The patient cohort

The cohort is individual patient data. It is available on request from the
authors, under the study's ethics approval, and is never committed:
`data/raw/`, `data/processed/` and `results/` are all gitignored.

### Expected file

Place one workbook, `data/raw/rafa_total.xlsx`, with one row per patient.
Every sheet is read and the sheets are stacked. A row is kept only if
`Forma Clínica` holds a Ridley-Jopling class (`TT`, `BT`, `BB`, `BL` or
`LL`). With the study workbook, 1,451 rows give 1,030 labelled records (Goiânia and unlabelled rows dropped).

### Columns read

| Column | Content | Becomes |
|---|---|---|
| `Forma Clínica` | Ridley-Jopling class | `class`, and its target 0 / 0.25 / 0.5 / 0.75 / 1 |
| `Número de Lesões` | `1 a 5`, `6 a 11` or `Infiltração Difusa` | `NU_LESOES` tier 1 / 2 / 3 |
| `Tipo de Reação` | `Reação Tipo 1`, `Reação Tipo 2` or `Sem Reação` | `REAC_TIPO_1`, `REAC_TIPO_2` (0/1) |
| `Intensidade do PGL-1+` | `Negativo`, `+` … `++++` | `PGL1` (0-4) |
| `Indice Baciloscópico` | `Negativo`, `+` … `6+` | `BACILOSCOP_N` (0-6) |
| `Indice Histológico` | `Negativo`, `+` … `6+` | `HISTOLOG` (0-6) |
| `Cor da Lesão` | eritematosa, hipocrômica, hipercrômica, normocrômica | one-hot `lesion_color__*` |
| `Tipo de Lesão` | placa, pápula, nódulo, mácula, infiltração difusa | one-hot `lesion_element__*` |
| `Sensibilidade` | alterada, duvidosa, normal | variable screen only (screened out) |
| `Forma de Tratamento` | 6- or 12-month regimen | `who_class` PB / MB, a validation target, never a finding |

A blank or unreadable cell becomes a missing value. Missing values are
expected: the estimator handles them through the selected missing-value
strategy.

The spreadsheet stacks the centres' databases in order, so a row's position
gives its centre: rows 1–372 are Bauru, rows 373–1,042 Fortaleza/Manaus and
rows 1,043–1,451 Goiânia (`SITE_BLOCKS` in `cohort.py`). The Goiânia sample is
not part of this study and is dropped while reading. The genotype panels are
not used to tell the centres apart, since Goiânia typed some of the same genes
as Bauru; no genotype value enters the analysis.

### What preprocessing writes

`python main.py preprocess` writes to `data/processed/`:

| File | Content |
|---|---|
| `raw_cohort.parquet` | 1,030 labelled records: `record_id`, the 15 schema columns, three `sensibility__*` columns, `class`, `target`, `site`, `who_class` |
| `cohort.parquet` | the 916-record analytic cohort, same columns without `sensibility__*` |
| `reference_cases.parquet` | 26 rows: `reference_id` (1-26), `class`, the 15 schema columns |
| `schema.json` | numeric and categorical columns, levels, class targets, reference classes |
| `exclusion_audit.csv` | one row per labelled record: whether it was removed, at which step and by which criterion |
| `exclusion_attrition.csv` | records removed and remaining after each step |
| `reference_case_criteria_check.csv` | reference cases whose class a criterion would rule out (empty) |
| `folds.csv` | `record_id`, `fold` (0-4): the shared split |
| `variable_screen*.csv`, `variable_screen_baseline.json` | the variable screen and its reaction tables |

The analytic cohort has 916 records: TT 62, BT 268, BB 70, BL 289, LL 227.

### The 15-column schema

| Column | Values |
|---|---|
| `NU_LESOES` | lesion count tier: 1 (1-5 lesions), 2 (6-11), 3 (diffuse infiltration) |
| `REAC_TIPO_1` | type-1 reaction history, 0/1 |
| `REAC_TIPO_2` | type-2 reaction history, 0/1 |
| `PGL1` | anti-PGL-1 titre, 0-4 |
| `BACILOSCOP_N` | bacilloscopy index, 0-6 |
| `HISTOLOG` | histological index, 0-6 |
| `lesion_color__eritematosa`, `__hipocromica`, `__hipercromica`, `__normocromica` | lesion colour, one-hot |
| `lesion_element__placa`, `__papula`, `__nodulo`, `__macula`, `__infiltracao_difusa` | lesion type, one-hot |

## The reference cases

`data/reference_cases/prototypes.xlsx` holds 26 descriptions of a typical
patient, each written by a leprosy specialist for one class: 9 TT, 8 BB and
9 LL. They anchor the position at 0 (TT), 0.5 (BB) and 1 (LL). The workbook
carries no specialist names.

Specialists often allowed several values for a finding ("1 ou 2", "0 a 4",
"eritematosa ou normocromica"). Each cell is collapsed to one value so a
reference case is a single record (`reference_cases.py`):

- a numeric finding takes the mean of the values allowed, hence the
  fractional grades below;
- the lesion count is bucketed into the cohort's three tiers first, then the
  tiers are averaged;
- a categorical finding spreads one unit evenly over the levels allowed, shown
  below as shares;
- a blank or unknown cell is missing.

The 26 cases as the estimator reads them:

| Case | Class | Lesion tier | Type-1 | Type-2 | Anti-PGL-1 | Bacilloscopy | Histology | Colour | Type |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | TT | 1 | 1 | 0 | 0 | 0 | 0 | erythematous 0.5, normochromic 0.5 | plaque |
| 2 | TT | 1 | 1 | 0 | 0.5 | 0.5 | 0 | erythematous | plaque |
| 3 | TT | 1 | 1 | 0 | 0 | 0 | 0 | erythematous | plaque 0.5, papule 0.5 |
| 4 | TT | 1 | 0 | 0 | 0 | 0 | 0.5 | erythematous | plaque |
| 5 | TT | 1 | 0 | 0 | 0 | 0 | 0 | erythematous | plaque |
| 6 | TT | 1 | 0 | 0 | 0 | 0 | 0 | erythematous 0.5, hyperchromic 0.5 | plaque |
| 7 | TT | 1 | 0 | 0 | 0 | 0 | 0 | erythematous | plaque |
| 8 | TT | 1 | 0 | 0 | 0 | 0 | 0 | erythematous | plaque |
| 9 | TT | 1 | 1 | 0 | 0.5 | 0 | 0 | erythematous 0.5, hyperchromic 0.5 | plaque 0.5, papule 0.5 |
| 10 | BB | 2 | 1 | 1 | 2 | 3 | 2.5 | erythematous 0.33, hypochromic 0.33, hyperchromic 0.33 | plaque |
| 11 | BB | 1.93 | 1 | 1 | 2 | 2.5 | 2.5 | erythematous 0.5, hyperchromic 0.5 | plaque 0.5, macule 0.5 |
| 12 | BB | 2.4 | 1 | 0 | 1.5 | 2.5 | 3.5 | erythematous 0.5, hyperchromic 0.5 | plaque 0.5, macule 0.5 |
| 13 | BB | 3 | 1 | 0 | 1 | 1 | 2.5 | erythematous | plaque |
| 14 | BB | 2 | 1 | 0 | 1 | 1 | 2.5 | erythematous 0.5, hypochromic 0.5 | plaque 0.5, macule 0.5 |
| 15 | BB | 2 | 0 | 0 | 2 | 2 | 0.5 | erythematous | plaque |
| 16 | BB | 2 | 1 | 0 | 3 | 3 | 2.5 | erythematous | plaque |
| 17 | BB | 2.27 | 1 | 0 | 2.5 | 3 | 3.5 | erythematous 0.5, hyperchromic 0.5 | plaque 0.5, papule 0.5 |
| 18 | LL | 3 | 0 | 1 | 4 | 6 | 5.5 | erythematous 0.5, hyperchromic 0.5 | plaque 0.33, nodule 0.33, diffuse infiltration 0.33 |
| 19 | LL | 3 | 0 | 1 | 3.5 | 5 | 5.5 | erythematous 0.5, hyperchromic 0.5 | plaque 0.25, papule 0.25, nodule 0.25, diffuse infiltration 0.25 |
| 20 | LL | 2.27 | 0 | 1 | 4 | 4.5 | 5.5 | hypochromic 0.5, hyperchromic 0.5 | papule 0.5, diffuse infiltration 0.5 |
| 21 | LL | 2.67 | 0 | 1 | 3.5 | 5 | 5.5 | erythematous 0.5, hyperchromic 0.5 | papule 0.33, nodule 0.33, diffuse infiltration 0.33 |
| 22 | LL | 3 | 0 | 1 | 4 | 6 | 5.5 | normochromic | diffuse infiltration |
| 23 | LL | 3 | 0 | 1 | 4 | 6 | 5.5 | erythematous 0.5, normochromic 0.5 | plaque 0.25, nodule 0.25, macule 0.25, diffuse infiltration 0.25 |
| 24 | LL | 3 | 0 | 0 | 4 | 6 | 5.5 | normochromic | diffuse infiltration |
| 25 | LL | 3 | 0 | 0 | 4 | 6 | 5.5 | erythematous | diffuse infiltration |
| 26 | LL | 3 | 0 | 1 | 3.5 | 5.5 | 4.5 | erythematous 0.5, hyperchromic 0.5 | papule 0.33, nodule 0.33, diffuse infiltration 0.33 |
