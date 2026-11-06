# Raw patient data (not in version control)

This folder holds the raw cohort spreadsheet. The data are individual patient
records, so they are never committed: everything in `data/raw/` except this
file is gitignored. The cohort is available on request from the authors, under
the study's ethics approval.

## Expected file

`rafa_total.xlsx`: one workbook, one row per patient. Every sheet is read and
the sheets are stacked. A row is kept only if `Forma Clínica` holds a
Ridley-Jopling class (`TT`, `BT`, `BB`, `BL` or `LL`). With the study
workbook, 1,451 rows give 1,030 labelled records.

## Columns read

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

A blank or unreadable cell becomes a missing value.

The spreadsheet stacks the centres' databases in order, so a row's position
gives its centre: rows 1–372 are Bauru, rows 373–1,042 Fortaleza/Manaus and
rows 1,043–1,451 Goiânia (`SITE_BLOCKS` in `cohort.py`). The Goiânia sample is
not part of this study and is dropped while reading. The genotype panels are
not used to tell the centres apart, since Goiânia typed some of the same genes
as Bauru; no genotype value enters the analysis.

## Then

```
python main.py preprocess
```

writes `data/processed/` (also gitignored): `raw_cohort.parquet` (1,030),
`cohort.parquet` (the 916-record analytic cohort), `reference_cases.parquet`,
`schema.json`, `exclusion_audit.csv`, `exclusion_attrition.csv`, `folds.csv`
and the variable-screen tables.
