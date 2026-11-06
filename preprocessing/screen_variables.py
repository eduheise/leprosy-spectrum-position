"""Variable screen: which candidate findings carry signal about the spectrum.

A variable earns a place in the schema if a tree ensemble trained on the
Ridley-Jopling label reads measurable signal from it. A model-based screen is
used rather than a correlation with the class target because a monotone
statistic scores zero for anything peaked in the middle: the type-1 reaction
is absent at both poles and peaks at mid-borderline, so its rank correlation
with the pole-to-pole axis is near zero although it marks the unstable middle.

Importance is grouped permutation importance on held-out folds: the one-hot
levels of a categorical are permuted together, and importance is the rise in
held-out multiclass log loss when the variable is scrambled. The backbone is a
random forest, not the selected configuration, because the screen runs before
the search and must not depend on a configuration tuned on its own output.
Missing values are filled with the deterministic kNN imputer the search also
offers. A variable is retained when its mean rise exceeds its fold-to-fold
standard deviation: a variable carrying nothing scatters around zero.

Input:  data/processed/raw_cohort.parquet, data/processed/exclusion_audit.csv,
        data/processed/cohort.parquet
Output: data/processed/variable_screen.csv
        data/processed/variable_screen_reaction_prevalence.csv
        data/processed/variable_screen_shape.csv
        data/processed/variable_screen_baseline.json (held-out log loss before any permutation)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from leprosy_similarity import paths
from leprosy_similarity.config import VariableScreenConfig, load_config
from leprosy_similarity.imputation import DeterministicKNNImputer
from leprosy_similarity.schema import CLASSES, FEATURES, SCREENED_OUT_FEATURES

INPUTS = (paths.RAW_COHORT, paths.EXCLUSION_AUDIT, paths.COHORT)
OUTPUTS = (
    paths.VARIABLE_SCREEN,
    paths.VARIABLE_SCREEN_REACTION_PREVALENCE,
    paths.VARIABLE_SCREEN_SHAPE,
    paths.VARIABLE_SCREEN_BASELINE,
)

CANDIDATES: tuple[str, ...] = FEATURES + SCREENED_OUT_FEATURES


def _levels(variable: str) -> list[str]:
    return [c for c in CANDIDATES if c.startswith(f"{variable}__")]


# Presentation name of each candidate variable and the columns it owns.
GROUPS: dict[str, list[str]] = {
    "Bacilloscopy index": ["BACILOSCOP_N"],
    "Lesion count": ["NU_LESOES"],
    "Anti-PGL-1 titre": ["PGL1"],
    "Histological index": ["HISTOLOG"],
    "Type-1 reaction": ["REAC_TIPO_1"],
    "Type-2 reaction": ["REAC_TIPO_2"],
    "Lesion colour": _levels("lesion_color"),
    "Lesion morphology": _levels("lesion_element"),
    "Sensibility": _levels("sensibility"),
}


def screening_cohort() -> pd.DataFrame:
    """The analytic cohort with the screened-out variables still attached.

    Rebuilt from the raw cohort and the exclusion audit, and checked against
    the persisted cohort so the two cannot drift apart.
    """
    raw = pd.read_parquet(paths.RAW_COHORT)
    audit = pd.read_csv(paths.EXCLUSION_AUDIT)
    kept = audit.loc[~audit["deleted"], "record_id"]
    screening = raw[raw["record_id"].isin(kept)].reset_index(drop=True)
    cohort = pd.read_parquet(paths.COHORT)
    if not screening[["record_id", *FEATURES]].equals(cohort[["record_id", *FEATURES]]):
        raise SystemExit("the screening cohort disagrees with data/processed/cohort.parquet")
    return screening


def permutation_screen(
    X: np.ndarray, y: np.ndarray, columns: list[str], settings: VariableScreenConfig
) -> tuple[dict[str, list[float]], float]:
    """Rise in held-out log loss per variable and fold, and the mean baseline."""
    rng = np.random.default_rng(settings.seed)
    rises: dict[str, list[float]] = {group: [] for group in GROUPS}
    baselines: list[float] = []
    one_hot = [i for i, c in enumerate(columns) if "__" in c]
    splitter = StratifiedKFold(settings.n_folds, shuffle=True, random_state=settings.seed)

    def impute(imputer: DeterministicKNNImputer, M: np.ndarray) -> np.ndarray:
        filled = imputer.transform(pd.DataFrame(M)).to_numpy().copy()
        filled[:, one_hot] = (filled[:, one_hot] >= 0.5).astype(float)
        return filled

    for train, test in splitter.split(X, y):
        imputer = DeterministicKNNImputer(k=settings.knn_neighbours).fit(pd.DataFrame(X[train]))
        x_train, x_test = impute(imputer, X[train]), impute(imputer, X[test])
        model = RandomForestClassifier(
            n_estimators=settings.n_trees,
            min_samples_leaf=settings.min_samples_leaf,
            max_features=settings.max_features,
            random_state=settings.seed,
            n_jobs=-1,
        ).fit(x_train, y[train])
        # Parallel prediction adds the trees' probabilities in whatever order
        # the threads finish, which moves the last bits of the log loss from
        # run to run. Predicting on one thread fixes the order; the fit, and
        # so every tree, is unchanged.
        model.set_params(n_jobs=1)

        def loss(x: np.ndarray) -> float:
            # predict_proba columns follow model.classes_, so the labels must too.
            return float(log_loss(y[test], model.predict_proba(x), labels=model.classes_))

        base = loss(x_test)
        baselines.append(base)
        for group, members in GROUPS.items():
            idx = [columns.index(c) for c in members]
            repeats = []
            for _ in range(settings.n_repeats):
                permuted = x_test.copy()
                order = rng.permutation(len(permuted))
                permuted[:, idx] = permuted[order][:, idx]
                repeats.append(loss(permuted) - base)
            rises[group].append(float(np.mean(repeats)))
    return rises, float(np.mean(baselines))


def screen_table(rises: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        mean = float(np.mean(rises[group]))
        sd = float(np.std(rises[group], ddof=1))
        rows.append(
            {"variable": group, "delta_logloss": round(mean, 4), "sd": round(sd, 4), "retained": bool(mean > sd)}
        )
    return pd.DataFrame(rows).sort_values("delta_logloss", ascending=False).reset_index(drop=True)


def reaction_shape(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Why the screen must be model-based: the reactions' prevalence by class.

    Contrasts each reaction's rank association with the class target against
    its association with distance from the centre of the spectrum.
    """
    prevalence = pd.DataFrame(
        [
            {
                "class": c,
                "n": int((cohort["class"] == c).sum()),
                "type1_pct": round(float(cohort.loc[cohort["class"] == c, "REAC_TIPO_1"].mean()) * 100, 1),
                "type2_pct": round(float(cohort.loc[cohort["class"] == c, "REAC_TIPO_2"].mean()) * 100, 1),
            }
            for c in CLASSES
        ]
    )
    target = cohort["target"].to_numpy(float)
    centrality = np.abs(target - 0.5)
    rows = []
    for column, name in (("REAC_TIPO_1", "Type-1 reaction"), ("REAC_TIPO_2", "Type-2 reaction")):
        values = cohort[column].to_numpy(float)
        ok = np.isfinite(values)
        rho_pole = float(spearmanr(values[ok], target[ok]).statistic)
        rho_mid = float(spearmanr(values[ok], centrality[ok]).statistic)
        rows.append(
            {
                "variable": name,
                "rho_vs_position": round(rho_pole, 3),
                "r2_vs_position_pct": round(rho_pole**2 * 100, 1),
                "rho_vs_centrality": round(rho_mid, 3),
                "r2_vs_centrality_pct": round(rho_mid**2 * 100, 1),
            }
        )
    return prevalence, pd.DataFrame(rows)


def run() -> None:
    settings = load_config().variable_screen
    cohort = screening_cohort()
    columns = list(CANDIDATES)
    rises, baseline = permutation_screen(
        cohort[columns].to_numpy(float), cohort["class"].to_numpy(), columns, settings
    )
    table = screen_table(rises)
    table.to_csv(paths.VARIABLE_SCREEN, index=False)
    prevalence, shape = reaction_shape(cohort)
    prevalence.to_csv(paths.VARIABLE_SCREEN_REACTION_PREVALENCE, index=False)
    shape.to_csv(paths.VARIABLE_SCREEN_SHAPE, index=False)
    baseline_summary = {"n_records": len(cohort), "n_candidates": len(columns), "baseline_logloss": baseline}
    paths.VARIABLE_SCREEN_BASELINE.write_text(json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8")

    print(f"screening cohort: {len(cohort)} records, {len(columns)} candidate columns")
    print(f"baseline held-out log loss: {baseline:.4f}")
    print(table.to_string(index=False))
    print(prevalence.to_string(index=False))
    print(shape.to_string(index=False))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
