"""Per-feature attribution: the beeswarm, the attribution table, observed against imputed.

Input:  results/attribution.npz, results/imputation_reconstruction.csv
Output: reports/figures/attribution.png  (fig:val:attribution)
        reports/tables/attribution.csv           (tab:val:attribution)
        reports/tables/observed_imputed.csv      (tab:val:observed-imputed)
        reports/numbers.json, keys ``attribution.*``

Each record's contributions come from the fold model that did not train on
it and are on that model's rescaled scale, so they sum to the record's
rescaled position minus the rescaled base.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from leprosy_similarity import paths
from leprosy_similarity.reports.numbers import record_numbers
from leprosy_similarity.reports.style import feature_name, figure_style, finding_name
from leprosy_similarity.schema import FEATURES, NUMERIC_FEATURES, categorical_blocks

INPUTS = (paths.ATTRIBUTION, paths.IMPUTATION_RECONSTRUCTION)
OUTPUTS = (paths.FIG_ATTRIBUTION, paths.TABLE_ATTRIBUTION, paths.TABLE_OBSERVED_IMPUTED)


def value_contribution_correlation(values: np.ndarray, contributions: np.ndarray) -> float:
    """Correlation of a feature's (filled) value with its contribution; 0 if either is constant."""
    if values.std() > 1e-9 and contributions.std() > 1e-9:
        return float(np.corrcoef(values, contributions)[0, 1])
    return 0.0


def attribution_table(contributions: np.ndarray, filled: np.ndarray) -> pd.DataFrame:
    """Mean |contribution| of each numeric finding, each categorical level and each block.

    A block's contribution is the absolute value of the sum over its levels,
    the total pull of the variable on the record.
    """
    rows = []
    for j, feature in enumerate(FEATURES):
        numeric = feature in NUMERIC_FEATURES
        rows.append(
            {
                "kind": "numeric" if numeric else "level",
                "feature": feature,
                "name": feature_name(feature),
                "mean_abs_contribution": float(np.abs(contributions[:, j]).mean()),
                "value_contribution_corr": (
                    value_contribution_correlation(filled[:, j], contributions[:, j]) if numeric else np.nan
                ),
            }
        )
    for variable, columns in categorical_blocks().items():
        index = [FEATURES.index(c) for c in columns]
        rows.append(
            {
                "kind": "block",
                "feature": variable,
                "name": finding_name(variable),
                "mean_abs_contribution": float(np.abs(contributions[:, index].sum(axis=1)).mean()),
                "value_contribution_corr": np.nan,
            }
        )
    table = pd.DataFrame(rows)
    order = {"numeric": 0, "block": 1, "level": 2}
    return table.sort_values(
        ["kind", "mean_abs_contribution"], key=lambda s: s.map(order) if s.name == "kind" else -s
    ).reset_index(drop=True)


def observed_imputed_table(
    contributions: np.ndarray, observed: np.ndarray, reconstruction: pd.DataFrame
) -> pd.DataFrame:
    """Mean |contribution| of each numeric finding where it was measured and where it was imputed."""
    rows = []
    for feature in NUMERIC_FEATURES:
        j = FEATURES.index(feature)
        seen, size = observed[:, j], np.abs(contributions[:, j])
        rows.append(
            {
                "feature": feature,
                "name": feature_name(feature),
                "observed_share": float(seen.mean()),
                "contribution_observed": float(size[seen].mean()) if seen.any() else np.nan,
                "contribution_imputed": float(size[~seen].mean()) if (~seen).any() else np.nan,
                "contribution_overall": float(size.mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["ratio_observed_to_imputed"] = table["contribution_observed"] / table["contribution_imputed"]
    rebuilt = reconstruction.set_index("feature")[["mae", "r2"]].add_prefix("reconstruction_")
    table = table.join(rebuilt, on="feature")
    return table.sort_values("contribution_observed", ascending=False).reset_index(drop=True)


def rank01(values: np.ndarray) -> np.ndarray:
    """Values rank-normalised to [0, 1] for colour, robust to skew and to binary findings."""
    if len(values) < 2:
        return np.full(len(values), 0.5)
    return np.argsort(np.argsort(values)).astype(float) / (len(values) - 1)


def swarm_offsets(x: np.ndarray, width: float = 0.42, n_bins: int = 40) -> np.ndarray:
    """Vertical offsets that spread the points of a beeswarm row by local density."""
    x = np.asarray(x, float)
    y = np.zeros(len(x))
    low, high = np.nanmin(x), np.nanmax(x)
    if high <= low:
        return y
    edges = np.linspace(low, high, n_bins + 1)
    bins = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        members = np.where(bins == b)[0]
        m = len(members)
        if m < 2:
            continue
        members = members[np.argsort(x[members])]
        half = min(width, 0.05 * (m - 1))
        y[members] = (np.arange(m) - (m - 1) / 2.0) / (m - 1) * 2 * half
    return y


def plot_beeswarm(contributions: np.ndarray, filled: np.ndarray, order: list[int]) -> None:
    """One row per numeric finding, one point per record, coloured by the record's value."""
    with figure_style():
        fig, ax = plt.subplots(figsize=(10, 5.4))
        points = None
        for row, j in enumerate(order):
            level = len(order) - 1 - row
            x = contributions[:, j]
            points = ax.scatter(x, level + swarm_offsets(x), c=rank01(filled[:, j]), cmap=plt.cm.coolwarm,
                                s=11, alpha=0.85, linewidths=0, vmin=0, vmax=1)
        ax.axvline(0, color="0.5", lw=0.9)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_name(FEATURES[j]) for j in order][::-1])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("contribution to the spectrum position "
                      "($-$ toward tuberculoid    |    $+$ toward lepromatous)")
        ax.set_title(f"Attribution of each finding to the spectrum position (n={len(contributions)})")
        ax.grid(axis="y", visible=False)
        bar = fig.colorbar(points, ax=ax, pad=0.015, fraction=0.045)
        bar.set_label("feature value")
        bar.set_ticks([0.02, 0.98])
        bar.set_ticklabels(["low", "high"])
        fig.tight_layout()
        fig.savefig(paths.FIG_ATTRIBUTION, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def numbers(store: Any, table: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    """Every number the thesis quotes from the attribution."""
    by_feature = table.set_index("feature")
    levels = by_feature[by_feature["kind"] == "level"]["mean_abs_contribution"]
    numeric = by_feature[by_feature["kind"] == "numeric"]
    return {
        "attribution.n_records": len(store["record_id"]),
        "attribution.n_reference_cases": len(store["reference_classes"]),
        "attribution.efficiency_gap_max": float(store["efficiency_gap"].max()),
        "attribution.efficiency_gap_raw_max": float(store["efficiency_gap_raw"].max()),
        "attribution.base_raw": float(store["base_raw"]),
        "attribution.base_rescaled_by_fold": sorted({round(float(b), 12) for b in store["base"]}),
        "attribution.n_clipped": int(((store["position"] == 0) | (store["position"] == 1)).sum()),
        "attribution.numeric_order": numeric.index.tolist(),
        "attribution.mean_abs_contribution": by_feature["mean_abs_contribution"].to_dict(),
        "attribution.value_contribution_corr": numeric["value_contribution_corr"].to_dict(),
        "attribution.level_order": levels.sort_values(ascending=False).index.tolist(),
        "attribution.observed_imputed": split.set_index("feature").drop(columns="name").to_dict(orient="index"),
    }


def run() -> None:
    store = np.load(paths.ATTRIBUTION, allow_pickle=False)
    contributions, filled, observed = store["contributions"], store["filled"], store["observed"]
    if list(store["features"]) != list(FEATURES):
        raise ValueError("the attribution was computed on a different schema")
    reconstruction = pd.read_csv(paths.IMPUTATION_RECONSTRUCTION)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    paths.TABLES.mkdir(parents=True, exist_ok=True)

    table = attribution_table(contributions, filled)
    split = observed_imputed_table(contributions, observed, reconstruction)
    table.to_csv(paths.TABLE_ATTRIBUTION, index=False)
    split.to_csv(paths.TABLE_OBSERVED_IMPUTED, index=False)

    importance = np.abs(contributions).mean(axis=0)
    numeric = [FEATURES.index(f) for f in NUMERIC_FEATURES]
    order = [numeric[i] for i in np.argsort(-importance[numeric])]
    plot_beeswarm(contributions, filled, order)
    record_numbers(numbers(store, table, split))
    print(table.to_string(index=False))
    print(split.to_string(index=False))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
