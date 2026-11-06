"""Response to each finding, and recovery of the WHO and Ridley-Jopling systems.

Input:  results/oof_positions.parquet, results/spectrum_position.json
Output: reports/figures/direction_sweeps.png              (fig:val:variable-behaviour)
        reports/figures/level_shifts.png                  (fig:val:categorical-behaviour)
        reports/figures/who_recovery.png                  (fig:val:who)
        reports/figures/position_by_class.png             (fig:val:who-kde)
        reports/tables/{recovery, class_confusion, direction_sweeps, level_shifts}.csv
        reports/numbers.json, keys ``position.*``

Every position here is out of fold: the record was positioned by the model
fitted on the other four folds of the shared split, on that model's
rescaled scale.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from leprosy_similarity import paths
from leprosy_similarity.reports.numbers import record_numbers
from leprosy_similarity.reports.style import (
    GROUP_NAMES,
    RJ_PALETTE,
    WHO_PALETTE,
    figure_style,
    finding_name,
    level_name,
    value_labels,
)
from leprosy_similarity.schema import CLASS_TARGETS, CLASSES, NUMERIC_FEATURES

INPUTS = (paths.OOF_POSITIONS, paths.SPECTRUM_POSITION)
OUTPUTS = (
    paths.FIG_DIRECTION_SWEEPS,
    paths.FIG_LEVEL_SHIFTS,
    paths.FIG_WHO_RECOVERY,
    paths.FIG_POSITION_BY_CLASS,
    paths.TABLE_RECOVERY,
    paths.TABLE_CLASS_CONFUSION,
    paths.TABLE_DIRECTION_SWEEPS,
    paths.TABLE_LEVEL_SHIFTS,
)


def plot_variable_behaviour(findings: dict[str, dict[str, Any]]) -> None:
    """Mean position against each numeric finding, panels ordered by impact."""
    by_impact = sorted(NUMERIC_FEATURES, key=lambda f: abs(findings[f]["delta"]), reverse=True)
    with figure_style():
        fig, axes = plt.subplots(3, 2, figsize=(11, 10))
        for ax, feature in zip(axes.ravel(), by_impact):
            sweep = findings[feature]
            x = list(range(len(sweep["values"])))
            mean, sd = np.array(sweep["mean"]), np.array(sweep["sd"])
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.18, color=sns.color_palette()[0], lw=0,
                            label=r"per-patient $\pm1\sigma$")
            sns.lineplot(x=x, y=mean, marker="o", ax=ax)
            ax.axhline(sweep["missing_mean"], color="0.4", ls="--", alpha=0.7,
                       label=f"missing = {sweep['missing_mean']:.2f}")
            for target in CLASS_TARGETS.values():
                ax.axhline(target, color="0.85", ls=":", alpha=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(value_labels(feature, sweep["values"]))
            ax.set_xlabel(finding_name(feature))
            ax.set_ylabel("mean position")
            ax.set_ylim(-0.02, 1.02)
            ax.margins(x=0.05)
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(paths.FIG_DIRECTION_SWEEPS, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot_categorical_behaviour(levels: pd.DataFrame) -> None:
    """Mean shift when each colour or morphology level is the only one present."""
    frame = pd.DataFrame(
        {
            "label": [level_name(c) for c in levels["column"]],
            "group": [GROUP_NAMES[v] for v in levels["variable"]],
            "delta": levels["mean_shift"],
            "sd": levels["sd_shift"],
        }
    )
    frame = frame.reindex(frame["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    with figure_style():
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.barplot(data=frame, y="label", x="delta", hue="group", dodge=False, palette="colorblind", ax=ax)
        ax.errorbar(frame["delta"], range(len(frame)), xerr=frame["sd"].abs(),
                    fmt="none", ecolor="0.3", elinewidth=1.0, capsize=2, zorder=5)
        ax.axvline(0.0, color="0.3", lw=1)
        ax.set_xlabel("Δ spectrum position when the category is present "
                      "(− toward tuberculoid, + toward lepromatous)")
        ax.set_ylabel("")
        ax.set_title("Categorical variable behaviour")
        ax.legend(title="", loc="lower right")
        fig.tight_layout()
        fig.savefig(paths.FIG_LEVEL_SHIFTS)
        plt.close(fig)


def plot_who(oof: pd.DataFrame, who: dict[str, Any]) -> None:
    """ROC of the position for multibacillary treatment, and the density by WHO class."""
    rated = oof[oof["who_class"].notna()]
    with figure_style():
        fig, axes = plt.subplot_mosaic([["roc", "kde"]], figsize=(12, 4.8))
        roc = axes["roc"]
        roc.plot(who["roc_fpr"], who["roc_tpr"], label=f"AUC = {who['auc']:.3f}")
        roc.plot([0, 1], [0, 1], "k:", alpha=0.4)
        roc.scatter([who["youden_fpr"]], [who["youden_tpr"]], color="black", zorder=5,
                    label=f"Youden thr = {who['youden_threshold']:.3f}")
        roc.set_xlabel("1 − specificity")
        roc.set_ylabel("sensitivity")
        roc.set_title("ROC: position → multibacillary")
        roc.legend(loc="lower right")

        kde = axes["kde"]
        sns.kdeplot(data=rated, x="position", hue="who_class", hue_order=["PB", "MB"], fill=True,
                    common_norm=False, alpha=0.4, palette=WHO_PALETTE, clip=(0, 1), ax=kde)
        for target in CLASS_TARGETS.values():
            kde.axvline(target, color="0.6", ls=":", alpha=0.6)
        if kde.get_legend() is not None:
            kde.get_legend().set_title("WHO class")
        kde.set_xlim(0, 1)
        kde.set_xlabel("out-of-fold spectrum position")
        kde.set_title(f"Position density by WHO class (all records, n={len(rated)})")
        fig.tight_layout()
        fig.savefig(paths.FIG_WHO_RECOVERY)
        plt.close(fig)


def plot_position_by_class(oof: pd.DataFrame) -> None:
    """Density of the position by Ridley-Jopling class, class means dotted."""
    with figure_style():
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.kdeplot(data=oof, x="position", hue="class", hue_order=list(CLASSES), fill=True,
                    common_norm=False, alpha=0.35, palette=RJ_PALETTE, clip=(0, 1), ax=ax)
        for c in CLASSES:
            ax.axvline(oof.loc[oof["class"] == c, "position"].mean(), color=RJ_PALETTE[c], ls=":", alpha=0.9, lw=1.8)
        if ax.get_legend() is not None:
            ax.get_legend().set_title("R&J class")
        ax.set_xlim(0, 1)
        ax.set_xlabel("out-of-fold spectrum position")
        ax.set_title(f"Position density by Ridley–Jopling class (all records, n={len(oof)})")
        fig.tight_layout()
        fig.savefig(paths.FIG_POSITION_BY_CLASS)
        plt.close(fig)


def recovery_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Every scalar of the WHO recovery and of the Ridley-Jopling ordering."""
    recovery, order = summary["recovery"], summary["ordering"]
    rows = [("who", k, v) for k, v in recovery["who"].items() if not isinstance(v, list)]
    rows += [("within_class", k, v) for k, v in recovery["within_class"].items()]
    for fill in ("zero", "median"):
        rows += [(f"bacilloscopy_{fill}_fill", k, v) for k, v in recovery["bacilloscopy_increment"][fill].items()]
    for key, value in order.items():
        if isinstance(value, dict):
            rows += [("ordering", f"{key}.{c}", v) for c, v in value.items()]
        elif not isinstance(value, list):
            rows.append(("ordering", key, value))
    return pd.DataFrame(rows, columns=["part", "metric", "value"])


def confusion_table(order: dict[str, Any]) -> pd.DataFrame:
    """Recorded class (rows) against the class of the nearest class target (columns)."""
    labels = order["confusion_labels"]
    return pd.DataFrame(order["confusion"], index=pd.Index(labels, name="recorded"), columns=labels)


def sweep_table(findings: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean position and spread at each swept value of each finding."""
    rows = []
    for sweep in findings:
        for value, mean, sd in zip(sweep["values"], sweep["mean"], sweep["sd"]):
            rows.append({"feature": sweep["feature"], "value": value, "mean_position": mean, "sd": sd})
        rows.append({"feature": sweep["feature"], "value": np.nan, "mean_position": sweep["missing_mean"],
                     "sd": sweep["missing_sd"]})
    return pd.DataFrame(rows)


def largest_confusion(order: dict[str, Any]) -> dict[str, Any]:
    """The largest off-diagonal cell of the confusion matrix."""
    matrix = np.array(order["confusion"])
    np.fill_diagonal(matrix, -1)
    i, j = np.unravel_index(int(matrix.argmax()), matrix.shape)
    labels = order["confusion_labels"]
    return {"recorded": labels[i], "read_as": labels[j], "count": int(matrix[i, j])}


def numbers(summary: dict[str, Any], oof: pd.DataFrame) -> dict[str, Any]:
    """Every number the thesis quotes from this area."""
    who = {k: v for k, v in summary["recovery"]["who"].items() if not isinstance(v, list)}
    within = summary["recovery"]["within_class"]
    increment = summary["recovery"]["bacilloscopy_increment"]
    order = summary["ordering"]
    sweeps = summary["sweeps"]
    findings = {s["feature"]: s for s in sweeps["findings"]}
    out: dict[str, Any] = {
        "position.n_folds": summary["n_folds"],
        "position.n_records": len(oof),
        "position.n_clipped_floor": int((oof["position"] == 0).sum()),
        "position.n_clipped_ceiling": int((oof["position"] == 1).sum()),
        "position.who": who,
        "position.who.within_class": within,
        "position.who.bacilloscopy_increment": increment,
        "position.order.spearman": order["spearman"],
        "position.order.quadratic_kappa": order["quadratic_kappa"],
        "position.order.mae": order["mae"],
        "position.order.mae_class_balanced": order["mae_class_balanced"],
        "position.order.mae_poles": order["mae_poles"],
        "position.order.mae_middle": order["mae_middle"],
        "position.order.per_class_mae": order["per_class_mae"],
        "position.order.per_class_mean": order["per_class_mean"],
        "position.order.per_class_median": order["per_class_median"],
        "position.order.steps_between_means": order["steps_between_means"],
        "position.order.silhouette": order["silhouette"],
        "position.order.per_class_silhouette": order["per_class_silhouette"],
        "position.order.n_exact": order["n_exact"],
        "position.order.exact_share": order["n_exact"] / order["n"],
        "position.order.n_within_one_step": order["n_within_one_step"],
        "position.order.within_one_step_share": order["n_within_one_step"] / order["n"],
        "position.order.largest_confusion": largest_confusion(order),
        "position.sweep.baseline_mean_position": sweeps["baseline_mean_position"],
        "position.sweep.violations": sweeps["violations"],
        "position.sweep.max_level_shift": max(abs(level["mean_shift"]) for level in sweeps["levels"]),
        "position.sweep.level_shift": {level["column"]: level["mean_shift"] for level in sweeps["levels"]},
    }
    for feature in NUMERIC_FEATURES:
        sweep = findings[feature]
        out[f"position.sweep.{feature}"] = {
            "delta": sweep["delta"],
            "values": sweep["values"],
            "mean": sweep["mean"],
            "spearman": sweep["spearman"],
            "monotonicity": sweep["monotonicity"],
            "direction": sweep["direction"],
            "missing_mean": sweep["missing_mean"],
        }
    return out


def run() -> None:
    summary = json.loads(paths.SPECTRUM_POSITION.read_text(encoding="utf-8"))
    oof = pd.read_parquet(paths.OOF_POSITIONS)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    paths.TABLES.mkdir(parents=True, exist_ok=True)

    findings = {s["feature"]: s for s in summary["sweeps"]["findings"]}
    plot_variable_behaviour(findings)
    plot_categorical_behaviour(pd.DataFrame(summary["sweeps"]["levels"]))
    plot_who(oof, summary["recovery"]["who"])
    plot_position_by_class(oof)

    recovery_table(summary).to_csv(paths.TABLE_RECOVERY, index=False)
    confusion_table(summary["ordering"]).to_csv(paths.TABLE_CLASS_CONFUSION)
    sweep_table(summary["sweeps"]["findings"]).to_csv(paths.TABLE_DIRECTION_SWEEPS, index=False)
    pd.DataFrame(summary["sweeps"]["levels"]).to_csv(paths.TABLE_LEVEL_SHIFTS, index=False)
    record_numbers(numbers(summary, oof))
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
