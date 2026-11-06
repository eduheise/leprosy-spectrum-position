"""Confidence under missingness: the imputer table, the band against the drift, k(p).

Input:  results/confidence.parquet, results/confidence_summary.json
Output: reports/tables/confidence_calibration.csv           (tab:val:confidence-calibration)
        reports/figures/confidence_band.png                 (fig:val:confidence-coverage)
        reports/figures/confidence_calibration_curve.png    (fig:val:coverage-target)
        reports/figures/confidence_coverage_heatmap.png     (fig:val:missingness)
        reports/numbers.json, keys ``confidence.*``

Every pair was scored by the fold model that held its record out. The
selected imputer and its multipliers are read from the summary the
experiment computed; nothing is refitted here.
"""
from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator

from leprosy_similarity import paths
from leprosy_similarity.confidence import (
    calibration_curve,
    case_split,
    coverage_by_size,
    fixed_band_by_size,
    held_out_ratio,
    quantile_levels,
)
from leprosy_similarity.config import ConfidenceConfig, load_config
from leprosy_similarity.reports.numbers import record_numbers
from leprosy_similarity.reports.style import figure_style

INPUTS = (paths.CONFIDENCE, paths.CONFIDENCE_SUMMARY, paths.CONFIDENCE_SINGLE_FINDING)
OUTPUTS = (
    paths.TABLE_CONFIDENCE_CALIBRATION,
    paths.FIG_CONFIDENCE_BAND,
    paths.FIG_CONFIDENCE_CALIBRATION_CURVE,
    paths.FIG_CONFIDENCE_COVERAGE_HEATMAP,
)

ARM_LABELS = {
    "knn_donor": "k-nearest-neighbour donors",
    "bayesian_ridge": "Bayesian-ridge MICE",
    "forest": "MICE-Forest",
}
# Held-out quantiles drawn as points on the calibration curve.
CURVE_POINTS = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
# Requested coverages marked with guide lines on the calibration curve.
CURVE_MARKS = (0.95, 0.99)


def calibration_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Delivered coverage (%) and mean band width per imputer and requested level."""
    table = pd.DataFrame(summary["sharpness"])
    wide = table.pivot(index="arm", columns="level", values=["coverage", "mean_width"])
    rows = []
    for arm in wide.index:
        row: dict[str, Any] = {"arm": arm, "method": ARM_LABELS[arm], "selected": arm == summary["selected_arm"]}
        for level in sorted(table["level"].unique()):
            row[f"coverage_{level:g}"] = 100.0 * wide.loc[arm, ("coverage", level)]
            row[f"width_{level:g}"] = wide.loc[arm, ("mean_width", level)]
        rows.append(row)
    order = table[table["level"] == table["level"].max()].sort_values("mean_width")["arm"].tolist()
    return pd.DataFrame(rows).set_index("arm").loc[order].reset_index()


def plot_coverage(by_size: pd.DataFrame, overall: float, multiplier: float) -> None:
    """The actual drift against the fixed band, and the band's coverage, by mask size."""
    with figure_style():
        fig, ax = plt.subplots(figsize=(9, 5.2))
        sizes = by_size["mask_size"]
        ax.fill_between(sizes, 0, by_size["drift_p90"], color="0.7", alpha=0.4,
                        label="actual drift (p90)")
        ax.plot(sizes, by_size["drift_median"], color="0.35", marker="o", label="actual drift (median)")
        ax.plot(sizes, by_size["half_width_mean"], color="tab:green", lw=2.4, marker="s",
                label=f"confidence safe range (±{multiplier:g}$\\sigma$)")
        ax.set_xlabel("number of hidden tests")
        ax.set_ylabel("position drift from the full-data answer")
        ax.set_ylim(bottom=0)
        ax.set_xticks(sizes)
        ax.yaxis.set_major_locator(MultipleLocator(0.025))
        twin = ax.twinx()
        twin.plot(sizes, 100 * by_size["coverage"], color="tab:blue", ls="--", marker="^", label="coverage")
        for x, value in zip(sizes, 100 * by_size["coverage"], strict=True):
            twin.annotate(f"{value:.1f}%", (x, value), textcoords="offset points", xytext=(0, 7),
                          ha="center", fontsize=8, color="tab:blue")
        twin.set_ylabel("coverage: % of drifts inside the range")
        twin.set_ylim(0, 108)
        twin.yaxis.set_major_locator(MultipleLocator(10))
        twin.axhline(100 * overall, color="tab:blue", ls=":", alpha=0.5)
        twin.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        ax.legend(handles + twin_handles, labels + twin_labels, loc="lower left", bbox_to_anchor=(0.02, 0.18),
                  framealpha=0.92)
        ax.set_title(f"Confidence brackets the missing-value volatility ({overall:.0%} coverage overall)")
        fig.tight_layout()
        fig.savefig(paths.FIG_CONFIDENCE_BAND)
        plt.close(fig)


def plot_curve(pairs: pd.DataFrame, arm: str, held: np.ndarray, settings: ConfidenceConfig, zero_share: float) -> None:
    """The calibration curve k(p), with the held-out empirical quantiles on it."""
    curve = calibration_curve(pairs, arm, held, settings.curve_grid)
    test = held_out_ratio(pairs, arm, held)
    with figure_style():
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        floor = 100.0 * zero_share
        ax.axvspan(0, floor, color="0.85", alpha=0.5)
        ax.text(floor / 2, float(curve(0.5)) * 1.02, "floor\n(zero-drift\nrecords)", fontsize=7.5, color="0.4",
                ha="center", va="bottom")
        requested = np.linspace(max(0.02, zero_share), settings.curve_grid.high, 300)
        ax.plot(100 * requested, curve(requested), "-", color="tab:red", lw=2.2,
                label="calibrated knob $k(p)$ (measured quantile)")
        ax.plot(100 * CURVE_POINTS, np.quantile(test, CURVE_POINTS), "o", color="0.35", ms=5, zorder=5,
                label="held-out empirical quantile")
        for p in CURVE_MARKS:
            k = float(curve(p))
            ax.plot([100 * p, 100 * p, 0], [0, k, k], ls=":", color="0.55", lw=0.9)
            ax.annotate(f"$k({p:g})\\approx{k:.1f}$", xy=(100 * p, k), xytext=(100 * p - 22, k + 0.25),
                        fontsize=8, color="0.25")
        ax.set_xlabel("requested coverage $p$ (%)")
        ax.set_ylabel(r"band multiplier $k$  (half-width $=k\,\sigma$)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, float(curve(0.995)) * 1.05)
        ax.set_title("The confidence band as a calibrated knob")
        ax.legend(loc="upper left", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(paths.FIG_CONFIDENCE_CALIBRATION_CURVE, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def plot_heatmap(sizes: np.ndarray, requested: np.ndarray, coverage: np.ndarray) -> None:
    """Held-out coverage of the calibrated band per requested level and mask size."""
    deviation = coverage - requested[:, None] * 100.0
    low = float(np.floor(coverage.min() / 10.0) * 10.0)
    extent = [sizes.min() - 0.5, sizes.max() + 0.5, requested.min() * 100, requested.max() * 100]
    red_white_green = LinearSegmentedColormap.from_list("rwg", ["#b2182b", "white", "#1a9850"])
    with figure_style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.0), sharey=True)
        image = left.imshow(coverage, origin="lower", extent=extent, aspect="auto", cmap="turbo",
                            interpolation="nearest", vmin=low, vmax=100)
        left.set_title("Achieved coverage")
        fig.colorbar(image, ax=left, fraction=0.046, pad=0.02, label="achieved coverage (%)")
        image = right.imshow(deviation, origin="lower", extent=extent, aspect="auto", cmap=red_white_green,
                             norm=Normalize(-25, 25), interpolation="nearest")
        right.set_title("Observed $-$ requested: green over-covers, red under-covers")
        fig.colorbar(image, ax=right, fraction=0.046, pad=0.02, extend="both", label="observed $-$ requested (pts)")
        for ax in (left, right):
            ax.set_xlabel("number of hidden tests")
            ax.grid(True, color=("0.5" if ax is right else "white"), alpha=0.3, lw=0.6)
            ax.set_axisbelow(False)
            ax.set_xlim(sizes.min() - 0.5, sizes.max() + 0.5)
            ax.set_xticks(sizes)
        left.set_ylabel("requested coverage band (%)")
        fig.tight_layout()
        fig.savefig(paths.FIG_CONFIDENCE_COVERAGE_HEATMAP, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def numbers(summary: dict[str, Any], pairs: pd.DataFrame, table: pd.DataFrame) -> dict[str, Any]:
    """Every confidence number the thesis quotes, under ``confidence.*``."""
    out: dict[str, Any] = {
        "confidence.n_records": summary["n_records"],
        "confidence.n_pairs": summary["n_pairs"],
        "confidence.held_out_records": summary["split"]["held_out_records"],
        "confidence.held_out_pairs": summary["split"]["held_out_pairs"],
        "confidence.calibration_pairs": summary["split"]["calibration_pairs"],
        "confidence.calibration_pairs_nonzero_sigma": summary["split"]["calibration_pairs_nonzero_sigma"],
        "confidence.selected_arm": summary["selected_arm"],
        "confidence.mean_confidence_poles": summary["mean_confidence"]["poles"],
        "confidence.mean_confidence_middle": summary["mean_confidence"]["middle"],
        "confidence.fixed_band_multiplier": summary["fixed_band"]["multiplier"],
        "confidence.fixed_band_coverage": summary["fixed_band"]["coverage"],
        "confidence.zero_drift_pairs": summary["zero_drift"]["n"],
        "confidence.zero_drift_share": summary["zero_drift"]["share"],
    }
    for size, n in pairs["mask_size"].value_counts().sort_index().items():
        out[f"confidence.pairs_by_size.{size}"] = int(n)
    for row in summary["fixed_band"]["by_size"]:
        size = int(row["mask_size"])
        for name in ("coverage", "drift_median", "drift_p90", "half_width_mean", "zero_drift"):
            out[f"confidence.fixed_band.{size}.{name}"] = row[name]
    for level, values in summary["multipliers"].items():
        out[f"confidence.k.{level}"] = values["k"]
        out[f"confidence.k_held_out_coverage.{level}"] = values["held_out_coverage"]
    for row in summary["sharpness"]:
        key = f"confidence.sharpness.{row['arm']}.{row['level']:g}"
        out[f"{key}.coverage"] = row["coverage"]
        out[f"{key}.mean_width"] = row["mean_width"]
    widths = pd.DataFrame(summary["sharpness"]).pivot(index="arm", columns="level", values="mean_width")
    selected = summary["selected_arm"]
    for level in widths.columns:
        others = widths[level].drop(selected).sort_values()
        out[f"confidence.runner_up.{level:g}"] = others.index[0]
        out[f"confidence.runner_up_extra_width.{level:g}"] = others.iloc[0] / widths.loc[selected, level] - 1.0
    for arm, rho in summary["rank_agreement"].items():
        out[f"confidence.rank_agreement.{arm}"] = rho
    return out


def run() -> None:
    settings = load_config().confidence
    pairs = pd.read_parquet(paths.CONFIDENCE)
    summary = json.loads(paths.CONFIDENCE_SUMMARY.read_text(encoding="utf-8"))
    arm = summary["selected_arm"]
    held = case_split(pairs["case"].to_numpy(), settings.calibration_split_seed)

    paths.TABLES.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    table = calibration_table(summary)
    table.to_csv(paths.TABLE_CONFIDENCE_CALIBRATION, index=False)
    multiplier = summary["fixed_band"]["multiplier"]
    single = fixed_band_by_size(pd.read_parquet(paths.CONFIDENCE_SINGLE_FINDING), arm, multiplier)
    single = single[single["mask_size"] == 1]
    by_size = (
        pd.concat([fixed_band_by_size(pairs, arm, multiplier), single], ignore_index=True)
        .sort_values("mask_size")
        .reset_index(drop=True)
    )
    plot_coverage(by_size, summary["fixed_band"]["coverage"], multiplier)
    plot_curve(pairs, arm, held, settings, summary["zero_drift"]["share"])
    requested = quantile_levels(settings.heatmap_grid)
    sizes, coverage = coverage_by_size(pairs, arm, held, requested)
    plot_heatmap(sizes, requested, coverage)
    out = numbers(summary, pairs, table)
    for name in ("coverage", "drift_median", "drift_p90", "half_width_mean", "zero_drift"):
        out[f"confidence.fixed_band.1.{name}"] = float(single.iloc[0][name])
    out["confidence.single_finding_pairs"] = int(single.iloc[0]["n"])
    record_numbers(out)

    print(f"selected arm {arm}; fixed ±{summary['fixed_band']['multiplier']:g} sigma covers "
          f"{summary['fixed_band']['coverage']:.1%}; zero drift {summary['zero_drift']['share']:.1%}")
    for path in OUTPUTS:
        print(f"wrote {path}")


if __name__ == "__main__":
    run()
