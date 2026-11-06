"""Confidence of a position: how far it can move because of what the record lacks.

A record's missing findings are completed ``M`` times by a multiple imputer,
the record is positioned on every completion, and ``sigma`` is the standard
deviation of the ``M`` positions (Rubin's between-imputation spread). The
rescaling is linear, so a spread is put on the rescaled scale by dividing it
by the model's span; the platform does the same. The band reported around
the position is ``+- k sigma``.

Three imputers can generate the completions:

- ``knn_donor``: whole-record hot-deck draws from the selected configuration's
  own kNN neighbourhood (``WholeRecordDonorDraw``), one donor per completion;
- ``bayesian_ridge``: chained Bayesian-ridge regressions sampling from their
  posterior, bounded to the observed range of each finding;
- ``forest``: chained gradient-boosting imputation (miceforest, LightGBM), one dataset per
  completion.

The multiplier ``k`` is not assumed. On a masking sweep, the drift of a
masked record (its distance from the position of the record with all its
findings) is divided by its ``sigma``; the ``p``-quantile of that ratio on one
half of the records is the ``k`` that covers a share ``p`` of drifts, and the
coverage it delivers is measured on the other half.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from leprosy_similarity.config import ConfidenceConfig, ImputationConfig, QuantileGrid
from leprosy_similarity.estimator import SpectrumEstimator
from leprosy_similarity.imputation import WholeRecordDonorDraw
from leprosy_similarity.rescaling import shared_leaf_counts
from leprosy_similarity.schema import CATEGORICAL_FEATURES, CLASS_TARGETS, FEATURES, NUMERIC_FEATURES

ARMS: tuple[str, ...] = ("knn_donor", "bayesian_ridge", "forest")
POLE_CLASSES: tuple[str, ...] = ("TT", "LL")

# A completion generator: fitted on training records, it returns M completed
# copies of the records it is given.
Completions = Callable[[pd.DataFrame], list[pd.DataFrame]]


def binarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Round the one-hot columns back to 0/1, so a drawn level is a recorded level."""
    frame[list(CATEGORICAL_FEATURES)] = (frame[list(CATEGORICAL_FEATURES)] >= 0.5).astype(float)
    return frame


def knn_donor_completions(pool: pd.DataFrame, draws: int, neighbours: int, seed: int) -> Completions:
    """Completion ``m`` copies each record's gaps from a donor drawn with seed ``seed + m``."""
    donors = [
        WholeRecordDonorDraw(k=neighbours, random_state=seed + m).fit(pool[list(FEATURES)]) for m in range(draws)
    ]

    def complete(records: pd.DataFrame) -> list[pd.DataFrame]:
        return [binarise(donor.transform(records[list(FEATURES)])) for donor in donors]

    return complete


def bayesian_ridge_completions(pool: pd.DataFrame, draws: int, max_iter: int, seed: int) -> Completions:
    """Posterior draws of a chained Bayesian-ridge imputer, one per seed ``seed + m``.

    Numeric findings and one-hot levels are imputed by separate chains. The
    numeric chain is bounded to the range observed in ``pool``: unbounded,
    the linear fill extrapolates to impossible values (lesion tiers of -25 on
    a 1..3 scale) and the posterior noise is added on top of them.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    numeric = pool[list(NUMERIC_FEATURES)]
    low = numeric.min().to_numpy(dtype=float)
    high = numeric.max().to_numpy(dtype=float)
    low = np.where(np.isfinite(low), low, -np.inf)
    high = np.where(np.isfinite(high), high, np.inf)
    chains = []
    for m in range(draws):
        numeric_chain = IterativeImputer(
            max_iter=max_iter, sample_posterior=True, random_state=seed + m, min_value=low, max_value=high
        ).fit(numeric)
        categorical_chain = IterativeImputer(
            max_iter=max_iter, sample_posterior=True, random_state=seed + m, min_value=0.0, max_value=1.0
        ).fit(pool[list(CATEGORICAL_FEATURES)])
        chains.append((numeric_chain, categorical_chain))

    def complete(records: pd.DataFrame) -> list[pd.DataFrame]:
        out = []
        for numeric_chain, categorical_chain in chains:
            filled = records[list(FEATURES)].copy()
            filled[list(NUMERIC_FEATURES)] = numeric_chain.transform(records[list(NUMERIC_FEATURES)])
            filled[list(CATEGORICAL_FEATURES)] = categorical_chain.transform(records[list(CATEGORICAL_FEATURES)])
            out.append(binarise(filled))
        return out

    return complete


def forest_completions(pool: pd.DataFrame, draws: int, iterations: int, seed: int) -> Completions:
    """The ``draws`` datasets of one miceforest kernel fitted on ``pool``.

    miceforest fills only the findings that had gaps when the kernel was
    fitted; a finding complete in ``pool`` (the reaction histories) stays
    missing in a completion and is then filled by the estimator's own
    missing-value strategy when the completion is positioned.
    """
    import miceforest as mf

    training = pool[list(FEATURES)].reset_index(drop=True).copy()
    kernel = mf.ImputationKernel(training, num_datasets=draws, random_state=seed)
    kernel.mice(iterations=iterations)

    def complete(records: pd.DataFrame) -> list[pd.DataFrame]:
        imputed = kernel.impute_new_data(records[list(FEATURES)].reset_index(drop=True).copy())
        return [
            binarise(imputed.complete_data(d)[list(FEATURES)].set_axis(records.index).astype(float))
            for d in range(draws)
        ]

    return complete


def make_completions(
    arm: str, pool: pd.DataFrame, settings: ConfidenceConfig, imputation: ImputationConfig, seed: int
) -> Completions:
    """The completion generator of ``arm``, fitted on the training records ``pool``."""
    draws = settings.donor_draws
    if arm == "knn_donor":
        return knn_donor_completions(pool, draws, settings.donor_neighbours, seed)
    if arm == "bayesian_ridge":
        return bayesian_ridge_completions(pool, draws, imputation.iterative.max_iter, seed)
    if arm == "forest":
        return forest_completions(pool, draws, imputation.miceforest.iterations, seed)
    raise ValueError(f"unknown confidence arm {arm!r}")


def exact_raw_position(estimator: SpectrumEstimator, records: pd.DataFrame) -> np.ndarray:
    """Raw position computed from whole shared-leaf counts.

    Equal to ``estimator.raw_position`` up to rounding, but exact in the sense
    that matters for a spread: the anchors are 0, 0.5 and 1, so the weighted
    count sum and the count total are exact, and two completions that land on
    the same position give the same number. Summing per-tree shares instead
    leaves last-bit differences that read as a spread of 1e-17, and a drift
    divided by such a spread is an absurd multiplier.
    """
    counts = shared_leaf_counts(estimator.leaves(records), estimator.reference_leaves)
    anchors = np.array([CLASS_TARGETS[c] for c in estimator.reference_classes], dtype=float)
    return (counts @ anchors) / counts.sum(axis=1)


def dispersion(estimator: SpectrumEstimator, completions: list[pd.DataFrame]) -> np.ndarray:
    """Between-completion standard deviation (``ddof=1``) of each record's position, rescaled.

    The spread is taken on the raw positions and divided by the span of the
    rescaling, as the platform does; taking it on the clipped rescaled
    positions would shrink the spread of the few completions beyond the
    reachable range.
    """
    positions = np.column_stack([exact_raw_position(estimator, completed) for completed in completions])
    return positions.std(axis=1, ddof=1) / estimator.scale.span


def case_split(cases: np.ndarray, seed: int) -> np.ndarray:
    """Held-out side of the calibration split: half of the sampled records.

    The split is by record, not by pair, so the masks of one record never
    fall on both sides. Returns a boolean mask over ``cases`` (one entry per
    pair).
    """
    unique = np.unique(cases)
    rng = np.random.default_rng(seed)
    held = rng.choice(unique, size=len(unique) // 2, replace=False)
    return np.isin(cases, held)


def calibration_ratio(sigma: np.ndarray, drift: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """``drift / sigma`` on the calibration side, over the pairs with a positive ``sigma``.

    A pair whose completions all land on one position has no finite ratio
    and cannot inform the multiplier, so it is left out of the fit.
    """
    usable = fit & np.isfinite(drift) & np.isfinite(sigma) & (sigma > 0)
    return drift[usable] / sigma[usable]


def covered(sigma: np.ndarray, drift: np.ndarray, k: np.ndarray | float) -> np.ndarray:
    """Whether each drift falls inside ``+- k sigma``; rows are pairs, columns values of ``k``.

    Every held-out pair counts. A pair with ``sigma`` zero has a band of zero
    width, which covers it only when it does not drift at all.
    """
    return drift[:, None] <= sigma[:, None] * np.atleast_1d(k)[None, :]


def band_at_level(
    sigma: np.ndarray, drift: np.ndarray, fit: np.ndarray, held: np.ndarray, level: float
) -> tuple[float, float, int]:
    """Coverage, mean width and size of the held-out side for a band fitted at ``level``.

    ``k`` is the ``level``-quantile of ``drift / sigma`` on the fitting side.
    """
    k = float(np.quantile(calibration_ratio(sigma, drift, fit), level))
    held = held & np.isfinite(drift) & np.isfinite(sigma)
    inside = covered(sigma[held], drift[held], k)[:, 0]
    return float(inside.mean()), float(np.mean(2 * k * sigma[held])), int(held.sum())


def sharpness_table(
    pairs: pd.DataFrame, arms: Sequence[str], levels: Sequence[float], held: np.ndarray
) -> pd.DataFrame:
    """Delivered coverage and mean band width of every arm at every requested level."""
    drift = pairs["drift"].to_numpy(float)
    rows = []
    for level in levels:
        for arm in arms:
            coverage, width, n = band_at_level(pairs[f"sigma_{arm}"].to_numpy(float), drift, ~held, held, level)
            rows.append({"level": level, "arm": arm, "coverage": coverage, "mean_width": width, "n": n})
    return pd.DataFrame(rows)


def select_arm(table: pd.DataFrame) -> str:
    """The arm with the narrowest band at the most requested levels.

    Every arm carries a calibrated multiplier, so every arm can be tuned to
    the coverage requested; what separates them is how wide the band must be
    to get there (sharpness subject to calibration, Gneiting et al. 2007).
    Ties go to the smaller summed width.
    """
    narrowest = table.loc[table.groupby("level")["mean_width"].idxmin(), "arm"].value_counts()
    total_width = table.groupby("arm")["mean_width"].sum()
    ranking = sorted(total_width.index, key=lambda arm: (-narrowest.get(arm, 0), total_width[arm]))
    return ranking[0]


def held_out_ratio(pairs: pd.DataFrame, arm: str, held: np.ndarray) -> np.ndarray:
    """``drift / sigma`` of the held-out pairs, the multiplier each one needs.

    A pair that does not drift needs no band (0); one that drifts with
    ``sigma`` zero cannot be covered by any multiplier (infinity).
    """
    drift = pairs["drift"].to_numpy(float)[held]
    sigma = pairs[f"sigma_{arm}"].to_numpy(float)[held]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(drift == 0, 0.0, drift / sigma)


def quantile_levels(grid: QuantileGrid) -> np.ndarray:
    return np.linspace(grid.low, grid.high, grid.points)


def calibration_curve(pairs: pd.DataFrame, arm: str, held: np.ndarray, grid: QuantileGrid) -> PchipInterpolator:
    """``k(p)``: a monotone spline through the empirical quantiles of the calibration ratio.

    The spline follows the measured quantiles exactly and adds no assumed
    shape between them.
    """
    ratio = calibration_ratio(pairs[f"sigma_{arm}"].to_numpy(float), pairs["drift"].to_numpy(float), ~held)
    levels = quantile_levels(grid)
    return PchipInterpolator(levels, np.quantile(ratio, levels))


def multipliers(
    pairs: pd.DataFrame, arm: str, held: np.ndarray, levels: Sequence[float], grid: QuantileGrid
) -> dict[float, tuple[float, float]]:
    """``k(p)`` fitted on the calibration side and its held-out coverage, per level."""
    curve = calibration_curve(pairs, arm, held, grid)
    k = np.array([float(curve(level)) for level in levels])
    inside = covered(pairs[f"sigma_{arm}"].to_numpy(float)[held], pairs["drift"].to_numpy(float)[held], k)
    return {level: (float(k[i]), float(inside[:, i].mean())) for i, level in enumerate(levels)}


def fixed_band_by_size(pairs: pd.DataFrame, arm: str, multiplier: float) -> pd.DataFrame:
    """Drift quantiles, band half-width and coverage of a fixed ``+- multiplier sigma`` band.

    One row per number of hidden tests, starting from zero (the record as
    recorded) when the sweep carries the unmasked spread.
    """
    frame = pairs.assign(half_width=multiplier * pairs[f"sigma_{arm}"])
    frame["inside"] = frame["drift"] <= frame["half_width"]
    by_size = (
        frame.groupby("mask_size")
        .agg(
            n=("drift", "size"),
            drift_median=("drift", "median"),
            drift_p10=("drift", lambda s: s.quantile(0.10)),
            drift_p25=("drift", lambda s: s.quantile(0.25)),
            drift_p75=("drift", lambda s: s.quantile(0.75)),
            drift_p90=("drift", lambda s: s.quantile(0.90)),
            half_width_mean=("half_width", "mean"),
            coverage=("inside", "mean"),
            zero_drift=("drift", lambda s: float((s == 0).mean())),
        )
        .reset_index()
    )
    full_sigma = f"sigma_full_{arm}"
    if full_sigma not in pairs:
        return by_size
    # With no test hidden a record does not move, so its drift is zero and any
    # band covers it; the band still has the width its own gaps give it.
    records = pairs.drop_duplicates("case")
    nothing_hidden = {
        "mask_size": 0,
        "n": len(records),
        "drift_median": 0.0,
        "drift_p10": 0.0,
        "drift_p25": 0.0,
        "drift_p75": 0.0,
        "drift_p90": 0.0,
        "half_width_mean": float((multiplier * records[full_sigma]).mean()),
        "coverage": 1.0,
        "zero_drift": 1.0,
    }
    return pd.concat([pd.DataFrame([nothing_hidden]), by_size], ignore_index=True)


def coverage_by_size(
    pairs: pd.DataFrame, arm: str, held: np.ndarray, requested: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out coverage (%) of the band calibrated at each requested level, per mask size.

    ``k`` for each requested level is the empirical quantile of the ratio on
    the calibration side; returns the mask sizes and a ``(levels, sizes)``
    matrix of achieved coverage.
    """
    sigma, drift = pairs[f"sigma_{arm}"].to_numpy(float), pairs["drift"].to_numpy(float)
    k = np.quantile(calibration_ratio(sigma, drift, ~held), requested)
    size = pairs["mask_size"].to_numpy()
    sizes = np.array(sorted(np.unique(size[held])))
    coverage = np.empty((len(requested), len(sizes)))
    for j, s in enumerate(sizes):
        rows = held & (size == s)
        coverage[:, j] = 100.0 * covered(sigma[rows], drift[rows], k).mean(axis=0)
    return sizes, coverage
