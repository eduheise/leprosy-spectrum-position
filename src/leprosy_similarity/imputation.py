"""Missing-value strategies the configuration search chooses between.

A strategy is fitted on the training records only and then fills (or flags)
the gaps of any record before it reaches the tree ensemble:

- ``native_routing``: leave NaN in place; the ensemble routes it at each split.
- ``mean_median``: median for numeric findings, most frequent level for one-hots.
- ``knn``: distance-weighted fill from the 5 nearest training records,
  measured over every finding the record observes (``DeterministicKNNImputer``).
- ``iterative_br``: round-robin Bayesian-ridge regressions, bounded to the
  observed range of each finding.
- ``miceforest``: chained gradient-boosting imputation (miceforest, LightGBM), averaged over datasets.
- ``mia_indicators``: leave NaN and add one missingness indicator per finding.

After any fill, the one-hot columns are made 0/1 again, so an imputed level
still means "this level was recorded".

The module also holds ``WholeRecordDonorDraw``, the random counterpart of the
kNN fill used to measure confidence: a record copies its missing findings from
one donor drawn among its nearest neighbours.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from leprosy_similarity.config import ImputationConfig
from leprosy_similarity.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, categorical_blocks


def nan_euclidean(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Distance from ``query`` to every pool row over the coordinates both observe.

    ``sqrt(sum_sq * n_columns / n_shared)``, scikit-learn's nan-Euclidean
    scaling, computed elementwise so it is identical on every machine; NaN
    where the two share no coordinate.
    """
    shared = ~np.isnan(pool) & ~np.isnan(query)[None, :]
    diff = np.where(shared, pool - np.nan_to_num(query)[None, :], 0.0)
    n_shared = shared.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.sqrt((diff * diff).sum(axis=1) * query.size / n_shared)
    distance[n_shared == 0] = np.nan
    return distance


def nearest(distance: np.ndarray, usable: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` usable pool rows closest to the query; ties go to the earlier row.

    A stable sort on elementwise distances makes the choice among equidistant
    donors the same on every CPU, which scikit-learn's ``KNNImputer`` does not
    guarantee. The cohort has many exact ties (records with identical
    findings), so this matters.
    """
    d = np.where(np.isnan(distance[usable]), np.inf, distance[usable])
    return usable[np.argsort(d, kind="stable")[:k]]


class DeterministicKNNImputer:
    """Distance-weighted kNN fill over every column of the fitted frame.

    One imputer covers the whole schema, so a record missing the morphology
    block is matched on its laboratory findings instead of receiving the
    column mode. For each missing cell the donors are the ``k`` nearest
    training rows that observed that column, weighted by ``1/d`` (or averaged
    equally over donors at distance zero). A record that shares no coordinate
    with any donor gets the column mean.
    """

    def __init__(self, k: int):
        self.k = k
        self.columns: list[str] = []
        self._pool = np.empty((0, 0))
        self._mean = np.empty(0)

    def fit(self, pool: pd.DataFrame) -> DeterministicKNNImputer:
        self.columns = list(pool.columns)
        self._pool = pool.to_numpy(dtype=float)
        self._mean = np.nanmean(self._pool, axis=0)
        return self

    def _fill_row(self, query: np.ndarray) -> np.ndarray:
        out = query.copy()
        distance = nan_euclidean(query, self._pool)
        for column in np.flatnonzero(np.isnan(query)):
            usable = np.flatnonzero(~np.isnan(self._pool[:, column]))
            if usable.size == 0 or np.all(np.isnan(distance[usable])):
                out[column] = self._mean[column]
                continue
            donors = nearest(distance, usable, self.k)
            d = distance[donors]
            donors, d = donors[np.isfinite(d)], d[np.isfinite(d)]
            values = self._pool[donors, column]
            if np.any(d == 0):
                out[column] = values[d == 0].mean()
            else:
                weights = 1.0 / d
                out[column] = float((weights * values).sum() / weights.sum())
        return out

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        queries = records[self.columns].to_numpy(dtype=float)
        filled = np.array([self._fill_row(q) if np.isnan(q).any() else q for q in queries])
        return pd.DataFrame(filled, columns=self.columns, index=records.index)


class WholeRecordDonorDraw:
    """One random hot-deck draw: a record copies its gaps from a single donor.

    The donor is drawn among the ``k`` nearest training records (same distance
    and tie rule as ``DeterministicKNNImputer``) that observed every finding
    the record is missing, so a drawn record is always a combination a real
    patient had. When no training record observed all of them (histology and
    the clinical examination come from different centres), the fill proceeds
    in rounds: each round draws among the ``k`` nearest records covering the
    largest share of the gaps still open and copies all of those.
    """

    def __init__(self, k: int, random_state: int):
        self.k = k
        self._rng = np.random.default_rng(random_state)
        self.columns: list[str] = []
        self._pool = np.empty((0, 0))
        self._mean = np.empty(0)

    def fit(self, pool: pd.DataFrame) -> WholeRecordDonorDraw:
        self.columns = list(pool.columns)
        self._pool = pool.to_numpy(dtype=float)
        self._mean = np.nanmean(self._pool, axis=0)
        return self

    def _draw_row(self, query: np.ndarray) -> np.ndarray:
        out = query.copy()
        distance = nan_euclidean(query, self._pool)
        gaps = np.isnan(out)
        while gaps.any():
            covered = (~np.isnan(self._pool) & gaps[None, :]).sum(axis=1)
            best = covered.max()
            if best == 0:
                out[gaps] = self._mean[gaps]
                break
            usable = np.flatnonzero(covered == best)
            donor = self._rng.choice(nearest(distance, usable, self.k))
            take = gaps & ~np.isnan(self._pool[donor])
            out[take] = self._pool[donor, take]
            gaps = np.isnan(out)
        return out

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        queries = records[self.columns].to_numpy(dtype=float)
        drawn = np.array([self._draw_row(q) if np.isnan(q).any() else q for q in queries])
        return pd.DataFrame(drawn, columns=self.columns, index=records.index)


class Strategy(Protocol):
    """A missing-value strategy: fitted on training records, applied to any."""

    name: str

    def fit(self, records: pd.DataFrame) -> Strategy: ...

    def transform(self, records: pd.DataFrame) -> pd.DataFrame: ...


def binarise_one_hots(frame: pd.DataFrame) -> pd.DataFrame:
    """Round every one-hot column back to 0/1 at 0.5."""
    frame[list(CATEGORICAL_FEATURES)] = (frame[list(CATEGORICAL_FEATURES)] >= 0.5).astype(float)
    return frame


class NativeRouting:
    name = "native_routing"

    def fit(self, records: pd.DataFrame) -> NativeRouting:
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        return records[list(FEATURES)].copy()


class MeanMedian:
    name = "mean_median"

    def __init__(self) -> None:
        from sklearn.impute import SimpleImputer

        self._numeric = SimpleImputer(strategy="median")
        self._categorical = SimpleImputer(strategy="most_frequent")

    def fit(self, records: pd.DataFrame) -> MeanMedian:
        self._numeric.fit(records[list(NUMERIC_FEATURES)])
        self._categorical.fit(records[list(CATEGORICAL_FEATURES)])
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        out = records[list(FEATURES)].copy()
        out[list(NUMERIC_FEATURES)] = self._numeric.transform(records[list(NUMERIC_FEATURES)])
        out[list(CATEGORICAL_FEATURES)] = self._categorical.transform(records[list(CATEGORICAL_FEATURES)])
        return out


class KNN:
    """Deterministic kNN over the whole schema, with whole-block categoricals.

    Every observed record carries exactly one level per one-hot block, so a
    block missing as a whole takes the single level with the largest
    neighbour weight; a partly observed block is rounded back to 0/1.
    """

    name = "knn"

    def __init__(self, k: int) -> None:
        self._imputer = DeterministicKNNImputer(k=k)

    def fit(self, records: pd.DataFrame) -> KNN:
        self._imputer.fit(records[list(FEATURES)])
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        out = self._imputer.transform(records[list(FEATURES)])
        for columns in categorical_blocks().values():
            block_missing = records[columns].isna().all(axis=1).to_numpy()
            weights = out[columns].to_numpy()
            rounded = (weights >= 0.5).astype(float)
            single = np.zeros_like(weights)
            single[np.arange(len(weights)), weights.argmax(axis=1)] = 1.0
            out[columns] = np.where(block_missing[:, None], single, rounded)
        return out


class IterativeBayesianRidge(MeanMedian):
    """Round-robin Bayesian-ridge regressions, bounded to the observed ranges.

    Unbounded, the linear fill extrapolates far outside each finding's scale
    (lesion tiers between -25 and +23 on a 1..3 scale), and the ensemble then
    reads the out-of-range values as a missingness flag rather than a value.
    The bounds come from the training records only.
    """

    name = "iterative_br"

    def __init__(self, max_iter: int, seed: int) -> None:
        self._max_iter = max_iter
        self._seed = seed

    def fit(self, records: pd.DataFrame) -> IterativeBayesianRidge:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        numeric = records[list(NUMERIC_FEATURES)]
        low = numeric.min().to_numpy(dtype=float)
        high = numeric.max().to_numpy(dtype=float)
        # A finding absent from every training record has no observed range.
        low = np.where(np.isfinite(low), low, -np.inf)
        high = np.where(np.isfinite(high), high, np.inf)
        self._numeric = IterativeImputer(
            max_iter=self._max_iter, random_state=self._seed, min_value=low, max_value=high
        )
        self._categorical = IterativeImputer(
            max_iter=self._max_iter, random_state=self._seed, min_value=0.0, max_value=1.0
        )
        self._numeric.fit(numeric)
        self._categorical.fit(records[list(CATEGORICAL_FEATURES)])
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        return binarise_one_hots(super().transform(records))


class MiceForest:
    """Chained gradient-boosting imputation (miceforest), averaged over several datasets.

    The kernel keeps one random stream across calls, so the fill a record
    receives depends on how many records were filled before it. Callers that
    need to reproduce a result must therefore transform in the same order.

    miceforest only fills columns that had gaps at fit time; a column complete
    in training (the reaction flags) comes back NaN when masked later, so it
    falls back to the training median.
    """

    name = "miceforest"

    def __init__(self, datasets: int, iterations: int, seed: int) -> None:
        self._datasets = datasets
        self._iterations = iterations
        self._seed = seed

    def fit(self, records: pd.DataFrame) -> MiceForest:
        import miceforest as mf

        clean = records[list(FEATURES)].reset_index(drop=True).copy()
        self._kernel = mf.ImputationKernel(clean, num_datasets=self._datasets, random_state=self._seed)
        self._kernel.mice(iterations=self._iterations)
        self._fallback = clean.median(numeric_only=True)
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        frame = records[list(FEATURES)].reset_index(drop=True).copy()
        completed = [
            self._kernel.impute_new_data(frame).complete_data(d).to_numpy() for d in range(self._datasets)
        ]
        out = pd.DataFrame(np.mean(np.stack(completed), axis=0), columns=list(FEATURES), index=records.index)
        if out.isna().any().any():
            out = out.fillna(self._fallback)
        return binarise_one_hots(out)


class MissingIndicators:
    """Keep NaN and append one 0/1 missingness indicator per finding."""

    name = "mia_indicators"
    indicator_columns: tuple[str, ...] = tuple(f"{c}__is_missing" for c in FEATURES)

    def fit(self, records: pd.DataFrame) -> MissingIndicators:
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        base = records[list(FEATURES)].astype(float).copy()
        indicators = base.isna().astype(float)
        indicators.columns = list(self.indicator_columns)
        return pd.concat([base, indicators], axis=1)


def make_strategy(name: str, settings: ImputationConfig) -> Strategy:
    """An unfitted missing-value strategy by name."""
    if name == "native_routing":
        return NativeRouting()
    if name == "mean_median":
        return MeanMedian()
    if name == "knn":
        return KNN(k=settings.knn_neighbours)
    if name == "iterative_br":
        return IterativeBayesianRidge(settings.iterative.max_iter, settings.iterative.seed)
    if name == "miceforest":
        m = settings.miceforest
        return MiceForest(m.datasets, m.iterations, m.seed)
    if name == "mia_indicators":
        return MissingIndicators()
    raise ValueError(f"unknown missing-value strategy {name!r}")
