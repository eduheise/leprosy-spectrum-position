"""The spectrum estimator: leaf co-occurrence similarity to the reference cases.

A tree ensemble is trained on the cohort's Ridley-Jopling labels. A record's
similarity to a reference case is the share of trees in which the two land in
the same leaf. The raw position is the similarity-weighted mean of the
reference anchors (a Nadaraya-Watson estimate),

    position(x) = sum_p s(x, p) * anchor(p) / sum_p s(x, p),

and the reported position rescales it to the model's reach (``rescaling.py``).

Usage::

    estimator = SpectrumEstimator(configuration, reference_cases, config).fit(train)
    estimator.position(records)        # rescaled, in [0, 1]
    estimator.raw_position(records)    # before rescaling
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leprosy_similarity import paths
from leprosy_similarity.backbones import encode_target, hyperparams_from_trial_params, leaf_indices, make_model
from leprosy_similarity.config import Config
from leprosy_similarity.imputation import Strategy, make_strategy
from leprosy_similarity.rescaling import PositionScale, fit_position_scale, recordable_grid
from leprosy_similarity.schema import CLASS_TARGETS, FEATURES


def leaf_cooccurrence_proximity(leaves_x: np.ndarray, leaves_p: np.ndarray) -> np.ndarray:
    """Share of trees in which row i of ``x`` and row j of ``p`` reach the same leaf."""
    leaves_x = np.asarray(leaves_x, dtype=np.int64)
    leaves_p = np.asarray(leaves_p, dtype=np.int64)
    if leaves_x.ndim != 2 or leaves_p.ndim != 2:
        raise ValueError("expected 2-D leaf-index matrices")
    n_trees = leaves_x.shape[1]
    if leaves_p.shape[1] != n_trees:
        raise ValueError(f"tree-count mismatch: x has {n_trees} trees, p has {leaves_p.shape[1]}")
    similarity = np.zeros((leaves_x.shape[0], leaves_p.shape[0]), dtype=float)
    for t in range(n_trees):
        similarity += (leaves_x[:, t][:, None] == leaves_p[:, t][None, :]).astype(float)
    similarity /= n_trees
    return similarity


def position_from_proximity(proximity: np.ndarray, reference_classes: np.ndarray) -> np.ndarray:
    """Similarity-weighted mean of the reference anchors; NaN where no similarity."""
    anchors = np.array([CLASS_TARGETS[c] for c in reference_classes], dtype=float)
    weights = np.nan_to_num(proximity, nan=0.0)
    total = weights.sum(axis=1)
    weighted = (weights * anchors[None, :]).sum(axis=1)
    return np.where(total > 0, weighted / np.where(total > 0, total, 1.0), np.nan)


@dataclass(frozen=True)
class Configuration:
    """One point of the search space: backbone, head, strategy, hyperparameters."""

    backbone: str
    target_head: str
    missing_values: str
    hyperparams: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trial_params(cls, params: Mapping[str, Any], space: Mapping[str, Any]) -> Configuration:
        """Configuration of a stored Optuna trial."""
        backbone = params["classifier"]
        return cls(
            backbone=backbone,
            target_head=params["target_head"],
            missing_values=params["mv_strategy"],
            hyperparams=hyperparams_from_trial_params(backbone, params, space),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "target_head": self.target_head,
            "missing_values": self.missing_values,
            "hyperparams": dict(self.hyperparams),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Configuration:
        return cls(
            backbone=data["backbone"],
            target_head=data["target_head"],
            missing_values=data["missing_values"],
            hyperparams=dict(data["hyperparams"]),
        )



def read_selection(path: Path = paths.SELECTED_CONFIGURATION) -> dict[str, Any]:
    """The selected configuration as stored: the configuration, its trial and objective values."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_configuration(path: Path = paths.SELECTED_CONFIGURATION) -> Configuration:
    """The configuration every experiment after the search refits."""
    return Configuration.from_dict(read_selection(path))


class SpectrumEstimator:
    """Missing-value strategy, tree ensemble and reference cases, fitted together."""

    def __init__(self, configuration: Configuration, reference_cases: pd.DataFrame, config: Config) -> None:
        self.configuration = configuration
        self.config = config
        self.reference_features = reference_cases[list(FEATURES)].astype(float)
        self.reference_classes = reference_cases["class"].astype(str).to_numpy()
        self.strategy: Strategy | None = None
        self.model: Any = None
        self.reference_leaves: np.ndarray | None = None
        self.scale: PositionScale | None = None

    def fit(self, train: pd.DataFrame, rescale: bool = True) -> SpectrumEstimator:
        """Fit the strategy and the ensemble on ``train``, then the position range.

        With ``rescale=False`` the range is left for ``fit_scale``. The search
        objective uses that order (raw positions first, range last) because
        the miceforest strategy draws from one random stream across calls.
        """
        c = self.configuration
        self.strategy = make_strategy(c.missing_values, self.config.imputation)
        self.strategy.fit(train[list(FEATURES)])
        features = self.strategy.transform(train[list(FEATURES)])
        target = encode_target(train["class"], c.target_head, c.backbone)
        self.model = make_model(c.backbone, c.target_head, c.hyperparams, self.config.backbones)
        self.model.fit(features.to_numpy(), target)
        self.reference_leaves = self.leaves(self.reference_features)
        self.scale = None
        if rescale:
            self.fit_scale(train)
        return self

    def leaves(self, records: pd.DataFrame) -> np.ndarray:
        """Leaf matrix of ``records`` after the fitted missing-value strategy."""
        if self.strategy is None:
            raise RuntimeError("the estimator is not fitted")
        features = self.strategy.transform(records[list(FEATURES)])
        return leaf_indices(self.configuration.backbone, self.model, features.to_numpy())

    def proximity(self, records: pd.DataFrame) -> np.ndarray:
        """Similarity of each record to each reference case, ``(n_records, 26)``."""
        return leaf_cooccurrence_proximity(self.leaves(records), self.reference_leaves)

    def raw_position(self, records: pd.DataFrame) -> np.ndarray:
        """Position before rescaling."""
        return position_from_proximity(self.proximity(records), self.reference_classes)

    def fit_scale(self, train: pd.DataFrame) -> PositionScale:
        """Fit the rescaling range over the recordable grid of ``train``."""
        self.scale = fit_position_scale(
            self.leaves,
            self.reference_leaves,
            self.reference_classes,
            recordable_grid(train[list(FEATURES)]),
            self.config.rescaling.batch_size,
        )
        return self.scale

    def position(self, records: pd.DataFrame) -> np.ndarray:
        """Rescaled position in [0, 1]."""
        if self.scale is None:
            raise RuntimeError("the position range is not fitted; call fit(rescale=True) or fit_scale")
        return self.scale(self.raw_position(records))
