"""Typed access to ``config.yaml``.

Every constant the pipeline uses is read from the YAML file at the project
root, so a reader can see in one place which numbers the results depend on.
The file is parsed once per process (``load_config`` is cached) and handed to
whatever needs it; nothing is copied into module-level globals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from leprosy_similarity import paths


@dataclass(frozen=True)
class FoldsConfig:
    n_folds: int
    seed: int


@dataclass(frozen=True)
class BackboneSettings:
    """How one tree library is called: its seed argument and fixed options."""

    seed_param: str
    fixed: dict[str, Any]


@dataclass(frozen=True)
class BackbonesConfig:
    seed: int
    settings: dict[str, BackboneSettings]


@dataclass(frozen=True)
class IterativeConfig:
    max_iter: int
    seed: int


@dataclass(frozen=True)
class MiceForestConfig:
    datasets: int
    iterations: int
    seed: int


@dataclass(frozen=True)
class ImputationConfig:
    knn_neighbours: int
    iterative: IterativeConfig
    miceforest: MiceForestConfig


@dataclass(frozen=True)
class SearchConfig:
    study_name: str
    n_trials: int
    sampler_seed: int
    masking_seed: int
    masking_sizes: tuple[int, ...]
    backbones: tuple[str, ...]
    target_heads: tuple[str, ...]
    missing_value_strategies: tuple[str, ...]
    infeasible: tuple[tuple[str, str], ...]
    space: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RescalingConfig:
    batch_size: int


@dataclass(frozen=True)
class VariableScreenConfig:
    seed: int
    n_folds: int
    n_repeats: int
    n_trees: int
    min_samples_leaf: int
    max_features: str
    knn_neighbours: int


@dataclass(frozen=True)
class QuantileGrid:
    """Evenly spaced requested coverages, ``points`` of them from ``low`` to ``high``."""

    low: float
    high: float
    points: int


@dataclass(frozen=True)
class ConfidenceConfig:
    records_per_class: int
    masks_per_record: int
    mask_sizes: tuple[int, ...]
    donor_draws: int
    donor_neighbours: int
    calibration_split_seed: int
    selection_levels: tuple[float, ...]
    levels: tuple[float, ...]
    curve_levels: tuple[float, ...]
    fixed_band_multiplier: float
    curve_grid: QuantileGrid
    heatmap_grid: QuantileGrid


def _confidence(section: dict[str, Any]) -> ConfidenceConfig:
    def levels(name: str) -> tuple[float, ...]:
        return tuple(float(v) for v in section[name])

    return ConfidenceConfig(
        records_per_class=int(section["records_per_class"]),
        masks_per_record=int(section["masks_per_record"]),
        mask_sizes=tuple(int(s) for s in section["mask_sizes"]),
        donor_draws=int(section["donor_draws"]),
        donor_neighbours=int(section["donor_neighbours"]),
        calibration_split_seed=int(section["calibration_split_seed"]),
        selection_levels=levels("selection_levels"),
        levels=levels("levels"),
        curve_levels=levels("curve_levels"),
        fixed_band_multiplier=float(section["fixed_band_multiplier"]),
        curve_grid=QuantileGrid(**section["curve_grid"]),
        heatmap_grid=QuantileGrid(**section["heatmap_grid"]),
    )


@dataclass(frozen=True)
class SelectionConfig:
    elbow_threshold: float


@dataclass(frozen=True)
class SearchReportConfig:
    hypervolume_margin: float
    visible_rise_share: float
    early_trials: int
    importance_seed: int
    hyperparameter_importance_trees: int


@dataclass(frozen=True)
class SpectrumPositionConfig:
    expected_lepromatous: tuple[str, ...]
    direction_tolerance: float
    logistic_max_iter: int


@dataclass(frozen=True)
class AttributionConfig:
    efficiency_tolerance: float
    n_jobs: int


@dataclass(frozen=True)
class Config:
    seed: int
    folds: FoldsConfig
    backbones: BackbonesConfig
    imputation: ImputationConfig
    search: SearchConfig
    rescaling: RescalingConfig
    variable_screen: VariableScreenConfig
    confidence: ConfidenceConfig
    selection: SelectionConfig
    search_report: SearchReportConfig
    spectrum_position: SpectrumPositionConfig
    attribution: AttributionConfig
    raw: dict[str, Any] = field(repr=False)


def _backbones(section: dict[str, Any]) -> BackbonesConfig:
    settings = {
        name: BackboneSettings(seed_param=spec["seed_param"], fixed=dict(spec["fixed"]))
        for name, spec in section.items()
        if name != "seed"
    }
    return BackbonesConfig(seed=int(section["seed"]), settings=settings)


def _search(section: dict[str, Any]) -> SearchConfig:
    return SearchConfig(
        study_name=section["study_name"],
        n_trials=int(section["n_trials"]),
        sampler_seed=int(section["sampler_seed"]),
        masking_seed=int(section["masking_seed"]),
        masking_sizes=tuple(int(s) for s in section["masking_sizes"]),
        backbones=tuple(section["backbones"]),
        target_heads=tuple(section["target_heads"]),
        missing_value_strategies=tuple(section["missing_value_strategies"]),
        infeasible=tuple(tuple(pair) for pair in section["infeasible"]),
        space=section["space"],
    )


def parse_config(raw: dict[str, Any]) -> Config:
    """Build the typed configuration from the parsed YAML mapping."""
    imputation = raw["imputation"]
    return Config(
        seed=int(raw["seed"]),
        folds=FoldsConfig(**raw["folds"]),
        backbones=_backbones(raw["backbones"]),
        imputation=ImputationConfig(
            knn_neighbours=int(imputation["knn_neighbours"]),
            iterative=IterativeConfig(**imputation["iterative"]),
            miceforest=MiceForestConfig(**imputation["miceforest"]),
        ),
        search=_search(raw["search"]),
        rescaling=RescalingConfig(**raw["rescaling"]),
        variable_screen=VariableScreenConfig(**raw["variable_screen"]),
        confidence=_confidence(raw["confidence"]),
        selection=SelectionConfig(elbow_threshold=float(raw["selection"]["elbow_threshold"])),
        search_report=SearchReportConfig(**raw["search_report"]),
        spectrum_position=SpectrumPositionConfig(
            expected_lepromatous=tuple(raw["spectrum_position"]["expected_lepromatous"]),
            direction_tolerance=float(raw["spectrum_position"]["direction_tolerance"]),
            logistic_max_iter=int(raw["spectrum_position"]["logistic_max_iter"]),
        ),
        attribution=AttributionConfig(**raw["attribution"]),
        raw=raw,
    )


@cache
def load_config(path: Path = paths.CONFIG) -> Config:
    """Read and validate ``config.yaml`` (once per process and path)."""
    with open(path, encoding="utf-8") as handle:
        return parse_config(yaml.safe_load(handle))
