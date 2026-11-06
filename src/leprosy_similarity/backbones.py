"""The tree ensembles the estimator can be built on, and their search space.

Four backbones (LightGBM, XGBoost, CatBoost, random forest) and two target
heads: ``anchored`` regresses the class target (TT 0 ... LL 1) and
``multiclass`` classifies the five Ridley-Jopling labels. The estimator never
uses the ensemble's prediction, only the leaf each record reaches in each
tree, so either head yields a similarity.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import optuna
import pandas as pd

from leprosy_similarity.config import BackbonesConfig
from leprosy_similarity.schema import CLASS_TARGETS


def make_model(backbone: str, target_head: str, hyperparams: Mapping[str, Any], settings: BackbonesConfig) -> Any:
    """An unfitted ensemble with the library's fixed options and seed applied."""
    library = settings.settings[backbone]
    params = {**library.fixed, **hyperparams, library.seed_param: settings.seed}
    regression = target_head == "anchored"
    if backbone == "lightgbm":
        import lightgbm as lgb

        return (lgb.LGBMRegressor if regression else lgb.LGBMClassifier)(**params)
    if backbone == "xgboost":
        import xgboost as xgb

        return (xgb.XGBRegressor if regression else xgb.XGBClassifier)(**params)
    if backbone == "catboost":
        import catboost as cb

        return (cb.CatBoostRegressor if regression else cb.CatBoostClassifier)(**params)
    if backbone == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return (RandomForestRegressor if regression else RandomForestClassifier)(**params)
    raise ValueError(f"unknown backbone {backbone!r}")


def leaf_indices(backbone: str, model: Any, features: np.ndarray) -> np.ndarray:
    """Leaf reached in every tree, as an ``(n_records, n_trees)`` integer matrix."""
    if backbone == "lightgbm":
        with warnings.catch_warnings():
            # LightGBM stores generic column names at fit time, and scikit-learn
            # then warns that a plain array has none. The leaves are unaffected.
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            leaves = model.predict(features, pred_leaf=True)
    elif backbone == "catboost":
        leaves = model.calc_leaf_indexes(features)
    elif backbone in ("xgboost", "rf"):
        leaves = model.apply(features)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    return np.asarray(leaves, dtype=np.int64)


def encode_target(classes: pd.Series, target_head: str, backbone: str) -> np.ndarray:
    """Training target for the head.

    ``anchored`` gives each record its class target. ``multiclass`` gives the
    label itself, except for XGBoost, which needs contiguous integers 0..K-1:
    the labels present are numbered in sorted order, so a training set missing
    a class still encodes contiguously.
    """
    if target_head == "anchored":
        return np.array([CLASS_TARGETS[c] for c in classes], dtype=float)
    if backbone == "xgboost":
        index = {label: i for i, label in enumerate(sorted(set(classes)))}
        return np.array([index[c] for c in classes], dtype=int)
    return np.asarray(classes.astype(str))


def suggest_hyperparams(trial: optuna.Trial, backbone: str, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Draw one backbone's hyperparameters from the search space, in declared order.

    Parameters are registered with Optuna under ``<prefix><name>`` so names
    stay unique across backbones; the returned dict uses the library's names.
    """
    spec = space[backbone]
    prefix = spec["prefix"]
    drawn: dict[str, Any] = {}
    for name, dist in spec["params"].items():
        key = f"{prefix}{name}"
        if dist["type"] == "int":
            drawn[name] = trial.suggest_int(key, dist["low"], dist["high"])
        elif dist["type"] == "float":
            drawn[name] = trial.suggest_float(key, dist["low"], dist["high"], log=dist.get("log", False))
        elif dist["type"] == "categorical":
            drawn[name] = trial.suggest_categorical(key, list(dist["choices"]))
        else:
            raise ValueError(f"unknown distribution {dist['type']!r} for {key}")
    return drawn


def hyperparams_from_trial_params(
    backbone: str, trial_params: Mapping[str, Any], space: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Library hyperparameters of a stored trial (its prefixed Optuna params)."""
    prefix = space[backbone]["prefix"]
    names = space[backbone]["params"]
    return {name: trial_params[f"{prefix}{name}"] for name in names}
