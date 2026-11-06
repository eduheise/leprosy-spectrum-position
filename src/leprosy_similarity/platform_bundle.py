"""The bundle the clinical platform serves: the final model and everything around it.

The platform scores one record at a time without this package installed, so
the bundle carries plain arrays alongside the fitted ensemble:

- the schema (feature order, numeric and one-hot columns, class anchors);
- the fitted ensemble, and its trees as flat arrays for the per-feature
  attribution;
- the 26 reference cases, as filled by the selected missing-value strategy
  and as recorded, with their numbers and classes;
- the donor pool of the kNN imputer and of the confidence draws (the analytic
  cohort's findings), with the number of neighbours, of draws and the seed;
- the rescaling range and the calibrated band multiplier per requested
  coverage.

``write_bundle`` saves it with joblib and writes a manifest holding the
SHA-256 of every file, so the platform can name the exact artefact it serves.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from leprosy_similarity.estimator import SpectrumEstimator
from leprosy_similarity.schema import CATEGORICAL_FEATURES, CLASS_TARGETS, FEATURES, NUMERIC_FEATURES


def xgboost_tree_table(model: Any) -> dict[str, np.ndarray]:
    """The booster's trees as flat arrays: one row per node.

    The platform rebuilds every tree from these arrays to run Tree Similarity
    SHAP without XGBoost's own tree walker. Leaves have feature ``-1`` and
    children ``-1``.
    """
    frame = model.get_booster().trees_to_dataframe()
    is_leaf = (frame["Feature"] == "Leaf").to_numpy()
    feature = np.where(is_leaf, -1, frame["Feature"].str[1:].where(~is_leaf, "-1").astype(int))

    def child(column: str) -> np.ndarray:
        node = frame[column].fillna("0-0").str.split("-").str[1].astype(int)
        return np.where(is_leaf, -1, node).astype(np.int64)

    return {
        "tree": frame["Tree"].to_numpy(dtype=np.int64),
        "node": frame["Node"].to_numpy(dtype=np.int64),
        "feature": feature.astype(np.int64),
        "threshold": frame["Split"].fillna(0.0).to_numpy(dtype=float),
        "yes": child("Yes"),
        "no": child("No"),
        "cover": frame["Cover"].to_numpy(dtype=float),
    }


def build_bundle(
    estimator: SpectrumEstimator,
    cohort: pd.DataFrame,
    reference_cases: pd.DataFrame,
    band_multipliers: Mapping[float, float],
    n_grid_records: int,
    donor_draws: int,
    donor_neighbours: int,
    donor_seed: int,
) -> dict[str, Any]:
    """Everything the platform needs to score, bound and explain one record."""
    if estimator.configuration.backbone != "xgboost":
        raise NotImplementedError(
            "the platform reads XGBoost tree dumps for its attribution; "
            f"a {estimator.configuration.backbone} bundle needs a matching tree export"
        )
    filled_references = estimator.strategy.transform(estimator.reference_features)
    return {
        "features": list(FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "class_indices": dict(CLASS_TARGETS),
        "model": estimator.model,
        "trees": xgboost_tree_table(estimator.model),
        "prototypes": filled_references[list(FEATURES)].to_numpy(dtype=float),
        "prototype_ids": reference_cases["reference_id"].astype(int).tolist(),
        "prototype_classes": [str(c) for c in estimator.reference_classes],
        "prototype_raw": reference_cases[list(FEATURES)].to_numpy(dtype=float),
        "donor_pool": cohort[list(FEATURES)].to_numpy(dtype=float),
        "donor_k": donor_neighbours,
        "donor_draws": donor_draws,
        "donor_seed": donor_seed,
        "scale": {"lo": estimator.scale.lo, "hi": estimator.scale.hi, "n_grid_records": n_grid_records},
        "band_k": {f"{level:.2f}": float(k) for level, k in band_multipliers.items()},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repository: Path) -> str | None:
    """Commit the bundle was built from, or None outside a repository with commits."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_bundle(
    bundle: Mapping[str, Any], bundle_path: Path, manifest_path: Path, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Save the bundle and a manifest of SHA-256 hashes beside it.

    ``sha256`` is the hash of the bundle file the platform loads (it reports
    it with every response); ``files`` hashes every file of the folder
    except the manifest itself.
    """
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), bundle_path, compress=3)
    files = {
        p.name: sha256(p) for p in sorted(bundle_path.parent.iterdir()) if p.is_file() and p != manifest_path
    }
    manifest = {"sha256": files[bundle_path.name], "files": files, **provenance}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
