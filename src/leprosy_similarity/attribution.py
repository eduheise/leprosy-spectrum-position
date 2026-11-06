"""Per-feature attribution of a position by Tree Similarity SHAP.

Tree Similarity SHAP (the ``shapley-values`` package) splits the similarity of
two records in one tree into one additive share per feature. Averaged over the
trees, it gives, for a record x and a reference case p, the attributions
phi(x, p, f), which sum to the pair's similarity minus a base similarity that
does not depend on the pair.

The position is a similarity-weighted mean of the reference anchors, so the
pair attributions project onto it exactly:

    contribution(x, f) = sum_p phi(x, p, f) * (anchor(p) - base) / sum_p s(x, p),

with ``base`` the mean anchor of the reference cases (the position of a record
about which nothing is known). Because the centred anchors sum to zero, the
contributions of a record sum to its position minus the base.

The trees are walked exactly as the library routes a record, so a pair's
attributions sum to the similarity the estimator uses; that is what makes the
identity exact rather than approximate.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from implementations._common import SimilarityTreeNode
from implementations.linear import _tree_explanation
from joblib import Parallel, delayed, effective_n_jobs

from leprosy_similarity.schema import CLASS_TARGETS


def lightgbm_tree(node: dict[str, Any]) -> SimilarityTreeNode:
    """A LightGBM dumped tree as a ``SimilarityTreeNode``; leaf counts as sample counts."""
    if "split_feature" not in node:
        return SimilarityTreeNode(sample_count=int(node.get("leaf_count", 0)))
    if node.get("decision_type", "<=") != "<=":
        raise NotImplementedError(f"categorical split {node['decision_type']!r} is not supported")
    return SimilarityTreeNode(
        sample_count=int(node.get("internal_count", 0)),
        feature=int(node["split_feature"]),
        threshold=float(node["threshold"]),
        left=lightgbm_tree(node["left_child"]),
        right=lightgbm_tree(node["right_child"]),
    )


def xgboost_trees(model: Any) -> list[SimilarityTreeNode]:
    """XGBoost trees as ``SimilarityTreeNode``s, routed as XGBoost routes them.

    XGBoost sends ``x < split`` to its ``Yes`` child, comparing float32
    values; the shared walker sends ``x <= threshold`` left. Storing each
    threshold as the largest float32 strictly below the split makes the two
    agree on float32-rounded inputs (see ``model_inputs``). XGBoost keeps a
    weighted mass per node, its cover, which stands in for the sample count;
    a leaf whose cover rounds below two holds no pair, and the efficiency
    identity then fails, which ``experiments/attribution.py`` checks.
    The models are fitted on unnamed arrays, so features are ``f0, f1, ...``.
    """
    frame = model.get_booster().trees_to_dataframe()
    trees = []
    for _, tree in frame.groupby("Tree", sort=True):
        rows = {row["ID"]: row for _, row in tree.iterrows()}
        trees.append(_xgboost_node(rows, tree.iloc[0]["ID"]))
    return trees


def _xgboost_node(rows: dict[str, Any], node_id: str) -> SimilarityTreeNode:
    """One node of a dumped XGBoost tree, with its subtree."""
    row = rows[node_id]
    count = round(float(row["Cover"]))
    if row["Feature"] == "Leaf":
        return SimilarityTreeNode(sample_count=count)
    return SimilarityTreeNode(
        sample_count=count,
        feature=int(row["Feature"][1:]),
        threshold=float(np.nextafter(np.float32(row["Split"]), np.float32(-np.inf))),
        left=_xgboost_node(rows, row["Yes"]),
        right=_xgboost_node(rows, row["No"]),
    )


def ensemble_trees(backbone: str, model: Any) -> list[SimilarityTreeNode]:
    """Every tree of a fitted ensemble, in the order the leaf matrix lists them."""
    if backbone == "xgboost":
        return xgboost_trees(model)
    if backbone == "lightgbm":
        return [lightgbm_tree(t["tree_structure"]) for t in model.booster_.dump_model()["tree_info"]]
    raise NotImplementedError(f"Tree Similarity SHAP is not wired for the {backbone!r} backbone")


def model_inputs(backbone: str, features: np.ndarray) -> np.ndarray:
    """Feature values as the library compares them: XGBoost rounds to float32 first."""
    if backbone == "xgboost":
        return np.asarray(features, dtype=np.float32).astype(float)
    return np.asarray(features, dtype=float)


def _record_attributions(x: np.ndarray, references: np.ndarray, trees: Sequence[SimilarityTreeNode]) -> np.ndarray:
    """Tree-averaged attributions of one record against every reference case, ``(n_P, F)``."""
    n_features = len(x)
    out = np.zeros((len(references), n_features))
    for j, p in enumerate(references):
        total = np.zeros(n_features)
        for tree in trees:
            total += np.asarray(_tree_explanation(x, p, tree, n_features).attributions, dtype=float)
        out[j] = total / len(trees)
    return out


def base_similarity(trees: Sequence[SimilarityTreeNode], x: np.ndarray, p: np.ndarray) -> float:
    """Tree-averaged base similarity: the chance two training records share a leaf.

    It depends only on the trees, not on the pair, so any pair gives it.
    """
    n_features = len(x)
    total = 0.0
    for tree in trees:
        total += float(_tree_explanation(x, p, tree, n_features).base_similarity)
    return total / len(trees)


def pair_attributions(
    trees: Sequence[SimilarityTreeNode], records: np.ndarray, references: np.ndarray, n_jobs: int
) -> tuple[np.ndarray, float]:
    """Attributions ``(n_records, n_references, n_features)`` and the base similarity.

    Records are explained independently, so they are spread over ``n_jobs``
    workers in contiguous chunks (the trees are shipped once per chunk); each
    record's sums run in the same order whatever the split.
    """
    n_chunks = min(len(records), 4 * effective_n_jobs(n_jobs))
    chunks = np.array_split(np.arange(len(records)), n_chunks)
    blocks = Parallel(n_jobs=n_jobs)(
        delayed(_chunk_attributions)(records[rows], references, trees) for rows in chunks
    )
    return np.concatenate(blocks), base_similarity(trees, records[0], references[0])


def _chunk_attributions(records: np.ndarray, references: np.ndarray, trees: Sequence[SimilarityTreeNode]) -> np.ndarray:
    return np.stack([_record_attributions(x, references, trees) for x in records])


def base_position(reference_classes: np.ndarray) -> float:
    """Position of a record about which nothing is known: the mean reference anchor."""
    return float(np.mean([CLASS_TARGETS[c] for c in reference_classes]))


def feature_contributions(
    attributions: np.ndarray, reference_classes: np.ndarray, similarity_base: float
) -> np.ndarray:
    """Signed contribution of each feature to each record's raw position, ``(n, F)``.

    Positive moves the record toward the lepromatous anchor, negative toward
    the tuberculoid. The anchors are centred on the fixed base, not on the
    record's own position: the decomposition explains the departure from the
    base, so every reference case's pull is measured from that one point.
    """
    anchors = np.array([CLASS_TARGETS[c] for c in reference_classes], dtype=float)
    similarity = attributions.sum(axis=2) + similarity_base
    total = similarity.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("a record has zero total similarity to the reference cases")
    centred = anchors[None, :, None] - anchors.mean()
    return (attributions * centred).sum(axis=1) / total[:, None]
