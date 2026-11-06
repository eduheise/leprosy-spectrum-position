"""How the out-of-fold position recovers the classifications the field uses.

Two comparisons, both on held-out positions:

- the WHO treatment class (paucibacillary versus multibacillary), which is
  neither an input nor a training target. Being binary, it is read as a
  ROC-AUC and as the agreement at the Youden-optimal cut. Because the WHO
  class is defined by a rule on the smear, the position is also tested for
  what it adds over bacilloscopy alone, with two logistic models compared
  by cross-validated AUC;
- the recorded Ridley-Jopling class, read for order (Spearman, quadratic
  kappa against the nearest class target), for error from the class targets, and
  for whether the classes form separate clusters (silhouette).
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from leprosy_similarity.schema import CLASS_TARGETS, CLASSES

Splits = Sequence[tuple[np.ndarray, np.ndarray]]


def nearest_class(position: np.ndarray) -> np.ndarray:
    """The Ridley-Jopling class whose target is closest to each position."""
    targets = np.array([CLASS_TARGETS[c] for c in CLASSES])
    return np.array(CLASSES)[np.abs(np.asarray(position)[:, None] - targets[None, :]).argmin(axis=1)]


def who_recovery(position: np.ndarray, is_multibacillary: np.ndarray) -> dict[str, Any]:
    """ROC-AUC of the position for multibacillary treatment, and the Youden cut.

    A position at or above the cut reads as multibacillary.
    """
    mb = np.asarray(is_multibacillary, dtype=int)
    fpr, tpr, thresholds = roc_curve(mb, position)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    predicted = (position >= threshold).astype(int)
    tp = int(((predicted == 1) & (mb == 1)).sum())
    fn = int(((predicted == 0) & (mb == 1)).sum())
    tn = int(((predicted == 0) & (mb == 0)).sum())
    fp = int(((predicted == 1) & (mb == 0)).sum())
    return {
        "n": len(mb),
        "n_multibacillary": int(mb.sum()),
        "n_paucibacillary": int((1 - mb).sum()),
        "auc": float(roc_auc_score(mb, position)),
        "youden_threshold": threshold,
        "youden_fpr": float(fpr[best]),
        "youden_tpr": float(tpr[best]),
        "n_agree": tp + tn,
        "agreement": float((predicted == mb).mean()),
        "true_positive": tp,
        "true_negative": tn,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }


def within_class_contrast(position: np.ndarray, is_multibacillary: np.ndarray) -> dict[str, Any]:
    """Multibacillary against paucibacillary positions within one class.

    One-sided Mann-Whitney (multibacillary higher) with the rank-biserial
    effect size, the U statistic mapped onto [-1, 1].
    """
    mb = position[is_multibacillary]
    pb = position[~is_multibacillary]
    test = mannwhitneyu(mb, pb, alternative="greater")
    return {
        "n_multibacillary": len(mb),
        "n_paucibacillary": len(pb),
        "median_multibacillary": float(np.median(mb)),
        "median_paucibacillary": float(np.median(pb)),
        "mannwhitney_u": float(test.statistic),
        "p_one_sided": float(test.pvalue),
        "rank_biserial": float(2.0 * test.statistic / (len(mb) * len(pb)) - 1.0),
    }


def cross_validated_auc(features: np.ndarray, is_multibacillary: np.ndarray, splits: Splits, max_iter: int) -> float:
    """Mean held-out ROC-AUC of a standardised logistic model over ``splits``."""
    model = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=max_iter))])
    labels = np.asarray(is_multibacillary, dtype=int)
    scores = cross_val_score(model, features, labels, cv=list(splits), scoring="roc_auc")
    return float(scores.mean())


def bacilloscopy_increment(
    position: np.ndarray,
    bacilloscopy: np.ndarray,
    is_multibacillary: np.ndarray,
    splits: Splits,
    max_iter: int,
) -> dict[str, Any]:
    """What the position adds to bacilloscopy alone in predicting the WHO class.

    Computed over every record, not only those with a smear, so the records
    the smear never reaches are not hidden. An absent smear is read as zero
    (no bacilli demonstrated, as the operational rule reads it) and, as a
    check that the conclusion does not rest on that choice, as the cohort
    median.
    """
    out: dict[str, Any] = {"n": len(position), "n_without_smear": int(np.isnan(bacilloscopy).sum())}
    fills = {"zero": 0.0, "median": float(np.nanmedian(bacilloscopy))}
    for name, fill in fills.items():
        smear = np.where(np.isnan(bacilloscopy), fill, bacilloscopy)
        alone = cross_validated_auc(smear.reshape(-1, 1), is_multibacillary, splits, max_iter)
        both = cross_validated_auc(np.column_stack([smear, position]), is_multibacillary, splits, max_iter)
        out[name] = {
            "fill": fill,
            "auc_bacilloscopy": alone,
            "auc_bacilloscopy_and_position": both,
            "increment": both - alone,
        }
    return out


def class_ordering(position: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    """Order, error and separation of the positions by recorded class."""
    target = np.array([CLASS_TARGETS[c] for c in classes])
    error = np.abs(position - target)
    present = [c for c in CLASSES if (classes == c).any()]
    predicted = nearest_class(position)
    true_index = np.array([CLASSES.index(c) for c in classes])
    predicted_index = np.array([CLASSES.index(c) for c in predicted])
    steps = np.abs(true_index - predicted_index)
    silhouette = silhouette_samples(position.reshape(-1, 1), classes, metric="euclidean")
    poles = np.isin(classes, [CLASSES[0], CLASSES[-1]])
    means = {c: float(position[classes == c].mean()) for c in present}
    per_class_mae = {c: float(error[classes == c].mean()) for c in present}
    return {
        "n": len(position),
        "spearman": float(spearmanr(target, position).statistic),
        "quadratic_kappa": float(cohen_kappa_score(true_index, predicted_index, weights="quadratic")),
        "mae": float(error.mean()),
        "mae_class_balanced": float(np.mean(list(per_class_mae.values()))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae_poles": float(error[poles].mean()),
        "mae_middle": float(error[~poles].mean()),
        "per_class_mae": per_class_mae,
        "per_class_mean": means,
        "per_class_median": {c: float(np.median(position[classes == c])) for c in present},
        "steps_between_means": [means[b] - means[a] for a, b in pairwise(present)],
        "silhouette": float(silhouette_score(position.reshape(-1, 1), classes, metric="euclidean")),
        "per_class_silhouette": {c: float(silhouette[classes == c].mean()) for c in present},
        "n_exact": int((steps == 0).sum()),
        "n_within_one_step": int((steps <= 1).sum()),
        "confusion_labels": list(CLASSES),
        "confusion": confusion_matrix(true_index, predicted_index, labels=list(range(len(CLASSES)))).tolist(),
    }
