import numpy as np

from leprosy_similarity.estimator import leaf_cooccurrence_proximity
from leprosy_similarity.rescaling import PositionScale, recordable_grid, shared_leaf_counts
from leprosy_similarity.schema import FEATURES


def test_shared_leaf_counts_match_the_proximity():
    rng = np.random.default_rng(0)
    x, p = rng.integers(0, 7, (20, 30)), rng.integers(0, 7, (5, 30))
    np.testing.assert_allclose(shared_leaf_counts(x, p) / 30, leaf_cooccurrence_proximity(x, p))


def test_grid_enumerates_every_recordable_combination(cohort):
    grid = recordable_grid(cohort[list(FEATURES)])
    n_numeric = np.prod([cohort[c].dropna().nunique() for c in FEATURES[:6]])
    assert len(grid) == n_numeric * 4 * 5
    assert not grid.isna().any().any()
    assert (grid[[c for c in FEATURES if c.startswith("lesion_color")]].sum(axis=1) == 1).all()


def test_scale_maps_the_range_onto_unit_interval_and_keeps_order():
    scale = PositionScale(lo=0.2, hi=0.7, argmin={}, argmax={})
    raw = np.array([0.1, 0.2, 0.45, 0.7, 0.9])
    out = scale(raw)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.all(np.diff(out) >= 0)
    assert PositionScale.from_dict(scale.to_dict()) == scale
