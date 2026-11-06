from leprosy_similarity.config import load_config
from leprosy_similarity.schema import CATEGORICAL_FEATURES, CLASS_TARGETS, CLASSES, FEATURES, NUMERIC_FEATURES


def test_schema_has_fifteen_columns():
    assert len(NUMERIC_FEATURES) == 6
    assert len(CATEGORICAL_FEATURES) == 9
    assert len(FEATURES) == 15


def test_class_targets_are_equally_spaced_in_order():
    assert CLASSES == ("TT", "BT", "BB", "BL", "LL")
    assert [CLASS_TARGETS[c] for c in CLASSES] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_config_loads_the_protocol_constants():
    config = load_config()
    assert config.folds.n_folds == 5 and config.folds.seed == 42
    assert config.search.masking_sizes == tuple(range(2, len(FEATURES) - 1, 2))
    assert set(config.search.space) == set(config.search.backbones)
