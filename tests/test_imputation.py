import numpy as np
import pandas as pd

from leprosy_similarity.imputation import DeterministicKNNImputer, WholeRecordDonorDraw, make_strategy
from leprosy_similarity.schema import CATEGORICAL_FEATURES, FEATURES, categorical_blocks


def test_knn_ties_go_to_the_earlier_donor():
    pool = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [10.0, 20.0, 30.0]})
    filled = DeterministicKNNImputer(k=1).fit(pool).transform(pd.DataFrame({"a": [1.0], "b": [np.nan]}))
    assert filled.loc[0, "b"] == 10.0


def test_knn_is_deterministic_and_complete(cohort, config):
    strategy = make_strategy("knn", config.imputation).fit(cohort[list(FEATURES)])
    first = strategy.transform(cohort[list(FEATURES)])
    second = make_strategy("knn", config.imputation).fit(cohort[list(FEATURES)]).transform(cohort[list(FEATURES)])
    assert not first.isna().any().any()
    pd.testing.assert_frame_equal(first, second)
    for columns in categorical_blocks().values():
        assert (first[columns].sum(axis=1) >= 1).all()
    assert set(np.unique(first[list(CATEGORICAL_FEATURES)])) <= {0.0, 1.0}


def test_every_strategy_fills_or_flags(cohort, config):
    for name in config.search.missing_value_strategies:
        if name == "miceforest":
            continue
        out = make_strategy(name, config.imputation).fit(cohort).transform(cohort)
        assert list(out.columns[: len(FEATURES)]) == list(FEATURES)
        if name in ("mean_median", "knn", "iterative_br"):
            assert not out.isna().any().any(), name


def test_donor_draws_repeat_with_the_seed(cohort):
    records = cohort[list(FEATURES)]
    a = WholeRecordDonorDraw(k=5, random_state=3).fit(records).transform(records)
    b = WholeRecordDonorDraw(k=5, random_state=3).fit(records).transform(records)
    pd.testing.assert_frame_equal(a, b)
    assert not a.isna().any().any()
