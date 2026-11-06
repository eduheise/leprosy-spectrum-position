import numpy as np
import pandas as pd

from leprosy_similarity.exclusion_criteria import CRITERIA, apply_exclusion_criteria
from leprosy_similarity.schema import FEATURES


def record(cls: str, **findings: float) -> dict:
    row = {c: np.nan for c in FEATURES}
    row.update(findings)
    row["class"] = cls
    return row


def test_contradicting_and_evidence_free_records_are_removed():
    raw = pd.DataFrame(
        [
            record("TT", BACILOSCOP_N=0, NU_LESOES=1),  # consistent
            record("LL", BACILOSCOP_N=0),  # negative smear rules out LL
            record("TT", BACILOSCOP_N=2),  # positive smear rules out TT
            record("BT", NU_LESOES=1),  # no laboratory evidence
            record("BB", HISTOLOG=3),  # histology alone is enough
        ]
    )
    raw.insert(0, "record_id", range(len(raw)))
    result = apply_exclusion_criteria(raw)
    assert result.cohort["record_id"].tolist() == [0, 4]
    assert result.audit["deleted"].tolist() == [False, True, True, True, False]
    steps = result.attrition.set_index("step")["removed"]
    assert steps["negative skin smear"] == 1 and steps["positive skin smear"] == 1
    assert steps.iloc[-1] == 1
    assert result.attrition["remaining"].iloc[-1] == 2
    assert len(result.attrition) == len(CRITERIA) + 1
