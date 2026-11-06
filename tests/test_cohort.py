import math

import pandas as pd

from leprosy_similarity.cohort import categorical_level, lesion_tier, load_raw_cohort, plus_scale, reaction_flag
from leprosy_similarity.reference_cases import categorical_share, numeric_value, parse_values


def test_graded_results():
    assert plus_scale("Negativo") == 0.0
    assert plus_scale("+++") == 3.0
    assert plus_scale("4+") == 4.0
    assert math.isnan(plus_scale(""))


def test_lesion_tiers_and_reactions():
    assert lesion_tier("1 a 5") == 1.0
    assert lesion_tier("6 a 11") == 2.0
    assert lesion_tier("Infiltração Difusa") == 3.0
    assert reaction_flag("Tipo 1")("Reação Tipo 1") == 1.0
    assert reaction_flag("Tipo 2")("Reação Tipo 1") == 0.0
    assert reaction_flag("Tipo 1")("Sem Reação") == 0.0


def test_categorical_spelling_variants():
    assert categorical_level("lesion_color", "Hipocrômica") == "hipocromica"
    assert categorical_level("lesion_color", "hiper") == "hipercromica"
    assert categorical_level("lesion_element", "Infiltração difusa") == "infiltracao_difusa"
    assert categorical_level("lesion_element", "unknown") is None


def test_reference_case_cells():
    assert parse_values("0 ou 1", cap=6) == [0, 1]
    assert parse_values("1 a 5", cap=15) == [1, 2, 3, 4, 5]
    assert parse_values("2026-05-01 00:00:00", cap=15) == [1, 2, 3, 4, 5]
    assert parse_values("4+", cap=6) == [4, 5, 6]
    assert numeric_value("NU_LESOES", "10 ou 15") == 2.5
    share = categorical_share("lesion_color", "eritematosa ou normocromica")
    assert share["lesion_color__eritematosa"] == 0.5 and share["lesion_color__hipocromica"] == 0.0


def test_raw_loader_keeps_labelled_rows_and_drops_goiania(tmp_path):
    raw = pd.DataFrame(
        {
            "Forma Clínica": ["TT", "LL", "Indeterminada", "BB"],
            "Número de Lesões": ["1 a 5", "Infiltração Difusa", "1 a 5", "6 a 11"],
            "Tipo de Reação": ["Sem Reação", "Reação Tipo 2", None, "Reação Tipo 1"],
            "Indice Baciloscópico": ["Negativo", "5+", None, "++"],
            "Forma de Tratamento": ["6 doses", "12 doses", None, "12 doses"],
        }
    )
    path = tmp_path / "raw.xlsx"
    raw.to_excel(path, index=False)
    cohort = load_raw_cohort([path], site_blocks=(("bauru", 3), ("goiania", 1)))
    assert cohort["class"].tolist() == ["TT", "LL"]
    assert cohort["site"].tolist() == ["bauru", "bauru"]
    assert cohort["BACILOSCOP_N"].tolist() == [0.0, 5.0]
    assert cohort["who_class"].tolist() == ["PB", "MB"]
    assert cohort["record_id"].tolist() == [0, 1]
