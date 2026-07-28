import pandas as pd

from nhanes_showcase.clean import (
    apply_clinical_plausibility_rules,
    choose_weight_column,
    standardise_missing_codes,
)


def test_standardise_missing_codes_only_changes_coded_columns():
    df = pd.DataFrame({"DIQ010": [1, 2, 7, 9], "RIDAGEYR": [18, 25, 35, 45]})
    out = standardise_missing_codes(df)
    assert out["DIQ010"].isna().sum() == 2
    assert out["RIDAGEYR"].isna().sum() == 0


def test_choose_weight_column_returns_first_available():
    df = pd.DataFrame({"WTMECPRP": [1.0, 2.0]})
    assert choose_weight_column(df) == "WTMECPRP"


def test_choose_weight_column_uses_fasting_weight_when_requested():
    df = pd.DataFrame({"WTMECPRP": [1.0], "WTSAFPRP": [2.0]})
    assert choose_weight_column(df, fasting_subsample=True) == "WTSAFPRP"


def test_clinical_plausibility_replaces_invalid_values():
    df = pd.DataFrame({"BMXBMI": [25.0, 150.0], "systolic_bp_mean": [120.0, 20.0]})
    out = apply_clinical_plausibility_rules(df)
    assert out["BMXBMI"].isna().sum() == 1
    assert out["systolic_bp_mean"].isna().sum() == 1
