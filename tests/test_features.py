import pandas as pd

from nhanes_showcase.features import define_target, engineer_features

def test_define_target_maps_diabetes_codes():
    df = pd.DataFrame({"DIQ010": [1, 2, 3]})
    out = define_target(df)
    assert out["target_diabetes"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(out.loc[2, "target_diabetes"])

def test_engineer_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "RIDAGEYR": [25, 65],
            "RIAGENDR": [1, 2],
            "RIDRETH3": [3, 4],
            "DMDEDUC2": [4, 5],
            "BMXBMI": [24.0, 33.0],
            "systolic_bp_mean": [118.0, 146.0],
            "diastolic_bp_mean": [76.0, 88.0],
            "INDFMPIR": [2.1, 1.0]
        }
    )
    out = engineer_features(df)
    
    assert {"age_group", "sex", "race_ethnicity", "education_level", "bmi_category"}.issubset(out.columns)