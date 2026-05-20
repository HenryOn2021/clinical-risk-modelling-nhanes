import pandas as pd

from nhanes_showcase.model_ml import build_ml_pipeline

def test_pipeline_fits_small_dataframe():
    X = pd.DataFrame(
        {
            "age_years": [25, 40, 60, 55],
            "bmi": [22.0, 27.0, 35.0, 30.0],
            "systolic_bp_mean": [110.0, 130.0, 150.0, 145.0],
            "sex": ["Male", "Female", "Male", "Female"],
            "age_group": ["18-39", "40-59", "60+", "40-59"]
        }
    )
    y = [0, 0, 1, 1]
    pipe = build_ml_pipeline(numeric_cols=["age_years", "bmi", "systolic_bp_mean"], categorical_cols=["sex", "age_group"])
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]
    assert len(probs) == 4