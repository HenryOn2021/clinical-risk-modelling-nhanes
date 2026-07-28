import math

import pandas as pd

from nhanes_showcase.stats import cohort_summary, weighted_mean


def test_weighted_mean_ignores_zero_and_missing_weights():
    value = weighted_mean(
        pd.Series([10.0, 20.0, 30.0]),
        pd.Series([1.0, 0.0, float("nan")]),
    )
    assert value == 10.0


def test_weighted_mean_returns_nan_without_valid_weights():
    assert math.isnan(weighted_mean(pd.Series([1.0]), pd.Series([0.0])))


def test_cohort_summary_includes_unweighted_values_without_weight():
    df = pd.DataFrame(
        {
            "target_diabetes": [0, 1],
            "age_years": [20.0, 60.0],
            "bmi": [22.0, 32.0],
        }
    )
    summary = cohort_summary(df)
    assert "age_years_mean" in summary["metric"].tolist()
    assert "target_prevalence" in summary["metric"].tolist()
