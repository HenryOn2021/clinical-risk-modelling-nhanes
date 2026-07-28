import pandas as pd
import pytest

from nhanes_showcase.fairness import disparity_table, fairness_report


def test_fairness_report_handles_multiple_groups_with_auc():
    report = fairness_report(
        y_true=[0, 1, 0, 1, 0, 1],
        y_pred=[0, 1, 1, 1, 0, 0],
        sensitive_features=["A", "A", "B", "B", "C", "C"],
        y_prob=[0.1, 0.9, 0.6, 0.8, 0.2, 0.4],
    )
    assert report.shape[0] == 3
    assert "roc_auc" in report.columns
    assert report["support"].sum() == 6


def test_disparity_table_computes_gap():
    report = pd.DataFrame({"tpr": [0.5, 0.8, 0.7]})
    gap = disparity_table(report, metric="tpr")
    assert gap.loc[0, "gap"] == pytest.approx(0.3)
