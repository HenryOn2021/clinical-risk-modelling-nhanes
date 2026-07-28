import numpy as np
import pytest

from nhanes_showcase.evaluate import (
    bootstrap_metric_intervals,
    build_prediction_frame,
    choose_threshold_for_sensitivity,
    compute_classification_metrics,
)


def test_metrics_dictionary_contains_core_keys():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.9]
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert "specificity" in metrics


def test_threshold_for_sensitivity_returns_probability():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.9]
    threshold = choose_threshold_for_sensitivity(y_true, y_prob, min_sensitivity=1.0)
    assert 0.0 <= threshold <= 1.0
    metrics = compute_classification_metrics(y_true, y_prob, threshold)
    assert metrics["recall_sensitivity"] == 1.0


def test_threshold_validation_rejects_invalid_target():
    with pytest.raises(ValueError, match="min_sensitivity"):
        choose_threshold_for_sensitivity([0, 1], [0.2, 0.8], min_sensitivity=0)


def test_prediction_frame_stores_operating_threshold():
    frame = build_prediction_frame([0, 1], [0.3, 0.7], threshold=0.6)
    assert frame["threshold"].nunique() == 1
    assert frame["threshold"].iloc[0] == 0.6
    assert frame["y_pred"].tolist() == [0, 1]


def test_bootstrap_intervals_have_valid_bounds():
    y_true = np.array([0] * 30 + [1] * 10)
    y_prob = np.linspace(0.05, 0.95, 40)
    intervals = bootstrap_metric_intervals(
        y_true,
        y_prob,
        threshold=0.5,
        n_bootstrap=100,
        random_state=1,
    )
    assert {"metric", "estimate", "ci_low", "ci_high"}.issubset(intervals.columns)
    assert (intervals["ci_low"] <= intervals["ci_high"]).all()
