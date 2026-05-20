from nhanes_showcase.evaluate import choose_threshold_for_sensitivity, compute_classification_metrics

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