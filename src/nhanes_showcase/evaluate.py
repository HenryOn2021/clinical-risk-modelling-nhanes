from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_binary_inputs(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    if truth.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional")
    if truth.shape[0] != probabilities.shape[0] or truth.shape[0] == 0:
        raise ValueError("y_true and y_prob must have the same non-zero length")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob contains non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("y_prob must be between 0 and 1")
    return truth, probabilities


def choose_threshold_for_sensitivity(y_true, y_prob, min_sensitivity: float = 0.80) -> float:
    if not 0 < min_sensitivity <= 1:
        raise ValueError("min_sensitivity must be in the interval (0, 1]")
    y_true, y_prob = _validated_binary_inputs(y_true, y_prob)
    if np.unique(y_true).size != 2:
        raise ValueError("Threshold selection requires both binary classes")

    thresholds = np.unique(y_prob)[::-1]
    best_threshold = float(thresholds[-1])
    best_specificity = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if sensitivity >= min_sensitivity and (
            specificity > best_specificity
            or (np.isclose(specificity, best_specificity) and t > best_threshold)
        ):
            best_threshold = float(t)
            best_specificity = specificity
    return best_threshold


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    y_true, y_prob = _validated_binary_inputs(y_true, y_prob)
    if np.unique(y_true).size != 2:
        raise ValueError("Metric calculation requires both binary classes")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity) if not np.isnan(specificity) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv) if not np.isnan(npv) else np.nan,
        "prevalence": float(y_true.mean()),
        "positive_prediction_rate": float(y_pred.mean()),
        "n": int(y_true.shape[0]),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_metric_intervals(
    y_true,
    y_prob,
    threshold: float,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    truth, probabilities = _validated_binary_inputs(y_true, y_prob)
    rng = np.random.default_rng(random_state)
    metric_names = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "npv",
    ]
    samples = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, truth.shape[0], size=truth.shape[0])
        sample_truth = truth[indices]
        if np.unique(sample_truth).size != 2:
            continue
        sample_metrics = compute_classification_metrics(
            sample_truth, probabilities[indices], threshold=threshold
        )
        for name in metric_names:
            samples[name].append(sample_metrics[name])

    alpha = (1 - confidence) / 2
    point_estimates = compute_classification_metrics(truth, probabilities, threshold)
    rows = []
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": float(point_estimates[name]),
                "ci_low": float(np.nanquantile(values, alpha)),
                "ci_high": float(np.nanquantile(values, 1 - alpha)),
                "confidence_level": float(confidence),
                "bootstrap_replicates": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def calibration_data(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positive": frac_pos})


def decision_curve_net_benefit(
    y_true, y_prob, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    y_true, y_prob = _validated_binary_inputs(y_true, y_prob)
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    n = len(y_true)
    rows = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        _, fp, _, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        model_nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        prevalence = y_true.mean()
        all_nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append(
            {
                "threshold": float(pt),
                "net_benefit_model": float(model_nb),
                "net_benefit_treat_all": float(all_nb),
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_frame(
    y_true, y_prob, threshold: float, meta: pd.DataFrame | None = None
) -> pd.DataFrame:
    y_true, y_prob = _validated_binary_inputs(y_true, y_prob)
    out = pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": (y_prob >= threshold).astype(int),
            "threshold": float(threshold),
        }
    )
    if meta is not None:
        meta = meta.reset_index(drop=True)
        out = pd.concat([out, meta], axis=1)
    return out


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
