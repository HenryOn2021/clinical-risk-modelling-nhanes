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
    roc_auc_score
)

def choose_threshold_for_sensitivity(y_true, y_prob, min_sensitivity: float = 0.80) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.unique(np.round(y_prob, 6))[::-1]
    best_threshold = 0.5
    best_specificity = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if sensitivity >= min_sensitivity and specificity > best_specificity:
            best_threshold = float(t)
            best_specificity = specificity
    return best_threshold

def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str,float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
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
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def calibration_data(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": mean_pred,
                         "fraction_positive": frac_pos})

def decision_curve_net_benefit(y_true, y_prob, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    n = len(y_true)
    rows = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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

def build_prediction_frame(y_true, y_prob, threshold: float, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "y_prob": np.asarray(y_prob).astype(float),
            "y_pred": (np.asarray(y_prob) >= threshold).astype(int),
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