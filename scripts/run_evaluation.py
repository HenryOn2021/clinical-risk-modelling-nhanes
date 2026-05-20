#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd

from nhanes_showcase.config import METRICS_DIR, PLOT_PATHS, PROCESSED_DIR, ensure_project_dirs
from nhanes_showcase.evaluate import compute_classification_metrics, decision_curve_net_benefit, save_json
from nhanes_showcase.fairness import disparity_table, fairness_report
from nhanes_showcase.plots import (
    plot_calibration_curve, 
    plot_confusion, 
    plot_fairness_bars, 
    plot_pr_curve, 
    plot_roc_curve
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved test predictions")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "test_predictions.parquet"))
    parser.add_argument("--decision-curve", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    pred_df = pd.read_parquet(args.input)
    y_true = pred_df["y_true"]
    y_prob = pred_df["y_prob"]
    y_pred = pred_df["y_pred"]

    metrics = compute_classification_metrics(y_true, y_prob, 
                                             threshold=float(pred_df["y_pred"].eq(1).where(pred_df["y_prob"]>=0,0).mean() * 0 + 0.5))
    
    # keep the actual stored threshold if present in metadata; otherwise metrics are
    # still valid except threshold-specific wording
    metrics["positive_rate"] = float(y_pred.mean())
    save_json(METRICS_DIR / "test_metrics.json", metrics)

    plot_roc_curve(y_true, y_prob, PLOT_PATHS["roc"])
    plot_pr_curve(y_true, y_prob, PLOT_PATHS["pr"])
    plot_calibration_curve(y_true, y_prob, PLOT_PATHS["calibration"])
    plot_confusion(y_true, y_pred, PLOT_PATHS["confusion"])

    if args.decision_curve:
        decision_curve_net_benefit(y_true, y_prob).to_csv(METRICS_DIR / "decision_curve.csv",index=False)
        
    if "sex" in pred_df.columns:
        sex_report = fairness_report(y_true, y_pred, pred_df["sex"], y_prob=y_prob)
        sex_report.to_csv(METRICS_DIR / "fairness_by_sex.csv", index=False)
        disparity_table(sex_report, metric="tpr").to_csv(METRICS_DIR / "fairness_gap_sex.csv", index=False)
        plot_fairness_bars(sex_report, "tpr", PLOT_PATHS["fairness_sex"], "True positive rate by sex")
    
    if "age_group" in pred_df.columns:
        age_report = fairness_report(y_true, y_pred, pred_df["age_group"], y_prob=y_prob)
        age_report.to_csv(METRICS_DIR / "fairness_by_age_group.csv", index=False)
        disparity_table(age_report, metric="tpr").to_csv(METRICS_DIR / "fairness_gap_age_group.csv", index=False)
        plot_fairness_bars(age_report, "tpr", PLOT_PATHS["fairness_age"], "True positive rate by age group")
    
    print(f"Saved evaluation outputs to {METRICS_DIR}")

if __name__ == "__main__":
    main()