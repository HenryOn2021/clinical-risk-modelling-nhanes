#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd

from nhanes_showcase.config import METRICS_DIR, PLOT_PATHS, PROCESSED_DIR, ensure_project_dirs
from nhanes_showcase.evaluate import (
    bootstrap_metric_intervals,
    compute_classification_metrics,
    decision_curve_net_benefit,
    save_json,
)
from nhanes_showcase.fairness import disparity_table, fairness_report
from nhanes_showcase.plots import (
    plot_calibration_curve,
    plot_confusion,
    plot_fairness_bars,
    plot_pr_curve,
    plot_roc_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved test predictions")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "test_predictions.parquet"))
    parser.add_argument("--decision-curve", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    ensure_project_dirs()
    pred_df = pd.read_parquet(args.input)
    y_true = pred_df["y_true"]
    y_prob = pred_df["y_prob"]
    y_pred = pred_df["y_pred"]

    if "threshold" not in pred_df.columns:
        raise ValueError("Prediction file is missing the stored decision threshold")
    thresholds = pred_df["threshold"].dropna().unique()
    if thresholds.shape[0] != 1:
        raise ValueError("Prediction file must contain exactly one decision threshold")
    threshold = float(thresholds[0])
    expected_pred = (y_prob >= threshold).astype(int)
    if not expected_pred.equals(y_pred.astype(int)):
        raise ValueError("Stored labels are inconsistent with probabilities and threshold")

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)
    save_json(METRICS_DIR / "test_metrics.json", metrics)
    bootstrap_metric_intervals(
        y_true,
        y_prob,
        threshold,
        n_bootstrap=args.bootstrap_samples,
    ).to_csv(METRICS_DIR / "test_metric_intervals.csv", index=False)

    plot_roc_curve(y_true, y_prob, PLOT_PATHS["roc"])
    plot_pr_curve(y_true, y_prob, PLOT_PATHS["pr"])
    plot_calibration_curve(y_true, y_prob, PLOT_PATHS["calibration"])
    plot_confusion(y_true, y_pred, PLOT_PATHS["confusion"])

    if args.decision_curve:
        decision_curve_net_benefit(y_true, y_prob).to_csv(
            METRICS_DIR / "decision_curve.csv", index=False
        )

    if "sex" in pred_df.columns:
        sex_report = fairness_report(y_true, y_pred, pred_df["sex"], y_prob=y_prob)
        sex_report.to_csv(METRICS_DIR / "fairness_by_sex.csv", index=False)
        disparity_table(sex_report, metric="tpr").to_csv(
            METRICS_DIR / "fairness_gap_sex.csv", index=False
        )
        plot_fairness_bars(
            sex_report, "tpr", PLOT_PATHS["fairness_sex"], "True positive rate by sex"
        )

    if "age_group" in pred_df.columns:
        age_report = fairness_report(y_true, y_pred, pred_df["age_group"], y_prob=y_prob)
        age_report.to_csv(METRICS_DIR / "fairness_by_age_group.csv", index=False)
        disparity_table(age_report, metric="tpr").to_csv(
            METRICS_DIR / "fairness_gap_age_group.csv", index=False
        )
        plot_fairness_bars(
            age_report, "tpr", PLOT_PATHS["fairness_age"], "True positive rate by age group"
        )

    if "race_ethnicity" in pred_df.columns:
        race_report = fairness_report(y_true, y_pred, pred_df["race_ethnicity"], y_prob=y_prob)
        race_report.to_csv(METRICS_DIR / "fairness_by_race_ethnicity.csv", index=False)
        disparity_table(race_report, metric="tpr").to_csv(
            METRICS_DIR / "fairness_gap_race_ethnicity.csv", index=False
        )
        plot_fairness_bars(
            race_report,
            "tpr",
            PLOT_PATHS["fairness_race"],
            "True positive rate by race and ethnicity",
        )

    print(f"Saved evaluation outputs to {METRICS_DIR}")


if __name__ == "__main__":
    main()
