#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nhanes_showcase.config import DOCS_DIR, METRICS_DIR, ensure_project_dirs


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_metric_table(intervals: pd.DataFrame) -> str:
    if intervals.empty:
        return "_Run `scripts/run_evaluation.py` to generate evaluation results._"
    labels = {
        "roc_auc": "ROC AUC",
        "average_precision": "Average precision",
        "brier_score": "Brier score",
        "accuracy": "Accuracy",
        "precision": "Precision / PPV",
        "recall_sensitivity": "Sensitivity / recall",
        "specificity": "Specificity",
        "f1": "F1 score",
        "npv": "Negative predictive value",
    }
    table = intervals.copy()
    table["Metric"] = table["metric"].map(labels).fillna(table["metric"])
    table["Estimate"] = table["estimate"].map(lambda value: f"{value:.3f}")
    table["95% CI"] = table.apply(lambda row: f"{row['ci_low']:.3f}-{row['ci_high']:.3f}", axis=1)
    return table[["Metric", "Estimate", "95% CI"]].to_markdown(index=False)


def format_fairness_table(report: pd.DataFrame) -> str:
    if report.empty:
        return "_No subgroup report is available._"
    table = report.copy()
    for column in ["selection_rate", "tpr", "fpr", "roc_auc"]:
        if column in table.columns:
            table[column] = table[column].map(
                lambda value: "NA" if pd.isna(value) else f"{value:.3f}"
            )
    return table.to_markdown(index=False)


def main() -> None:
    ensure_project_dirs()

    metrics = load_json(METRICS_DIR / "test_metrics.json")
    training = load_json(METRICS_DIR / "training_summary.json")
    intervals = load_csv(METRICS_DIR / "test_metric_intervals.csv")
    fairness_reports = {
        "Sex": load_csv(METRICS_DIR / "fairness_by_sex.csv"),
        "Age group": load_csv(METRICS_DIR / "fairness_by_age_group.csv"),
        "Race and ethnicity": load_csv(METRICS_DIR / "fairness_by_race_ethnicity.csv"),
    }

    model_card_lines = [
        "# Model card: NHANES diabetes screening classifier",
        "",
        "## Model overview",
        "",
        "A calibrated logistic-regression classifier that estimates whether an adult",
        "NHANES participant reported a clinician diagnosis of diabetes (`DIQ010 = 1`).",
        "It is an educational portfolio model, not a diagnostic device or a prospective",
        "risk model.",
        "",
        "## Intended use",
        "",
        "- Demonstrate reproducible clinical-data ingestion, validation and modelling.",
        "- Demonstrate leakage-safe preprocessing, calibration and threshold selection.",
        "- Support technical review of code, assumptions, evaluation and limitations.",
        "- **Not intended for individual clinical decisions or deployment.**",
        "",
        "## Data",
        "",
        "- Source: NHANES 2017-March 2020 pre-pandemic public-use release.",
        f"- Analysis cohort: {training.get('analysis_rows', 'NA')} adults with a usable target.",
        f"- Training set: {training.get('training_rows', 'NA')} participants.",
        f"- Held-out test set: {training.get('test_rows', 'NA')} participants.",
        "- Target: self-reported diagnosed diabetes, not laboratory-confirmed diabetes.",
        "- Default predictors: age, BMI, mean systolic and diastolic blood pressure,",
        "  poverty-income ratio, sex, race/ethnicity and education.",
        "- Fasting glucose is excluded by default because it is target-adjacent and only",
        "  available in a subsample.",
        "",
        "## Training and threshold selection",
        "",
        "- Median imputation and standardisation for numeric variables.",
        "- Most-frequent imputation and one-hot encoding for categorical variables.",
        "- Class-balanced logistic regression.",
        "- Five-fold stratified CV tunes regularisation strength using ROC AUC.",
        "- Sigmoid calibration is fitted within cross-validation.",
        "- The operating threshold is chosen from out-of-fold training probabilities to",
        "  meet the prespecified sensitivity target while maximising specificity.",
        "- The test set remains untouched until final evaluation.",
        "",
        "## Held-out performance",
        "",
        format_metric_table(intervals),
        "",
        (
            f"Operating threshold: **{metrics['threshold']:.4f}**."
            if "threshold" in metrics
            else "Operating threshold: **NA**."
        ),
        "",
        "Percentile confidence intervals are based on 1,000 bootstrap resamples of the",
        "held-out test set. They quantify sampling variation in this test split only.",
        "",
        "## Subgroup evaluation",
        "",
        "These are descriptive checks, not proof of fairness. Small group counts and the",
        "absence of uncertainty intervals mean differences should be interpreted cautiously.",
    ]

    for title, report in fairness_reports.items():
        model_card_lines.extend(
            [
                "",
                f"### {title}",
                "",
                format_fairness_table(report),
            ]
        )

    model_card_lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Cross-sectional classification cannot estimate future diabetes risk.",
            "- The self-reported target is vulnerable to recall error and undiagnosed disease.",
            "- A single random hold-out split is used; there is no external validation.",
            "- NHANES survey weights are used for descriptive prevalence estimates, but the",
            "  predictive model is an unweighted individual-level portfolio analysis.",
            "- The statsmodels baseline uses complete cases and heteroskedasticity-robust",
            "  standard errors; it does not implement the full NHANES complex survey design.",
            "- Race/ethnicity and sex are survey variables with limited categories and should",
            "  not be interpreted as biological causes.",
            "- Threshold utility depends on the costs of false positives and false negatives.",
            "",
            "## Reproducibility",
            "",
            "Run the scripts in the order shown in the repository README. Generated metrics",
            "and figures are versioned for transparent review. The seed, split, features,",
            "hyperparameters and selected operating threshold are recorded in",
            "`artifacts/metrics/training_summary.json`.",
            "",
            "## Figures",
            "",
            "- `reports/figures/roc_test.png`",
            "- `reports/figures/pr_test.png`",
            "- `reports/figures/calibration_test.png`",
            "- `reports/figures/confusion_matrix_test.png`",
        ]
    )

    model_card_path = DOCS_DIR / "model_card.md"
    model_card_path.write_text("\n".join(model_card_lines) + "\n", encoding="utf-8")

    results_lines = [
        "# Reproducible results summary",
        "",
        "This file is generated from the saved evaluation artifacts.",
        "",
        "## Held-out performance",
        "",
        format_metric_table(intervals),
        "",
        f"- Test-set size: {metrics.get('n', 'NA')}",
        f"- Observed prevalence: {metrics.get('prevalence', 'NA')}",
        (
            f"- Selected threshold: {metrics['threshold']:.4f}"
            if "threshold" in metrics
            else "- Selected threshold: NA"
        ),
        f"- TP / FP / TN / FN: {metrics.get('tp', 'NA')} / {metrics.get('fp', 'NA')} / "
        f"{metrics.get('tn', 'NA')} / {metrics.get('fn', 'NA')}",
        "",
        "See [the model card](model_card.md) for intended use and limitations.",
    ]
    results_path = DOCS_DIR / "results_summary.md"
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")

    print(f"Saved model card to {model_card_path}")
    print(f"Saved results summary to {results_path}")


if __name__ == "__main__":
    main()
