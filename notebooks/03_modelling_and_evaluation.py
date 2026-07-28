# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (.venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 - Modelling and held-out evaluation
#
# This notebook inspects the verified modelling run and reconstructs its key
# checks from saved artifacts.
#
# Sequence:
#
# 1. select predictive features and aligned metadata;
# 2. reproduce the stratified 80/20 split;
# 3. inspect cross-validated regularisation selection;
# 4. inspect calibration and out-of-fold threshold selection;
# 5. recompute held-out metrics;
# 6. review bootstrap uncertainty and subgroup diagnostics.

# %%
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display


def find_project_root() -> Path:
    candidate = Path.cwd().resolve()
    if (candidate / "pyproject.toml").exists():
        return candidate
    if (candidate.parent / "pyproject.toml").exists():
        return candidate.parent
    raise FileNotFoundError("Run this notebook from the repository root or notebooks/.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nhanes_showcase.evaluate import compute_classification_metrics
from nhanes_showcase.features import select_model_data
from nhanes_showcase.model_ml import split_dataset

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# %% [markdown]
# ## 1. Feature set and reproducible split

# %%
analysis = pd.read_parquet(PROCESSED_DIR / "analysis_dataset.parquet")
X, y, numeric_features, categorical_features, metadata = select_model_data(analysis)

X_train, X_test, y_train, y_test, metadata_train, metadata_test = split_dataset(
    X,
    y,
    metadata,
)

split_summary = pd.DataFrame(
    [
        {
            "partition": "Training",
            "rows": X_train.shape[0],
            "positive_prevalence": y_train.mean(),
            "role": "preprocessing, tuning, calibration and threshold selection",
        },
        {
            "partition": "Held-out test",
            "rows": X_test.shape[0],
            "positive_prevalence": y_test.mean(),
            "role": "final evaluation only",
        },
    ]
)
display(split_summary.style.format({"positive_prevalence": "{:.3%}"}))
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# %% [markdown]
# All learned preprocessing is inside the scikit-learn pipeline. Numeric
# medians and scaling parameters, categorical modes and one-hot levels are
# fitted from training folds only.

# %% [markdown]
# ## 2. Cross-validated model selection

# %%
cv_results = pd.read_csv(METRICS_DIR / "cv_results.csv")
cv_view = cv_results[
    ["param_model__C", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score")
display(
    cv_view.style.format(
        {
            "mean_test_score": "{:.3f}",
            "std_test_score": "{:.3f}",
        }
    )
)

# %%
with (METRICS_DIR / "training_summary.json").open(encoding="utf-8") as file:
    training_summary = json.load(file)

training_record = pd.Series(
    {
        "Best regularisation C": training_summary["best_parameters"]["model__C"],
        "Mean CV ROC AUC": training_summary["best_cv_roc_auc_mean"],
        "CV ROC AUC standard deviation": training_summary["best_cv_roc_auc_std"],
        "CV folds": training_summary["cv_splits"],
        "Sensitivity target": training_summary["minimum_sensitivity_target"],
        "Selected threshold": training_summary["selected_threshold"],
    },
    name="value",
)
display(training_record.to_frame())

# %% [markdown]
# ## 3. Calibration and threshold selection
#
# Sigmoid calibration is fitted within stratified cross-validation. The
# operating threshold is then chosen from out-of-fold calibrated training
# probabilities: among thresholds meeting sensitivity >= 0.80, the code selects
# the one with the greatest specificity. The held-out test set is not used in
# this decision.

# %%
oof_metrics = pd.Series(training_summary["oof_threshold_metrics"], name="OOF estimate")
display(
    oof_metrics.loc[
        [
            "threshold",
            "roc_auc",
            "average_precision",
            "recall_sensitivity",
            "specificity",
            "npv",
        ]
    ].to_frame()
)

# %% [markdown]
# ## 4. Held-out metric recomputation

# %%
predictions = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
threshold_values = predictions["threshold"].dropna().unique()
assert threshold_values.shape[0] == 1
threshold = float(threshold_values[0])

expected_labels = predictions["y_prob"].ge(threshold).astype(int)
pd.testing.assert_series_equal(
    expected_labels.reset_index(drop=True),
    predictions["y_pred"].astype(int).reset_index(drop=True),
    check_names=False,
)

recomputed_metrics = compute_classification_metrics(
    predictions["y_true"],
    predictions["y_prob"],
    threshold=threshold,
)

with (METRICS_DIR / "test_metrics.json").open(encoding="utf-8") as file:
    saved_metrics = json.load(file)

for metric_name, metric_value in recomputed_metrics.items():
    assert abs(metric_value - saved_metrics[metric_name]) < 1e-12

metric_table = pd.Series(
    recomputed_metrics,
    name="Held-out estimate",
).loc[
    [
        "roc_auc",
        "average_precision",
        "brier_score",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "npv",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
]
display(metric_table.to_frame())
print("Saved and recomputed held-out metrics match exactly.")

# %% [markdown]
# ## 5. Discrimination, calibration and classification counts

# %%
for figure_name, title in [
    ("roc_test.png", "ROC curve"),
    ("pr_test.png", "Precision-recall curve"),
    ("calibration_test.png", "Calibration curve"),
    ("confusion_matrix_test.png", "Confusion matrix"),
]:
    display(Markdown(f"### {title}"))
    display(Image(filename=str(FIGURES_DIR / figure_name), width=700))

# %% [markdown]
# ## 6. Bootstrap uncertainty
#
# The intervals are percentile intervals from 1,000 seeded bootstrap resamples
# of the held-out test set. They quantify sample variability for this test
# population; they do not address transportability to another dataset.

# %%
intervals = pd.read_csv(METRICS_DIR / "test_metric_intervals.csv")
display(
    intervals[["metric", "estimate", "ci_low", "ci_high", "bootstrap_replicates"]].style.format(
        {
            "estimate": "{:.3f}",
            "ci_low": "{:.3f}",
            "ci_high": "{:.3f}",
        }
    )
)

# %% [markdown]
# ## 7. Subgroup diagnostics

# %%
for group_name in ["sex", "age_group", "race_ethnicity"]:
    report_path = METRICS_DIR / f"fairness_by_{group_name}.csv"
    display(Markdown(f"### {group_name.replace('_', ' ').title()}"))
    display(pd.read_csv(report_path))

# %% [markdown]
# These are descriptive diagnostics, not proof of fairness. The strong
# age-threshold interaction is especially important: a single global threshold
# yields substantially different false-positive rates across age groups.

# %% [markdown]
# ## 8. Optional full model rerun
#
# Leave this flag as `False` when reviewing the repository. Set it to `True` only
# when you intentionally want to overwrite the saved model, predictions, metrics
# and viewer-facing assets with a fresh verified run.

# %%
RUN_TRAINING = False

if RUN_TRAINING:
    commands = [
        [sys.executable, "scripts/run_ml.py", "--minimum-sensitivity", "0.80", "--n-jobs", "1"],
        [
            sys.executable,
            "scripts/run_evaluation.py",
            "--decision-curve",
            "--bootstrap-samples",
            "1000",
        ],
        [sys.executable, "scripts/export_assets.py"],
    ]
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
else:
    print("Training rerun skipped. Set RUN_TRAINING = True to opt in.")

# %% [markdown]
# ## Final interpretation
#
# The calibrated logistic regression achieved ROC AUC 0.796 on 1,885 held-out
# adults. At the training-selected threshold of 0.1357, sensitivity was 0.811,
# specificity 0.662 and negative predictive value 0.951. These estimates support
# the repository's technical demonstration; they do not establish clinical
# utility, diagnostic safety or external validity.
