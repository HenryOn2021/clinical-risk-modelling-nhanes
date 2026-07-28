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
# # 00 - Project overview
#
# This notebook is the quickest route through the repository. It explains the
# modelling question, locates the main artifacts and reproduces the headline
# result table from files created by the tested command-line pipeline.
#
# **Outcome:** self-reported diagnosed diabetes at the NHANES examination.
#
# **Important:** this is cross-sectional classification, not future-risk
# prediction, diagnosis or a clinical decision tool.

# %%
import json
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

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## Repository map
#
# - `src/nhanes_showcase/`: reusable ingestion, cleaning, feature, modelling and
#   evaluation functions.
# - `scripts/`: ordered command-line pipeline.
# - `data/`: raw, interim and processed data layers.
# - `artifacts/`: baseline, model and metric outputs.
# - `reports/figures/`: viewer-facing plots.
# - `docs/`: methodology, results summary, model card and audit notes.
# - `tests/`: 28 unit and integration-oriented checks.

# %%
important_paths = [
    "README.md",
    "docs/methodology.md",
    "docs/model_card.md",
    "artifacts/metrics/training_summary.json",
    "artifacts/metrics/test_metrics.json",
    "data/processed/test_predictions.parquet",
]

path_check = []
for relative_path in important_paths:
    full_path = PROJECT_ROOT / relative_path
    path_check.append(
        {
            "path": relative_path,
            "exists": full_path.exists(),
            "size_kb": round(full_path.stat().st_size / 1024, 1) if full_path.exists() else None,
        }
    )

display(pd.DataFrame(path_check))

# %% [markdown]
# ## Verified headline results

# %%
with (METRICS_DIR / "test_metrics.json").open(encoding="utf-8") as file:
    test_metrics = json.load(file)

intervals = pd.read_csv(METRICS_DIR / "test_metric_intervals.csv")
metric_labels = {
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "brier_score": "Brier score",
    "recall_sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "npv": "Negative predictive value",
    "accuracy": "Accuracy",
}

result_rows = []
for metric_name, metric_label in metric_labels.items():
    interval_row = intervals.loc[intervals["metric"].eq(metric_name)].iloc[0]
    result_rows.append(
        {
            "Metric": metric_label,
            "Estimate": test_metrics[metric_name],
            "95% CI lower": interval_row["ci_low"],
            "95% CI upper": interval_row["ci_high"],
        }
    )

headline_results = pd.DataFrame(result_rows)
display(headline_results.style.format(precision=3))

# %%
summary = (
    f"**Held-out test set:** {test_metrics['n']:,} adults  \n"
    f"**Selected threshold:** {test_metrics['threshold']:.4f}  \n"
    f"**Confusion matrix counts:** TP={test_metrics['tp']}, "
    f"FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}"
)
display(Markdown(summary))

# %% [markdown]
# ## Core figures

# %%
display(Markdown("### Held-out discrimination"))
display(Image(filename=str(FIGURES_DIR / "roc_test.png"), width=650))
display(Image(filename=str(FIGURES_DIR / "pr_test.png"), width=650))

# %%
display(Markdown("### Probability calibration and operating-point counts"))
display(Image(filename=str(FIGURES_DIR / "calibration_test.png"), width=650))
display(Image(filename=str(FIGURES_DIR / "confusion_matrix_test.png"), width=600))

# %% [markdown]
# ## Correct interpretation
#
# The model separates adults who reported diagnosed diabetes from those who did
# not. The ROC AUC quantifies ranking performance across thresholds; sensitivity
# and specificity describe the selected operating point. Neither quantity shows
# that the model predicts future disease or improves clinical outcomes.
#
# Continue to `01_data_and_cohort.ipynb` for the source-to-cohort pipeline.
