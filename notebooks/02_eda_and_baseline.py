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
# # 02 - Exploratory analysis and statistical baseline
#
# This notebook separates two related tasks:
#
# - **description:** missingness, distributions and weighted prevalence;
# - **association modelling:** complete-case logistic regression with robust
#   standard errors.
#
# The descriptive survey weights improve population summaries. They do not turn
# the predictive model or bootstrap evaluation into full complex-survey
# inference.

# %%
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

from nhanes_showcase.clean import choose_weight_column
from nhanes_showcase.stats import cohort_summary, describe_missingness, summarise_subgroups

ANALYSIS_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_dataset.parquet"
BASELINE_DIR = PROJECT_ROOT / "artifacts" / "baseline"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

analysis = pd.read_parquet(ANALYSIS_PATH)
weight_column = choose_weight_column(analysis)

print(f"Analysis rows: {analysis.shape[0]:,}")
print(f"Examination weight: {weight_column}")

# %% [markdown]
# ## 1. Data types and missingness

# %%
schema_view = pd.DataFrame(
    {
        "column": analysis.columns,
        "dtype": analysis.dtypes.astype(str).values,
        "non_missing": analysis.notna().sum().values,
        "unique_values": analysis.nunique(dropna=True).values,
    }
)
display(schema_view)

# %%
missingness = describe_missingness(analysis)
display(missingness)
display(Image(filename=str(FIGURES_DIR / "missingness_heatmap.png"), width=850))

# %% [markdown]
# Missing predictive values are retained at this stage. Median and
# most-frequent imputers are learned later inside each training fold, preventing
# information from the validation or test sets from leaking into preprocessing.

# %% [markdown]
# ## 2. Weighted and unweighted cohort summaries

# %%
display(cohort_summary(analysis, weight_col=weight_column))

# %%
for group_column in ["sex", "age_group", "race_ethnicity"]:
    display(Markdown(f"### Diabetes prevalence by `{group_column}`"))
    group_summary = summarise_subgroups(
        analysis,
        group_column,
        weight_col=weight_column,
    )
    display(group_summary)

# %% [markdown]
# The NHANES examination-weighted diabetes prevalence is lower than the
# unweighted cohort prevalence. The weights are used for population description,
# whereas the classifier is trained to predict individuals in the analytic
# sample.

# %% [markdown]
# ## 3. Distribution plots

# %%
display(Image(filename=str(FIGURES_DIR / "target_balance.png"), width=650))
display(Image(filename=str(FIGURES_DIR / "bmi_by_target.png"), width=750))

# %% [markdown]
# ## 4. Interpretable statistical baseline
#
# The baseline is a separate complete-case `statsmodels` logistic regression.
# It estimates adjusted associations, reports HC3 robust standard errors and is
# not used to generate the held-out machine-learning predictions.

# %%
baseline_cohort = pd.read_json(BASELINE_DIR / "baseline_cohort.json", typ="series")
display(baseline_cohort.rename("value").to_frame())

odds_ratios = pd.read_csv(BASELINE_DIR / "odds_ratios.csv")
selected_terms = odds_ratios.loc[
    odds_ratios["term"].isin(["age_years", "bmi", "C(sex)[T.Male]"]),
    ["term", "odds_ratio", "ci_low", "ci_high", "p_value"],
]
display(selected_terms.style.format(precision=3))

# %%
display(Image(filename=str(FIGURES_DIR / "odds_ratios.png"), width=850))

# %% [markdown]
# ## Interpretation boundaries
#
# An odds ratio above one is an adjusted association with the contemporaneous
# self-reported outcome. It is not a causal effect, a risk ratio or proof that
# changing the predictor would change diabetes status.
#
# Continue to `03_modelling_and_evaluation.ipynb` for leakage-safe modelling,
# calibration, thresholding and held-out evaluation.
