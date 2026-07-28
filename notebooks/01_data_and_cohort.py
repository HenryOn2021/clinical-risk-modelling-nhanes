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
# # 01 - Data ingestion and cohort construction
#
# This notebook walks through the first half of the data pipeline using the
# package functions that also power `scripts/build_dataset.py`.
#
# Stages:
#
# 1. identify the required NHANES source tables;
# 2. load local XPT files;
# 3. validate schemas and one-row-per-person keys;
# 4. merge components and construct the adult cohort;
# 5. define the binary outcome;
# 6. engineer reporting and modelling variables;
# 7. create the final analysis table.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display


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

from nhanes_showcase.clean import build_adult_analysis_frame, summarise_qc
from nhanes_showcase.data_catalog import get_file_catalog
from nhanes_showcase.features import define_target, engineer_features, finalise_analysis_dataset
from nhanes_showcase.ingest import load_local_tables
from nhanes_showcase.schema import validate_tables

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# %% [markdown]
# ## 1. Source catalogue
#
# Fasting glucose is excluded from the default model. It is target-adjacent,
# available only for a fasting subsample and therefore changes the analytic
# interpretation and weighting requirements.

# %%
catalog = get_file_catalog(include_glucose=False)
catalog_table = pd.DataFrame(
    {
        "component": list(catalog.keys()),
        "source_url": list(catalog.values()),
        "local_path": [str(RAW_DIR / f"{name}.xpt") for name in catalog],
    }
)
display(catalog_table)

# %% [markdown]
# ## 2. Load the local XPT tables
#
# `load_local_tables` standardises variable names to uppercase and decodes any
# byte-valued object columns. The ingestion script downloads missing files
# atomically; this notebook remains read-only and uses the supplied local files.

# %%
tables = load_local_tables(raw_dir=RAW_DIR, names=list(catalog))

source_rows = []
for name, table in tables.items():
    source_rows.append(
        {
            "component": name,
            "rows": table.shape[0],
            "columns": table.shape[1],
            "duplicate_person_ids": int(table["SEQN"].duplicated().sum()),
        }
    )

display(pd.DataFrame(source_rows))

# %% [markdown]
# ## 3. Validate before linkage
#
# Validation fails early if a required variable is absent, a participant key is
# duplicated or a protected numeric field lies outside the broad expected
# source range. The one-to-one merge then prevents accidental row multiplication.

# %%
validate_tables(tables)
print("Schema, key uniqueness and source-range validation passed.")

# %% [markdown]
# ## 4. Adult cohort construction
#
# `build_adult_analysis_frame` applies questionnaire missing-code handling,
# one-to-one table linkage, mean blood-pressure calculation, clinical
# plausibility rules and the age restriction (`age >= 18`).

# %%
adult = build_adult_analysis_frame(tables)
print(f"Adults after linkage and age restriction: {adult.shape[0]:,}")
print(f"Columns after linkage and cleaning: {adult.shape[1]}")

# %%
selected_source_columns = [
    "SEQN",
    "RIDAGEYR",
    "DIQ010",
    "BMXBMI",
    "systolic_bp_mean",
    "diastolic_bp_mean",
]
display(adult[selected_source_columns].head())

# %% [markdown]
# ## 5. Target definition
#
# `DIQ010 == 1` maps to positive and `DIQ010 == 2` maps to negative. Borderline,
# refused, unknown and missing responses do not receive a target and are removed
# only when the final analysis table is created.

# %%
adult_with_target = define_target(adult)

target_mapping_check = (
    adult_with_target.groupby("DIQ010", dropna=False)["target_diabetes"]
    .agg(rows="size", mapped_target="mean", mapped_rows="count")
    .reset_index()
)
display(target_mapping_check)

# %% [markdown]
# ## 6. Feature engineering
#
# Human-readable categories support EDA and subgroup analysis. Continuous age
# and BMI remain the predictive variables, avoiding duplicate representations
# of the same information inside the default model.

# %%
featured = engineer_features(adult_with_target)

feature_preview_columns = [
    "age_years",
    "age_group",
    "sex",
    "race_ethnicity",
    "education_level",
    "bmi",
    "bmi_category",
    "poverty_income_ratio",
]
display(featured[feature_preview_columns].head())

# %% [markdown]
# ## 7. Final analysis table and cohort flow

# %%
analysis = finalise_analysis_dataset(featured, include_glucose=False)
qc_summary = summarise_qc(adult_with_target)

cohort_flow = pd.DataFrame(
    [
        {"stage": "Demographics source file", "participants": tables["DEMO"].shape[0]},
        {"stage": "Adults after linkage and age restriction", "participants": adult.shape[0]},
        {"stage": "Adults with usable target", "participants": analysis.shape[0]},
        {
            "stage": "Diabetes-positive",
            "participants": int(analysis["target_diabetes"].sum()),
        },
        {
            "stage": "Diabetes-negative",
            "participants": int(analysis["target_diabetes"].eq(0).sum()),
        },
    ]
)
display(cohort_flow)
display(pd.Series(qc_summary, name="value").to_frame())

# %%
saved_analysis = pd.read_parquet(PROCESSED_DIR / "analysis_dataset.parquet")
pd.testing.assert_frame_equal(
    analysis.reset_index(drop=True),
    saved_analysis.reset_index(drop=True),
    check_dtype=False,
)
print("Reconstructed analysis table matches the saved processed dataset.")

# %% [markdown]
# ## Stage output
#
# The verified default cohort contains **9,421 adults**, including **1,423**
# positive and **7,998** negative examples. Continue to
# `02_eda_and_baseline.ipynb` to inspect missingness, survey-weighted summaries
# and the interpretable association model.
