from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import choose_weight_column
from .config import ID_COL, TARGET_COL

SEX_MAP = {1: "Male", 2: "Female"}
RACE_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multiracial",
}

EDUCATION_MAP = {
    1: "<9th grade",
    2: "9-11th grade",
    3: "High school/GED",
    4: "Some college/AA",
    5: "College graduate",
}


def define_target(df: pd.DataFrame, source_col: str = "DIQ010") -> pd.DataFrame:
    if source_col not in df.columns:
        raise KeyError(f"Cannot define target: missing source column {source_col}")
    out = df.copy()
    source = pd.to_numeric(out[source_col], errors="coerce")
    out[TARGET_COL] = source.map({1: 1.0, 2: 0.0})
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_years"] = out["RIDAGEYR"]
    out["age_group"] = pd.cut(
        out["age_years"], bins=[18, 40, 60, np.inf], labels=["18-39", "40-59", "60+"], right=False
    )
    out["sex"] = out["RIAGENDR"].map(SEX_MAP)
    if "RIDRETH3" in out.columns:
        out["race_ethnicity"] = out["RIDRETH3"].map(RACE_MAP)
    if "DMDEDUC2" in out.columns:
        out["education_level"] = out["DMDEDUC2"].map(EDUCATION_MAP)
    out["bmi"] = out["BMXBMI"]
    out["bmi_category"] = pd.cut(
        out["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    )
    out["poverty_income_ratio"] = out["INDFMPIR"] if "INDFMPIR" in out.columns else np.nan
    if "LBXGLU" in out.columns:
        out["fasting_glucose"] = out["LBXGLU"]
    return out


def finalise_analysis_dataset(df: pd.DataFrame, include_glucose: bool = False) -> pd.DataFrame:
    keep = [
        ID_COL,
        TARGET_COL,
        "age_years",
        "age_group",
        "sex",
        "race_ethnicity",
        "education_level",
        "bmi",
        "bmi_category",
        "systolic_bp_mean",
        "diastolic_bp_mean",
        "poverty_income_ratio",
    ]
    if include_glucose and "fasting_glucose" in df.columns:
        keep.append("fasting_glucose")
    weight_col = choose_weight_column(df)
    if weight_col:
        keep.append(weight_col)
    fasting_weight_col = choose_weight_column(df, fasting_subsample=True)
    if include_glucose and fasting_weight_col:
        keep.append(fasting_weight_col)
    existing = [c for c in keep if c in df.columns]
    out = df[existing].copy()
    out = out.loc[out[TARGET_COL].notna()].reset_index(drop=True)
    return out


def get_model_feature_lists(
    df: pd.DataFrame, include_glucose: bool = False
) -> tuple[list[str], list[str]]:
    numeric = [
        "age_years",
        "bmi",
        "systolic_bp_mean",
        "diastolic_bp_mean",
        "poverty_income_ratio",
    ]
    categorical = ["sex", "race_ethnicity", "education_level"]
    if include_glucose and "fasting_glucose" in df.columns:
        numeric.append("fasting_glucose")
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def select_model_data(
    df: pd.DataFrame, include_glucose: bool = False
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], pd.DataFrame]:
    numeric_cols, categorical_cols = get_model_feature_lists(df, include_glucose=include_glucose)
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        raise ValueError("No model features are available in the analysis dataset")
    if TARGET_COL not in df.columns:
        raise KeyError(f"Analysis dataset is missing target column {TARGET_COL}")
    meta_cols = [c for c in [ID_COL, "sex", "race_ethnicity", "age_group"] if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).copy()
    if y.nunique() != 2:
        raise ValueError("The target must contain both binary classes")
    meta = df[meta_cols].copy()
    return X, y, numeric_cols, categorical_cols, meta
