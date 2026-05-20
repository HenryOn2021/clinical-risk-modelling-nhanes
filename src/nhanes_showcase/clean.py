from __future__ import annotations

from mimetypes import init
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ID_COL, WEIGHT_CANDIDATES
from .schema import validate_tables

QUESTIONNAIRE_CODE_COLUMNS = {"DIQ010", "DMDEDUC2"}
CLINICAL_RANGES = {
    "RIDAGEYR":(18,120),
    "BMXBMI":(10,100),
    "LBXGLU":(20,600),
    "systolic_bp_mean":(60,260),
    "diastolic_bp_mean":(30,180),
}

def standardise_missing_codes(
        df:pd.DataFrame,
        coded_columns:Iterable[str] | None=None,
) -> pd.DataFrame:
    coded_columns = set(coded_columns or QUESTIONNAIRE_CODE_COLUMNS)
    out = df.copy()
    missing_codes = {7, 9, 77, 99, 777, 999, 7777, 9999}
    for col in coded_columns:
        if col in out.columns:
            out[col] = out[col].replace(list(missing_codes), np.nan)
    return out

def merge_tables(tables:dict[str,pd.DataFrame]) -> pd.DataFrame:
    merged = tables["DEMO"].copy()
    for name in ["DIQ", "BMX", "BPXO", "GLU"]:
        if name not in tables:
            continue
        other = tables[name].copy()
        keep_cols = [c for c in other.columns if c == ID_COL or c not in merged.columns]
        merged = merged.merge(other[keep_cols], on=ID_COL, how="left", validate="one_to_one")
    return merged

def choose_weight_column(df:pd.DataFrame) -> str | None:
    for col in WEIGHT_CANDIDATES:
        if col in df.columns:
            return col
    return None

def add_mean_blood_pressure(df:pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sys_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in out.columns]
    dia_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in out.columns]
    if sys_cols:
        out["systolic_bp_mean"] = out[sys_cols].mean(axis=1, skipna=True)
    if dia_cols:
        out["diastolic_bp_mean"] = out[dia_cols].mean(axis=1, skipna=True)
    return out

def apply_clinical_plausibility_rules(df:pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (low, high) in CLINICAL_RANGES.items():
        if col in out.columns:
            out.loc[~out[col].between(low, high, inclusive="both"), col] = np.nan
    return out

def flag_outliers_iqr(df:pd.DataFrame, numeric_cols:list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        out[f"{col}_outlier"] = series.lt(low) | series.gt(high)
    return out

def check_unique_person_key(df:pd.DataFrame, key:str=ID_COL) -> None:
    if df[key].duplicated().any():
        raise ValueError(f"Duplicate rows found for {key}")

def build_adult_analysis_frame(tables:dict[str,pd.DataFrame], min_age:int=18) -> pd.DataFrame:
    validate_tables(tables)
    cleaned = {name: standardise_missing_codes(df) for name, df in tables.items()}
    merged = merge_tables(cleaned)
    merged = add_mean_blood_pressure(merged)
    merged = apply_clinical_plausibility_rules(merged)
    merged = merged.loc[merged["RIDAGEYR"].ge(min_age)].copy()
    check_unique_person_key(merged)
    return merged

def summarise_qc(df:pd.DataFrame) -> dict[str,int]:
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_missing_bmi": int(df["BMXBMI"].isna().sum()) if "BMXBMI" in df else 0,
        "n_missing_bp": int(df.get("systolic_bp_mean", pd.Series(dtype=float)).isna().sum()),
    }