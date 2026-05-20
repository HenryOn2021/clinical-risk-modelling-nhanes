from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COL

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))

def weighted_prevalence(
        df: pd.DataFrame, 
        group_col: str, 
        target_col: str = TARGET_COL, 
        weight_col: str | None = None
) -> pd.DataFrame:
    records = []
    for group, gdf in df.groupby(group_col, dropna=False):
        if weight_col and weight_col in df.columns:
            prev = weighted_mean(gdf[target_col], gdf[weight_col])
        else:
            prev = float(gdf[target_col].mean())
        records.append(
            {
                "group": group, 
                "n": int(gdf.shape[0]), 
                "prevalence": prev, 
                "positive_cases": int(gdf[target_col].sum()),
            }
        )
    return pd.DataFrame(records).reset_index(drop=True)

def describe_missingness(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": df.isna().sum().values, 
            "pct_missing": (df.isna().mean().values * 100).round(2),
        }
    )
    return out.sort_values(["pct_missing", "column"], ascending=[False, True]).reset_index(drop=True)

def cohort_summary(df: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    rows = [{"metric": "n_rows", "value": int(df.shape[0])}]
    for col in ["age_years", "bmi", "systolic_bp_mean"]:
        if col in df.columns:
            if weight_col and weight_col in df.columns:
                value = weighted_mean(df[col], df[weight_col])
                rows.append({"metric": f"{col}_weighted_mean", "value": round(value,3)})
                rows.append({"metric":f"{col}_mean", "value": round(float(df[col].mean()),3)})
    return pd.DataFrame(rows)

def summarise_subgroups(
        df: pd.DataFrame, 
        group_col: str, 
        target_col: str = TARGET_COL, 
        weight_col: str | None = None,
) -> pd.DataFrame:
    summary=weighted_prevalence(df, group_col, target_col=target_col, weight_col=weight_col)
    summary=summary.rename(columns={"group":group_col})
    return summary