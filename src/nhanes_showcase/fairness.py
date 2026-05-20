from __future__ import annotations

import pandas as pd
from pandas.core import series
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from sklearn.metrics import roc_auc_score

def _safe_group_auc(y_true:pd.Series,y_prob:pd.Series)->float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true,y_prob))

def fairness_report(
        y_true,
        y_pred,
        sensitive_features,
        y_prob=None,
) -> pd.DataFrame:
    sensitive= pd.Series(sensitive_features).fillna("Missing").reset_index(drop=True)
    y_true=pd.Series(y_true).reset_index(drop=True)
    y_pred=pd.Series(y_pred).reset_index(drop=True)

    mf = MetricFrame(
        metrics={
            "selection_rate":selection_rate,
            "tpr":true_positive_rate,
            "fpr":false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    report = mf.by_group.reset_index()
    report.columns=["group", "selection_rate", "tpr", "fpr"]
    report["support"] = sensitive.value_counts(dropna=False).reindex(report["group"]).values
    
    if y_prob is not None:
        probs = pd.Series(y_prob).reset_index(drop=True)
        auc_rows = []
        grouped = pd.DataFrame({"group": sensitive,"y_true": y_true,"y_prob": probs}).groupby("group")
        for group, gdf in grouped:
            auc_rows.append({"group": group, "roc_auc": _safe_group_auc(gdf["y_true"],gdf["y_prob"])})
            report = report.merge(pd.DataFrame(auc_rows), on="group", how="left")
            
    return report.sort_values("group", na_position="last").reset_index(drop=True)

def disparity_table(report_df: pd.DataFrame, metric:str = "tpr") -> pd.DataFrame:
    series = report_df[metric].dropna()
    return pd.DataFrame(
        {
            "metric": [metric],
            "min": [float(series.min()) if not series.empty else float("nan")], 
            "max": [float(series.max()) if not series.empty else float("nan")], 
            "gap": [float(series.max() - series.min()) if len(series) else float("nan")],
        }
    )