from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid")

def savefig(fig, path:str | Path) -> None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path,dpi=300,bbox_inches="tight")

def plot_target_balance(df: pd.DataFrame, target_col: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts(dropna=False).sort_index()
    ax.bar([str(x) for x in counts.index], counts.values)
    ax.set_title("Target balance")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    savefig(fig, path)
    plt.close(fig)

def plot_missingness_heatmap(df: pd.DataFrame, path: str | Path, max_rows: int = 300) -> None:
    sample = df.head(max_rows).isna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(sample.T, cbar=False, ax=ax)
    ax.set_title("Missingness heatmap")
    ax.set_xlabel("Row sample")
    ax.set_ylabel("Column")
    savefig(fig, path)
    plt.close(fig)

def plot_bmi_by_target(df: pd.DataFrame, path: str | Path) -> None:
    if not {"bmi", "target_diabetes"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="bmi", hue="target_diabetes", kde=True, ax=ax)
    ax.set_title("BMI distribution by target")
    savefig(fig, path)
    plt.close(fig)

def plot_odds_ratios(or_df: pd.DataFrame, path: str | Path, top_n: int = 15) -> None:
    plot_df = or_df.head(top_n).sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(plot_df) + 1)))
    ax.errorbar(x=plot_df["odds_ratio"], y=plot_df["term"], xerr=[plot_df["odds_ratio"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["odds_ratio"]], fmt="o")
    ax.axvline(1.0, linestyle="--")
    ax.set_title("Baseline odds ratios")
    ax.set_xlabel("Odds ratio")
    savefig(fig, path)
    plt.close(fig)

def plot_roc_curve(y_true, y_prob, path: str | Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    savefig(fig, path)
    plt.close(fig)

def plot_pr_curve(y_true, y_prob, path: str | Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    savefig(fig, path)
    plt.close(fig)

def plot_calibration_curve(y_true, y_prob, path: str | Path, n_bins: int = 10) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration curve")
    savefig(fig, path)
    plt.close(fig)

def plot_confusion(y_true, y_pred, path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax)
    ax.set_title("Confusion matrix")
    savefig(fig, path)
    plt.close(fig)

def plot_fairness_bars(report_df: pd.DataFrame, metric_col: str, path: str | Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = report_df.sort_values(metric_col)
        ax.bar(plot_df["group"].astype(str), plot_df[metric_col].astype(float))
        ax.set_title(title)
        ax.set_ylabel(metric_col)
        ax.set_xlabel("Group")
        ax.tick_params(axis="x",rotation=30)
        savefig(fig, path)
        plt.close(fig)