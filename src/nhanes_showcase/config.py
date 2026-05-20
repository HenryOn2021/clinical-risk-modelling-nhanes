from __future__ import annotations

from pathlib import Path

ID_COL = "SEQN"
TARGET_COL = "target_diabetes"
RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT_DIR / "docs"

WEIGHT_CANDIDATES = ["WTMECPRP", "WTMEC2YR", "WTINTPRP", "WTINT2YR"]

PLOT_PATHS={
    "target_balance": FIGURES_DIR / "target_balance.png",
    "missingness": FIGURES_DIR / "missingness_heatmap.png",
    "bmi_by_target": FIGURES_DIR / "bmi_by_target.png",
    "odds_ratios": FIGURES_DIR / "odds_ratios.png",
    "roc": FIGURES_DIR / "roc_test.png",
    "pr": FIGURES_DIR / "pr_test.png",
    "calibration": FIGURES_DIR / "calibration_test.png",
    "confusion": FIGURES_DIR / "confusion_matrix_test.png",
    "fairness_sex": FIGURES_DIR / "fairness_tpr_by_sex.png",
    "fairness_age": FIGURES_DIR / "fairness_tpr_by_age_group.png",
}

def ensure_project_dirs()->None:
    for path in [
        RAW_DIR, 
        INTERIM_DIR, 
        PROCESSED_DIR,
        MODELS_DIR, 
        BASELINE_DIR, 
        METRICS_DIR, 
        FIGURES_DIR, 
        DOCS_DIR
    ]:
        path.mkdir(parents=True, exist_ok=True)