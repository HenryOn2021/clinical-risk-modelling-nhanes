#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nhanes_showcase.config import DOCS_DIR, FIGURES_DIR, METRICS_DIR, ensure_project_dirs

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    ensure_project_dirs()

    metrics = load_json(METRICS_DIR / "test_metrics.json")
    sex_fairness = (
        pd.read_csv(METRICS_DIR / "fairness_by_sex.csv")
        if (METRICS_DIR / "fairness_by_sex.csv").exists()
        else pd.DataFrame()
    )

    lines = [
        "# Model card", 
        "", 
        "## Model details", 
        "Binary classifier for self-reported diabetes in NHANES adults.", 
        "", 
        "## Intended use", 
        "Portfolio demonstration of an end-to-end clinical data science", "workflow. Not for clinical deployment.", 
        "", 
        "## Data",
        "NHANES 2017–March 2020 pre-pandemic release. Adult cohort. Target = self-reported diabetes (assumption for this showcase).", 
        "", 
        "## Evaluation summary", 
        f"- ROC AUC: {metrics.get('roc_auc', 'NA')}", 
        f"- Average precision: {metrics.get('average_precision', 'NA')}", 
        f"- Recall / sensitivity: {metrics.get('recall_sensitivity', 'NA')}", 
        f"- Specificity: {metrics.get('specificity', 'NA')}", 
        "", 
        "## Fairness and subgroup notes", 
    ]

    if not sex_fairness.empty:
        lines.extend(["", sex_fairness.to_markdown(index=False), ""])
    else:
        lines.extend(["- Add subgroup tables after running `scripts/run_evaluation.py`."])
        
    lines.extend(
        [
            "",
            "## Limitations",
            "- Cross-sectional survey data, not a longitudinal clinical deployment setting.",
            "- Self-reported target may misclassify true diabetes status.",
            "- Weighted descriptive analyses are appropriate for population summaries; predictive modelling here is a pragmatic portfolio exercise.",
            "- External validation is not included.",
            "",
            "## Saved figures",
            f"- `{FIGURES_DIR/'roc_test.png'}`",
            f"- `{FIGURES_DIR/'calibration_test.png'}`",
            f"- `{FIGURES_DIR/'fairness_tpr_by_sex.png'}`",
        ]
    )
    
    out_path = DOCS_DIR / "model_card.md"
    out_path.write_text("\n".join(lines),encoding="utf-8")
    print(f"Saved model card to {out_path}")

if __name__ == "__main__":
    main()