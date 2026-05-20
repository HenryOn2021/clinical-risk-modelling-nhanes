#!/usr/bin/env python
from __future__ import annotations

import argparse
from nhanes_showcase.clean import choose_weight_column
from nhanes_showcase.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DIR, PLOT_PATHS, ensure_project_dirs
from nhanes_showcase.plots import plot_bmi_by_target, plot_missingness_heatmap, plot_target_balance
from nhanes_showcase.stats import cohort_summary, describe_missingness, summarise_subgroups
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight EDA and save figures")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "analysis_dataset.parquet"))
    args = parser.parse_args()

    ensure_project_dirs()
    df = pd.read_parquet(args.input)
    weight_col = choose_weight_column(df)

    describe_missingness(df).to_csv(METRICS_DIR / "missingness_report.csv", index=False)
    cohort_summary(df, weight_col=weight_col).to_csv(METRICS_DIR / "cohort_summary.csv", index=False)

    if "sex" in df.columns:
        summarise_subgroups(df, "sex", weight_col=weight_col).to_csv(METRICS_DIR / "subgroup_sex.csv", index=False)

    if "age_group" in df.columns:
        summarise_subgroups(df, "age_group", weight_col=weight_col).to_csv(METRICS_DIR / "subgroup_age_group.csv", index=False)

    plot_target_balance(df, "target_diabetes", PLOT_PATHS["target_balance"])
    plot_missingness_heatmap(df, PLOT_PATHS["missingness"])
    plot_bmi_by_target(df, PLOT_PATHS["bmi_by_target"])

    print(f"Saved EDA outputs to {FIGURES_DIR} and {METRICS_DIR}")
    
if __name__ == "__main__":
    main()