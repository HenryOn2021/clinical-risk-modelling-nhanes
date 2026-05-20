#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from nhanes_showcase.config import BASELINE_DIR, PLOT_PATHS, PROCESSED_DIR, ensure_project_dirs
from nhanes_showcase.model_baseline import fit_logit_formula, odds_ratio_table, prepare_baseline_frame,save_baseline_outputs
from nhanes_showcase.plots import plot_odds_ratios

def main()->None:
    parser = argparse.ArgumentParser(description="Fit interpretable baseline logistic regression")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "analysis_dataset.parquet"))
    args = parser.parse_args()

    ensure_project_dirs()
    df = pd.read_parquet(args.input)
    baseline_df = prepare_baseline_frame(df)
    result = fit_logit_formula(baseline_df)
    save_baseline_outputs(result, BASELINE_DIR)

    or_df = odds_ratio_table(result)
    or_df.to_csv(BASELINE_DIR / "odds_ratios.csv", index=False)
    plot_odds_ratios(or_df, PLOT_PATHS["odds_ratios"])

    print(f"Saved baseline outputs to {BASELINE_DIR}")
    
if __name__ == "__main__":
    main()