#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle

import pandas as pd

from nhanes_showcase.config import METRICS_DIR, MODELS_DIR,PROCESSED_DIR, ensure_project_dirs
from nhanes_showcase.evaluate import build_prediction_frame, choose_threshold_for_sensitivity
from nhanes_showcase.features import select_model_data
from nhanes_showcase.model_ml import calibrate_estimator, fit_search, predict_scores, split_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn pipeline and save predictions")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "analysis_dataset.parquet"))
    parser.add_argument("--include-glucose", action="store_true")
    parser.add_argument("--minimum-sensitivity", type=float, default=0.80)
    args = parser.parse_args()

    ensure_project_dirs()
    df = pd.read_parquet(args.input)
    X, y, numeric_cols, categorical_cols, meta = select_model_data(df, include_glucose=args.include_glucose)
    
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(X, y, meta)
    search = fit_search(X_train, y_train, numeric_cols, categorical_cols)
    calibrated = calibrate_estimator(search.best_estimator_, X_train, y_train)

    train_prob = predict_scores(calibrated, X_train)
    threshold = choose_threshold_for_sensitivity(y_train, train_prob, min_sensitivity=args.minimum_sensitivity)
    
    test_prob = predict_scores(calibrated, X_test)
    pred_df = build_prediction_frame(y_test, test_prob, threshold=threshold, meta=meta_test)
    pred_df.to_parquet(PROCESSED_DIR/"test_predictions.parquet",index=False)
    
    cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(METRICS_DIR / "cv_results.csv", index=False)

    bundle = {
        "model":calibrated,
        "threshold":threshold,
        "numeric_cols":numeric_cols,
        "categorical_cols":categorical_cols,
        "best_params":search.best_params_,
    }
    with open(MODELS_DIR/"calibrated_logreg.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved model bundle to {MODELS_DIR / 'calibrated_logreg.pkl'}")
    print(f"Saved test predictions to {PROCESSED_DIR / 'test_predictions.parquet'}")

if __name__=="__main__":
    main()