#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd

from nhanes_showcase.config import (
    DEFAULT_CV_SPLITS,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    ensure_project_dirs,
)
from nhanes_showcase.evaluate import (
    build_prediction_frame,
    choose_threshold_for_sensitivity,
    compute_classification_metrics,
)
from nhanes_showcase.features import select_model_data
from nhanes_showcase.model_ml import (
    calibrate_estimator,
    fit_search,
    make_calibrated_estimator,
    out_of_fold_scores,
    predict_scores,
    split_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn pipeline and save predictions")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "analysis_dataset.parquet"))
    parser.add_argument("--include-glucose", action="store_true")
    parser.add_argument("--minimum-sensitivity", type=float, default=0.80)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for cross-validation; 1 is safest in constrained environments",
    )
    args = parser.parse_args()
    if not 0 < args.minimum_sensitivity <= 1:
        parser.error("--minimum-sensitivity must be in the interval (0, 1]")
    if args.cv_splits < 3:
        parser.error("--cv-splits must be at least 3")

    ensure_project_dirs()
    df = pd.read_parquet(args.input)
    X, y, numeric_cols, categorical_cols, meta = select_model_data(
        df, include_glucose=args.include_glucose
    )

    X_train, X_test, y_train, y_test, _meta_train, meta_test = split_dataset(X, y, meta)
    search = fit_search(
        X_train,
        y_train,
        numeric_cols,
        categorical_cols,
        n_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )
    calibration_template = make_calibrated_estimator(
        search.best_estimator_, n_splits=args.cv_splits
    )
    train_oof_prob = out_of_fold_scores(
        calibration_template,
        X_train,
        y_train,
        n_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )
    threshold = choose_threshold_for_sensitivity(
        y_train,
        train_oof_prob,
        min_sensitivity=args.minimum_sensitivity,
    )
    threshold_metrics = compute_classification_metrics(y_train, train_oof_prob, threshold=threshold)
    calibrated = calibrate_estimator(
        search.best_estimator_,
        X_train,
        y_train,
        n_splits=args.cv_splits,
    )

    test_prob = predict_scores(calibrated, X_test)
    pred_df = build_prediction_frame(y_test, test_prob, threshold=threshold, meta=meta_test)
    pred_df.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(METRICS_DIR / "cv_results.csv", index=False)
    best_index = int(search.best_index_)
    training_summary = {
        "analysis_rows": int(df.shape[0]),
        "training_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "training_prevalence": float(y_train.mean()),
        "test_prevalence": float(y_test.mean()),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "best_parameters": search.best_params_,
        "best_cv_roc_auc_mean": float(search.cv_results_["mean_test_score"][best_index]),
        "best_cv_roc_auc_std": float(search.cv_results_["std_test_score"][best_index]),
        "minimum_sensitivity_target": float(args.minimum_sensitivity),
        "selected_threshold": float(threshold),
        "threshold_selection_source": "out-of-fold calibrated training predictions",
        "oof_threshold_metrics": threshold_metrics,
        "random_state": 42,
        "cv_splits": int(args.cv_splits),
    }
    (METRICS_DIR / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )

    bundle = {
        "model": calibrated,
        "threshold": threshold,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "best_params": search.best_params_,
        "include_glucose": args.include_glucose,
    }
    with open(MODELS_DIR / "calibrated_logreg.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved model bundle to {MODELS_DIR / 'calibrated_logreg.pkl'}")
    print(
        f"Selected threshold {threshold:.4f} "
        f"(OOF sensitivity={threshold_metrics['recall_sensitivity']:.3f}, "
        f"specificity={threshold_metrics['specificity']:.3f})"
    )
    print(f"Saved test predictions to {PROCESSED_DIR / 'test_predictions.parquet'}")


if __name__ == "__main__":
    main()
