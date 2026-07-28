from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_CV_SPLITS, DEFAULT_TEST_SIZE, RANDOM_STATE


def make_stratified_cv(
    n_splits: int = DEFAULT_CV_SPLITS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def split_dataset(
    X,
    y,
    meta,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    return train_test_split(
        X,
        y,
        meta,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_ml_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def make_search(
    pipeline: Pipeline,
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    n_jobs: int = 1,
) -> GridSearchCV:
    cv = make_stratified_cv(n_splits=n_splits)
    param_grid = {
        "model__C": [0.1, 1.0, 3.0, 10.0],
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )


def fit_search(
    X_train,
    y_train,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    n_jobs: int = 1,
) -> GridSearchCV:
    pipeline = build_ml_pipeline(numeric_cols, categorical_cols)
    search = make_search(pipeline, n_splits=n_splits, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def make_calibrated_estimator(
    estimator,
    *,
    method: str = "sigmoid",
    n_splits: int = DEFAULT_CV_SPLITS,
) -> CalibratedClassifierCV:
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'")
    return CalibratedClassifierCV(
        estimator=clone(estimator),
        cv=make_stratified_cv(n_splits=n_splits),
        method=method,
    )


def calibrate_estimator(
    best_estimator,
    X_train,
    y_train,
    *,
    method: str = "sigmoid",
    n_splits: int = DEFAULT_CV_SPLITS,
):
    calibrated = make_calibrated_estimator(best_estimator, method=method, n_splits=n_splits)
    calibrated.fit(X_train, y_train)
    return calibrated


def out_of_fold_scores(
    estimator,
    X,
    y,
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    n_jobs: int = 1,
):
    probabilities = cross_val_predict(
        clone(estimator),
        X,
        y,
        cv=make_stratified_cv(n_splits=n_splits),
        method="predict_proba",
        n_jobs=n_jobs,
    )
    return probabilities[:, 1]


def predict_scores(estimator, X):
    return estimator.predict_proba(X)[:, 1]


def predict_labels(y_prob, threshold: float):
    return (y_prob >= threshold).astype(int)
