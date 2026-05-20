from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE

def split_dataset(X, y, meta, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    return train_test_split(
        X, 
        y, 
        meta, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
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
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def build_ml_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    estimator = LogisticRegression(
        max_iter=2000, 
        class_weight="balanced", 
        random_state=RANDOM_STATE
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

def make_search(pipeline: Pipeline) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {
        "model__C": [0.1, 1.0, 3.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    }
    return GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        scoring="roc_auc", 
        cv=cv, 
        n_jobs=-1, 
        refit=True, 
        verbose=0
    )

def fit_search(X_train, y_train, numeric_cols: list[str], categorical_cols: list[str]) -> GridSearchCV:
    pipeline = build_ml_pipeline(numeric_cols, categorical_cols)
    search = make_search(pipeline)
    search.fit(X_train, y_train)
    return search

def calibrate_estimator(best_estimator, X_train, y_train, method: str = "sigmoid"):
    calibrated = CalibratedClassifierCV(best_estimator, cv=5, method=method)
    calibrated.fit(X_train, y_train)
    return calibrated

def predict_scores(estimator, X):
    return estimator.predict_proba(X)[:, 1]

def predict_labels(y_prob, threshold: float):
    return (y_prob >= threshold).astype(int)