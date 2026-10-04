"""Data preparation, model training, and inference for churn prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _is_identifier(column: str) -> bool:
    normalized = column.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"id", "customerid", "customer_id"} or normalized.endswith(
        "_id"
    )


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    features.columns = features.columns.astype(str).str.strip()
    features = features.loc[:, ~features.columns.map(_is_identifier)]
    features = features.replace(r"^\s*$", np.nan, regex=True)

    for column in features.columns:
        if not (
            pd.api.types.is_object_dtype(features[column].dtype)
            or pd.api.types.is_string_dtype(features[column].dtype)
        ):
            continue
        converted = pd.to_numeric(features[column], errors="coerce")
        non_missing = features[column].notna().sum()
        if non_missing and converted.notna().sum() / non_missing >= 0.9:
            features[column] = converted
    return features


def _encode_target(target: pd.Series) -> pd.Series:
    if target.isna().any():
        raise ValueError("The target column contains missing values.")

    normalized = target.astype(str).str.strip().str.lower()
    values = set(normalized.unique())
    if values == {"yes", "no"}:
        return normalized.map({"no": 0, "yes": 1}).astype(int)
    if values == {"true", "false"}:
        return normalized.map({"false": 0, "true": 1}).astype(int)
    if values == {"0", "1"}:
        return normalized.astype(int)
    if len(values) != 2:
        raise ValueError(
            f"The target must have exactly two classes; found {sorted(values)}."
        )

    first, second = sorted(values)
    print(f"Target classes mapped as {first!r} -> 0, {second!r} -> 1.")
    return normalized.map({first: 0, second: 1}).astype(int)


def _build_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def train_model(
    data_path: str | Path,
    target_column: str = "Churn",
    output_dir: str | Path = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare three classifiers, select by churn F1, and save model and metrics."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    frame.columns = frame.columns.astype(str).str.strip()
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column {target_column!r} not found. "
            f"Available columns: {frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError("The dataset has no rows.")

    y = _encode_target(frame.pop(target_column))
    X = _prepare_features(frame)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding IDs.")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class must contain at least two rows.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError as error:
        raise ValueError(f"Could not split the dataset: {error}") from error

    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in estimators.items():
        model = _build_pipeline(X_train, estimator)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        fitted_models[name] = model

    best_name = max(
        results, key=lambda name: (results[name]["f1"], results[name]["roc_auc"])
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": fitted_models[best_name],
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "selected_model": best_name,
        },
        output_path / "churn_model.joblib",
    )
    report = {
        "dataset": str(data_path),
        "rows": int(len(frame)),
        "target_column": target_column,
        "positive_class": "churn",
        "selected_model": best_name,
        "test_size": test_size,
        "random_state": random_state,
        "metrics": results,
    }
    (output_path / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def predict_file(
    data_path: str | Path,
    model_path: str | Path = "artifacts/churn_model.joblib",
    output_path: str | Path = "artifacts/predictions.csv",
) -> Path:
    """Predict churn labels and probabilities for rows in a CSV file."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Prediction dataset not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    artifact = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    features = _prepare_features(frame)
    expected_columns = artifact["feature_columns"]
    features = features.reindex(columns=expected_columns)

    model = artifact["model"]
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    results = pd.DataFrame(
        {
            "churn_prediction": np.where(predictions == 1, "Yes", "No"),
            "churn_probability": probabilities,
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path
