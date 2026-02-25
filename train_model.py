from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Weight (kg)",
    "Daily Water Intake (liters)",
    "Physical Activity Level",
    "Weather",
]

TARGET_CANDIDATES = [
    "Hydration Level",
    "Hydration_Level",
    "hydration_level",
    "HydrationStatus",
    "Hydration Status",
    "Target",
    "Label",
]



def _find_target_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for name in TARGET_CANDIDATES:
        if name in cols:
            return name
    raise ValueError(
        "Could not find the target column. Expected one of: "
        f"{', '.join(TARGET_CANDIDATES)}. "
        f"Found columns: {cols}"
    )


def build_pipeline() -> Pipeline:
    numeric_features = ["Age", "Weight (kg)", "Daily Water Intake (liters)"]
    categorical_features = ["Gender", "Physical Activity Level", "Weather"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=2000)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hydration classifier and save it with joblib."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the Kaggle CSV (daily_water_intake). Example: data/daily_water_intake.csv",
    )
    parser.add_argument(
        "--out",
        default="model/hydration_model.joblib",
        help="Output path for the saved pipeline.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    target_col = _find_target_column(df)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing required feature columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, preds))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"\nSaved model pipeline to: {out_path}")


if __name__ == "__main__":
    main()

