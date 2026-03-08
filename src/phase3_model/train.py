"""Train an XGBoost binary classifier on the feature matrix."""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils import load_config, resolve_path


def train_model(config: dict | None = None) -> XGBClassifier:
    cfg = config or load_config()
    model_cfg = cfg["model"]

    matrix_path = resolve_path(cfg["paths"]["feature_matrix_csv"])
    print(f"[Phase3] Loading feature matrix from {matrix_path} …")
    df = pd.read_csv(matrix_path)

    # Separate features and label
    feature_cols = [c for c in df.columns if c not in ("latitude", "longitude", "label")]
    X = df[feature_cols]
    y = df["label"]

    print(f"[Phase3] Features ({len(feature_cols)}): {feature_cols}")
    print(f"[Phase3] Label distribution: {y.value_counts().to_dict()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_seed"],
        stratify=y,
    )

    # Train XGBoost
    xgb_params = model_cfg["xgb_params"].copy()
    n_estimators = xgb_params.pop("n_estimators", 300)

    model = XGBClassifier(n_estimators=n_estimators, **xgb_params, random_state=model_cfg["random_seed"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Save model
    model_path = resolve_path(cfg["paths"]["model_file"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"[Phase3] Model saved to {model_path}")

    # Save test split for evaluation
    test_path = resolve_path(cfg["paths"]["processed_data"]) / "test_split.csv"
    test_df = X_test.copy()
    test_df["label"] = y_test.values
    test_df["latitude"] = df.loc[X_test.index, "latitude"].values
    test_df["longitude"] = df.loc[X_test.index, "longitude"].values
    test_df.to_csv(test_path, index=False)

    return model


if __name__ == "__main__":
    train_model()
