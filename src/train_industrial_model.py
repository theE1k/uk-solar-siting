#!/usr/bin/env python3
"""
Train an industrial self-consumption solar siting model.

Uses FiT non-domestic PV installations as positives and OSM industrial
sites as negatives. Reuses all Phase 2 feature extractors.

Outputs:
  data/processed/industrial_feature_matrix.csv
  data/output/industrial_xgb_model.json
  data/output/industrial_evaluation_report.json
  data/output/industrial_feature_importance.png
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

from src.utils import load_config, resolve_path
from src.phase2_features.extract_solar_irradiance import extract_era5_features
from src.phase2_features.extract_pvgis import extract_pvgis_features
from src.phase2_features.extract_terrain import extract_terrain_features
from src.phase2_features.extract_infrastructure import extract_infrastructure_distances
from src.phase2_features.extract_grid_connection import extract_grid_connection
from src.phase2_features.extract_land_constraints import extract_land_constraints

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT  = Path(__file__).resolve().parent.parent / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def build_feature_matrix(cfg) -> pd.DataFrame:
    matrix_path = PROC / "industrial_feature_matrix.csv"
    if matrix_path.exists():
        print(f"[Industrial] Loading cached feature matrix from {matrix_path}")
        return pd.read_csv(matrix_path)

    labeled_path = PROC / "industrial_labeled_sites.csv"
    sites = pd.read_csv(labeled_path)
    print(f"[Industrial] Building features for {len(sites)} sites …")

    era5    = extract_era5_features(sites, cfg)
    pvgis   = extract_pvgis_features(sites, cfg)
    terrain = extract_terrain_features(sites, cfg)
    infra   = extract_infrastructure_distances(sites, cfg)
    grid    = extract_grid_connection(sites, cfg)
    land    = extract_land_constraints(sites, cfg)

    matrix = pd.concat([sites, era5, pvgis, terrain, infra, grid, land], axis=1)
    before = len(matrix)
    matrix = matrix.dropna(subset=["ghi_pvgis_kwh_m2_day", "elevation_m"]).reset_index(drop=True)
    if before != len(matrix):
        print(f"  Dropped {before - len(matrix)} rows with missing critical features.")

    matrix.to_csv(matrix_path, index=False)
    print(f"[Industrial] Feature matrix: {matrix.shape[0]} rows × {matrix.shape[1]} cols → {matrix_path}")
    return matrix


def train_and_evaluate(matrix: pd.DataFrame, cfg):
    feature_cols = [c for c in matrix.columns
                    if c not in ("latitude", "longitude", "label")]
    X = matrix[feature_cols]
    y = matrix["label"]

    model_cfg = cfg["model"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg["test_size"],
        random_state=model_cfg["random_seed"], stratify=y
    )

    params = {k: v for k, v in model_cfg["xgb_params"].items() if k != "eval_metric"}
    model = xgb.XGBClassifier(**params,
                               random_state=model_cfg["random_seed"],
                               use_label_encoder=False)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    print("[Industrial] Evaluation:", report)

    # Save model
    model_path = OUT / "industrial_xgb_model.json"
    model.save_model(str(model_path))
    print(f"[Industrial] Model saved → {model_path}")

    # Save report
    report_path = OUT / "industrial_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Feature importance plot
    scores = pd.Series(model.get_booster().get_fscore()).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, len(scores) * 0.4)))
    scores.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Industrial Model — Feature Importance (F-score)")
    ax.set_xlabel("F-score")
    plt.tight_layout()
    fig.savefig(OUT / "industrial_feature_importance.png", dpi=150)
    plt.close()
    print(f"[Industrial] Feature importance chart saved")

    return model, report


def main():
    cfg = load_config()
    matrix = build_feature_matrix(cfg)
    model, report = train_and_evaluate(matrix, cfg)

    print("\n=== Industrial Model Summary ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
