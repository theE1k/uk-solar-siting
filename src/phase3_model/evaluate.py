"""Evaluate the trained model: metrics, ROC curve, and feature importance."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, roc_curve,
)
from xgboost import XGBClassifier

from src.utils import load_config, resolve_path


def evaluate_model(config: dict | None = None):
    cfg = config or load_config()

    # Load model
    model_path = resolve_path(cfg["paths"]["model_file"])
    model = XGBClassifier()
    model.load_model(str(model_path))
    print(f"[Phase3] Loaded model from {model_path}")

    # Load test data
    test_path = resolve_path(cfg["paths"]["processed_data"]) / "test_split.csv"
    df = pd.read_csv(test_path)
    feature_cols = [c for c in df.columns if c not in ("latitude", "longitude", "label")]
    X_test = df[feature_cols]
    y_test = df["label"]

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    print("[Phase3] Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n" + classification_report(y_test, y_pred, target_names=["Not suitable", "Suitable"]))

    # Save metrics
    report_path = resolve_path(cfg["paths"]["evaluation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Solar Site Suitability")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_path = resolve_path(cfg["paths"]["roc_curve_png"])
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase3] ROC curve saved to {roc_path}")

    # --- Feature Importance ---
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.Series(importance)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.4)))
    imp_df.plot.barh(ax=ax, color="#2196F3")
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (Gain)")
    ax.grid(axis="x", alpha=0.3)
    fi_path = resolve_path(cfg["paths"]["feature_importance_png"])
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase3] Feature importance chart saved to {fi_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
