"""
sequence_metrics.py — training.sequence_metrics
Threshold sweep, evaluation helpers, class weights, and comparison-frame builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_class_weights(y: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        return {0: 1.0, 1: 1.0}
    total = counts.sum()
    return {
        0: float(total / (2.0 * counts[0])),
        1: float(total / (2.0 * counts[1])),
    }


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    pr_auc = safe_pr_auc(y_true, y_score)
    roc_auc = safe_roc_auc(y_true, y_score)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
    }


def get_label_column(meta_df: pd.DataFrame) -> str:
    """Return the locked future-horizon label column when present."""
    if "target_label" in meta_df.columns:
        return "target_label"
    if "future_label" in meta_df.columns:
        return "future_label"
    return "last_label"


def sweep_thresholds(y_true: np.ndarray, y_score: np.ndarray, thresholds) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= float(threshold)).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "accuracy"])
    return pd.DataFrame(rows)


def pick_best_threshold(sweep_df: pd.DataFrame) -> dict:
    if sweep_df.empty:
        raise ValueError("Threshold sweep is empty.")
    ranked = sweep_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def safe_mean(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return float(numeric.mean())


def classifier_comparison_frame(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["model_name", "threshold", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
        )
    return pd.DataFrame(rows).sort_values(
        ["f1", "pr_auc", "roc_auc"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def autoencoder_comparison_frame(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "scope",
                "asset_count",
                "threshold",
                "macro_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "macro_pr_auc",
                "macro_roc_auc",
                "pooled_accuracy",
                "pooled_precision",
                "pooled_recall",
                "pooled_f1",
                "pooled_pr_auc",
                "pooled_roc_auc",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "macro_pr_auc", "macro_roc_auc"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def compute_autoencoder_architecture_summary(model_name: str, asset_results: list) -> dict:
    if not asset_results:
        raise ValueError(f"No asset results available for {model_name}.")

    asset_df = pd.DataFrame([row["summary"] for row in asset_results])
    pooled_scores = np.concatenate([row["test_window_scores"] for row in asset_results]).astype(np.float32)
    pooled_labels = np.concatenate([row["test_window_labels"] for row in asset_results]).astype(np.int8)
    pooled_pred = np.concatenate([row["test_window_pred"] for row in asset_results]).astype(np.int8)

    pooled_precision = precision_score(pooled_labels, pooled_pred, zero_division=0)
    pooled_recall = recall_score(pooled_labels, pooled_pred, zero_division=0)
    pooled_f1 = f1_score(pooled_labels, pooled_pred, zero_division=0)
    pooled_accuracy = float((pooled_labels == pooled_pred).mean()) if len(pooled_labels) else 0.0

    macro_accuracy = float(asset_df["accuracy"].mean()) if "accuracy" in asset_df.columns else None
    macro_pr_auc = safe_mean(asset_df["pr_auc"]) if "pr_auc" in asset_df.columns else None
    macro_roc_auc = safe_mean(asset_df["roc_auc"]) if "roc_auc" in asset_df.columns else None

    return {
        "model_name": model_name,
        "scope": "per_asset",
        "asset_count": int(len(asset_df)),
        "macro_accuracy": macro_accuracy,
        "macro_precision": float(asset_df["precision"].mean()),
        "macro_recall": float(asset_df["recall"].mean()),
        "macro_f1": float(asset_df["f1"].mean()),
        "macro_pr_auc": macro_pr_auc,
        "macro_roc_auc": macro_roc_auc,
        "pooled_accuracy": pooled_accuracy,
        "pooled_precision": float(pooled_precision),
        "pooled_recall": float(pooled_recall),
        "pooled_f1": float(pooled_f1),
        "pooled_pr_auc": safe_pr_auc(pooled_labels, pooled_scores),
        "pooled_roc_auc": safe_roc_auc(pooled_labels, pooled_scores),
    }
