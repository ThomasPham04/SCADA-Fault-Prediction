"""
sequence_metrics.py - training.sequence_metrics
Threshold sweep, evaluation helpers, class weights, and comparison frames.
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
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "support": 0,
            "positive_count": 0,
            "negative_count": 0,
        }
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
    """Return the locked label column when present."""
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


def classifier_comparison_frame(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["f1", "pr_auc", "roc_auc"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def aggregate_event_scores(meta_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Group window-level scores to one row per (asset_id, sequence_id) event."""
    label_col = get_label_column(meta_df)
    df = meta_df[["asset_id", "sequence_id", label_col]].copy()
    df["score"] = np.asarray(scores, dtype=float)
    return (
        df.groupby(["asset_id", "sequence_id"], as_index=False, sort=False)
        .agg(score=("score", "max"), true_label=(label_col, "max"))
    )
