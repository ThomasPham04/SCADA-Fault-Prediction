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
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "pr_auc": float("nan"), "roc_auc": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "support": 0, "positive_count": 0, "negative_count": 0,
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


def aggregate_event_scores(meta_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Group window-level scores to one row per (asset_id, sequence_id) event."""
    label_col = get_label_column(meta_df)
    df = meta_df[["asset_id", "sequence_id", label_col]].copy()
    df["score"] = np.asarray(scores, dtype=float)
    return (
        df.groupby(["asset_id", "sequence_id"], as_index=False, sort=False)
        .agg(score=("score", "max"), true_label=(label_col, "max"))
    )


def sweep_thresholds_event_level(
    val_meta: pd.DataFrame, val_scores: np.ndarray, thresholds
) -> pd.DataFrame:
    """Threshold sweep where each sequence counts as one prediction (max-score aggregation)."""
    event_df = aggregate_event_scores(val_meta, val_scores)
    return sweep_thresholds(
        event_df["true_label"].to_numpy(dtype=int),
        event_df["score"].to_numpy(dtype=float),
        thresholds,
    )


def evaluate_event_level_from_sequences(
    test_sequences: list,
    all_window_scores: list,
    threshold: float,
) -> dict:
    """Event-level evaluation: flag a sequence as anomaly if max window score >= threshold."""
    event_true = []
    event_scores = []
    for seq, scores in zip(test_sequences, all_window_scores):
        event_true.append(int(np.asarray(seq["y"]).max()))
        event_scores.append(float(np.asarray(scores).max()))
    return evaluate_at_threshold(
        np.array(event_true, dtype=np.int8),
        np.array(event_scores, dtype=np.float32),
        threshold,
    )


def compute_lead_time_minutes(
    test_sequences: list,
    all_window_scores: list,
    threshold: float,
) -> float | None:
    """
    Mean lead time (minutes) for correctly detected anomaly events.

    Lead time = first_future_anomaly_time − end_time of the first flagged window.
    Only sequences that are positive AND are detected contribute.
    """
    import pandas as _pd

    lead_times = []
    for seq, scores in zip(test_sequences, all_window_scores):
        y = np.asarray(seq["y"])
        if int(y.max()) == 0:
            continue
        scores = np.asarray(scores)
        flagged = scores >= threshold
        if not flagged.any():
            continue
        first_idx = int(flagged.argmax())
        end_times = seq.get("end_time")
        anomaly_times = seq.get("first_future_anomaly_time")
        if end_times is None or anomaly_times is None:
            continue
        try:
            t_detect = _pd.Timestamp(end_times[first_idx])
            pos_anomaly_times = [t for t in anomaly_times[y == 1] if t and t != ""]
            if not pos_anomaly_times:
                continue
            t_fault = _pd.Timestamp(pos_anomaly_times[0])
            lead_times.append((t_fault - t_detect).total_seconds() / 60.0)
        except Exception:
            continue
    return float(np.mean(lead_times)) if lead_times else None


def compute_false_alarm_stats(
    test_sequences: list,
    all_window_scores: list,
    threshold: float,
) -> dict:
    """False alarm count and mean flagged-window count for normal (y=0) sequences."""
    false_alarm_count = 0
    false_alarm_windows: list[int] = []
    total_normal = 0
    for seq, scores in zip(test_sequences, all_window_scores):
        y = np.asarray(seq["y"])
        if int(y.max()) != 0:
            continue
        total_normal += 1
        n_flagged = int((np.asarray(scores) >= threshold).sum())
        if n_flagged > 0:
            false_alarm_count += 1
            false_alarm_windows.append(n_flagged)
    return {
        "false_alarm_count": false_alarm_count,
        "total_normal_events": total_normal,
        "false_alarm_rate": float(false_alarm_count / total_normal) if total_normal else 0.0,
        "false_alarm_windows_mean": float(np.mean(false_alarm_windows)) if false_alarm_windows else 0.0,
    }


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
