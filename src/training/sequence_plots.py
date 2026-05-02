"""
sequence_plots.py — training.sequence_plots
Plotting and prediction-frame helpers: loss history, threshold sweep,
confusion matrix, metrics bar, PR/ROC curves, and prediction DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def save_history_csv_and_plot(
    history_df: pd.DataFrame, csv_path: Path, png_path: Path, title: str
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["loss"], label="train_loss", linewidth=2)
    if "val_loss" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def save_threshold_plot(
    sweep_df: pd.DataFrame, best_threshold: float, output_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", linewidth=2)
    ax.axvline(
        best_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Best threshold={best_threshold:.3f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(conf_matrix, output_path: Path, title: str) -> None:
    cm = np.asarray(conf_matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metrics_bar_plot(metrics: dict, output_path: Path, title: str) -> None:
    labels = ["Precision", "Recall", "F1"]
    values = [metrics.get("precision", 0.0), metrics.get("recall", 0.0), metrics.get("f1", 0.0)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=["#4E79A7", "#59A14F", "#E15759"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_pr_curve_plot(
    y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "PR curve needs both classes", ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, linewidth=2, label=f"PR-AUC={pr_auc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve_plot(
    y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "ROC curve needs both classes", ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_threshold_csv(sweep_df: pd.DataFrame, output_path: Path) -> None:
    sweep_df.to_csv(output_path, index=False)


def build_prediction_frame(meta_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    pred_df = meta_df.copy()
    pred_df["score"] = np.asarray(scores, dtype=float)
    pred_df["predicted_label"] = (pred_df["score"] >= threshold).astype(int)
    pred_df["threshold"] = float(threshold)
    return pred_df
