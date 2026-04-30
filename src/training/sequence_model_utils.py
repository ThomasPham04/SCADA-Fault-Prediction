"""
Shared helpers for sequence model training.

This module contains the main-project training helpers for sequence exports:
- load exported sequence data
- build models
- train and evaluate classifiers / autoencoders
- save metrics and plots
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import callbacks, layers, models


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def cleanup_tf() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_int_list(value):
    if value is None:
        return None
    return [int(item) for item in value]


def list_asset_dirs(autoencoder_root: Path, asset_filter=None):
    filter_ids = set(to_int_list(asset_filter) or [])
    asset_dirs = []
    for asset_dir in sorted(autoencoder_root.glob("asset_*")):
        try:
            asset_id = int(asset_dir.name.replace("asset_", ""))
        except ValueError:
            continue
        if filter_ids and asset_id not in filter_ids:
            continue
        asset_dirs.append(asset_dir)
    return asset_dirs


def load_classifier_bundle(classifier_dir: Path) -> dict:
    if not classifier_dir.exists():
        raise FileNotFoundError(f"Classifier export not found: {classifier_dir}")

    bundle = {
        "X_train": np.load(classifier_dir / "X_train.npy", mmap_mode="r"),
        "y_train": np.load(classifier_dir / "y_train.npy", mmap_mode="r"),
        "X_val": np.load(classifier_dir / "X_val.npy", mmap_mode="r"),
        "y_val": np.load(classifier_dir / "y_val.npy", mmap_mode="r"),
        "X_test": np.load(classifier_dir / "X_test.npy", mmap_mode="r"),
        "y_test": np.load(classifier_dir / "y_test.npy", mmap_mode="r"),
        "train_meta": pd.read_csv(classifier_dir / "train_meta.csv"),
        "val_meta": pd.read_csv(classifier_dir / "val_meta.csv"),
        "test_meta": pd.read_csv(classifier_dir / "test_meta.csv"),
        "metadata": load_json(classifier_dir / "metadata.json"),
    }
    return bundle


def load_scaler_if_present(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def load_autoencoder_asset_bundle(asset_dir: Path) -> dict:
    if not asset_dir.exists():
        raise FileNotFoundError(f"Autoencoder asset export not found: {asset_dir}")

    asset_id = int(asset_dir.name.replace("asset_", ""))
    return {
        "asset_id": asset_id,
        "X_train": np.load(asset_dir / "X_train.npy", mmap_mode="r"),
        "X_val_normal": np.load(asset_dir / "X_val.npy", mmap_mode="r"),
        "metadata": load_json(asset_dir / "metadata.json"),
        "test_dir": asset_dir / "test_by_sequence",
        "scaler": load_scaler_if_present(asset_dir / "scaler.pkl"),
    }


def build_classifier_model(model_name: str, input_shape: tuple):
    inputs = layers.Input(shape=input_shape, name="input_sequence")

    if model_name == "lstm":
        x = layers.LSTM(96, return_sequences=True)(inputs)
        x = layers.Dropout(0.25)(x)
        x = layers.LSTM(48)(x)
        x = layers.Dropout(0.25)(x)
    elif model_name == "gru":
        x = layers.GRU(96, return_sequences=True)(inputs)
        x = layers.Dropout(0.25)(x)
        x = layers.GRU(48)(x)
        x = layers.Dropout(0.25)(x)
    elif model_name == "cnn_lstm":
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.25)(x)
    elif model_name == "cnn_gru":
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.GRU(64)(x)
        x = layers.Dropout(0.25)(x)
    else:
        raise ValueError(f"Unsupported classifier model: {model_name}")

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def build_autoencoder_model(model_name: str, input_shape: tuple):
    time_steps, feature_count = input_shape
    inputs = layers.Input(shape=input_shape, name="input_sequence")

    if model_name == "lstm_ae":
        x = layers.LSTM(96, return_sequences=True)(inputs)
        x = layers.Dropout(0.20)(x)
        x = layers.LSTM(48, return_sequences=False)(x)
        x = layers.RepeatVector(time_steps)(x)
        x = layers.LSTM(48, return_sequences=True)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.LSTM(96, return_sequences=True)(x)
    elif model_name == "gru_ae":
        x = layers.GRU(96, return_sequences=True)(inputs)
        x = layers.Dropout(0.20)(x)
        x = layers.GRU(48, return_sequences=False)(x)
        x = layers.RepeatVector(time_steps)(x)
        x = layers.GRU(48, return_sequences=True)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.GRU(96, return_sequences=True)(x)
    else:
        raise ValueError(f"Unsupported autoencoder model: {model_name}")

    outputs = layers.TimeDistributed(layers.Dense(feature_count))(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def classifier_callbacks(model_path: Path):
    return [
        callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]


def autoencoder_callbacks(model_path: Path):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
    ]


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
    rows = []
    for threshold in thresholds:
        metrics = evaluate_at_threshold(y_true, y_score, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
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


def history_to_frame(history) -> pd.DataFrame:
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "epoch"
    return history_df.reset_index()


def load_best_model(model_path: Path):
    if model_path.exists():
        return tf.keras.models.load_model(model_path, compile=False)
    raise FileNotFoundError(f"Saved model not found: {model_path}")


def reconstruction_scores(model, X, batch_size: int) -> np.ndarray:
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(np.asarray(X) - recon), axis=(1, 2))
    return errors.astype(np.float32)


def load_asset_val_classifier_slice(bundle: dict, asset_id: int):
    val_meta = bundle["val_meta"]
    mask = pd.to_numeric(val_meta["asset_id"], errors="coerce").astype(int).eq(int(asset_id)).to_numpy()
    return (
        np.asarray(bundle["X_val"][mask], dtype=np.float32),
        np.asarray(bundle["y_val"][mask], dtype=np.int8),
    )


def load_autoencoder_test_sequences(asset_dir: Path):
    sequence_rows = []
    test_dir = asset_dir / "test_by_sequence"
    for npz_path in sorted(test_dir.glob("sequence_*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            label_key = "target_label" if "target_label" in data.files else "y"
            sequence_rows.append(
                {
                    "asset_id": int(np.asarray(data["asset_id"]).reshape(-1)[0]),
                    "sequence_id": int(np.asarray(data["sequence_id"]).reshape(-1)[0]),
                    "X": np.asarray(data["X"], dtype=np.float32),
                    "y": np.asarray(data[label_key], dtype=np.int8),
                }
            )
    return sequence_rows


def save_history_csv_and_plot(history_df: pd.DataFrame, csv_path: Path, png_path: Path, title: str) -> None:
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


def save_threshold_plot(sweep_df: pd.DataFrame, best_threshold: float, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", linewidth=2)
    ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Best threshold={best_threshold:.3f}")
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


def save_pr_curve_plot(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
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


def save_roc_curve_plot(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
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


def safe_mean(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return float(numeric.mean())


def run_classifier_experiment(
    model_name: str,
    bundle: dict,
    output_dir: Path,
    random_seed: int,
    epochs: int,
    batch_size: int,
    overwrite: bool,
    save_predictions: bool,
) -> dict:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        loaded = load_json(metrics_path)
        return {
            "summary": loaded["summary"],
            "metrics": loaded,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tf()
    set_random_seed(random_seed)

    X_train = bundle["X_train"]
    y_train = np.asarray(bundle["y_train"]).astype(np.int8)
    X_val = bundle["X_val"]
    y_val = np.asarray(bundle["y_val"]).astype(np.int8)
    X_test = bundle["X_test"]
    y_test = np.asarray(bundle["y_test"]).astype(np.int8)

    input_shape = (int(X_train.shape[1]), int(X_train.shape[2]))
    model = build_classifier_model(model_name, input_shape)
    model_path = output_dir / "model.keras"

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=compute_class_weights(y_train),
        callbacks=classifier_callbacks(model_path),
        verbose=1,
    )

    best_model = load_best_model(model_path)
    history_df = history_to_frame(history)
    save_history_csv_and_plot(
        history_df,
        output_dir / "history.csv",
        output_dir / "loss_history.png",
        f"{model_name.upper()} Train / Val Loss",
    )

    val_scores = best_model.predict(X_val, batch_size=batch_size, verbose=0).reshape(-1)
    test_scores = best_model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)

    threshold_grid = np.arange(0.10, 0.91, 0.05)
    sweep_df = sweep_thresholds(y_val, val_scores, threshold_grid)
    best_row = pick_best_threshold(sweep_df)
    best_threshold = float(best_row["threshold"])

    save_threshold_csv(sweep_df, output_dir / "threshold_sweep_val.csv")
    save_threshold_plot(
        sweep_df,
        best_threshold,
        output_dir / "threshold_sweep_val.png",
        f"{model_name.upper()} Validation Threshold Sweep",
    )

    val_metrics = evaluate_at_threshold(y_val, val_scores, best_threshold)
    test_metrics = evaluate_at_threshold(y_test, test_scores, best_threshold)

    if save_predictions:
        build_prediction_frame(bundle["val_meta"], val_scores, best_threshold).to_csv(
            output_dir / "val_predictions.csv",
            index=False,
        )
        build_prediction_frame(bundle["test_meta"], test_scores, best_threshold).to_csv(
            output_dir / "test_predictions.csv",
            index=False,
        )

    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        output_dir / "confusion_matrix_test.png",
        f"{model_name.upper()} Test Confusion Matrix",
    )
    save_metrics_bar_plot(
        test_metrics,
        output_dir / "test_metrics_bar.png",
        f"{model_name.upper()} Test Metrics",
    )
    save_pr_curve_plot(
        y_test,
        test_scores,
        output_dir / "pr_curve_test.png",
        f"{model_name.upper()} Test PR Curve",
    )
    save_roc_curve_plot(
        y_test,
        test_scores,
        output_dir / "roc_curve_test.png",
        f"{model_name.upper()} Test ROC Curve",
    )

    summary = {
        "model_name": model_name,
        "threshold": best_threshold,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
    }

    payload = {
        "model_name": model_name,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "selected_threshold": best_threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "summary": summary,
    }
    save_json(metrics_path, payload)

    del model, best_model, history
    cleanup_tf()

    return {
        "summary": summary,
        "metrics": payload,
    }


def run_autoencoder_experiment(
    model_name: str,
    asset_dir: Path,
    classifier_bundle: dict,
    output_dir: Path,
    random_seed: int,
    epochs: int,
    batch_size: int,
    overwrite: bool,
    save_predictions: bool,
) -> dict:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        loaded = load_json(metrics_path)
        preds = pd.read_csv(output_dir / "test_window_predictions.csv")
        return {
            "summary": loaded["summary"],
            "metrics": loaded,
            "test_window_scores": preds["score"].to_numpy(dtype=np.float32),
            "test_window_labels": preds["true_label"].to_numpy(dtype=np.int8),
            "test_window_pred": preds["predicted_label"].to_numpy(dtype=np.int8),
        }

    asset_bundle = load_autoencoder_asset_bundle(asset_dir)
    asset_id = asset_bundle["asset_id"]

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tf()
    set_random_seed(random_seed)

    X_train = asset_bundle["X_train"]
    X_val_normal = asset_bundle["X_val_normal"]
    input_shape = (int(X_train.shape[1]), int(X_train.shape[2]))

    X_val_labeled, y_val_labeled = load_asset_val_classifier_slice(classifier_bundle, asset_id)
    if len(X_val_labeled) == 0:
        raise RuntimeError(f"No classifier validation windows found for asset {asset_id}.")

    model = build_autoencoder_model(model_name, input_shape)
    model_path = output_dir / "model.keras"

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val_normal, X_val_normal),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=autoencoder_callbacks(model_path),
        verbose=1,
    )

    best_model = load_best_model(model_path)
    history_df = history_to_frame(history)
    save_history_csv_and_plot(
        history_df,
        output_dir / "history.csv",
        output_dir / "loss_history.png",
        f"{model_name.upper()} Asset {asset_id} Train / Val Loss",
    )

    val_scores = reconstruction_scores(best_model, X_val_labeled, batch_size=batch_size)
    np.save(output_dir / "val_scores.npy", val_scores)

    threshold_candidates = np.unique(
        np.quantile(val_scores, np.linspace(0.80, 0.995, 40))
    )
    if len(threshold_candidates) == 0:
        raise RuntimeError(f"Threshold sweep candidates are empty for asset {asset_id}.")

    sweep_df = sweep_thresholds(y_val_labeled, val_scores, threshold_candidates)
    best_row = pick_best_threshold(sweep_df)
    best_threshold = float(best_row["threshold"])

    save_threshold_csv(sweep_df, output_dir / "threshold_sweep_val.csv")
    save_threshold_plot(
        sweep_df,
        best_threshold,
        output_dir / "threshold_sweep_val.png",
        f"{model_name.upper()} Asset {asset_id} Validation Threshold Sweep",
    )

    val_metrics = evaluate_at_threshold(y_val_labeled, val_scores, best_threshold)

    test_sequences = load_autoencoder_test_sequences(asset_dir)
    all_window_scores = []
    all_window_labels = []

    for sequence_data in test_sequences:
        sequence_scores = reconstruction_scores(best_model, sequence_data["X"], batch_size=batch_size)
        all_window_scores.append(sequence_scores)
        all_window_labels.append(sequence_data["y"])

    if not all_window_scores:
        raise RuntimeError(f"No test sequences found for asset {asset_id}.")

    flat_window_scores = np.concatenate(all_window_scores).astype(np.float32)
    flat_window_labels = np.concatenate(all_window_labels).astype(np.int8)
    flat_window_pred = (flat_window_scores >= best_threshold).astype(np.int8)

    test_metrics = evaluate_at_threshold(flat_window_labels, flat_window_scores, best_threshold)

    pd.DataFrame(
        {
            "true_label": flat_window_labels,
            "score": flat_window_scores,
            "predicted_label": flat_window_pred,
        }
    ).to_csv(output_dir / "test_window_predictions.csv", index=False)

    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        output_dir / "confusion_matrix_test.png",
        f"{model_name.upper()} Asset {asset_id} Test Confusion Matrix",
    )
    save_metrics_bar_plot(
        test_metrics,
        output_dir / "test_metrics_bar.png",
        f"{model_name.upper()} Asset {asset_id} Test Metrics",
    )
    save_pr_curve_plot(
        flat_window_labels,
        flat_window_scores,
        output_dir / "pr_curve_test.png",
        f"{model_name.upper()} Asset {asset_id} Test PR Curve",
    )
    save_roc_curve_plot(
        flat_window_labels,
        flat_window_scores,
        output_dir / "roc_curve_test.png",
        f"{model_name.upper()} Asset {asset_id} Test ROC Curve",
    )

    save_json(
        output_dir / "threshold.json",
        {
            "asset_id": asset_id,
            "model_name": model_name,
            "selected_threshold": best_threshold,
            "threshold_source": "validation_f1_sweep",
            "threshold_candidates": threshold_candidates.tolist(),
        },
    )

    summary = {
        "model_name": model_name,
        "asset_id": asset_id,
        "threshold": best_threshold,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
    }

    payload = {
        "model_name": model_name,
        "asset_id": asset_id,
        "train_shape": list(X_train.shape),
        "val_normal_shape": list(X_val_normal.shape),
        "val_labeled_shape": list(X_val_labeled.shape),
        "selected_threshold": best_threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "summary": summary,
    }
    save_json(metrics_path, payload)

    del model, best_model, history
    cleanup_tf()

    return {
        "summary": summary,
        "metrics": payload,
        "test_window_scores": flat_window_scores,
        "test_window_labels": flat_window_labels,
        "test_window_pred": flat_window_pred,
    }


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
                "asset_count",
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
