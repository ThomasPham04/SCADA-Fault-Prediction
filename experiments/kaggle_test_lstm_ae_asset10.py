"""
Small standalone script to test the trained LSTM-AE on asset 10 test events.

Edit the settings block if your folders live somewhere else, then run:

    from kaggle_test_lstm_ae_asset10 import main
    main()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score


# ============================================================================
# SETTINGS
# ============================================================================

MODEL_DIR = "sequence_training_results/window_36h/autoencoder/asset_10/lstm_ae"
ASSET_EXPORT_DIR = "sequence_exports/window_36h/autoencoder/asset_10"
OUTPUT_DIR = "asset_10_lstm_ae_test"
EVENT_IDS = None
BATCH_SIZE = 128


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reconstruction_scores(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    scores = np.mean(np.square(X - recon), axis=(1, 2))
    return scores.astype(np.float32)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
    }


def select_sequence_files(test_dir: Path):
    files = sorted(test_dir.glob("sequence_*.npz"))
    if EVENT_IDS is None:
        return files

    selected_ids = {int(event_id) for event_id in EVENT_IDS}
    return [path for path in files if int(path.stem.replace("sequence_", "")) in selected_ids]


def main() -> None:
    model_dir = Path(MODEL_DIR)
    asset_export_dir = Path(ASSET_EXPORT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_event_dir = output_dir / "per_event_windows"
    per_event_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.keras"
    threshold_path = model_dir / "threshold.json"
    test_dir = asset_export_dir / "test_by_sequence"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test event folder not found: {test_dir}")

    threshold_info = load_json(threshold_path)
    threshold = float(threshold_info["selected_threshold"])
    model = tf.keras.models.load_model(model_path, compile=False)

    sequence_files = select_sequence_files(test_dir)
    if not sequence_files:
        raise RuntimeError("No matching sequence files found.")

    event_rows = []
    all_window_rows = []

    for npz_path in sequence_files:
        with np.load(npz_path, allow_pickle=True) as data:
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int8)
            end_time = np.asarray(data["end_time"]).astype(str)
            asset_id = int(np.asarray(data["asset_id"]).reshape(-1)[0])
            sequence_id = int(np.asarray(data["sequence_id"]).reshape(-1)[0])

        scores = reconstruction_scores(model, X, batch_size=BATCH_SIZE)
        predicted_window = (scores >= threshold).astype(np.int8)

        window_df = pd.DataFrame(
            {
                "asset_id": asset_id,
                "sequence_id": sequence_id,
                "window_index": np.arange(len(scores)),
                "end_time": end_time,
                "true_label": y,
                "score": scores,
                "predicted_label": predicted_window,
                "threshold": threshold,
            }
        )
        window_df.to_csv(per_event_dir / f"sequence_{sequence_id}_windows.csv", index=False)
        all_window_rows.append(window_df)

        event_score = float(scores.max())
        event_true = int(y.max())
        event_pred = int(event_score >= threshold)
        event_rows.append(
            {
                "asset_id": asset_id,
                "sequence_id": sequence_id,
                "true_label": event_true,
                "score": event_score,
                "predicted_label": event_pred,
                "threshold": threshold,
                "window_count": int(len(scores)),
                "last_end_time": end_time[-1] if len(end_time) else "",
            }
        )

    event_df = pd.DataFrame(event_rows).sort_values("sequence_id", kind="mergesort").reset_index(drop=True)
    window_df = pd.concat(all_window_rows, ignore_index=True)

    event_df.to_csv(output_dir / "asset_10_event_scores.csv", index=False)
    window_df.to_csv(output_dir / "asset_10_window_scores.csv", index=False)

    event_metrics = evaluate_binary(event_df["true_label"].to_numpy(), event_df["predicted_label"].to_numpy())
    window_metrics = evaluate_binary(window_df["true_label"].to_numpy(), window_df["predicted_label"].to_numpy())

    summary = {
        "asset_id": 10,
        "model_dir": str(model_dir),
        "asset_export_dir": str(asset_export_dir),
        "threshold": threshold,
        "event_metrics": event_metrics,
        "window_metrics": window_metrics,
        "tested_sequences": event_df["sequence_id"].tolist(),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 70)
    print("Asset 10 LSTM-AE Test Run")
    print("=" * 70)
    print(f"Threshold       : {threshold:.6f}")
    print(f"Tested events   : {event_df['sequence_id'].tolist()}")
    print(f"Event precision : {event_metrics['precision']:.4f}")
    print(f"Event recall    : {event_metrics['recall']:.4f}")
    print(f"Event F1        : {event_metrics['f1']:.4f}")
    print(f"Window precision: {window_metrics['precision']:.4f}")
    print(f"Window recall   : {window_metrics['recall']:.4f}")
    print(f"Window F1       : {window_metrics['f1']:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
