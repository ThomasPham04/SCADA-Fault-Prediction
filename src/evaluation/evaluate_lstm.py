"""
Evaluate the legacy LSTM prediction detector.

Usage:
    python -m src.evaluation.evaluate_lstm
    python -m src.evaluation.evaluate_lstm --per_asset
    python -m src.evaluation.evaluate_lstm --per_asset --assets 0 10
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import WIND_FARM_A_PROCESSED
from evaluation.evaluator import LSTMEvaluator, smooth_mae


def load_events(data_split: str = "test") -> dict:
    """Load per-event NPZ files from the processed global split."""
    split_dir = os.path.join(WIND_FARM_A_PROCESSED, "global", f"{data_split}_by_event")
    events = {}

    print(f"Loading {data_split} events from: {split_dir}")
    if not os.path.exists(split_dir):
        print(f"  ERROR: Directory not found: {split_dir}")
        return events

    for filename in os.listdir(split_dir):
        if not filename.endswith(".npz"):
            continue
        try:
            event_id = int(filename.split("_")[1].split(".")[0])
            data = np.load(os.path.join(split_dir, filename), allow_pickle=True)
            if "X" in data and "y" in data:
                events[event_id] = {
                    "X": data["X"],
                    "y": data["y"],
                    "label": str(data["label"]),
                }
        except Exception as exc:
            print(f"  Error loading {filename}: {exc}")

    print(f"  Loaded {len(events)} {data_split} events")
    return events


def _compute_event_scores(model, events: dict, smoothing_window: int = 3) -> tuple:
    scores, labels = [], []
    for _, data in sorted(events.items()):
        y_pred = model.predict(data["X"], verbose=0, batch_size=256)
        mae = np.mean(np.abs(data["y"] - y_pred), axis=1)
        mae = smooth_mae(mae, window=smoothing_window)
        scores.append(np.percentile(mae, 95))
        labels.append(1 if data["label"] == "anomaly" else 0)
    return np.asarray(scores), np.asarray(labels)


def determine_threshold(
    model,
    val_events: dict,
    train_events: dict,
    min_recall: float = 0.7,
    smoothing_window: int = 3,
    iqr_multiplier: float = 1.5,
) -> tuple:
    """Choose a global MAE threshold from validation PR and normal-train IQR."""
    from sklearn.metrics import precision_recall_curve

    val_scores, val_labels = _compute_event_scores(model, val_events, smoothing_window)
    train_scores, train_labels = _compute_event_scores(model, train_events, smoothing_window)

    if len(val_scores) == 0:
        return 0.0, 0.0

    if len(np.unique(val_labels)) < 2:
        pr_upper = float(np.percentile(val_scores, 95))
    else:
        precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        valid_idx = np.where(recall >= min_recall)[0]
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])] if len(valid_idx) else int(np.argmax(f1_scores))
        threshold_idx = min(best_idx, len(thresholds) - 1)
        pr_upper = float(thresholds[threshold_idx]) if len(thresholds) else float(np.percentile(val_scores, 95))

    normal_scores = train_scores[train_labels == 0]
    if len(normal_scores) == 0:
        normal_scores = train_scores
    if len(normal_scores) == 0:
        return pr_upper, 0.0

    q1, q3 = np.percentile(normal_scores, [25, 75])
    iqr_upper = float(q3 + iqr_multiplier * (q3 - q1))
    return min(pr_upper, iqr_upper), 0.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the legacy LSTM prediction detector.")
    parser.add_argument(
        "--per_asset",
        action="store_true",
        help="Evaluate per-asset LSTM models instead of the global one.",
    )
    parser.add_argument(
        "--assets",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Asset IDs to evaluate when --per_asset is set.",
    )
    args = parser.parse_args()

    evaluator = LSTMEvaluator()

    if args.per_asset:
        evaluator.evaluate_per_asset(asset_filter=args.assets)
        return

    from config import MODELS_DIR
    from tensorflow import keras

    print("=" * 100)
    print("LSTM global final test-set evaluation")
    print("=" * 100)

    model_path = os.path.join(MODELS_DIR, "lstm_7day.keras")
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        print("Use the current classifier pipeline: python src/main.py train-sequences --windows 24")
        return

    model = keras.models.load_model(model_path)
    train_events = load_events("train")
    val_events = load_events("val")
    test_events = load_events("test")
    if not val_events or not test_events:
        print("Error: missing val or test event files.")
        return

    upper_th, lower_th = determine_threshold(model, val_events, train_events, min_recall=0.7)
    evaluator.evaluate_test(model, test_events, upper_th, lower_th)


if __name__ == "__main__":
    main()
