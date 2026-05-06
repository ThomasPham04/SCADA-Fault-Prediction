"""
Train and evaluate a normal-behavior sequence autoencoder from the
Wind Farm A normal-behavior Parquet handoff.

The handoff is already feature-engineered, imputed, scaled, and split into:
  - train_normal.parquet
  - validation_normal.parquet
  - test_prediction.parquet

Labels are not used for fitting. They are used only to evaluate test-window
reconstruction scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.sequence_metrics import evaluate_at_threshold
from training.sequence_models import autoencoder_callbacks, build_autoencoder_model
from training.sequence_plots import (
    save_confusion_matrix_plot,
    save_history_csv_and_plot,
    save_metrics_bar_plot,
    save_pr_curve_plot,
    save_roc_curve_plot,
)
from training.sequence_utils import (
    cleanup_tf,
    history_to_frame,
    load_best_model,
    reconstruction_scores,
    save_json,
    save_model_summary,
    set_random_seed,
)


DEFAULT_DATA_DIR = (
    "/home/terry/Project/C2C_Windturbine/"
    "normal_behavior_full_prediction-20260505T162159Z-3-001/"
    "normal_behavior_full_prediction"
)

LABEL_COLUMNS = [
    "label",
    "label_prediction_period_anomaly",
    "label_timestamp_interval_anomaly",
    "label_normal_status",
    "label_status_anomaly",
    "label_eval_mask_normal_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a normal-behavior autoencoder from the Parquet handoff."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "normal_behavior_handoff",
    )
    parser.add_argument(
        "--model-name",
        choices=["lstm_ae", "gru_ae", "dense_ae", "cnn_lstm", "cnn_gru"],
        default="lstm_ae",
    )
    parser.add_argument("--window-size", type=int, default=144)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--cadence-minutes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-stddev", type=float, default=0.0)
    parser.add_argument("--encoder-units", type=int, default=64)
    parser.add_argument("--bottleneck-units", type=int, default=16)
    parser.add_argument(
        "--lstm-units",
        type=int,
        nargs=5,
        default=None,
        metavar=("ENC1", "ENC2", "BOTTLENECK", "DEC1", "DEC2"),
        help=(
            "Optional custom LSTM AE stack. Example: "
            "--lstm-units 44 25 8 26 44 creates encoder 44->25->8 "
            "and decoder 26->44 before the output projection."
        ),
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--event-consecutive-windows", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs. Uses chronological first N windows.",
    )
    parser.add_argument(
        "--max-val-windows",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs. Uses chronological first N windows.",
    )
    parser.add_argument(
        "--max-test-windows",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs. Uses chronological first N windows.",
    )
    parser.add_argument(
        "--save-window-arrays",
        action="store_true",
        help="Persist X_train/X_validation/X_test arrays. These are large.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_custom_lstm_autoencoder(
    input_shape: tuple[int, int],
    lstm_units: list[int],
    learning_rate: float,
    noise_stddev: float,
):
    if len(lstm_units) != 5:
        raise ValueError("lstm_units must contain exactly five integers.")
    time_steps, feature_count = input_shape
    enc1, enc2, bottleneck, dec1, dec2 = [int(unit) for unit in lstm_units]

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    x = layers.GaussianNoise(noise_stddev)(inputs) if noise_stddev > 0.0 else inputs
    x = layers.LSTM(enc1, return_sequences=True, name=f"encoder_lstm_{enc1}")(x)
    x = layers.Dropout(0.20, name="encoder_dropout_1")(x)
    x = layers.LSTM(enc2, return_sequences=True, name=f"encoder_lstm_{enc2}")(x)
    x = layers.Dropout(0.20, name="encoder_dropout_2")(x)
    x = layers.LSTM(bottleneck, return_sequences=False, name=f"bottleneck_lstm_{bottleneck}")(x)
    x = layers.RepeatVector(time_steps, name="repeat_bottleneck")(x)
    x = layers.LSTM(dec1, return_sequences=True, name=f"decoder_lstm_{dec1}")(x)
    x = layers.Dropout(0.20, name="decoder_dropout_1")(x)
    x = layers.LSTM(dec2, return_sequences=True, name=f"decoder_lstm_{dec2}")(x)
    outputs = layers.TimeDistributed(
        layers.Dense(feature_count),
        name="reconstruction",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="lstm_ae_custom")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn_recurrent_autoencoder(
    input_shape: tuple[int, int],
    conv_filters: int,
    bottleneck_units: int,
    learning_rate: float,
    noise_stddev: float,
    recurrent_kind: str,
):
    if recurrent_kind not in {"lstm", "gru"}:
        raise ValueError("recurrent_kind must be 'lstm' or 'gru'.")

    recurrent_layer = layers.LSTM if recurrent_kind == "lstm" else layers.GRU
    time_steps, feature_count = input_shape
    encoded_steps = int(np.ceil(np.ceil(time_steps / 2) / 2))
    decoded_steps = encoded_steps * 4

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    x = layers.GaussianNoise(noise_stddev)(inputs) if noise_stddev > 0.0 else inputs
    x = layers.Conv1D(
        conv_filters,
        5,
        padding="same",
        activation="relu",
        name=f"encoder_conv1d_{conv_filters}_k5",
    )(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="encoder_pool_1")(x)
    x = layers.Dropout(0.20, name="encoder_dropout_1")(x)
    x = layers.Conv1D(
        conv_filters,
        3,
        padding="same",
        activation="relu",
        name=f"encoder_conv1d_{conv_filters}_k3",
    )(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="encoder_pool_2")(x)
    x = recurrent_layer(
        bottleneck_units,
        return_sequences=False,
        name=f"bottleneck_{recurrent_kind}",
    )(x)

    x = layers.RepeatVector(encoded_steps, name="repeat_bottleneck")(x)
    x = recurrent_layer(
        conv_filters,
        return_sequences=True,
        name=f"decoder_{recurrent_kind}",
    )(x)
    x = layers.UpSampling1D(size=2, name="decoder_upsample_1")(x)
    x = layers.Conv1D(
        conv_filters,
        3,
        padding="same",
        activation="relu",
        name=f"decoder_conv1d_{conv_filters}_k3",
    )(x)
    x = layers.UpSampling1D(size=2, name="decoder_upsample_2")(x)
    x = layers.Conv1D(
        conv_filters,
        5,
        padding="same",
        activation="relu",
        name=f"decoder_conv1d_{conv_filters}_k5",
    )(x)
    if decoded_steps > time_steps:
        x = layers.Cropping1D((0, decoded_steps - time_steps), name="crop_to_window")(x)
    outputs = layers.Conv1D(feature_count, 1, padding="same", name="reconstruction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"cnn_{recurrent_kind}_ae")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def load_feature_list(data_dir: Path) -> list[str]:
    path = data_dir / "selected_features.csv"
    features = pd.read_csv(path)
    col = "feature" if "feature" in features.columns else features.columns[0]
    feature_cols = features[col].dropna().astype(str).tolist()
    if not feature_cols:
        raise ValueError(f"No features found in {path}")
    return feature_cols


def load_split(data_dir: Path, split_name: str, feature_cols: list[str]) -> pd.DataFrame:
    path = data_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    missing_features = sorted(set(feature_cols) - set(df.columns))
    if missing_features:
        raise ValueError(f"{split_name} is missing selected features: {missing_features}")
    required_meta = {"time_stamp", "asset_id", "sequence_id", "row_id"}
    missing_meta = sorted(required_meta - set(df.columns))
    if missing_meta:
        raise ValueError(f"{split_name} is missing metadata columns: {missing_meta}")
    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    if df["time_stamp"].isna().any():
        bad_count = int(df["time_stamp"].isna().sum())
        raise ValueError(f"{split_name} has {bad_count} invalid time_stamp values.")
    for col in LABEL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    df[feature_cols] = df[feature_cols].astype(np.float32)
    return df


def handoff_audit(frames: dict[str, pd.DataFrame], feature_cols: list[str]) -> dict:
    audit = {"feature_count": len(feature_cols), "splits": {}}
    for split_name, df in frames.items():
        label_counts = {
            col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}
            for col in LABEL_COLUMNS
            if col in df.columns
        }
        audit["splits"][split_name] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "selected_feature_missing_values": int(df[feature_cols].isna().sum().sum()),
            "asset_rows": {
                str(k): int(v) for k, v in df["asset_id"].value_counts().sort_index().to_dict().items()
            },
            "time_start": str(df["time_stamp"].min()),
            "time_end": str(df["time_stamp"].max()),
            "label_counts": label_counts,
        }
    return audit


def _as_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    window_size: int,
    stride: int,
    cadence_minutes: int,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    expected_gap = pd.Timedelta(minutes=cadence_minutes)
    x_parts: list[np.ndarray] = []
    rows: list[dict] = []
    segment_count = 0
    skipped_short_segments = 0

    sort_cols = ["asset_id", "sequence_id", "time_stamp", "row_id"]
    for (_, _), group in df.sort_values(sort_cols, kind="mergesort").groupby(
        ["asset_id", "sequence_id"],
        sort=False,
    ):
        group = group.reset_index(drop=True)
        new_segment = group["time_stamp"].diff().ne(expected_gap)
        new_segment.iloc[0] = True
        segment_ids = new_segment.cumsum()

        for _, segment in group.groupby(segment_ids, sort=False):
            segment_count += 1
            if len(segment) < window_size:
                skipped_short_segments += 1
                continue

            values = segment[feature_cols].to_numpy(dtype=np.float32, copy=False)
            max_start = len(segment) - window_size + 1
            for start in range(0, max_start, stride):
                end = start + window_size
                window = segment.iloc[start:end]
                first = window.iloc[0]
                last = window.iloc[-1]
                meta = {
                    "split": split_name,
                    "asset_id": _as_scalar(last["asset_id"]),
                    "sequence_id": _as_scalar(last["sequence_id"]),
                    "start_time": str(first["time_stamp"]),
                    "end_time": str(last["time_stamp"]),
                    "start_row_id": _as_scalar(first["row_id"]),
                    "end_row_id": _as_scalar(last["row_id"]),
                    "window_steps": int(window_size),
                }
                for optional in [
                    "event_id",
                    "event_label",
                    "event_type",
                    "event_description",
                    "event_start",
                    "event_end",
                    "event_start_id",
                    "event_end_id",
                    "status_type_id",
                    "source_event_split",
                    "train_test",
                    "row_phase",
                ]:
                    if optional in window.columns:
                        meta[optional] = _as_scalar(last[optional])

                if "label" in window.columns:
                    labels = window["label"].to_numpy(dtype=np.int8)
                    meta["window_label_any_event_interval"] = int(labels.max())
                    meta["window_label_last_event_interval"] = int(labels[-1])
                if "label_status_anomaly" in window.columns:
                    status_anomaly = window["label_status_anomaly"].to_numpy(dtype=np.int8)
                    meta["window_has_status_anomaly"] = int(status_anomaly.max())
                if "label_eval_mask_normal_status" in window.columns:
                    normal_status = window["label_eval_mask_normal_status"].to_numpy(dtype=np.int8)
                    meta["window_all_normal_status"] = int(normal_status.min())

                x_parts.append(values[start:end])
                rows.append(meta)

    if x_parts:
        x = np.stack(x_parts).astype(np.float32, copy=False)
    else:
        x = np.empty((0, window_size, len(feature_cols)), dtype=np.float32)

    summary = {
        "split": split_name,
        "rows": int(len(df)),
        "segments": int(segment_count),
        "skipped_short_segments": int(skipped_short_segments),
        "windows": int(len(x)),
    }
    return x, pd.DataFrame(rows), summary


def percentile_summary(scores: np.ndarray) -> dict:
    return {
        "p95": float(np.quantile(scores, 0.95)),
        "p97_5": float(np.quantile(scores, 0.975)),
        "p99": float(np.quantile(scores, 0.99)),
        "p99_5": float(np.quantile(scores, 0.995)),
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
    }


def consecutive_positive_max(values: pd.Series) -> int:
    best = 0
    current = 0
    for value in values.astype(int):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def classification_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
    }


def build_event_summary(pred_df: pd.DataFrame, consecutive_windows: int) -> pd.DataFrame:
    rows = []
    for event_id, group in pred_df.sort_values(["event_id", "end_time"], kind="mergesort").groupby("event_id"):
        predicted = group["predicted_label"].astype(int)
        positive = group["window_label_any_event_interval"].astype(int)
        first_alarm = group.loc[predicted.eq(1), "end_time"].min() if predicted.any() else ""
        rows.append(
            {
                "event_id": event_id,
                "asset_id": group["asset_id"].iloc[-1],
                "sequence_id": group["sequence_id"].iloc[-1],
                "event_label": group["event_label"].iloc[-1] if "event_label" in group else "",
                "event_description": group["event_description"].iloc[-1] if "event_description" in group else "",
                "window_count": int(len(group)),
                "positive_window_count": int(positive.sum()),
                "true_event_label": int(positive.max()),
                "max_score": float(group["score"].max()),
                "mean_score": float(group["score"].mean()),
                "predicted_window_count": int(predicted.sum()),
                "first_predicted_anomaly_time": first_alarm,
                "max_consecutive_predicted_windows": consecutive_positive_max(predicted),
                "event_detected_any_window": int(predicted.any()),
                "event_detected_consecutive": int(
                    consecutive_positive_max(predicted) >= consecutive_windows
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("event_id", kind="mergesort").reset_index(drop=True)


def threshold_sweep_rows(
    pred_df: pd.DataFrame,
    scores: np.ndarray,
    val_scores: np.ndarray,
    quantiles: list[float],
    consecutive_windows: int,
) -> list[dict]:
    rows = []
    y_any = pred_df["window_label_any_event_interval"].to_numpy(dtype=np.int8)
    y_last = pred_df["window_label_last_event_interval"].to_numpy(dtype=np.int8)
    for quantile in quantiles:
        threshold = float(np.quantile(val_scores, quantile))
        metrics_any = evaluate_at_threshold(y_any, scores, threshold)
        metrics_last = evaluate_at_threshold(y_last, scores, threshold)

        sweep_pred = pred_df.copy()
        sweep_pred["predicted_label"] = (scores >= threshold).astype(np.int8)
        event_summary = build_event_summary(sweep_pred, consecutive_windows)
        event_metrics = classification_metrics_from_predictions(
            event_summary["true_event_label"].to_numpy(dtype=np.int8),
            event_summary["event_detected_consecutive"].to_numpy(dtype=np.int8),
        )

        rows.append(
            {
                "threshold_source": "validation_quantile",
                "quantile": float(quantile),
                "threshold": threshold,
                "window_precision": metrics_any["precision"],
                "window_recall": metrics_any["recall"],
                "window_f1": metrics_any["f1"],
                "window_accuracy": metrics_any["accuracy"],
                "window_last_precision": metrics_last["precision"],
                "window_last_recall": metrics_last["recall"],
                "window_last_f1": metrics_last["f1"],
                "event_consecutive_precision": event_metrics["precision"],
                "event_consecutive_recall": event_metrics["recall"],
                "event_consecutive_f1": event_metrics["f1"],
                "predicted_window_count": int((scores >= threshold).sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    window_hours = args.window_size * args.cadence_minutes / 60.0
    stride_hours = args.stride * args.cadence_minutes / 60.0
    model_dir_name = args.model_name
    if args.lstm_units is not None:
        model_dir_name = f"{args.model_name}_{'_'.join(str(unit) for unit in args.lstm_units)}"
    run_dir = (
        args.output_dir
        / f"window_{window_hours:g}h_stride_{stride_hours:g}h"
        / model_dir_name
    )
    if run_dir.exists() and not args.overwrite and (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Run output already exists: {run_dir}. Use --overwrite to rerun.")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.random_seed)
    cleanup_tf()

    feature_cols = load_feature_list(data_dir)
    frames = {
        "train_normal": load_split(data_dir, "train_normal", feature_cols),
        "validation_normal": load_split(data_dir, "validation_normal", feature_cols),
        "test_prediction": load_split(data_dir, "test_prediction", feature_cols),
    }
    audit = handoff_audit(frames, feature_cols)
    save_json(run_dir / "handoff_audit.json", audit)
    pd.DataFrame({"feature": feature_cols}).to_csv(run_dir / "feature_list.csv", index=False)

    X_train, train_meta, train_window_summary = make_windows(
        frames["train_normal"],
        feature_cols,
        "train_normal",
        args.window_size,
        args.stride,
        args.cadence_minutes,
    )
    X_val, val_meta, val_window_summary = make_windows(
        frames["validation_normal"],
        feature_cols,
        "validation_normal",
        args.window_size,
        args.stride,
        args.cadence_minutes,
    )
    X_test, test_meta, test_window_summary = make_windows(
        frames["test_prediction"],
        feature_cols,
        "test_prediction",
        args.window_size,
        args.stride,
        args.cadence_minutes,
    )

    if args.max_train_windows is not None:
        X_train = X_train[: args.max_train_windows]
        train_meta = train_meta.iloc[: args.max_train_windows].reset_index(drop=True)
    if args.max_val_windows is not None:
        X_val = X_val[: args.max_val_windows]
        val_meta = val_meta.iloc[: args.max_val_windows].reset_index(drop=True)
    if args.max_test_windows is not None:
        X_test = X_test[: args.max_test_windows]
        test_meta = test_meta.iloc[: args.max_test_windows].reset_index(drop=True)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError("Train, validation, and test windows must all be non-empty.")

    train_meta.to_csv(run_dir / "train_window_metadata.csv", index=False)
    val_meta.to_csv(run_dir / "validation_window_metadata.csv", index=False)
    test_meta.to_csv(run_dir / "test_window_metadata.csv", index=False)

    if args.save_window_arrays:
        np.save(run_dir / "X_train.npy", X_train)
        np.save(run_dir / "X_validation.npy", X_val)
        np.save(run_dir / "X_test.npy", X_test)

    input_shape = (int(X_train.shape[1]), int(X_train.shape[2]))
    if args.lstm_units is not None:
        if args.model_name != "lstm_ae":
            raise ValueError("--lstm-units is only supported with --model-name lstm_ae.")
        model = build_custom_lstm_autoencoder(
            input_shape,
            args.lstm_units,
            learning_rate=args.learning_rate,
            noise_stddev=args.noise_stddev,
        )
    elif args.model_name in {"cnn_lstm", "cnn_gru"}:
        model = build_cnn_recurrent_autoencoder(
            input_shape,
            conv_filters=args.encoder_units,
            bottleneck_units=args.bottleneck_units,
            learning_rate=args.learning_rate,
            noise_stddev=args.noise_stddev,
            recurrent_kind="lstm" if args.model_name == "cnn_lstm" else "gru",
        )
    else:
        model = build_autoencoder_model(
            args.model_name,
            input_shape,
            encoder_units=args.encoder_units,
            bottleneck_units=args.bottleneck_units,
            learning_rate=args.learning_rate,
            noise_stddev=args.noise_stddev,
        )
    save_model_summary(model, run_dir / "model_summary.txt")
    model_path = run_dir / "model.keras"

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=autoencoder_callbacks(model_path),
        verbose=2,
    )

    best_model = load_best_model(model_path)
    history_df = history_to_frame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    save_history_csv_and_plot(
        history_df,
        run_dir / "history.csv",
        run_dir / "loss_history.png",
        f"{args.model_name.upper()} Normal-Behavior Handoff Loss",
    )

    val_scores = reconstruction_scores(best_model, X_val, batch_size=args.batch_size)
    threshold = float(np.quantile(val_scores, args.threshold_quantile))
    test_scores = reconstruction_scores(best_model, X_test, batch_size=args.batch_size)

    pred_df = test_meta.copy()
    pred_df["score"] = test_scores.astype(np.float32)
    pred_df["threshold"] = threshold
    pred_df["predicted_label"] = (pred_df["score"] >= threshold).astype(np.int8)
    pred_df.to_csv(run_dir / "test_window_predictions.csv", index=False)
    pd.DataFrame({"score": val_scores.astype(np.float32)}).to_csv(
        run_dir / "validation_scores.csv",
        index=False,
    )

    y_any = pred_df["window_label_any_event_interval"].to_numpy(dtype=np.int8)
    y_last = pred_df["window_label_last_event_interval"].to_numpy(dtype=np.int8)
    y_status = pred_df.get("window_has_status_anomaly", pd.Series(np.zeros(len(pred_df)))).to_numpy(dtype=np.int8)

    metrics_any = evaluate_at_threshold(y_any, test_scores, threshold)
    metrics_last = evaluate_at_threshold(y_last, test_scores, threshold)
    metrics_status = evaluate_at_threshold(y_status, test_scores, threshold)

    sweep_quantiles = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
    sweep_df = pd.DataFrame(
        threshold_sweep_rows(
            pred_df,
            test_scores,
            val_scores,
            sweep_quantiles,
            args.event_consecutive_windows,
        )
    )
    sweep_df.to_csv(run_dir / "threshold_sweep_validation_quantiles.csv", index=False)

    event_summary = build_event_summary(pred_df, args.event_consecutive_windows)
    event_summary.to_csv(run_dir / "event_summary.csv", index=False)
    event_metrics_any_score = evaluate_at_threshold(
        event_summary["true_event_label"].to_numpy(dtype=np.int8),
        event_summary["max_score"].to_numpy(dtype=np.float32),
        threshold,
    )
    event_metrics_consecutive = classification_metrics_from_predictions(
        event_summary["true_event_label"].to_numpy(dtype=np.int8),
        event_summary["event_detected_consecutive"].to_numpy(dtype=np.int8),
    )

    save_confusion_matrix_plot(
        metrics_any["confusion_matrix"],
        run_dir / "confusion_matrix_test_any_label.png",
        f"{args.model_name.upper()} Test Window Confusion Matrix",
    )
    save_metrics_bar_plot(
        metrics_any,
        run_dir / "test_metrics_bar_any_label.png",
        f"{args.model_name.upper()} Test Window Metrics",
    )
    save_pr_curve_plot(
        y_any,
        test_scores,
        run_dir / "pr_curve_test_any_label.png",
        f"{args.model_name.upper()} Test PR Curve",
    )
    save_roc_curve_plot(
        y_any,
        test_scores,
        run_dir / "roc_curve_test_any_label.png",
        f"{args.model_name.upper()} Test ROC Curve",
    )

    window_summary = {
        "data_dir": str(data_dir),
        "model_name": args.model_name,
        "model_dir_name": model_dir_name,
        "custom_lstm_units": args.lstm_units,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "cadence_minutes": int(args.cadence_minutes),
        "feature_count": int(len(feature_cols)),
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "train_window_summary": train_window_summary,
        "validation_window_summary": val_window_summary,
        "test_window_summary": test_window_summary,
        "threshold_quantile": float(args.threshold_quantile),
        "threshold": threshold,
        "validation_score_percentiles": percentile_summary(val_scores),
        "test_score_percentiles": percentile_summary(test_scores),
    }
    save_json(run_dir / "window_summary.json", window_summary)

    payload = {
        **window_summary,
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "encoder_units": int(args.encoder_units),
        "bottleneck_units": int(args.bottleneck_units),
        "custom_lstm_units": args.lstm_units,
        "window_metrics_any_event_interval": metrics_any,
        "window_metrics_last_event_interval": metrics_last,
        "window_metrics_status_anomaly": metrics_status,
        "event_metrics_any_window_score": event_metrics_any_score,
        "event_metrics_consecutive_rule": event_metrics_consecutive,
        "event_consecutive_windows": int(args.event_consecutive_windows),
    }
    save_json(run_dir / "metrics.json", payload)

    print("\n" + "=" * 70)
    print("Normal-behavior handoff run complete")
    print("=" * 70)
    print(f"Output dir : {run_dir}")
    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")
    print(f"Threshold  : {threshold:.6f} ({args.threshold_quantile:.3f} val quantile)")
    print(
        "Window any-label metrics: "
        f"precision={metrics_any['precision']:.4f}, "
        f"recall={metrics_any['recall']:.4f}, "
        f"f1={metrics_any['f1']:.4f}, "
        f"pr_auc={metrics_any['pr_auc']}, "
        f"roc_auc={metrics_any['roc_auc']}"
    )
    print(
        "Event 12-consecutive metrics: "
        f"precision={event_metrics_consecutive['precision']:.4f}, "
        f"recall={event_metrics_consecutive['recall']:.4f}, "
        f"f1={event_metrics_consecutive['f1']:.4f}"
    )

    del model, best_model, X_train, X_val, X_test
    cleanup_tf()


if __name__ == "__main__":
    main()
