"""
sequence_experiments.py — training.sequence_experiments
High-level experiment runners: classifier, per-asset autoencoder, and global autoencoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.sequence_io import (
    load_asset_val_classifier_slice,
    load_autoencoder_asset_bundle,
    load_autoencoder_global_bundle,
    load_autoencoder_test_sequences,
    load_classifier_val_slice,
    load_classifier_val_slice_with_meta,
)
from training.sequence_metrics import (
    aggregate_event_scores,
    compute_class_weights,
    compute_false_alarm_stats,
    compute_lead_time_minutes,
    evaluate_at_threshold,
    evaluate_event_level_from_sequences,
    pick_best_threshold,
    safe_mean,
    sweep_thresholds,
    sweep_thresholds_event_level,
)
from training.sequence_models import (
    autoencoder_callbacks,
    build_autoencoder_model,
    build_classifier_model,
    build_threshold_nn,
    classifier_callbacks,
    threshold_nn_callbacks,
)
from training.sequence_plots import (
    build_prediction_frame,
    save_confusion_matrix_plot,
    save_history_csv_and_plot,
    save_metrics_bar_plot,
    save_pr_curve_plot,
    save_roc_curve_plot,
    save_threshold_csv,
    save_threshold_plot,
)
from training.sequence_utils import (
    adaptive_reconstruction_scores,
    cleanup_tf,
    history_to_frame,
    load_best_model,
    load_json,
    per_timestep_l2_scores,
    reconstruction_scores,
    save_json,
    save_model_summary,
    set_random_seed,
    to_int_list,
)


def _select_ae_threshold(
    best_model,
    X_val_normal: np.ndarray,
    val_scores: np.ndarray,
    y_val_labeled: np.ndarray,
    batch_size: int,
    identifier: str,
    val_meta: pd.DataFrame | None = None,
) -> tuple:
    """Returns (best_threshold, threshold_source, sweep_df).

    When val_meta with sequence_id is provided and the val set contains faults,
    the threshold is selected to maximise event-level F1 (one prediction per
    sequence via max-score aggregation) instead of window-level F1.
    """
    no_fault_val = (
        len(val_scores) == 0
        or len(y_val_labeled) == 0
        or int(y_val_labeled.max()) == 0
    )
    if no_fault_val:
        normal_val_scores = reconstruction_scores(best_model, X_val_normal, batch_size=batch_size)
        best_threshold = float(np.quantile(normal_val_scores, 0.99))
        threshold_source = "normal_val_99th_percentile"
        sweep_df = pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "accuracy"])
    else:
        threshold_candidates = np.unique(
            np.quantile(val_scores, np.linspace(0.80, 0.995, 40))
        )
        if len(threshold_candidates) == 0:
            raise RuntimeError(f"Threshold sweep candidates are empty for {identifier}.")
        use_event_level = (
            val_meta is not None
            and "sequence_id" in val_meta.columns
            and "asset_id" in val_meta.columns
        )
        if use_event_level:
            sweep_df = sweep_thresholds_event_level(val_meta, val_scores, threshold_candidates)
            threshold_source = "event_level_f1_sweep"
        else:
            sweep_df = sweep_thresholds(y_val_labeled, val_scores, threshold_candidates)
            threshold_source = "validation_f1_sweep"
        best_row = pick_best_threshold(sweep_df)
        best_threshold = float(best_row["threshold"])
    return best_threshold, threshold_source, sweep_df


def run_classifier_experiment(
    model_name: str,
    bundle: dict,
    output_dir: Path,
    random_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float | None,
    l2_strength: float,
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
    model = build_classifier_model(
        model_name,
        input_shape,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_strength=l2_strength,
    )
    model.summary()
    save_model_summary(model, output_dir / "model_summary.txt")
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
    val_meta = bundle["val_meta"]
    has_seq_col = "sequence_id" in val_meta.columns and "asset_id" in val_meta.columns
    has_pos_val = int(y_val.max()) > 0 if len(y_val) > 0 else False
    if has_seq_col and has_pos_val:
        sweep_df = sweep_thresholds_event_level(val_meta, val_scores, threshold_grid)
        threshold_source = "event_level_f1_sweep"
    else:
        sweep_df = sweep_thresholds(y_val, val_scores, threshold_grid)
        threshold_source = "validation_f1_sweep"
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

    # Event-level evaluation on test set
    test_meta = bundle["test_meta"]
    event_test_metrics: dict | None = None
    event_test_df: pd.DataFrame | None = None
    if "sequence_id" in test_meta.columns and "asset_id" in test_meta.columns:
        event_test_df = aggregate_event_scores(test_meta, test_scores)
        event_test_metrics = evaluate_at_threshold(
            event_test_df["true_label"].to_numpy(dtype=np.int8),
            event_test_df["score"].to_numpy(dtype=np.float32),
            best_threshold,
        )

    if save_predictions:
        build_prediction_frame(val_meta, val_scores, best_threshold).to_csv(
            output_dir / "val_predictions.csv", index=False
        )
        build_prediction_frame(test_meta, test_scores, best_threshold).to_csv(
            output_dir / "test_predictions.csv", index=False
        )
        if event_test_df is not None:
            event_test_df["predicted_label"] = (
                event_test_df["score"] >= best_threshold
            ).astype(np.int8)
            event_test_df.to_csv(output_dir / "test_event_predictions.csv", index=False)

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
        y_test, test_scores, output_dir / "pr_curve_test.png",
        f"{model_name.upper()} Test PR Curve",
    )
    save_roc_curve_plot(
        y_test, test_scores, output_dir / "roc_curve_test.png",
        f"{model_name.upper()} Test ROC Curve",
    )

    summary = {
        "model_name": model_name,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
    }
    if event_test_metrics is not None:
        summary["event_precision"] = event_test_metrics["precision"]
        summary["event_recall"] = event_test_metrics["recall"]
        summary["event_f1"] = event_test_metrics["f1"]
        summary["event_count"] = event_test_metrics["support"]

    payload = {
        "model_name": model_name,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "hyperparameters": {
            "learning_rate": float(learning_rate),
            "dropout_rate": None if dropout_rate is None else float(dropout_rate),
            "l2_strength": float(l2_strength),
        },
        "selected_threshold": best_threshold,
        "threshold_source": threshold_source,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "event_test_metrics": event_test_metrics,
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
    learning_rate: float = 1e-3,
    noise_stddev: float = 0.0,
    use_adaptive_threshold: bool = False,
    gamma: float = 0.344,
    threshold_nn_units: int = 23,
    encoder_units: int | None = None,
    bottleneck_units: int | None = None,
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

    X_val_labeled, y_val_labeled, val_meta = load_classifier_val_slice_with_meta(
        classifier_bundle, asset_filter=[asset_id]
    )
    # Empty labeled val is fine — threshold selection will use 99th-percentile fallback.
    has_labeled_val = len(X_val_labeled) > 0

    model = build_autoencoder_model(
        model_name,
        input_shape,
        encoder_units=encoder_units,
        bottleneck_units=bottleneck_units,
        learning_rate=learning_rate,
        noise_stddev=noise_stddev,
    )
    model.summary()
    save_model_summary(model, output_dir / "model_summary.txt")
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

    # Adaptive threshold: train NN regression on validation data to predict expected RE
    threshold_nn = None
    if use_adaptive_threshold:
        N_val, T_val, F_val = X_val_normal.shape
        X_val_flat = X_val_normal.reshape(N_val * T_val, F_val)
        val_ts_l2 = per_timestep_l2_scores(best_model, X_val_normal, batch_size).reshape(-1)
        split = max(1, int(len(X_val_flat) * 0.8))
        threshold_nn = build_threshold_nn(F_val, hidden_units=threshold_nn_units)
        tnn_path = output_dir / "threshold_nn.keras"
        threshold_nn.fit(
            X_val_flat[:split], val_ts_l2[:split],
            validation_data=(X_val_flat[split:], val_ts_l2[split:]),
            epochs=300,
            batch_size=batch_size,
            callbacks=threshold_nn_callbacks(tnn_path),
            verbose=0,
        )
        if tnn_path.exists():
            threshold_nn = load_best_model(tnn_path)

    if has_labeled_val:
        if use_adaptive_threshold and threshold_nn is not None:
            val_scores = adaptive_reconstruction_scores(best_model, threshold_nn, X_val_labeled, batch_size)
        else:
            val_scores = reconstruction_scores(best_model, X_val_labeled, batch_size)
        np.save(output_dir / "val_scores.npy", val_scores)
    else:
        val_scores = np.empty(0, dtype=np.float32)

    if use_adaptive_threshold:
        best_threshold = gamma
        threshold_source = "adaptive_threshold_nn"
        sweep_df = pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "accuracy"])
    else:
        best_threshold, threshold_source, sweep_df = _select_ae_threshold(
            best_model, X_val_normal, val_scores, y_val_labeled, batch_size,
            f"asset {asset_id}", val_meta=val_meta,
        )

    save_threshold_csv(sweep_df, output_dir / "threshold_sweep_val.csv")
    if not sweep_df.empty:
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
        if use_adaptive_threshold and threshold_nn is not None:
            seq_scores = adaptive_reconstruction_scores(best_model, threshold_nn, sequence_data["X"], batch_size)
        else:
            seq_scores = reconstruction_scores(best_model, sequence_data["X"], batch_size=batch_size)
        all_window_scores.append(seq_scores)
        all_window_labels.append(sequence_data["y"])

    if not all_window_scores:
        raise RuntimeError(f"No test sequences found for asset {asset_id}.")

    flat_window_scores = np.concatenate(all_window_scores).astype(np.float32)
    flat_window_labels = np.concatenate(all_window_labels).astype(np.int8)
    flat_window_pred = (flat_window_scores >= best_threshold).astype(np.int8)

    test_metrics = evaluate_at_threshold(flat_window_labels, flat_window_scores, best_threshold)

    # Event-level evaluation
    event_test_metrics = evaluate_event_level_from_sequences(
        test_sequences, all_window_scores, best_threshold
    )
    lead_time_minutes = compute_lead_time_minutes(
        test_sequences, all_window_scores, best_threshold
    )
    false_alarm_stats = compute_false_alarm_stats(
        test_sequences, all_window_scores, best_threshold
    )

    asset_ids_per_window = np.concatenate([
        np.full(len(s["y"]), int(s["asset_id"]), dtype=np.int64)
        for s in test_sequences
    ])
    sequence_ids_per_window = np.concatenate([
        np.full(len(s["y"]), int(s["sequence_id"]), dtype=np.int64)
        for s in test_sequences
    ])
    pd.DataFrame(
        {
            "asset_id": asset_ids_per_window,
            "sequence_id": sequence_ids_per_window,
            "true_label": flat_window_labels,
            "score": flat_window_scores,
            "predicted_label": flat_window_pred,
        }
    ).to_csv(output_dir / "test_window_predictions.csv", index=False)

    # Event-level predictions CSV
    event_rows = [
        {
            "asset_id": int(s["asset_id"]),
            "sequence_id": int(s["sequence_id"]),
            "event_true_label": int(np.asarray(s["y"]).max()),
            "event_score": float(np.asarray(sc).max()),
            "event_predicted_label": int(np.asarray(sc).max() >= best_threshold),
        }
        for s, sc in zip(test_sequences, all_window_scores)
    ]
    pd.DataFrame(event_rows).to_csv(output_dir / "test_event_predictions.csv", index=False)

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
        flat_window_labels, flat_window_scores, output_dir / "pr_curve_test.png",
        f"{model_name.upper()} Asset {asset_id} Test PR Curve",
    )
    save_roc_curve_plot(
        flat_window_labels, flat_window_scores, output_dir / "roc_curve_test.png",
        f"{model_name.upper()} Asset {asset_id} Test ROC Curve",
    )

    save_json(
        output_dir / "threshold.json",
        {
            "asset_id": asset_id,
            "model_name": model_name,
            "selected_threshold": best_threshold,
            "threshold_source": threshold_source,
        },
    )

    summary = {
        "model_name": model_name,
        "asset_id": asset_id,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
        "event_precision": event_test_metrics["precision"],
        "event_recall": event_test_metrics["recall"],
        "event_f1": event_test_metrics["f1"],
        "event_count": event_test_metrics["support"],
        "lead_time_minutes": lead_time_minutes,
        "false_alarm_count": false_alarm_stats["false_alarm_count"],
        "false_alarm_rate": false_alarm_stats["false_alarm_rate"],
    }

    payload = {
        "model_name": model_name,
        "asset_id": asset_id,
        "train_shape": list(X_train.shape),
        "val_normal_shape": list(X_val_normal.shape),
        "val_labeled_shape": list(X_val_labeled.shape),
        "selected_threshold": best_threshold,
        "threshold_source": threshold_source,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "event_test_metrics": event_test_metrics,
        "lead_time_minutes": lead_time_minutes,
        "false_alarm_stats": false_alarm_stats,
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


def run_global_autoencoder_experiment(
    model_name: str,
    autoencoder_root: Path,
    classifier_bundle: dict,
    output_dir: Path,
    random_seed: int,
    epochs: int,
    batch_size: int,
    overwrite: bool,
    save_predictions: bool,
    asset_filter=None,
    learning_rate: float = 1e-3,
    noise_stddev: float = 0.0,
    use_adaptive_threshold: bool = False,
    gamma: float = 0.344,
    threshold_nn_units: int = 23,
    encoder_units: int | None = None,
    bottleneck_units: int | None = None,
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

    global_bundle = load_autoencoder_global_bundle(autoencoder_root, asset_filter=asset_filter)

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tf()
    set_random_seed(random_seed)

    X_train = global_bundle["X_train"]
    X_val_normal = global_bundle["X_val_normal"]
    if len(X_train) == 0:
        raise RuntimeError("Global autoencoder has no normal training windows.")
    if len(X_val_normal) == 0:
        raise RuntimeError("Global autoencoder has no normal validation windows.")

    input_shape = (int(X_train.shape[1]), int(X_train.shape[2]))
    X_val_labeled, y_val_labeled, val_meta = load_classifier_val_slice_with_meta(
        classifier_bundle, asset_filter=asset_filter
    )
    has_labeled_val = len(X_val_labeled) > 0

    model = build_autoencoder_model(
        model_name,
        input_shape,
        encoder_units=encoder_units,
        bottleneck_units=bottleneck_units,
        learning_rate=learning_rate,
        noise_stddev=noise_stddev,
    )
    model.summary()
    save_model_summary(model, output_dir / "model_summary.txt")
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
        f"{model_name.upper()} Global Train / Val Loss",
    )

    threshold_nn = None
    if use_adaptive_threshold:
        N_val, T_val, F_val = X_val_normal.shape
        X_val_flat = X_val_normal.reshape(N_val * T_val, F_val)
        val_ts_l2 = per_timestep_l2_scores(best_model, X_val_normal, batch_size).reshape(-1)
        split = max(1, int(len(X_val_flat) * 0.8))
        threshold_nn = build_threshold_nn(F_val, hidden_units=threshold_nn_units)
        tnn_path = output_dir / "threshold_nn.keras"
        threshold_nn.fit(
            X_val_flat[:split], val_ts_l2[:split],
            validation_data=(X_val_flat[split:], val_ts_l2[split:]),
            epochs=300,
            batch_size=batch_size,
            callbacks=threshold_nn_callbacks(tnn_path),
            verbose=0,
        )
        if tnn_path.exists():
            threshold_nn = load_best_model(tnn_path)

    if has_labeled_val:
        if use_adaptive_threshold and threshold_nn is not None:
            val_scores = adaptive_reconstruction_scores(best_model, threshold_nn, X_val_labeled, batch_size)
        else:
            val_scores = reconstruction_scores(best_model, X_val_labeled, batch_size=batch_size)
        np.save(output_dir / "val_scores.npy", val_scores)
    else:
        val_scores = np.empty(0, dtype=np.float32)

    if use_adaptive_threshold:
        best_threshold = gamma
        threshold_source = "adaptive_threshold_nn"
        sweep_df = pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "accuracy"])
    else:
        best_threshold, threshold_source, sweep_df = _select_ae_threshold(
            best_model, X_val_normal, val_scores, y_val_labeled, batch_size,
            "global autoencoder", val_meta=val_meta,
        )

    save_threshold_csv(sweep_df, output_dir / "threshold_sweep_val.csv")
    if not sweep_df.empty:
        save_threshold_plot(
            sweep_df,
            best_threshold,
            output_dir / "threshold_sweep_val.png",
            f"{model_name.upper()} Global Validation Threshold Sweep",
        )

    val_metrics = evaluate_at_threshold(y_val_labeled, val_scores, best_threshold)

    test_sequences = []
    for test_dir in global_bundle["test_dirs"]:
        test_sequences.extend(load_autoencoder_test_sequences(test_dir, asset_filter=asset_filter))
    if not test_sequences:
        raise RuntimeError("No test sequences found for global autoencoder.")

    prediction_frames = []
    all_seq_scores = []
    for sequence_data in test_sequences:
        if use_adaptive_threshold and threshold_nn is not None:
            sequence_scores = adaptive_reconstruction_scores(best_model, threshold_nn, sequence_data["X"], batch_size)
        else:
            sequence_scores = reconstruction_scores(best_model, sequence_data["X"], batch_size=batch_size)
        all_seq_scores.append(sequence_scores)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "asset_id": int(sequence_data["asset_id"]),
                    "sequence_id": int(sequence_data["sequence_id"]),
                    "true_label": sequence_data["y"].astype(np.int8),
                    "score": sequence_scores.astype(np.float32),
                }
            )
        )

    pred_df = pd.concat(prediction_frames, ignore_index=True)
    pred_df["predicted_label"] = (pred_df["score"] >= best_threshold).astype(np.int8)

    flat_window_scores = pred_df["score"].to_numpy(dtype=np.float32)
    flat_window_labels = pred_df["true_label"].to_numpy(dtype=np.int8)
    flat_window_pred = pred_df["predicted_label"].to_numpy(dtype=np.int8)

    test_metrics = evaluate_at_threshold(flat_window_labels, flat_window_scores, best_threshold)
    pred_df.to_csv(output_dir / "test_window_predictions.csv", index=False)

    # Event-level evaluation
    event_df = (
        pred_df.groupby(["asset_id", "sequence_id"], as_index=False, sort=False)
        .agg(event_score=("score", "max"), event_true_label=("true_label", "max"))
    )
    event_df["event_predicted_label"] = (event_df["event_score"] >= best_threshold).astype(np.int8)
    event_df.to_csv(output_dir / "test_event_predictions.csv", index=False)
    event_test_metrics = evaluate_at_threshold(
        event_df["event_true_label"].to_numpy(dtype=np.int8),
        event_df["event_score"].to_numpy(dtype=np.float32),
        best_threshold,
    )
    lead_time_minutes = compute_lead_time_minutes(test_sequences, all_seq_scores, best_threshold)
    false_alarm_stats = compute_false_alarm_stats(test_sequences, all_seq_scores, best_threshold)

    asset_rows = []
    for asset_id, asset_df in pred_df.groupby("asset_id", sort=False):
        asset_metrics = evaluate_at_threshold(
            asset_df["true_label"].to_numpy(dtype=np.int8),
            asset_df["score"].to_numpy(dtype=np.float32),
            best_threshold,
        )
        asset_rows.append(
            {
                "model_name": model_name,
                "scope": "global",
                "asset_id": int(asset_id),
                "threshold": best_threshold,
                "accuracy": asset_metrics["accuracy"],
                "precision": asset_metrics["precision"],
                "recall": asset_metrics["recall"],
                "f1": asset_metrics["f1"],
                "pr_auc": asset_metrics["pr_auc"],
                "roc_auc": asset_metrics["roc_auc"],
            }
        )

    asset_summary_df = pd.DataFrame(asset_rows).sort_values("asset_id", kind="mergesort")
    asset_summary_df.to_csv(output_dir / "asset_summary.csv", index=False)

    summary = {
        "model_name": model_name,
        "scope": "global",
        "source": global_bundle["source"],
        "asset_count": int(len(asset_summary_df)),
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "macro_accuracy": float(asset_summary_df["accuracy"].mean()),
        "macro_precision": float(asset_summary_df["precision"].mean()),
        "macro_recall": float(asset_summary_df["recall"].mean()),
        "macro_f1": float(asset_summary_df["f1"].mean()),
        "macro_pr_auc": safe_mean(asset_summary_df["pr_auc"]),
        "macro_roc_auc": safe_mean(asset_summary_df["roc_auc"]),
        "pooled_accuracy": test_metrics["accuracy"],
        "pooled_precision": test_metrics["precision"],
        "pooled_recall": test_metrics["recall"],
        "pooled_f1": test_metrics["f1"],
        "pooled_pr_auc": test_metrics["pr_auc"],
        "pooled_roc_auc": test_metrics["roc_auc"],
        "event_precision": event_test_metrics["precision"],
        "event_recall": event_test_metrics["recall"],
        "event_f1": event_test_metrics["f1"],
        "event_count": event_test_metrics["support"],
        "lead_time_minutes": lead_time_minutes,
        "false_alarm_count": false_alarm_stats["false_alarm_count"],
        "false_alarm_rate": false_alarm_stats["false_alarm_rate"],
    }

    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        output_dir / "confusion_matrix_test.png",
        f"{model_name.upper()} Global Test Confusion Matrix",
    )
    save_metrics_bar_plot(
        {
            "precision": summary["pooled_precision"],
            "recall": summary["pooled_recall"],
            "f1": summary["pooled_f1"],
        },
        output_dir / "test_metrics_bar.png",
        f"{model_name.upper()} Global Test Metrics",
    )
    save_pr_curve_plot(
        flat_window_labels, flat_window_scores, output_dir / "pr_curve_test.png",
        f"{model_name.upper()} Global Test PR Curve",
    )
    save_roc_curve_plot(
        flat_window_labels, flat_window_scores, output_dir / "roc_curve_test.png",
        f"{model_name.upper()} Global Test ROC Curve",
    )

    save_json(
        output_dir / "threshold.json",
        {
            "scope": "global",
            "model_name": model_name,
            "selected_threshold": best_threshold,
            "threshold_source": threshold_source,
            "threshold_candidates": sweep_df["threshold"].tolist() if threshold_source in (
                "validation_f1_sweep", "event_level_f1_sweep"
            ) else [],
            "asset_filter": to_int_list(asset_filter),
        },
    )

    payload = {
        "model_name": model_name,
        "scope": "global",
        "source": global_bundle["source"],
        "asset_ids": global_bundle["asset_ids"],
        "asset_filter": to_int_list(asset_filter),
        "train_shape": list(X_train.shape),
        "val_normal_shape": list(X_val_normal.shape),
        "val_labeled_shape": list(X_val_labeled.shape),
        "selected_threshold": best_threshold,
        "threshold_source": threshold_source,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "event_test_metrics": event_test_metrics,
        "lead_time_minutes": lead_time_minutes,
        "false_alarm_stats": false_alarm_stats,
        "asset_summaries": asset_summary_df.to_dict(orient="records"),
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
