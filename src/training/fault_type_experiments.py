"""
fault_type_experiments.py — training.fault_type_experiments
Per-fault-type LSTM-AE training with dual-alarm scoring (raw + EWMA).

Wraps the lower-level helpers from sequence_experiments / sequence_utils
to train a narrow-feature autoencoder for a single (asset, fault_type) pair
and evaluate both:
  - raw reconstruction error threshold alarm  (detects advanced degradation)
  - EWMA drift alarm                          (detects gradual 14-day drift)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fault_type_config import EWMA_ALPHA, get_feature_indices, is_low_confidence
from training.sequence_experiments import _select_ae_threshold
from training.sequence_io import (
    load_autoencoder_asset_bundle,
    load_autoencoder_test_sequences,
    load_classifier_val_slice_with_meta,
)
from training.sequence_metrics import (
    compute_false_alarm_stats,
    compute_lead_time_minutes,
    evaluate_at_threshold,
    evaluate_event_level_from_sequences,
)
from training.sequence_models import autoencoder_callbacks, build_autoencoder_model
from training.sequence_plots import save_history_csv_and_plot
from training.sequence_utils import (
    cleanup_tf,
    compute_ewma_scores,
    history_to_frame,
    load_best_model,
    load_json,
    reconstruction_scores,
    save_json,
    save_model_summary,
    set_random_seed,
)


# ---------------------------------------------------------------------------
# EWMA score plot
# ---------------------------------------------------------------------------

def _save_ewma_score_plot(
    test_sequences: list[dict],
    all_raw_scores: list[np.ndarray],
    all_ewma_scores: list[np.ndarray],
    raw_threshold: float,
    ewma_threshold: float,
    output_path: Path,
    title: str,
) -> None:
    """Plot raw reconstruction error and EWMA side-by-side for each test sequence."""
    n_seq = len(test_sequences)
    if n_seq == 0:
        return

    fig, axes = plt.subplots(n_seq, 1, figsize=(12, 3 * n_seq), squeeze=False)
    for idx, (seq, raw_sc, ewma_sc) in enumerate(
        zip(test_sequences, all_raw_scores, all_ewma_scores)
    ):
        ax = axes[idx, 0]
        x = np.arange(len(raw_sc))
        true_label = int(np.asarray(seq["y"]).max())
        label_str = "anomaly" if true_label == 1 else "normal"

        ax.plot(x, raw_sc, alpha=0.6, linewidth=1, color="steelblue", label="Raw error")
        ax.plot(x, ewma_sc, linewidth=1.8, color="darkorange", label="EWMA")
        ax.axhline(raw_threshold, linestyle="--", linewidth=1, color="steelblue",
                   alpha=0.8, label=f"θ_raw={raw_threshold:.4f}")
        ax.axhline(ewma_threshold, linestyle="--", linewidth=1, color="darkorange",
                   alpha=0.8, label=f"θ_ewma={ewma_threshold:.4f}")

        # Shade anomaly windows
        y_arr = np.asarray(seq["y"])
        for j, lbl in enumerate(y_arr):
            if lbl == 1:
                ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.08)

        seq_id = int(seq["sequence_id"])
        ax.set_title(f"Sequence {seq_id} ({label_str})", fontsize=9)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Score")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_fault_type_detector_experiment(
    asset_id: int,
    fault_type: str,
    asset_dir: Path,
    classifier_bundle: dict,
    output_dir: Path,
    random_seed: int = 42,
    model_name: str = "lstm_ae",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    encoder_units: int = 32,
    bottleneck_units: int = 16,
    overwrite: bool = False,
) -> dict:
    """Train a fault-type specific LSTM-AE and evaluate with dual-alarm scoring.

    Parameters
    ----------
    asset_id:
        Turbine asset ID (0, 10, 11, 13, or 21 for Wind Farm A).
    fault_type:
        One of "hydraulic", "gearbox", "generator_bearing", "transformer".
    asset_dir:
        Path to the per-asset autoencoder export directory
        (e.g. sequence_exports/window_24h/autoencoder/asset_0/).
    classifier_bundle:
        Loaded classifier bundle (used to get the labeled validation slice
        for threshold selection).
    output_dir:
        Where to save model, metrics, and plots.
    encoder_units / bottleneck_units:
        Smaller than the default full-feature LSTM-AE since we train on 4–7
        features instead of 21.
    """
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        loaded = load_json(metrics_path)
        return {"summary": loaded["summary"], "metrics": loaded}

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tf()
    set_random_seed(random_seed)

    # ------------------------------------------------------------------
    # Load full asset bundle and slice to fault-type feature subset
    # ------------------------------------------------------------------
    asset_bundle = load_autoencoder_asset_bundle(asset_dir)
    metadata = asset_bundle["metadata"]
    all_feature_cols: list[str] = metadata.get("feature_cols", [])

    feature_indices = get_feature_indices(fault_type, all_feature_cols)
    if not feature_indices:
        raise ValueError(
            f"No matching features found for fault_type='{fault_type}' "
            f"in asset {asset_id}. Available cols: {all_feature_cols}"
        )
    selected_features = [all_feature_cols[i] for i in feature_indices]

    X_train = np.asarray(asset_bundle["X_train"], dtype=np.float32)[:, :, feature_indices]
    X_val_normal = np.asarray(asset_bundle["X_val_normal"], dtype=np.float32)[:, :, feature_indices]
    n_features = len(feature_indices)
    input_shape = (int(X_train.shape[1]), n_features)

    # Labeled validation slice for threshold selection
    X_val_labeled, y_val_labeled, val_meta = load_classifier_val_slice_with_meta(
        classifier_bundle, asset_filter=[asset_id]
    )
    if len(X_val_labeled) > 0:
        X_val_labeled = X_val_labeled[:, :, feature_indices]

    # ------------------------------------------------------------------
    # Build and train LSTM-AE
    # ------------------------------------------------------------------
    model = build_autoencoder_model(
        model_name,
        input_shape,
        encoder_units=encoder_units,
        bottleneck_units=bottleneck_units,
        learning_rate=learning_rate,
    )
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
        f"{model_name.upper()} Asset {asset_id} [{fault_type}]",
    )

    # ------------------------------------------------------------------
    # Raw reconstruction error — threshold selection on validation
    # ------------------------------------------------------------------
    val_scores = (
        reconstruction_scores(best_model, X_val_labeled, batch_size)
        if len(X_val_labeled) > 0
        else np.empty(0, dtype=np.float32)
    )

    raw_threshold, threshold_source, sweep_df = _select_ae_threshold(
        best_model,
        X_val_normal,
        val_scores,
        y_val_labeled,
        batch_size,
        f"asset {asset_id} [{fault_type}]",
        val_meta=val_meta,
    )

    val_metrics = evaluate_at_threshold(y_val_labeled, val_scores, raw_threshold)

    # ------------------------------------------------------------------
    # Score all test sequences (raw)
    # ------------------------------------------------------------------
    test_sequences = load_autoencoder_test_sequences(asset_dir)
    if not test_sequences:
        raise RuntimeError(f"No test sequences found for asset {asset_id}.")

    all_raw_scores: list[np.ndarray] = []
    all_window_labels: list[np.ndarray] = []
    for seq in test_sequences:
        X_seq = seq["X"][:, :, feature_indices]
        scores = reconstruction_scores(best_model, X_seq, batch_size)
        all_raw_scores.append(scores)
        all_window_labels.append(seq["y"])

    flat_raw_scores = np.concatenate(all_raw_scores).astype(np.float32)
    flat_labels = np.concatenate(all_window_labels).astype(np.int8)

    raw_test_metrics = evaluate_at_threshold(flat_labels, flat_raw_scores, raw_threshold)
    raw_event_metrics = evaluate_event_level_from_sequences(
        test_sequences, all_raw_scores, raw_threshold
    )
    raw_lead_time = compute_lead_time_minutes(test_sequences, all_raw_scores, raw_threshold)
    raw_false_alarm = compute_false_alarm_stats(test_sequences, all_raw_scores, raw_threshold)

    # ------------------------------------------------------------------
    # EWMA scoring
    # ------------------------------------------------------------------
    ewma_alpha = EWMA_ALPHA[fault_type]

    # Warm-start EWMA with the median of normal validation errors to avoid
    # a cold-start spike when streaming starts on a fresh turbine.
    normal_val_raw = reconstruction_scores(best_model, X_val_normal, batch_size)
    ewma_init = float(np.median(normal_val_raw))

    # EWMA threshold — 95th percentile of EWMA over normal val windows
    ewma_val_normal = compute_ewma_scores(normal_val_raw, ewma_alpha, init_value=ewma_init)
    ewma_threshold = float(np.quantile(ewma_val_normal, 0.95))

    # Compute EWMA per test sequence (reset between sequences)
    all_ewma_scores: list[np.ndarray] = []
    for raw_sc in all_raw_scores:
        ewma_sc = compute_ewma_scores(raw_sc, ewma_alpha, init_value=ewma_init)
        all_ewma_scores.append(ewma_sc)

    flat_ewma_scores = np.concatenate(all_ewma_scores).astype(np.float32)

    ewma_test_metrics = evaluate_at_threshold(flat_labels, flat_ewma_scores, ewma_threshold)
    ewma_event_metrics = evaluate_event_level_from_sequences(
        test_sequences, all_ewma_scores, ewma_threshold
    )
    ewma_lead_time = compute_lead_time_minutes(test_sequences, all_ewma_scores, ewma_threshold)
    ewma_false_alarm = compute_false_alarm_stats(test_sequences, all_ewma_scores, ewma_threshold)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    low_conf = is_low_confidence(asset_id, fault_type)

    # Predictions CSV — includes both raw and EWMA columns
    asset_ids_col = np.concatenate(
        [np.full(len(s["y"]), int(s["asset_id"]), dtype=np.int64) for s in test_sequences]
    )
    seq_ids_col = np.concatenate(
        [np.full(len(s["y"]), int(s["sequence_id"]), dtype=np.int64) for s in test_sequences]
    )
    pd.DataFrame(
        {
            "asset_id": asset_ids_col,
            "sequence_id": seq_ids_col,
            "true_label": flat_labels,
            "raw_score": flat_raw_scores,
            "raw_predicted": (flat_raw_scores >= raw_threshold).astype(np.int8),
            "ewma_score": flat_ewma_scores,
            "ewma_predicted": (flat_ewma_scores >= ewma_threshold).astype(np.int8),
        }
    ).to_csv(output_dir / "test_window_predictions.csv", index=False)

    # EWMA score series plot
    _save_ewma_score_plot(
        test_sequences,
        all_raw_scores,
        all_ewma_scores,
        raw_threshold,
        ewma_threshold,
        output_dir / "ewma_score_series.png",
        f"Asset {asset_id} [{fault_type}] — Raw vs EWMA Reconstruction Error",
    )

    # Threshold JSON files
    save_json(
        output_dir / "threshold_raw.json",
        {"threshold": raw_threshold, "source": threshold_source},
    )
    save_json(
        output_dir / "threshold_ewma.json",
        {
            "threshold": ewma_threshold,
            "alpha": ewma_alpha,
            "source": "normal_val_95th_percentile_ewma",
            "ewma_init_value": ewma_init,
        },
    )

    # Summary
    summary = {
        "model_name": model_name,
        "asset_id": asset_id,
        "fault_type": fault_type,
        "n_features": n_features,
        "selected_features": selected_features,
        "raw_threshold": raw_threshold,
        "ewma_threshold": ewma_threshold,
        "ewma_alpha": ewma_alpha,
        "raw_f1": raw_test_metrics["f1"],
        "raw_pr_auc": raw_test_metrics["pr_auc"],
        "raw_roc_auc": raw_test_metrics["roc_auc"],
        "ewma_f1": ewma_test_metrics["f1"],
        "ewma_pr_auc": ewma_test_metrics["pr_auc"],
        "ewma_roc_auc": ewma_test_metrics["roc_auc"],
        "raw_event_f1": raw_event_metrics["f1"],
        "ewma_event_f1": ewma_event_metrics["f1"],
        "raw_event_recall": raw_event_metrics["recall"],
        "ewma_event_recall": ewma_event_metrics["recall"],
        "lead_time_raw_minutes": raw_lead_time,
        "lead_time_ewma_minutes": ewma_lead_time,
        "raw_false_alarm_rate": raw_false_alarm["false_alarm_rate"],
        "ewma_false_alarm_rate": ewma_false_alarm["false_alarm_rate"],
        "low_confidence": low_conf,
    }

    payload = {
        "model_name": model_name,
        "asset_id": asset_id,
        "fault_type": fault_type,
        "feature_indices": list(feature_indices),
        "selected_features": selected_features,
        "n_features": n_features,
        "ewma_alpha": ewma_alpha,
        "train_shape": list(X_train.shape),
        "val_normal_shape": list(X_val_normal.shape),
        "val_labeled_shape": list(X_val_labeled.shape) if len(X_val_labeled) > 0 else [0, 0, n_features],
        "raw_threshold": raw_threshold,
        "raw_threshold_source": threshold_source,
        "ewma_threshold": ewma_threshold,
        "ewma_threshold_source": "normal_val_95th_percentile_ewma",
        "ewma_init_value": ewma_init,
        "low_confidence": low_conf,
        "validation_metrics": val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "ewma_test_metrics": ewma_test_metrics,
        "raw_event_test_metrics": raw_event_metrics,
        "ewma_event_test_metrics": ewma_event_metrics,
        "lead_time_raw_minutes": raw_lead_time,
        "lead_time_ewma_minutes": ewma_lead_time,
        "raw_false_alarm_stats": raw_false_alarm,
        "ewma_false_alarm_stats": ewma_false_alarm,
        "summary": summary,
    }
    save_json(metrics_path, payload)

    del model, best_model, history
    cleanup_tf()

    return {"summary": summary, "metrics": payload}
