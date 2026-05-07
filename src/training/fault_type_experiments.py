"""
fault_type_experiments.py - training.fault_type_experiments
Per-fault-type autoencoder training with dual-alarm scoring (raw + EWMA).

Supports two experiment scopes:
  - per_asset: train one detector for one (asset, fault_type) pair.
  - fault_type: train one pooled detector for one fault type across assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fault_type_config import (
    EWMA_ALPHA,
    fault_type_from_description,
    get_feature_indices,
    is_low_confidence,
)
from training.sequence_experiments import _select_ae_threshold
from training.sequence_io import (
    load_autoencoder_asset_bundle,
    load_autoencoder_global_bundle,
    load_autoencoder_test_sequences,
    load_autoencoder_test_sequences_from_dirs,
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


TEST_EVENT_SCOPES = {"all", "matching-fault", "matching-anomaly-only"}


# ---------------------------------------------------------------------------
# Event-type filtering
# ---------------------------------------------------------------------------

def _normalize_test_event_scope(test_event_scope: str) -> str:
    normalized = str(test_event_scope).strip().replace("_", "-")
    if normalized not in TEST_EVENT_SCOPES:
        raise ValueError(
            f"Unknown test_event_scope='{test_event_scope}'. "
            f"Expected one of {sorted(TEST_EVENT_SCOPES)}."
        )
    return normalized


def _load_event_lookup(event_info_path: Path | None) -> dict[tuple[int, int], dict]:
    """Read Wind Farm A event_info.csv into a (asset_id, event_id) lookup."""
    if event_info_path is None:
        return {}
    path = Path(event_info_path)
    if not path.exists():
        raise FileNotFoundError(f"event_info.csv not found: {path}")

    event_df = pd.read_csv(path, sep=None, engine="python")
    required_cols = {"asset", "event_id", "event_label"}
    missing = required_cols.difference(event_df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    lookup: dict[tuple[int, int], dict] = {}
    for _, row in event_df.iterrows():
        asset_id = int(row["asset"])
        sequence_id = int(row["event_id"])
        event_label = str(row.get("event_label", "")).strip().lower()
        description = row.get("event_description", "")
        lookup[(asset_id, sequence_id)] = {
            "asset_id": asset_id,
            "sequence_id": sequence_id,
            "event_label": event_label,
            "event_description": "" if pd.isna(description) else str(description),
            "event_fault_type": fault_type_from_description(description),
        }
    return lookup


def _event_key_from_sequence(seq: dict) -> tuple[int, int]:
    return int(seq["asset_id"]), int(seq["sequence_id"])


def _sequence_event_info(seq: dict, event_lookup: dict[tuple[int, int], dict]) -> dict | None:
    return event_lookup.get(_event_key_from_sequence(seq))


def _matches_event_scope(
    event_info: dict | None,
    fault_type: str,
    test_event_scope: str,
) -> bool:
    if test_event_scope == "all":
        return True
    if event_info is None:
        return False
    if event_info.get("event_fault_type") == fault_type:
        return True
    if test_event_scope == "matching-fault" and event_info.get("event_label") == "normal":
        return True
    return False


def _enrich_sequence_with_event_info(seq: dict, event_info: dict | None) -> dict:
    enriched = dict(seq)
    if event_info:
        enriched["event_label"] = event_info.get("event_label")
        enriched["event_description"] = event_info.get("event_description")
        enriched["event_fault_type"] = event_info.get("event_fault_type")
    else:
        enriched["event_label"] = None
        enriched["event_description"] = None
        enriched["event_fault_type"] = None
    return enriched


def _format_event_keys(event_keys: set[tuple[int, int]]) -> list[str]:
    return [f"{asset}:{sequence}" for asset, sequence in sorted(event_keys)]


def _filter_test_sequences_by_event_scope(
    test_sequences: list[dict],
    fault_type: str,
    test_event_scope: str,
    event_lookup: dict[tuple[int, int], dict],
) -> tuple[list[dict], dict]:
    """Keep only matching fault events, with optional normal events as negatives."""
    if test_event_scope == "all":
        included_events = {_event_key_from_sequence(seq) for seq in test_sequences}
        enriched = [
            _enrich_sequence_with_event_info(seq, _sequence_event_info(seq, event_lookup))
            for seq in test_sequences
        ]
        return enriched, {
            "scope": test_event_scope,
            "input_sequence_count": int(len(test_sequences)),
            "selected_sequence_count": int(len(enriched)),
            "included_event_count": int(len(included_events)),
            "excluded_event_count": 0,
            "missing_event_info_count": 0,
            "matching_fault_event_count": 0,
            "normal_event_count": 0,
            "included_events": _format_event_keys(included_events),
            "matching_fault_events": [],
            "normal_events": [],
        }

    selected: list[dict] = []
    included_events: set[tuple[int, int]] = set()
    excluded_events: set[tuple[int, int]] = set()
    missing_events: set[tuple[int, int]] = set()
    matching_fault_events: set[tuple[int, int]] = set()
    normal_events: set[tuple[int, int]] = set()

    for seq in test_sequences:
        event_key = _event_key_from_sequence(seq)
        event_info = _sequence_event_info(seq, event_lookup)
        if event_info is None:
            missing_events.add(event_key)

        if _matches_event_scope(event_info, fault_type, test_event_scope):
            selected.append(_enrich_sequence_with_event_info(seq, event_info))
            included_events.add(event_key)
            if event_info and event_info.get("event_fault_type") == fault_type:
                matching_fault_events.add(event_key)
            if event_info and event_info.get("event_label") == "normal":
                normal_events.add(event_key)
        else:
            excluded_events.add(event_key)

    return selected, {
        "scope": test_event_scope,
        "input_sequence_count": int(len(test_sequences)),
        "selected_sequence_count": int(len(selected)),
        "included_event_count": int(len(included_events)),
        "excluded_event_count": int(len(excluded_events)),
        "missing_event_info_count": int(len(missing_events)),
        "matching_fault_event_count": int(len(matching_fault_events)),
        "normal_event_count": int(len(normal_events)),
        "included_events": _format_event_keys(included_events),
        "matching_fault_events": _format_event_keys(matching_fault_events),
        "normal_events": _format_event_keys(normal_events),
    }


def _filter_validation_by_event_scope(
    X_val_labeled: np.ndarray,
    y_val_labeled: np.ndarray,
    val_meta: pd.DataFrame,
    fault_type: str,
    test_event_scope: str,
    event_lookup: dict[tuple[int, int], dict],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    if test_event_scope == "all" or len(val_meta) == 0:
        y_arr = np.asarray(y_val_labeled, dtype=np.int8)
        return X_val_labeled, y_arr, val_meta, {
            "scope": test_event_scope,
            "input_window_count": int(len(y_arr)),
            "selected_window_count": int(len(y_arr)),
            "positive_count": int((y_arr == 1).sum()) if len(y_arr) else 0,
            "negative_count": int((y_arr == 0).sum()) if len(y_arr) else 0,
            "missing_event_info_count": 0,
        }

    keep_mask: list[bool] = []
    missing_count = 0
    for _, row in val_meta.iterrows():
        event_key = (int(row["asset_id"]), int(row["sequence_id"]))
        event_info = event_lookup.get(event_key)
        if event_info is None:
            missing_count += 1
        keep_mask.append(_matches_event_scope(event_info, fault_type, test_event_scope))

    mask = np.asarray(keep_mask, dtype=bool)
    filtered_X = np.asarray(X_val_labeled[mask], dtype=np.float32)
    filtered_y = np.asarray(y_val_labeled[mask], dtype=np.int8)
    filtered_meta = val_meta.loc[mask].reset_index(drop=True)
    return filtered_X, filtered_y, filtered_meta, {
        "scope": test_event_scope,
        "input_window_count": int(len(y_val_labeled)),
        "selected_window_count": int(len(filtered_y)),
        "positive_count": int((filtered_y == 1).sum()),
        "negative_count": int((filtered_y == 0).sum()),
        "missing_event_info_count": int(missing_count),
    }


def _load_test_sequences_from_bundle(data_bundle: dict, asset_filter: list[int]) -> list[dict]:
    if "test_dirs" in data_bundle:
        return load_autoencoder_test_sequences_from_dirs(
            data_bundle["test_dirs"], asset_filter=asset_filter
        )
    return load_autoencoder_test_sequences(data_bundle["test_dir"], asset_filter=asset_filter)


# ---------------------------------------------------------------------------
# Plotting
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
        ax.axhline(
            raw_threshold,
            linestyle="--",
            linewidth=1,
            color="steelblue",
            alpha=0.8,
            label=f"raw={raw_threshold:.4f}",
        )
        ax.axhline(
            ewma_threshold,
            linestyle="--",
            linewidth=1,
            color="darkorange",
            alpha=0.8,
            label=f"ewma={ewma_threshold:.4f}",
        )

        y_arr = np.asarray(seq["y"])
        for j, lbl in enumerate(y_arr):
            if lbl == 1:
                ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.08)

        seq_id = int(seq["sequence_id"])
        fault_label = seq.get("event_fault_type") or "unknown"
        ax.set_title(f"Sequence {seq_id} ({label_str}, {fault_label})", fontsize=9)
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
# Experiment runners
# ---------------------------------------------------------------------------

def _run_fault_type_detector_from_bundle(
    *,
    scope: str,
    fault_type: str,
    data_bundle: dict,
    classifier_bundle: dict,
    output_dir: Path,
    asset_filter: list[int],
    title_label: str,
    asset_id: int | None = None,
    event_info_path: Path | None = None,
    test_event_scope: str = "all",
    random_seed: int = 42,
    model_name: str = "lstm_ae",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    encoder_units: int = 32,
    bottleneck_units: int = 16,
    overwrite: bool = False,
) -> dict:
    """Shared implementation for per-asset and pooled fault-type detectors."""
    test_event_scope = _normalize_test_event_scope(test_event_scope)
    asset_filter = sorted({int(asset) for asset in asset_filter})

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        loaded = load_json(metrics_path)
        return {"summary": loaded["summary"], "metrics": loaded}

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tf()
    set_random_seed(random_seed)

    event_lookup = (
        _load_event_lookup(event_info_path)
        if test_event_scope != "all" or event_info_path is not None
        else {}
    )

    metadata = data_bundle["metadata"]
    all_feature_cols: list[str] = metadata.get("feature_cols", [])
    feature_indices = get_feature_indices(fault_type, all_feature_cols)
    if not feature_indices:
        raise ValueError(
            f"No matching features found for fault_type='{fault_type}' "
            f"in {title_label}. Available cols: {all_feature_cols}"
        )
    selected_features = [all_feature_cols[i] for i in feature_indices]

    X_train = np.asarray(data_bundle["X_train"], dtype=np.float32)[:, :, feature_indices]
    X_val_normal = np.asarray(data_bundle["X_val_normal"], dtype=np.float32)[:, :, feature_indices]
    if len(X_train) == 0:
        raise RuntimeError(f"No normal training windows for {title_label} [{fault_type}].")
    if len(X_val_normal) == 0:
        raise RuntimeError(f"No normal validation windows for {title_label} [{fault_type}].")

    n_features = len(feature_indices)
    input_shape = (int(X_train.shape[1]), n_features)

    X_val_labeled, y_val_labeled, val_meta = load_classifier_val_slice_with_meta(
        classifier_bundle, asset_filter=asset_filter
    )
    X_val_labeled, y_val_labeled, val_meta, val_filter_summary = (
        _filter_validation_by_event_scope(
            X_val_labeled,
            y_val_labeled,
            val_meta,
            fault_type,
            test_event_scope,
            event_lookup,
        )
    )
    if len(X_val_labeled) > 0:
        X_val_labeled = np.asarray(X_val_labeled, dtype=np.float32)[:, :, feature_indices]

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
        f"{model_name.upper()} {title_label} [{fault_type}]",
    )

    val_scores = (
        reconstruction_scores(best_model, X_val_labeled, batch_size)
        if len(X_val_labeled) > 0
        else np.empty(0, dtype=np.float32)
    )
    raw_threshold, threshold_source, _sweep_df = _select_ae_threshold(
        best_model,
        X_val_normal,
        val_scores,
        y_val_labeled,
        batch_size,
        f"{title_label} [{fault_type}]",
        val_meta=val_meta,
    )
    val_metrics = evaluate_at_threshold(y_val_labeled, val_scores, raw_threshold)

    test_sequences = _load_test_sequences_from_bundle(data_bundle, asset_filter=asset_filter)
    test_sequences, test_filter_summary = _filter_test_sequences_by_event_scope(
        test_sequences,
        fault_type,
        test_event_scope,
        event_lookup,
    )
    if not test_sequences:
        raise RuntimeError(
            f"No test sequences found for {title_label} [{fault_type}] "
            f"with test_event_scope='{test_event_scope}'."
        )
    if test_event_scope != "all" and test_filter_summary["matching_fault_event_count"] == 0:
        raise RuntimeError(
            f"No matching {fault_type} anomaly events found for {title_label}. "
            f"Check --event-info-csv and --assets."
        )

    all_raw_scores: list[np.ndarray] = []
    all_window_labels: list[np.ndarray] = []
    for seq in test_sequences:
        X_seq = np.asarray(seq["X"], dtype=np.float32)[:, :, feature_indices]
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

    ewma_alpha = EWMA_ALPHA[fault_type]
    normal_val_raw = reconstruction_scores(best_model, X_val_normal, batch_size)
    ewma_init = float(np.median(normal_val_raw))
    ewma_val_normal = compute_ewma_scores(normal_val_raw, ewma_alpha, init_value=ewma_init)
    ewma_threshold = float(np.quantile(ewma_val_normal, 0.95))

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

    low_confidence_assets = [
        int(asset) for asset in asset_filter if is_low_confidence(int(asset), fault_type)
    ]
    low_conf = bool(low_confidence_assets)

    asset_ids_col = np.concatenate(
        [np.full(len(s["y"]), int(s["asset_id"]), dtype=np.int64) for s in test_sequences]
    )
    seq_ids_col = np.concatenate(
        [np.full(len(s["y"]), int(s["sequence_id"]), dtype=np.int64) for s in test_sequences]
    )
    event_fault_type_col = np.concatenate(
        [
            np.full(len(s["y"]), s.get("event_fault_type") or "", dtype=object)
            for s in test_sequences
        ]
    )
    event_description_col = np.concatenate(
        [
            np.full(len(s["y"]), s.get("event_description") or "", dtype=object)
            for s in test_sequences
        ]
    )
    pd.DataFrame(
        {
            "asset_id": asset_ids_col,
            "sequence_id": seq_ids_col,
            "event_fault_type": event_fault_type_col,
            "event_description": event_description_col,
            "true_label": flat_labels,
            "raw_score": flat_raw_scores,
            "raw_predicted": (flat_raw_scores >= raw_threshold).astype(np.int8),
            "ewma_score": flat_ewma_scores,
            "ewma_predicted": (flat_ewma_scores >= ewma_threshold).astype(np.int8),
        }
    ).to_csv(output_dir / "test_window_predictions.csv", index=False)

    event_rows = []
    for seq, raw_sc, ewma_sc in zip(test_sequences, all_raw_scores, all_ewma_scores):
        raw_event_score = float(np.asarray(raw_sc).max())
        ewma_event_score = float(np.asarray(ewma_sc).max())
        event_rows.append(
            {
                "asset_id": int(seq["asset_id"]),
                "sequence_id": int(seq["sequence_id"]),
                "event_fault_type": seq.get("event_fault_type") or "",
                "event_description": seq.get("event_description") or "",
                "event_true_label": int(np.asarray(seq["y"]).max()),
                "raw_event_score": raw_event_score,
                "raw_event_predicted": int(raw_event_score >= raw_threshold),
                "ewma_event_score": ewma_event_score,
                "ewma_event_predicted": int(ewma_event_score >= ewma_threshold),
            }
        )
    pd.DataFrame(event_rows).to_csv(output_dir / "test_event_predictions.csv", index=False)

    _save_ewma_score_plot(
        test_sequences,
        all_raw_scores,
        all_ewma_scores,
        raw_threshold,
        ewma_threshold,
        output_dir / "ewma_score_series.png",
        f"{title_label} [{fault_type}] - Raw vs EWMA Reconstruction Error",
    )

    save_json(
        output_dir / "threshold_raw.json",
        {
            "threshold": raw_threshold,
            "source": threshold_source,
            "scope": scope,
            "asset_id": asset_id,
            "asset_ids": asset_filter,
            "fault_type": fault_type,
            "test_event_scope": test_event_scope,
        },
    )
    save_json(
        output_dir / "threshold_ewma.json",
        {
            "threshold": ewma_threshold,
            "alpha": ewma_alpha,
            "source": "normal_val_95th_percentile_ewma",
            "ewma_init_value": ewma_init,
            "scope": scope,
            "asset_id": asset_id,
            "asset_ids": asset_filter,
            "fault_type": fault_type,
            "test_event_scope": test_event_scope,
        },
    )

    summary = {
        "model_name": model_name,
        "scope": scope,
        "asset_id": asset_id,
        "asset_ids": asset_filter,
        "asset_count": len(asset_filter),
        "fault_type": fault_type,
        "test_event_scope": test_event_scope,
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
        "low_confidence_assets": low_confidence_assets,
        "test_matching_fault_events": test_filter_summary["matching_fault_event_count"],
        "test_normal_events": test_filter_summary["normal_event_count"],
    }

    payload = {
        "model_name": model_name,
        "scope": scope,
        "asset_id": asset_id,
        "asset_ids": asset_filter,
        "fault_type": fault_type,
        "test_event_scope": test_event_scope,
        "feature_indices": list(feature_indices),
        "selected_features": selected_features,
        "n_features": n_features,
        "ewma_alpha": ewma_alpha,
        "train_shape": list(X_train.shape),
        "val_normal_shape": list(X_val_normal.shape),
        "val_labeled_shape": (
            list(X_val_labeled.shape) if len(X_val_labeled) > 0 else [0, 0, n_features]
        ),
        "raw_threshold": raw_threshold,
        "raw_threshold_source": threshold_source,
        "ewma_threshold": ewma_threshold,
        "ewma_threshold_source": "normal_val_95th_percentile_ewma",
        "ewma_init_value": ewma_init,
        "low_confidence": low_conf,
        "low_confidence_assets": low_confidence_assets,
        "validation_event_filter": val_filter_summary,
        "test_event_filter": test_filter_summary,
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
    event_info_path: Path | None = None,
    test_event_scope: str = "all",
) -> dict:
    """Train one per-asset fault-type detector."""
    asset_bundle = load_autoencoder_asset_bundle(asset_dir)
    return _run_fault_type_detector_from_bundle(
        scope="per_asset",
        fault_type=fault_type,
        data_bundle=asset_bundle,
        classifier_bundle=classifier_bundle,
        output_dir=output_dir,
        asset_filter=[int(asset_id)],
        title_label=f"Asset {asset_id}",
        asset_id=int(asset_id),
        event_info_path=event_info_path,
        test_event_scope=test_event_scope,
        random_seed=random_seed,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        encoder_units=encoder_units,
        bottleneck_units=bottleneck_units,
        overwrite=overwrite,
    )


def run_cross_asset_fault_type_detector_experiment(
    fault_type: str,
    autoencoder_root: Path,
    classifier_bundle: dict,
    output_dir: Path,
    asset_ids: list[int],
    random_seed: int = 42,
    model_name: str = "lstm_ae",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    encoder_units: int = 32,
    bottleneck_units: int = 16,
    overwrite: bool = False,
    event_info_path: Path | None = None,
    test_event_scope: str = "matching-fault",
) -> dict:
    """Train one pooled model for a fault type across matching assets."""
    asset_filter = sorted({int(asset_id) for asset_id in asset_ids})
    if not asset_filter:
        raise ValueError(f"No assets were provided for fault_type='{fault_type}'.")

    global_bundle = load_autoencoder_global_bundle(autoencoder_root, asset_filter=asset_filter)
    return _run_fault_type_detector_from_bundle(
        scope="fault_type",
        fault_type=fault_type,
        data_bundle=global_bundle,
        classifier_bundle=classifier_bundle,
        output_dir=output_dir,
        asset_filter=asset_filter,
        title_label=f"Fault type {fault_type} assets {asset_filter}",
        asset_id=None,
        event_info_path=event_info_path,
        test_event_scope=test_event_scope,
        random_seed=random_seed,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        encoder_units=encoder_units,
        bottleneck_units=bottleneck_units,
        overwrite=overwrite,
    )
