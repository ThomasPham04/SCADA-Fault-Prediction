"""
Analyze why a normal-behavior autoencoder does or does not separate anomalies.

This script loads a saved handoff autoencoder run, reconstructs validation/test
windows, and writes feature-level reconstruction-error diagnostics. Labels are
used only for analysis, not for model fitting or threshold calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.sequence_utils import cleanup_tf, save_json, set_random_seed

from experiments.train_normal_behavior_handoff import (
    DEFAULT_DATA_DIR,
    load_feature_list,
    load_split,
    make_windows,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "results" / "normal_behavior_handoff" / "window_24h_stride_1h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature-level reconstruction errors for handoff AE runs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--models", nargs="+", default=["dense_ae", "cnn_gru"])
    parser.add_argument("--window-size", type=int, default=144)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--cadence-minutes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-quantile", type=float, default=0.80)
    parser.add_argument("--top-k-features", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_windows(args: argparse.Namespace, feature_cols: list[str]):
    val_df = load_split(args.data_dir, "validation_normal", feature_cols)
    test_df = load_split(args.data_dir, "test_prediction", feature_cols)
    X_val, val_meta, _ = make_windows(
        val_df,
        feature_cols,
        "validation_normal",
        args.window_size,
        args.stride,
        args.cadence_minutes,
    )
    X_test, test_meta, _ = make_windows(
        test_df,
        feature_cols,
        "test_prediction",
        args.window_size,
        args.stride,
        args.cadence_minutes,
    )
    return X_val, val_meta, X_test, test_meta


def score_variants_from_error(error: np.ndarray) -> dict[str, np.ndarray]:
    squared = np.square(error)
    absolute = np.abs(error)
    per_timestep_l2 = np.sqrt(np.sum(squared, axis=2))
    per_feature_mse = squared.mean(axis=1)
    return {
        "mean_l2": per_timestep_l2.mean(axis=1),
        "mean_mse": squared.mean(axis=(1, 2)),
        "mean_abs": absolute.mean(axis=(1, 2)),
        "max_timestep_l2": per_timestep_l2.max(axis=1),
        "p95_timestep_l2": np.quantile(per_timestep_l2, 0.95, axis=1),
        "last_24_mean_l2": per_timestep_l2[:, -24:].mean(axis=1),
        "last_72_mean_l2": per_timestep_l2[:, -72:].mean(axis=1),
        "max_feature_mse": per_feature_mse.max(axis=1),
        "top5_feature_mse": np.sort(per_feature_mse, axis=1)[:, -5:].mean(axis=1),
    }


def feature_frame(
    feature_cols: list[str],
    val_feature_mse: np.ndarray,
    test_feature_mse: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    normal_mask = y_true == 0
    anomaly_mask = y_true == 1
    val_mean = val_feature_mse.mean(axis=0)
    val_std = val_feature_mse.std(axis=0) + 1e-8
    val_p95 = np.quantile(val_feature_mse, 0.95, axis=0) + 1e-8

    rows = []
    for idx, feature in enumerate(feature_cols):
        values = test_feature_mse[:, idx]
        normal_values = values[normal_mask]
        anomaly_values = values[anomaly_mask]
        if len(np.unique(y_true)) > 1:
            pr_auc = float(average_precision_score(y_true, values))
            roc_auc = float(roc_auc_score(y_true, values))
        else:
            pr_auc = np.nan
            roc_auc = np.nan
        rows.append(
            {
                "feature": feature,
                "feature_index": idx,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "validation_mean_mse": float(val_mean[idx]),
                "validation_p95_mse": float(val_p95[idx]),
                "normal_median_mse": float(np.median(normal_values)),
                "anomaly_median_mse": float(np.median(anomaly_values)),
                "normal_p90_mse": float(np.quantile(normal_values, 0.90)),
                "anomaly_p90_mse": float(np.quantile(anomaly_values, 0.90)),
                "anomaly_over_normal_median": float(
                    np.median(anomaly_values) / (np.median(normal_values) + 1e-8)
                ),
                "anomaly_over_validation_p95": float(
                    np.median(anomaly_values) / val_p95[idx]
                ),
                "mean_z_gap": float(
                    ((anomaly_values.mean() - normal_values.mean()) / val_std[idx])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["roc_auc", "pr_auc", "anomaly_over_normal_median"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def score_sweep_frame(
    val_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    quantiles: list[float],
) -> pd.DataFrame:
    rows = []
    for score_name, val_score in val_scores.items():
        test_score = test_scores[score_name]
        pr_auc = float(average_precision_score(y_true, test_score))
        roc_auc = float(roc_auc_score(y_true, test_score))
        for quantile in quantiles:
            threshold = float(np.quantile(val_score, quantile))
            y_pred = (test_score >= threshold).astype(np.int8)
            rows.append(
                {
                    "score": score_name,
                    "quantile": float(quantile),
                    "threshold": threshold,
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "predicted_window_count": int(y_pred.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["f1", "precision"], ascending=[False, False])


def top_feature_dict(
    feature_cols: list[str],
    values: np.ndarray,
    top_k: int,
) -> dict:
    order = np.argsort(values)[::-1][:top_k]
    return {
        f"top_{rank + 1}_feature": feature_cols[idx]
        for rank, idx in enumerate(order)
    } | {
        f"top_{rank + 1}_value": float(values[idx])
        for rank, idx in enumerate(order)
    }


def group_feature_frame(
    feature_cols: list[str],
    feature_ratio: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    groups = {
        "true_positive": (y_true == 1) & (y_pred == 1),
        "false_negative": (y_true == 1) & (y_pred == 0),
        "false_positive": (y_true == 0) & (y_pred == 1),
        "true_negative": (y_true == 0) & (y_pred == 0),
    }
    rows = []
    for group_name, mask in groups.items():
        if not mask.any():
            rows.append({"group": group_name, "window_count": 0})
            continue
        mean_values = feature_ratio[mask].mean(axis=0)
        row = {
            "group": group_name,
            "window_count": int(mask.sum()),
            "mean_top_feature_ratio": float(np.max(mean_values)),
        }
        row.update(top_feature_dict(feature_cols, mean_values, top_k))
        rows.append(row)
    return pd.DataFrame(rows)


def event_feature_frame(
    feature_cols: list[str],
    test_meta: pd.DataFrame,
    feature_ratio: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    analysis_df = test_meta.copy()
    analysis_df["score"] = scores.astype(np.float32)
    analysis_df["predicted_label"] = y_pred.astype(np.int8)
    rows = []
    for event_id, group in analysis_df.sort_values(["event_id", "end_time"]).groupby("event_id"):
        idx = group.index.to_numpy()
        positive_mask = group["window_label_any_event_interval"].to_numpy(dtype=np.int8) == 1
        ratio_idx = idx[positive_mask] if positive_mask.any() else idx
        mean_values = feature_ratio[ratio_idx].mean(axis=0)
        row = {
            "event_id": event_id,
            "asset_id": group["asset_id"].iloc[-1],
            "event_label": group["event_label"].iloc[-1] if "event_label" in group else "",
            "event_description": group["event_description"].iloc[-1] if "event_description" in group else "",
            "window_count": int(len(group)),
            "positive_window_count": int(positive_mask.sum()),
            "predicted_window_count": int(group["predicted_label"].sum()),
            "mean_score": float(group["score"].mean()),
            "max_score": float(group["score"].max()),
            "mean_top_feature_ratio": float(np.max(mean_values)),
        }
        row.update(top_feature_dict(feature_cols, mean_values, top_k))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["event_label", "mean_top_feature_ratio"],
        ascending=[True, False],
        kind="mergesort",
    )


def analyze_model(
    model_name: str,
    run_dir: Path,
    X_val: np.ndarray,
    X_test: np.ndarray,
    test_meta: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict:
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir = run_dir / "error_analysis"
    if output_dir.exists() and not args.overwrite and (output_dir / "summary.json").exists():
        raise FileExistsError(f"Analysis already exists: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    val_recon = model.predict(X_val, batch_size=args.batch_size, verbose=0).astype(np.float32)
    test_recon = model.predict(X_test, batch_size=args.batch_size, verbose=0).astype(np.float32)

    val_error = X_val.astype(np.float32) - val_recon
    test_error = X_test.astype(np.float32) - test_recon
    val_feature_mse = np.square(val_error).mean(axis=1)
    test_feature_mse = np.square(test_error).mean(axis=1)
    val_feature_p95 = np.quantile(val_feature_mse, 0.95, axis=0) + 1e-8
    feature_ratio = test_feature_mse / val_feature_p95

    y_true = test_meta["window_label_any_event_interval"].to_numpy(dtype=np.int8)
    val_scores = score_variants_from_error(val_error)
    test_scores = score_variants_from_error(test_error)

    score_sweep = score_sweep_frame(
        val_scores,
        test_scores,
        y_true,
        [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99],
    )
    score_sweep.to_csv(output_dir / "score_variant_sweep.csv", index=False)

    feature_metrics = feature_frame(feature_cols, val_feature_mse, test_feature_mse, y_true)
    feature_metrics.to_csv(output_dir / "feature_separation.csv", index=False)

    primary_score = "mean_l2"
    threshold = float(np.quantile(val_scores[primary_score], args.threshold_quantile))
    y_pred = (test_scores[primary_score] >= threshold).astype(np.int8)

    group_features = group_feature_frame(
        feature_cols,
        feature_ratio,
        y_true,
        y_pred,
        args.top_k_features,
    )
    group_features.to_csv(output_dir / "confusion_group_top_features.csv", index=False)

    event_features = event_feature_frame(
        feature_cols,
        test_meta,
        feature_ratio,
        test_scores[primary_score],
        y_pred,
        args.top_k_features,
    )
    event_features.to_csv(output_dir / "event_top_features.csv", index=False)

    best_score_row = score_sweep.iloc[0].to_dict()
    best_score_name = str(best_score_row["score"])
    best_threshold = float(best_score_row["threshold"])
    best_score_values = test_scores[best_score_name]
    best_y_pred = (best_score_values >= best_threshold).astype(np.int8)
    best_pred_df = test_meta.copy()
    best_pred_df["score_name"] = best_score_name
    best_pred_df["score"] = best_score_values.astype(np.float32)
    best_pred_df["threshold"] = best_threshold
    best_pred_df["threshold_quantile"] = float(best_score_row["quantile"])
    best_pred_df["predicted_label"] = best_y_pred
    best_pred_df.to_csv(output_dir / "best_score_variant_predictions.csv", index=False)

    primary_metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "threshold_quantile": float(args.threshold_quantile),
        "predicted_window_count": int(y_pred.sum()),
    }
    summary = {
        "model_name": model_name,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "test_window_count": int(len(y_true)),
        "positive_window_count": int(y_true.sum()),
        "negative_window_count": int((y_true == 0).sum()),
        "primary_score": primary_score,
        "primary_score_metrics": primary_metrics,
        "best_score_variant": best_score_row,
        "top_features_by_roc_auc": feature_metrics.head(args.top_k_features).to_dict(
            orient="records"
        ),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_seed)
    cleanup_tf()

    feature_cols = load_feature_list(args.data_dir)
    X_val, _, X_test, test_meta = load_windows(args, feature_cols)
    test_meta = test_meta.reset_index(drop=True)

    summaries = []
    for model_name in args.models:
        run_dir = args.run_root / model_name
        print(f"\n[Analyze] {model_name}: {run_dir}")
        summary = analyze_model(
            model_name,
            run_dir,
            X_val,
            X_test,
            test_meta,
            feature_cols,
            args,
        )
        summaries.append(
            {
                "model_name": model_name,
                "primary_precision": summary["primary_score_metrics"]["precision"],
                "primary_recall": summary["primary_score_metrics"]["recall"],
                "primary_f1": summary["primary_score_metrics"]["f1"],
                "best_score": summary["best_score_variant"]["score"],
                "best_score_quantile": summary["best_score_variant"]["quantile"],
                "best_score_precision": summary["best_score_variant"]["precision"],
                "best_score_recall": summary["best_score_variant"]["recall"],
                "best_score_f1": summary["best_score_variant"]["f1"],
                "best_score_pr_auc": summary["best_score_variant"]["pr_auc"],
                "best_score_roc_auc": summary["best_score_variant"]["roc_auc"],
            }
        )
    comparison = pd.DataFrame(summaries).sort_values(
        ["best_score_f1", "primary_f1"],
        ascending=[False, False],
        kind="mergesort",
    )
    output_path = args.run_root / "error_analysis_comparison.csv"
    if output_path.exists():
        previous = pd.read_csv(output_path)
        comparison = (
            pd.concat(
                [previous[~previous["model_name"].isin(comparison["model_name"])], comparison],
                ignore_index=True,
            )
            .sort_values(
                ["best_score_f1", "primary_f1"],
                ascending=[False, False],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
    comparison.to_csv(output_path, index=False)
    print("\nError-analysis comparison")
    print(comparison.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
