"""
train_fault_type_detectors.py
CLI entry point for training per-fault-type LSTM-AE detectors.

Usage
-----
    python src/training/scripts/train_fault_type_detectors.py

    # Train specific assets and fault types only
    python src/training/scripts/train_fault_type_detectors.py \\
        --assets 10 --fault-types hydraulic generator_bearing

    # Full run, overwrite existing results
    python src/training/scripts/train_fault_type_detectors.py \\
        --assets 0 10 11 13 21 --overwrite

Output
------
    results/fault_type_detectors/window_24h/asset_<N>/<fault_type>/lstm_ae/
        model.keras
        metrics.json          (raw + EWMA metrics, lead times, false-alarm stats)
        threshold_raw.json
        threshold_ewma.json
        history.csv / loss_history.png
        ewma_score_series.png
        test_window_predictions.csv

    results/fault_type_detectors/window_24h/summary.csv   (all models ranked)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path when run from the repo root
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from config import PROCESSED_DATA_DIR, RANDOM_SEED, RESULTS_DIR
from fault_type_config import ASSET_FAULT_TYPES
from training.fault_type_experiments import run_fault_type_detector_experiment
from training.sequence_io import load_classifier_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-fault-type LSTM-AE detectors with dual-alarm scoring."
    )
    parser.add_argument(
        "--exports-dir",
        default=str(Path(PROCESSED_DATA_DIR) / "sequence_exports"),
        help="Root of the windowed sequence exports (default: Dataset/processed/sequence_exports).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(RESULTS_DIR) / "fault_type_detectors"),
        help="Output root directory (default: results/fault_type_detectors).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Window size in hours (must match an existing export; default: 24).",
    )
    parser.add_argument(
        "--assets",
        type=int,
        nargs="+",
        default=None,
        help="Asset IDs to train.  Default: all assets in ASSET_FAULT_TYPES.",
    )
    parser.add_argument(
        "--fault-types",
        nargs="+",
        default=None,
        help="Fault types to train.  Default: all types for each asset.",
    )
    parser.add_argument(
        "--model",
        default="lstm_ae",
        choices=["lstm_ae", "gru_ae"],
        help="Autoencoder architecture (default: lstm_ae).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-units", type=int, default=32)
    parser.add_argument("--bottleneck-units", type=int, default=16)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-train even if metrics.json already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    exports_dir = Path(args.exports_dir)
    results_dir = Path(args.results_dir)
    window_tag = f"window_{args.window}h"

    classifier_dir = exports_dir / window_tag / "classifier"
    ae_root = exports_dir / window_tag / "autoencoder"

    if not classifier_dir.exists():
        print(f"[ERROR] Classifier export not found: {classifier_dir}")
        sys.exit(1)

    classifier_bundle = load_classifier_bundle(classifier_dir)

    # Determine which (asset, fault_type) pairs to train
    asset_ids = args.assets if args.assets is not None else list(ASSET_FAULT_TYPES.keys())
    fault_type_filter = set(args.fault_types) if args.fault_types else None

    summary_rows = []

    for asset_id in asset_ids:
        if asset_id not in ASSET_FAULT_TYPES:
            print(f"[WARN] asset {asset_id} not in ASSET_FAULT_TYPES, skipping.")
            continue

        fault_types = ASSET_FAULT_TYPES[asset_id]
        if fault_type_filter:
            fault_types = [ft for ft in fault_types if ft in fault_type_filter]

        asset_dir = ae_root / f"asset_{asset_id}"
        if not asset_dir.exists():
            print(f"[WARN] Asset export not found: {asset_dir}, skipping.")
            continue

        for fault_type in fault_types:
            out_dir = results_dir / window_tag / f"asset_{asset_id}" / fault_type / args.model
            print(f"\n{'='*60}")
            print(f"  Asset {asset_id}  |  fault_type: {fault_type}  |  model: {args.model}")
            print(f"  Output: {out_dir}")
            print(f"{'='*60}")

            try:
                result = run_fault_type_detector_experiment(
                    asset_id=asset_id,
                    fault_type=fault_type,
                    asset_dir=asset_dir,
                    classifier_bundle=classifier_bundle,
                    output_dir=out_dir,
                    random_seed=args.seed,
                    model_name=args.model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    encoder_units=args.encoder_units,
                    bottleneck_units=args.bottleneck_units,
                    overwrite=args.overwrite,
                )
                summary_rows.append(result["summary"])
                s = result["summary"]
                print(
                    f"  raw  F1={s['raw_f1']:.3f}  ROC-AUC={s['raw_roc_auc']}  "
                    f"lead={s['lead_time_raw_minutes']} min  "
                    f"FA={s['raw_false_alarm_rate']:.2f}"
                )
                print(
                    f"  ewma F1={s['ewma_f1']:.3f}  ROC-AUC={s['ewma_roc_auc']}  "
                    f"lead={s['lead_time_ewma_minutes']} min  "
                    f"FA={s['ewma_false_alarm_rate']:.2f}"
                )
                if s.get("low_confidence"):
                    print("  [!] Low-confidence result — only 1 anomaly event.")
            except Exception as exc:
                print(f"  [ERROR] {exc}")

    # Save combined summary CSV
    if summary_rows:
        summary_path = results_dir / window_tag / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
