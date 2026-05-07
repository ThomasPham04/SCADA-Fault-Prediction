"""
train_fault_type_detectors.py
CLI entry point for per-asset and cross-asset fault-type autoencoder detectors.

Examples
--------
    # Existing behavior: per-asset hydraulic detector for asset 10, evaluated
    # on all asset-10 test events.
    python src/training/scripts/train_fault_type_detectors.py \
        --assets 10 --fault-types hydraulic

    # Per-asset detector, but evaluate only hydraulic anomaly events plus
    # normal events for asset 10.
    python src/training/scripts/train_fault_type_detectors.py \
        --assets 10 --fault-types hydraulic --test-event-scope matching-fault

    # New experiment mode: one pooled hydraulic model across all hydraulic
    # assets, tested only on hydraulic anomaly events plus normal events.
    python src/training/scripts/train_fault_type_detectors.py \
        --experiment-mode fault-type --fault-types hydraulic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path when run from the repo root.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from config import PROCESSED_DATA_DIR, RANDOM_SEED, RESULTS_DIR
from fault_type_config import ASSET_FAULT_TYPES, FAULT_TYPE_FEATURE_GROUPS
from training.fault_type_experiments import (
    run_cross_asset_fault_type_detector_experiment,
    run_fault_type_detector_experiment,
)
from training.sequence_io import load_classifier_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fault-type autoencoder detectors with raw + EWMA scoring."
    )
    parser.add_argument(
        "--experiment-mode",
        choices=["per-asset", "fault-type"],
        default="per-asset",
        help=(
            "per-asset keeps the existing one asset/fault model layout. "
            "fault-type trains one pooled model per fault type across matching assets."
        ),
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
        help=(
            "Asset IDs to include. In per-asset mode this trains those assets. "
            "In fault-type mode this limits the pooled assets."
        ),
    )
    parser.add_argument(
        "--fault-types",
        nargs="+",
        default=None,
        help=(
            "Fault types to train. Default: all valid fault types for the selected mode."
        ),
    )
    parser.add_argument(
        "--test-event-scope",
        choices=["all", "matching-fault", "matching-anomaly-only"],
        default=None,
        help=(
            "Test/validation event filter. Default: all in per-asset mode, "
            "matching-fault in fault-type mode."
        ),
    )
    parser.add_argument(
        "--event-info-csv",
        default=str(Path("Dataset") / "raw" / "Wind Farm A" / "event_info.csv"),
        help="Event metadata CSV used for fault-type filtering.",
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


def _selected_fault_types(args: argparse.Namespace) -> list[str]:
    if args.fault_types:
        return list(dict.fromkeys(args.fault_types))
    if args.experiment_mode == "fault-type":
        return list(FAULT_TYPE_FEATURE_GROUPS.keys())
    return []


def _mode_test_event_scope(args: argparse.Namespace) -> str:
    if args.test_event_scope:
        return args.test_event_scope
    return "matching-fault" if args.experiment_mode == "fault-type" else "all"


def _matching_assets_for_fault_type(
    fault_type: str,
    requested_assets: list[int] | None,
) -> list[int]:
    requested = set(requested_assets or [])
    asset_ids = []
    for asset_id, fault_types in ASSET_FAULT_TYPES.items():
        if fault_type not in fault_types:
            continue
        if requested and asset_id not in requested:
            continue
        asset_ids.append(int(asset_id))
    return sorted(asset_ids)


def _print_summary(s: dict) -> None:
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
    print(
        f"  test events: fault={s.get('test_matching_fault_events', 0)}  "
        f"normal={s.get('test_normal_events', 0)}  "
        f"scope={s.get('test_event_scope')}"
    )
    if s.get("low_confidence"):
        print(f"  [!] Low-confidence assets: {s.get('low_confidence_assets')}")


def _run_per_asset_mode(
    args: argparse.Namespace,
    classifier_bundle: dict,
    ae_root: Path,
    results_dir: Path,
    window_tag: str,
    test_event_scope: str,
) -> list[dict]:
    asset_ids = args.assets if args.assets is not None else list(ASSET_FAULT_TYPES.keys())
    fault_type_filter = set(args.fault_types) if args.fault_types else None
    event_info_path = Path(args.event_info_csv) if test_event_scope != "all" else None
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
            model_dir_name = (
                args.model
                if test_event_scope == "all"
                else f"{args.model}__{test_event_scope.replace('-', '_')}"
            )
            out_dir = results_dir / window_tag / f"asset_{asset_id}" / fault_type / model_dir_name
            print(f"\n{'='*60}")
            print(
                f"  Mode: per-asset  |  asset: {asset_id}  |  "
                f"fault_type: {fault_type}  |  model: {args.model}"
            )
            print(f"  Test event scope: {test_event_scope}")
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
                    event_info_path=event_info_path,
                    test_event_scope=test_event_scope,
                )
                summary_rows.append(result["summary"])
                _print_summary(result["summary"])
            except Exception as exc:
                print(f"  [ERROR] {exc}")

    return summary_rows


def _run_fault_type_mode(
    args: argparse.Namespace,
    classifier_bundle: dict,
    ae_root: Path,
    results_dir: Path,
    window_tag: str,
    test_event_scope: str,
) -> list[dict]:
    fault_types = _selected_fault_types(args)
    event_info_path = Path(args.event_info_csv)
    summary_rows = []

    for fault_type in fault_types:
        if fault_type not in FAULT_TYPE_FEATURE_GROUPS:
            print(f"[WARN] Unknown fault type {fault_type}, skipping.")
            continue

        asset_ids = _matching_assets_for_fault_type(fault_type, args.assets)
        if not asset_ids:
            print(f"[WARN] No matching assets for fault_type={fault_type}, skipping.")
            continue

        asset_tag = "assets_" + "_".join(str(asset_id) for asset_id in asset_ids)
        out_dir = (
            results_dir
            / window_tag
            / f"fault_type_{fault_type}"
            / asset_tag
            / args.model
        )
        print(f"\n{'='*60}")
        print(
            f"  Mode: fault-type  |  fault_type: {fault_type}  |  "
            f"assets: {asset_ids}  |  model: {args.model}"
        )
        print(f"  Test event scope: {test_event_scope}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")

        try:
            result = run_cross_asset_fault_type_detector_experiment(
                fault_type=fault_type,
                autoencoder_root=ae_root,
                classifier_bundle=classifier_bundle,
                output_dir=out_dir,
                asset_ids=asset_ids,
                random_seed=args.seed,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                encoder_units=args.encoder_units,
                bottleneck_units=args.bottleneck_units,
                overwrite=args.overwrite,
                event_info_path=event_info_path,
                test_event_scope=test_event_scope,
            )
            summary_rows.append(result["summary"])
            _print_summary(result["summary"])
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    return summary_rows


def main() -> None:
    args = _parse_args()

    exports_dir = Path(args.exports_dir)
    results_dir = Path(args.results_dir)
    window_tag = f"window_{args.window}h"
    test_event_scope = _mode_test_event_scope(args)

    classifier_dir = exports_dir / window_tag / "classifier"
    ae_root = exports_dir / window_tag / "autoencoder"

    if not classifier_dir.exists():
        print(f"[ERROR] Classifier export not found: {classifier_dir}")
        sys.exit(1)

    classifier_bundle = load_classifier_bundle(classifier_dir)

    if args.experiment_mode == "fault-type":
        summary_rows = _run_fault_type_mode(
            args,
            classifier_bundle,
            ae_root,
            results_dir,
            window_tag,
            test_event_scope,
        )
    else:
        summary_rows = _run_per_asset_mode(
            args,
            classifier_bundle,
            ae_root,
            results_dir,
            window_tag,
            test_event_scope,
        )

    if summary_rows:
        suffix = "fault_type" if args.experiment_mode == "fault-type" else "per_asset"
        summary_path = results_dir / window_tag / f"summary_{suffix}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
