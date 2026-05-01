"""
main.py - SCADA Fault Prediction Pipeline
Single entry point for the combined-sequence training flow.

Usage examples:
    # 1. Prepare windowed sequence exports from one combined CSV
    python src/main.py prepare --csv df_final.csv --feature-file final_features.csv --window-hours 24

    # 2. Train sequence classifiers and per-asset autoencoders from those exports
    python src/main.py train-sequences --windows 24

    # 3. Train a single model (auto-routes to classifier or autoencoder)
    python src/main.py train-sequences --windows 24 --model lstm
    python src/main.py train-sequences --windows 24 --model lstm gru_ae
"""

from __future__ import annotations

import argparse
import os
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def run_prepare(args: argparse.Namespace) -> None:
    """Prepare classifier and autoencoder exports from a combined CSV."""
    if not args.csv:
        raise SystemExit(
            "prepare requires --csv. The CSV must contain time_stamp, asset_id, "
            "train_test, sequence_id, and label or status_type_id."
        )

    from data_pipeline.preprocessing.combined_sequence_pipeline import (
        CombinedSequencePipeline,
        looks_like_combined_sequence_csv,
    )

    if not looks_like_combined_sequence_csv(args.csv):
        raise SystemExit(
            "Unsupported CSV schema. This CLI now supports only the combined-sequence "
            "schema: time_stamp, asset_id, train_test, sequence_id, and label or "
            "status_type_id."
        )

    pipeline = CombinedSequencePipeline(
        csv_path=args.csv,
        feature_file=args.feature_file,
        output_dir=args.sequence_output_dir,
        selected_windows_hours=args.window_hours,
        window_candidates_hours=args.window_candidates_hours,
        top_k_windows=args.top_k_windows,
        expected_feature_count=args.expected_feature_count,
        scaler_type=args.combined_scaler,
        validation_source=args.validation_source,
        prediction_val_ratio=args.prediction_val_ratio,
        run_window_search=not args.skip_window_search,
        random_seed=args.seed,
    )
    pipeline.run()


def run_train_sequences(args: argparse.Namespace) -> None:
    """Train global sequence classifiers and per-asset sequence autoencoders."""
    from config import PROCESSED_DATA_DIR, RESULTS_DIR
    from training.sequence_model_trainer import SequenceModelTrainer, SequenceTrainingConfig

    exports_dir = args.exports_dir or os.path.join(PROCESSED_DATA_DIR, "sequence_exports")
    results_dir = args.results_dir or os.path.join(RESULTS_DIR, "sequence_training_results")

    _classifier_choices = {"lstm", "gru", "cnn_lstm", "cnn_gru"}
    _autoencoder_choices = {"lstm_ae", "gru_ae"}

    if args.model:
        unknown = [m for m in args.model if m not in _classifier_choices | _autoencoder_choices]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Choose from: "
                             f"{sorted(_classifier_choices | _autoencoder_choices)}")
        classifier_models = [m for m in args.model if m in _classifier_choices]
        autoencoder_models = [m for m in args.model if m in _autoencoder_choices]
    else:
        classifier_models = [] if args.skip_classifiers else args.classifier_models
        autoencoder_models = [] if args.skip_autoencoders else args.autoencoder_models

    config = SequenceTrainingConfig(
        exports_dir=exports_dir,
        results_dir=results_dir,
        windows=args.windows,
        classifier_models=classifier_models,
        autoencoder_models=autoencoder_models,
        asset_filter=args.assets,
        random_seed=args.seed,
        overwrite=args.overwrite,
        save_predictions=not args.no_save_predictions,
        classifier_epochs=args.classifier_epochs,
        classifier_batch_size=args.classifier_batch_size,
        autoencoder_epochs=args.autoencoder_epochs,
        autoencoder_batch_size=args.autoencoder_batch_size,
    )
    SequenceModelTrainer(config).run()


def add_combined_csv_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        metavar="PATH",
        help="Combined sequence CSV with asset_id and sequence_id boundaries.",
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Feature list CSV for combined-sequence exports, e.g. final_features.csv.",
    )
    parser.add_argument(
        "--sequence-output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output root for sequence exports.",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        nargs="+",
        default=None,
        metavar="H",
        help="Export these window sizes directly and skip window search.",
    )
    parser.add_argument(
        "--window-candidates-hours",
        type=int,
        nargs="+",
        default=None,
        metavar="H",
        help="Window sizes to test during window search.",
    )
    parser.add_argument(
        "--top-k-windows",
        type=int,
        default=1,
        help="Number of best windows to export after window search.",
    )
    parser.add_argument(
        "--skip-window-search",
        action="store_true",
        help="Skip probe-model search and export the first candidate window only.",
    )
    parser.add_argument(
        "--expected-feature-count",
        type=int,
        default=None,
        help="Optional guard for the selected feature count.",
    )
    parser.add_argument(
        "--combined-scaler",
        type=str,
        default="minmax",
        choices=["minmax", "standard"],
        help="Scaler for combined CSV exports.",
    )
    parser.add_argument(
        "--validation-source",
        type=str,
        default="train_tail",
        choices=["train_tail", "prediction"],
        help=(
            "How to create the validation split. 'train_tail' keeps the legacy "
            "behavior by using the tail of each train segment. 'prediction' "
            "uses the first part of each prediction segment for validation and "
            "the rest for test."
        ),
    )
    parser.add_argument(
        "--prediction-val-ratio",
        type=float,
        default=0.5,
        help=(
            "When --validation-source prediction is used, fraction of each "
            "asset_id + sequence_id prediction segment assigned to validation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)


def add_sequence_training_flags(parser: argparse.ArgumentParser) -> None:
    classifier_choices = ["lstm", "gru", "cnn_lstm", "cnn_gru"]
    autoencoder_choices = ["lstm_ae", "gru_ae"]
    all_model_choices = classifier_choices + autoencoder_choices
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        choices=all_model_choices,
        metavar="MODEL",
        help=(
            f"Train only these model(s). Automatically routes classifiers "
            f"({', '.join(classifier_choices)}) and autoencoders "
            f"({', '.join(autoencoder_choices)}). "
            f"Overrides --classifier-models, --autoencoder-models, "
            f"--skip-classifiers, and --skip-autoencoders."
        ),
    )
    parser.add_argument(
        "--exports-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Root folder containing window_<H>h sequence exports.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output folder for sequence training results.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[24],
        metavar="H",
        help="Window sizes to train.",
    )
    parser.add_argument(
        "--classifier-models",
        type=str,
        nargs="+",
        default=classifier_choices,
        choices=classifier_choices,
        help="Supervised sequence classifiers to train.",
    )
    parser.add_argument(
        "--autoencoder-models",
        type=str,
        nargs="+",
        default=autoencoder_choices,
        choices=autoencoder_choices,
        help="Per-asset sequence autoencoder architectures to train.",
    )
    parser.add_argument(
        "--assets",
        type=int,
        nargs="+",
        default=None,
        help="Restrict autoencoder training to specific asset IDs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even if metrics already exist.",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Skip verbose prediction CSV outputs.",
    )
    parser.add_argument(
        "--skip-classifiers",
        action="store_true",
        help="Do not train supervised classifiers.",
    )
    parser.add_argument(
        "--skip-autoencoders",
        action="store_true",
        help="Do not train per-asset sequence autoencoders.",
    )
    parser.add_argument("--classifier-epochs", type=int, default=25)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--autoencoder-epochs", type=int, default=30)
    parser.add_argument("--autoencoder-batch-size", type=int, default=128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SCADA Fault Prediction - combined-sequence pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare sequence exports from a combined CSV.",
    )
    add_combined_csv_flags(prepare_parser)

    train_sequences_parser = subparsers.add_parser(
        "train-sequences",
        help="Train sequence classifiers and autoencoders from exports.",
    )
    add_sequence_training_flags(train_sequences_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "train-sequences":
        run_train_sequences(args)


if __name__ == "__main__":
    main()
