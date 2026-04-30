"""
Train sequence classifiers and autoencoders from combined CSV exports.

Requires sequence exports from:
    python src/main.py prepare --csv df_final.csv --feature-file final_features.csv

Usage:
    python -m training.scripts.train_sequence_models --windows 24
    python src/main.py train-sequences --windows 24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

DEFAULT_SEQUENCE_WINDOWS = [24]
DEFAULT_CLASSIFIER_MODELS = ["lstm", "gru", "cnn_lstm", "cnn_gru"]
DEFAULT_AUTOENCODER_MODELS = ["lstm_ae", "gru_ae"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train sequence models from exported classifier/autoencoder NPY bundles."
    )
    ap.add_argument(
        "--exports-dir",
        type=Path,
        default=Path(PROCESSED_DATA_DIR) / "sequence_exports",
        help="Root folder containing window_<H>h sequence exports.",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "sequence_training_results",
        help="Output folder for trained models, metrics, and plots.",
    )
    ap.add_argument("--windows", type=int, nargs="+", default=DEFAULT_SEQUENCE_WINDOWS)
    ap.add_argument(
        "--classifier-models",
        type=str,
        nargs="+",
        default=DEFAULT_CLASSIFIER_MODELS,
        choices=DEFAULT_CLASSIFIER_MODELS,
    )
    ap.add_argument(
        "--autoencoder-models",
        type=str,
        nargs="+",
        default=DEFAULT_AUTOENCODER_MODELS,
        choices=DEFAULT_AUTOENCODER_MODELS,
    )
    ap.add_argument("--assets", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--skip-classifiers", action="store_true")
    ap.add_argument("--skip-autoencoders", action="store_true")
    ap.add_argument("--classifier-epochs", type=int, default=25)
    ap.add_argument("--classifier-batch-size", type=int, default=256)
    ap.add_argument("--autoencoder-epochs", type=int, default=30)
    ap.add_argument("--autoencoder-batch-size", type=int, default=128)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from training.sequence_model_trainer import SequenceModelTrainer, SequenceTrainingConfig

    config = SequenceTrainingConfig(
        exports_dir=args.exports_dir,
        results_dir=args.results_dir,
        windows=args.windows,
        classifier_models=[] if args.skip_classifiers else args.classifier_models,
        autoencoder_models=[] if args.skip_autoencoders else args.autoencoder_models,
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


if __name__ == "__main__":
    main()
