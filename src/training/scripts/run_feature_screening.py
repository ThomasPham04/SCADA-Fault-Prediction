"""
Run Feature Screening - training.scripts.run_feature_screening
Entry-point script for cross-dataset feature screening on SCADA CSV files.

Usage:
    python -m src.training.scripts.run_feature_screening
    python -m src.training.scripts.run_feature_screening --dataset-dir "path\\to\\datasets"
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RESULTS_DIR, WIND_FARM_A_DATASETS
from data_pipeline.preprocessing.feature_screening import FeatureScreening


DROP_COLUMNS = {
    "time_stamp",
    "asset_id",
    "train_test",
    "status_type_id",
    "sequence_id",
    "id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature screening using point-biserial and Spearman aggregation.",
    )
    parser.add_argument("--dataset-dir", type=str, default=WIND_FARM_A_DATASETS)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "feature_screening"),
    )
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--target-col", type=str, default="label")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-mean-abs-pb-r", type=float, default=0.05)
    parser.add_argument("--min-pb-sign-consistency", type=float, default=0.60)
    parser.add_argument("--min-pb-sig-ratio", type=float, default=0.30)
    parser.add_argument("--ff-threshold", type=float, default=0.85)
    parser.add_argument("--sample-per-file", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    screening = FeatureScreening(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        csv_sep=args.sep,
        target_col=args.target_col,
        drop_columns=DROP_COLUMNS,
        alpha=args.alpha,
        min_mean_abs_pb_r=args.min_mean_abs_pb_r,
        min_pb_sign_consistency=args.min_pb_sign_consistency,
        min_pb_sig_ratio=args.min_pb_sig_ratio,
        ff_threshold=args.ff_threshold,
    )

    results = screening.run_pipeline(sample_per_file=args.sample_per_file)

    print("=" * 70)
    print("Feature Screening Complete")
    print("=" * 70)
    print(f"Selected features: {len(results['selected'])}")
    print(f"Final features after redundancy removal: {len(results['final_features'])}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
