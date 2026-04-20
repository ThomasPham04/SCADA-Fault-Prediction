"""
Kaggle-ready training script for sequence models.

Notebook usage:
    If both files are in the same Kaggle folder:
    from kaggle_train_sequence_models import main
    main()

    If you keep them inside an experiments folder:
    from experiments.kaggle_train_sequence_models import main
    main()
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_train_utils import (
    autoencoder_comparison_frame,
    classifier_comparison_frame,
    cleanup_tf,
    compute_autoencoder_architecture_summary,
    list_asset_dirs,
    load_classifier_bundle,
    run_autoencoder_experiment,
    run_classifier_experiment,
    save_json,
    set_random_seed,
)


# ============================================================================
# SETTINGS
# ============================================================================

EXPORTS_DIR = "/kaggle/input/notebooks/thuyennn/multi-train-1/sequence_exports"
RESULTS_DIR = "sequence_training_results"
WINDOWS_TO_RUN = [36]
CLASSIFIER_MODELS = ["lstm", "gru", "cnn_lstm", "cnn_gru"]
AUTOENCODER_MODELS = ["lstm_ae", "gru_ae"]
ASSET_FILTER = None
RANDOM_SEED = 42
OVERWRITE = False
SAVE_PREDICTIONS = True
CLASSIFIER_EPOCHS = 25
CLASSIFIER_BATCH_SIZE = 256
AUTOENCODER_EPOCHS = 30
AUTOENCODER_BATCH_SIZE = 128


def train_window(window_hours: int, exports_root: Path, results_root: Path) -> dict | None:
    export_window_dir = exports_root / f"window_{window_hours}h"
    if not export_window_dir.exists():
        print(f"[SKIP] Export folder not found for {window_hours}h: {export_window_dir}")
        return None

    print("\n" + "=" * 70)
    print(f"Training Window: {window_hours}h")
    print("=" * 70)

    results_window_dir = results_root / f"window_{window_hours}h"
    results_window_dir.mkdir(parents=True, exist_ok=True)

    classifier_export_dir = export_window_dir / "classifier"
    classifier_results_dir = results_window_dir / "classifier"
    classifier_results_dir.mkdir(parents=True, exist_ok=True)

    classifier_bundle = load_classifier_bundle(classifier_export_dir)
    classifier_rows = []
    classifier_payload = []

    for model_name in CLASSIFIER_MODELS:
        print(f"\n[Classifier] Training {model_name}")
        model_output_dir = classifier_results_dir / model_name
        result = run_classifier_experiment(
            model_name=model_name,
            bundle=classifier_bundle,
            output_dir=model_output_dir,
            random_seed=RANDOM_SEED,
            epochs=CLASSIFIER_EPOCHS,
            batch_size=CLASSIFIER_BATCH_SIZE,
            overwrite=OVERWRITE,
            save_predictions=SAVE_PREDICTIONS,
        )
        classifier_rows.append(result["summary"])
        classifier_payload.append(result["metrics"])

    classifier_comparison = classifier_comparison_frame(classifier_rows)
    classifier_comparison.to_csv(classifier_results_dir / "model_comparison.csv", index=False)

    autoencoder_export_dir = export_window_dir / "autoencoder"
    autoencoder_results_dir = results_window_dir / "autoencoder"
    autoencoder_results_dir.mkdir(parents=True, exist_ok=True)

    asset_dirs = list_asset_dirs(autoencoder_export_dir, ASSET_FILTER)
    autoencoder_rows = []
    autoencoder_payload = []

    if not asset_dirs:
        print("[SKIP] No autoencoder asset folders matched the current ASSET_FILTER.")

    for model_name in AUTOENCODER_MODELS:
        print(f"\n[Autoencoder] Training {model_name}")
        asset_results = []

        for asset_dir in asset_dirs:
            asset_id = int(asset_dir.name.replace("asset_", ""))
            print(f"  - Asset {asset_id}")
            model_output_dir = autoencoder_results_dir / asset_dir.name / model_name
            result = run_autoencoder_experiment(
                model_name=model_name,
                asset_dir=asset_dir,
                classifier_bundle=classifier_bundle,
                output_dir=model_output_dir,
                random_seed=RANDOM_SEED,
                epochs=AUTOENCODER_EPOCHS,
                batch_size=AUTOENCODER_BATCH_SIZE,
                overwrite=OVERWRITE,
                save_predictions=SAVE_PREDICTIONS,
            )
            asset_results.append(result)

        if not asset_results:
            continue

        asset_summary_df = pd.DataFrame([item["summary"] for item in asset_results]).sort_values("asset_id", kind="mergesort")
        asset_summary_df.to_csv(autoencoder_results_dir / f"{model_name}_asset_summary.csv", index=False)

        architecture_summary = compute_autoencoder_architecture_summary(model_name, asset_results)
        autoencoder_rows.append(architecture_summary)
        autoencoder_payload.append(
            {
                "architecture_summary": architecture_summary,
                "asset_summaries": asset_summary_df.to_dict(orient="records"),
            }
        )
        save_json(
            autoencoder_results_dir / f"{model_name}_summary.json",
            {
                "architecture_summary": architecture_summary,
                "asset_summaries": asset_summary_df.to_dict(orient="records"),
            },
        )

    autoencoder_comparison = autoencoder_comparison_frame(autoencoder_rows)
    autoencoder_comparison.to_csv(autoencoder_results_dir / "model_comparison.csv", index=False)

    best_classifier = classifier_comparison.iloc[0].to_dict() if not classifier_comparison.empty else None
    best_autoencoder = autoencoder_comparison.iloc[0].to_dict() if not autoencoder_comparison.empty else None

    window_summary = {
        "window_hours": window_hours,
        "classifier_models": classifier_payload,
        "autoencoder_models": autoencoder_payload,
        "best_classifier": best_classifier,
        "best_autoencoder": best_autoencoder,
    }
    save_json(results_window_dir / "window_summary.json", window_summary)

    del classifier_bundle
    cleanup_tf()
    gc.collect()

    return window_summary


def main() -> None:
    set_random_seed(RANDOM_SEED)
    exports_root = Path(EXPORTS_DIR)
    results_root = Path(RESULTS_DIR)
    results_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Kaggle Sequence Model Training")
    print("=" * 70)
    print(f"Exports folder : {exports_root}")
    print(f"Results folder : {results_root}")
    print(f"Windows to run : {WINDOWS_TO_RUN}")
    print(f"Classifier set : {CLASSIFIER_MODELS}")
    print(f"Autoencoder set: {AUTOENCODER_MODELS}")

    run_summary = {
        "exports_dir": str(exports_root),
        "results_dir": str(results_root),
        "windows_requested": WINDOWS_TO_RUN,
        "windows_completed": [],
    }

    for window_hours in WINDOWS_TO_RUN:
        window_summary = train_window(window_hours, exports_root, results_root)
        if window_summary is not None:
            run_summary["windows_completed"].append(window_summary)

    save_json(results_root / "run_summary.json", run_summary)

    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)
    for window_summary in run_summary["windows_completed"]:
        window_hours = window_summary["window_hours"]
        best_classifier = window_summary.get("best_classifier")
        best_autoencoder = window_summary.get("best_autoencoder")
        print(f"Window {window_hours}h:")
        if best_classifier:
            print(
                f"  Best classifier : {best_classifier['model_name']} "
                f"(event_f1={best_classifier['event_f1']:.4f})"
            )
        if best_autoencoder:
            print(
                f"  Best autoencoder: {best_autoencoder['model_name']} "
                f"(macro_event_f1={best_autoencoder['macro_event_f1']:.4f})"
            )

    print(f"Saved outputs to : {results_root}")


if __name__ == "__main__":
    main()
