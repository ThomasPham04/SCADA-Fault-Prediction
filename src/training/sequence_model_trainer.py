"""
Sequence model training orchestration.

Consumes the windowed exports produced by
data_pipeline.preprocessing.combined_sequence_pipeline and trains:
- global supervised sequence classifiers
- per-asset sequence autoencoders
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import PROCESSED_DATA_DIR, RANDOM_SEED, RESULTS_DIR
from training.sequence_experiments import (
    run_autoencoder_experiment,
    run_classifier_experiment,
    run_global_autoencoder_experiment,
)
from training.sequence_io import (
    empty_classifier_bundle,
    list_asset_dirs,
    load_classifier_bundle,
)
from training.sequence_metrics import (
    autoencoder_comparison_frame,
    classifier_comparison_frame,
    compute_autoencoder_architecture_summary,
)
from training.sequence_utils import (
    cleanup_tf,
    save_json,
    set_random_seed,
    to_int_list,
)


DEFAULT_SEQUENCE_WINDOWS = [24]
DEFAULT_CLASSIFIER_MODELS = ["lstm", "gru", "cnn_lstm", "cnn_gru"]
DEFAULT_AUTOENCODER_MODELS = ["lstm_ae", "gru_ae"]


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


@dataclass
class SequenceTrainingConfig:
    """Configuration for training exported sequence datasets."""

    exports_dir: Path = field(
        default_factory=lambda: Path(PROCESSED_DATA_DIR) / "sequence_exports"
    )
    results_dir: Path = field(
        default_factory=lambda: Path(RESULTS_DIR) / "sequence_training_results"
    )
    windows: list[int] = field(default_factory=lambda: list(DEFAULT_SEQUENCE_WINDOWS))
    classifier_models: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSIFIER_MODELS))
    autoencoder_models: list[str] = field(default_factory=lambda: list(DEFAULT_AUTOENCODER_MODELS))
    asset_filter: list[int] | None = None
    random_seed: int = RANDOM_SEED
    overwrite: bool = False
    save_predictions: bool = True
    classifier_epochs: int = 25
    classifier_batch_size: int = 256
    classifier_learning_rate: float = 1e-3
    classifier_dropout: float | None = None
    classifier_l2: float = 0.0
    autoencoder_epochs: int = 30
    autoencoder_batch_size: int = 128
    autoencoder_scope: str = "per_asset"
    autoencoder_learning_rate: float = 1e-3
    autoencoder_noise: float = 0.0
    autoencoder_use_adaptive_threshold: bool = False
    autoencoder_gamma: float = 0.344
    autoencoder_threshold_nn_units: int = 23

    def __post_init__(self) -> None:
        self.exports_dir = _as_path(self.exports_dir)
        self.results_dir = _as_path(self.results_dir)
        self.windows = [int(window) for window in self.windows]
        self.classifier_models = [str(model) for model in self.classifier_models]
        self.autoencoder_models = [str(model) for model in self.autoencoder_models]
        self.asset_filter = to_int_list(self.asset_filter)
        self.classifier_learning_rate = float(self.classifier_learning_rate)
        self.classifier_l2 = float(self.classifier_l2)
        self.autoencoder_learning_rate = float(self.autoencoder_learning_rate)
        self.autoencoder_noise = float(self.autoencoder_noise)
        self.autoencoder_gamma = float(self.autoencoder_gamma)
        valid_scopes = {"per_asset", "global", "both"}
        if self.autoencoder_scope not in valid_scopes:
            raise ValueError(f"autoencoder_scope must be one of {sorted(valid_scopes)}")


class SequenceModelTrainer:
    """Train classifiers and autoencoders from combined sequence exports."""

    def __init__(self, config: SequenceTrainingConfig | None = None) -> None:
        self.config = config or SequenceTrainingConfig()

    def train_window(self, window_hours: int) -> dict | None:
        """Train all configured models for one exported window size."""
        cfg = self.config
        export_window_dir = cfg.exports_dir / f"window_{window_hours}h"
        if not export_window_dir.exists():
            print(f"[SKIP] Export folder not found for {window_hours}h: {export_window_dir}")
            return None

        print("\n" + "=" * 70)
        print(f"Training Window: {window_hours}h")
        print("=" * 70)

        results_window_dir = cfg.results_dir / f"window_{window_hours}h"
        results_window_dir.mkdir(parents=True, exist_ok=True)

        classifier_export_dir = export_window_dir / "classifier"
        classifier_results_dir = results_window_dir / "classifier"
        classifier_results_dir.mkdir(parents=True, exist_ok=True)

        need_classifier_bundle = bool(cfg.classifier_models) or cfg.autoencoder_models
        if classifier_export_dir.exists():
            classifier_bundle = load_classifier_bundle(classifier_export_dir)
        elif need_classifier_bundle:
            print(
                f"[INFO] No classifier export at {classifier_export_dir}. "
                "Autoencoder threshold will use 99th-percentile fallback."
            )
            classifier_bundle = empty_classifier_bundle()
        else:
            classifier_bundle = empty_classifier_bundle()

        classifier_rows = []
        classifier_payload = []

        for model_name in cfg.classifier_models:
            print(f"\n[Classifier] Training {model_name}")
            model_output_dir = classifier_results_dir / model_name
            result = run_classifier_experiment(
                model_name=model_name,
                bundle=classifier_bundle,
                output_dir=model_output_dir,
                random_seed=cfg.random_seed,
                epochs=cfg.classifier_epochs,
                batch_size=cfg.classifier_batch_size,
                learning_rate=cfg.classifier_learning_rate,
                dropout_rate=cfg.classifier_dropout,
                l2_strength=cfg.classifier_l2,
                overwrite=cfg.overwrite,
                save_predictions=cfg.save_predictions,
            )
            classifier_rows.append(result["summary"])
            classifier_payload.append(result["metrics"])

        classifier_comparison = classifier_comparison_frame(classifier_rows)
        classifier_comparison.to_csv(classifier_results_dir / "model_comparison.csv", index=False)

        autoencoder_export_dir = export_window_dir / "autoencoder"
        autoencoder_results_dir = results_window_dir / "autoencoder"
        autoencoder_results_dir.mkdir(parents=True, exist_ok=True)

        asset_dirs = list_asset_dirs(autoencoder_export_dir, cfg.asset_filter)
        autoencoder_rows = []
        autoencoder_payload = []
        run_per_asset_autoencoders = cfg.autoencoder_scope in {"per_asset", "both"}
        run_global_autoencoders = cfg.autoencoder_scope in {"global", "both"}

        if cfg.autoencoder_models and run_per_asset_autoencoders and not asset_dirs:
            print("[SKIP] No autoencoder asset folders matched the current asset filter.")

        for model_name in cfg.autoencoder_models:
            if run_per_asset_autoencoders:
                print(f"\n[Autoencoder] Training {model_name} per asset")
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
                        random_seed=cfg.random_seed,
                        epochs=cfg.autoencoder_epochs,
                        batch_size=cfg.autoencoder_batch_size,
                        overwrite=cfg.overwrite,
                        save_predictions=cfg.save_predictions,
                        learning_rate=cfg.autoencoder_learning_rate,
                        noise_stddev=cfg.autoencoder_noise,
                        use_adaptive_threshold=cfg.autoencoder_use_adaptive_threshold,
                        gamma=cfg.autoencoder_gamma,
                        threshold_nn_units=cfg.autoencoder_threshold_nn_units,
                    )
                    asset_results.append(result)

                if asset_results:
                    asset_summary_df = (
                        pd.DataFrame([item["summary"] for item in asset_results])
                        .sort_values("asset_id", kind="mergesort")
                    )
                    asset_summary_df.to_csv(
                        autoencoder_results_dir / f"{model_name}_asset_summary.csv",
                        index=False,
                    )

                    architecture_summary = compute_autoencoder_architecture_summary(
                        model_name,
                        asset_results,
                    )
                    payload = {
                        "architecture_summary": architecture_summary,
                        "asset_summaries": asset_summary_df.to_dict(orient="records"),
                    }
                    autoencoder_rows.append(architecture_summary)
                    autoencoder_payload.append(payload)
                    save_json(autoencoder_results_dir / f"{model_name}_summary.json", payload)

            if run_global_autoencoders:
                print(f"\n[Autoencoder] Training {model_name} global pooled")
                model_output_dir = autoencoder_results_dir / "global" / model_name
                result = run_global_autoencoder_experiment(
                    model_name=model_name,
                    autoencoder_root=autoencoder_export_dir,
                    classifier_bundle=classifier_bundle,
                    output_dir=model_output_dir,
                    random_seed=cfg.random_seed,
                    epochs=cfg.autoencoder_epochs,
                    batch_size=cfg.autoencoder_batch_size,
                    overwrite=cfg.overwrite,
                    save_predictions=cfg.save_predictions,
                    asset_filter=cfg.asset_filter,
                    learning_rate=cfg.autoencoder_learning_rate,
                    noise_stddev=cfg.autoencoder_noise,
                    use_adaptive_threshold=cfg.autoencoder_use_adaptive_threshold,
                    gamma=cfg.autoencoder_gamma,
                    threshold_nn_units=cfg.autoencoder_threshold_nn_units,
                )
                autoencoder_rows.append(result["summary"])
                autoencoder_payload.append(result["metrics"])
                save_json(
                    autoencoder_results_dir / f"{model_name}_global_summary.json",
                    result["metrics"],
                )

        autoencoder_comparison = autoencoder_comparison_frame(autoencoder_rows)
        autoencoder_comparison.to_csv(autoencoder_results_dir / "model_comparison.csv", index=False)

        best_classifier = (
            classifier_comparison.iloc[0].to_dict()
            if not classifier_comparison.empty
            else None
        )
        best_autoencoder = (
            autoencoder_comparison.iloc[0].to_dict()
            if not autoencoder_comparison.empty
            else None
        )

        window_summary = {
            "window_hours": int(window_hours),
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

    def run(self) -> dict:
        """Run sequence training for every configured window."""
        cfg = self.config
        set_random_seed(cfg.random_seed)
        cfg.results_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Sequence Model Training")
        print("=" * 70)
        print(f"Exports folder : {cfg.exports_dir}")
        print(f"Results folder : {cfg.results_dir}")
        print(f"Windows to run : {cfg.windows}")
        print(f"Classifier set : {cfg.classifier_models}")
        print(f"Classifier lr  : {cfg.classifier_learning_rate}")
        print(f"Classifier drop: {cfg.classifier_dropout}")
        print(f"Classifier L2  : {cfg.classifier_l2}")
        print(f"Autoencoder set: {cfg.autoencoder_models}")
        print(f"Autoenc. scope : {cfg.autoencoder_scope}")
        print(f"Autoenc. lr    : {cfg.autoencoder_learning_rate}")
        print(f"Autoenc. noise : {cfg.autoencoder_noise}")
        print(f"Adaptive thresh: {cfg.autoencoder_use_adaptive_threshold} (gamma={cfg.autoencoder_gamma})")
        print(f"Asset filter   : {cfg.asset_filter}")

        run_summary = {
            "exports_dir": str(cfg.exports_dir),
            "results_dir": str(cfg.results_dir),
            "windows_requested": cfg.windows,
            "windows_completed": [],
        }

        for window_hours in cfg.windows:
            window_summary = self.train_window(window_hours)
            if window_summary is not None:
                run_summary["windows_completed"].append(window_summary)

        save_json(cfg.results_dir / "run_summary.json", run_summary)

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
                    f"(f1={best_classifier['f1']:.4f})"
                )
            if best_autoencoder:
                print(
                    f"  Best autoencoder: {best_autoencoder['model_name']} "
                    f"(macro_f1={best_autoencoder['macro_f1']:.4f})"
                )

        print(f"Saved outputs to : {cfg.results_dir}")
        return run_summary
