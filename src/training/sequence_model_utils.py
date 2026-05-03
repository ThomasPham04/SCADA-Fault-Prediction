"""
sequence_model_utils.py — backward-compatibility re-export shim.

All symbols have moved to focused sub-modules:
  training.sequence_utils      — seeding, TF cleanup, JSON I/O, model persistence
  training.sequence_io         — data loading (bundles, NPZ test sequences)
  training.sequence_models     — Keras model builders and callbacks
  training.sequence_metrics    — threshold sweep, evaluation, comparison frames
  training.sequence_plots      — plotting and prediction-frame helpers
  training.sequence_experiments — experiment runners (classifier / autoencoder)

Import directly from those modules for new code.
"""

from training.sequence_experiments import (
    run_autoencoder_experiment,
    run_classifier_experiment,
    run_global_autoencoder_experiment,
)
from training.sequence_io import (
    _filter_array_by_asset_meta,
    empty_classifier_bundle,
    list_asset_dirs,
    load_asset_val_classifier_slice,
    load_autoencoder_asset_bundle,
    load_autoencoder_global_bundle,
    load_autoencoder_test_sequences,
    load_classifier_bundle,
    load_classifier_val_slice,
    load_scaler_if_present,
)
from training.sequence_metrics import (
    autoencoder_comparison_frame,
    classifier_comparison_frame,
    compute_autoencoder_architecture_summary,
    compute_class_weights,
    evaluate_at_threshold,
    get_label_column,
    pick_best_threshold,
    safe_mean,
    safe_pr_auc,
    safe_roc_auc,
    sweep_thresholds,
)
from training.sequence_models import (
    autoencoder_callbacks,
    build_autoencoder_model,
    build_classifier_model,
    classifier_callbacks,
)
from training.sequence_plots import (
    build_prediction_frame,
    save_confusion_matrix_plot,
    save_history_csv_and_plot,
    save_metrics_bar_plot,
    save_pr_curve_plot,
    save_roc_curve_plot,
    save_threshold_csv,
    save_threshold_plot,
)
from training.sequence_utils import (
    cleanup_tf,
    history_to_frame,
    json_default,
    load_best_model,
    load_json,
    reconstruction_scores,
    save_json,
    save_model_summary,
    set_random_seed,
    to_int_list,
)

__all__ = [
    # sequence_utils
    "cleanup_tf",
    "history_to_frame",
    "json_default",
    "load_best_model",
    "load_json",
    "reconstruction_scores",
    "save_json",
    "save_model_summary",
    "set_random_seed",
    "to_int_list",
    # sequence_io
    "empty_classifier_bundle",
    "list_asset_dirs",
    "load_asset_val_classifier_slice",
    "load_autoencoder_asset_bundle",
    "load_autoencoder_global_bundle",
    "load_autoencoder_test_sequences",
    "load_classifier_bundle",
    "load_classifier_val_slice",
    "load_scaler_if_present",
    # sequence_models
    "autoencoder_callbacks",
    "build_autoencoder_model",
    "build_classifier_model",
    "classifier_callbacks",
    # sequence_metrics
    "autoencoder_comparison_frame",
    "classifier_comparison_frame",
    "compute_autoencoder_architecture_summary",
    "compute_class_weights",
    "evaluate_at_threshold",
    "get_label_column",
    "pick_best_threshold",
    "safe_mean",
    "safe_pr_auc",
    "safe_roc_auc",
    "sweep_thresholds",
    # sequence_plots
    "build_prediction_frame",
    "save_confusion_matrix_plot",
    "save_history_csv_and_plot",
    "save_metrics_bar_plot",
    "save_pr_curve_plot",
    "save_roc_curve_plot",
    "save_threshold_csv",
    "save_threshold_plot",
    # sequence_experiments
    "run_autoencoder_experiment",
    "run_classifier_experiment",
    "run_global_autoencoder_experiment",
]
