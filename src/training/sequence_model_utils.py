"""
sequence_model_utils.py - backward-compatibility re-export shim.

All symbols have moved to focused sub-modules:
  training.sequence_utils       - seeding, TF cleanup, JSON I/O, model persistence
  training.sequence_io          - classifier data loading
  training.sequence_models      - Keras classifier builders and callbacks
  training.sequence_metrics     - threshold sweep, evaluation, comparison frames
  training.sequence_plots       - plotting and prediction-frame helpers
  training.sequence_experiments - experiment runners

Import directly from those modules for new code.
"""

from training.sequence_experiments import run_classifier_experiment
from training.sequence_io import (
    load_classifier_bundle,
    load_classifier_val_slice,
    load_classifier_val_slice_with_meta,
)
from training.sequence_metrics import (
    aggregate_event_scores,
    classifier_comparison_frame,
    compute_class_weights,
    evaluate_at_threshold,
    get_label_column,
    pick_best_threshold,
    safe_pr_auc,
    safe_roc_auc,
    sweep_thresholds,
)
from training.sequence_models import (
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
    save_json,
    save_model_summary,
    set_random_seed,
    to_int_list,
)

__all__ = [
    "aggregate_event_scores",
    "build_classifier_model",
    "build_prediction_frame",
    "classifier_callbacks",
    "classifier_comparison_frame",
    "cleanup_tf",
    "compute_class_weights",
    "evaluate_at_threshold",
    "get_label_column",
    "history_to_frame",
    "json_default",
    "load_best_model",
    "load_classifier_bundle",
    "load_classifier_val_slice",
    "load_classifier_val_slice_with_meta",
    "load_json",
    "pick_best_threshold",
    "run_classifier_experiment",
    "safe_pr_auc",
    "safe_roc_auc",
    "save_confusion_matrix_plot",
    "save_history_csv_and_plot",
    "save_json",
    "save_metrics_bar_plot",
    "save_model_summary",
    "save_pr_curve_plot",
    "save_roc_curve_plot",
    "save_threshold_csv",
    "save_threshold_plot",
    "set_random_seed",
    "sweep_thresholds",
    "to_int_list",
]
