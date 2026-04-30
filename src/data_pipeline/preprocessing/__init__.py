"""Preprocessing sub-package with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "engineer_angle_features",
    "drop_counter_features",
    "get_feature_columns",
    "preprocess_features",
    "wrap_angle_deg",
    "fill_missing_by_group",
    "process_all_events_train",
    "process_all_events_test",
    "temporal_split_train_val",
    "normalize_data",
    "normalize_asset",
]

_EXPORTS = {
    "engineer_angle_features": ("data_pipeline.preprocessing.feature_engineering", "engineer_angle_features"),
    "drop_counter_features": ("data_pipeline.preprocessing.feature_engineering", "drop_counter_features"),
    "get_feature_columns": ("data_pipeline.preprocessing.feature_engineering", "get_feature_columns"),
    "preprocess_features": ("data_pipeline.preprocessing.feature_engineering", "preprocess_features"),
    "wrap_angle_deg": ("data_pipeline.preprocessing.feature_engineering", "FeatureEngineer"),
    "fill_missing_by_group": ("data_pipeline.preprocessing.feature_engineering", "FeatureEngineer"),
    "process_all_events_train": ("data_pipeline.preprocessing.splitter", "process_all_events_train"),
    "process_all_events_test": ("data_pipeline.preprocessing.splitter", "process_all_events_test"),
    "temporal_split_train_val": ("data_pipeline.preprocessing.splitter", "temporal_split_train_val"),
    "normalize_data": ("data_pipeline.preprocessing.normalizer", "normalize_data"),
    "normalize_asset": ("data_pipeline.preprocessing.normalizer", "normalize_asset"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    attr = getattr(module, attr_name)

    if name == "wrap_angle_deg":
        attr = attr.wrap_angle_deg
    elif name == "fill_missing_by_group":
        attr = attr.fill_missing_by_group

    globals()[name] = attr
    return attr
