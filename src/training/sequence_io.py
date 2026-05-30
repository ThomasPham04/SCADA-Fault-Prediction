"""
sequence_io.py - training.sequence_io
Data loading helpers for classifier sequence exports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.sequence_utils import load_json, to_int_list


def _resolve_filter_ids(asset_filter) -> set:
    return set(to_int_list(asset_filter) or [])


def load_classifier_bundle(classifier_dir: Path) -> dict:
    if not classifier_dir.exists():
        raise FileNotFoundError(f"Classifier export not found: {classifier_dir}")
    return {
        "X_train": np.load(classifier_dir / "X_train.npy", mmap_mode="r"),
        "y_train": np.load(classifier_dir / "y_train.npy", mmap_mode="r"),
        "X_val": np.load(classifier_dir / "X_val.npy", mmap_mode="r"),
        "y_val": np.load(classifier_dir / "y_val.npy", mmap_mode="r"),
        "X_test": np.load(classifier_dir / "X_test.npy", mmap_mode="r"),
        "y_test": np.load(classifier_dir / "y_test.npy", mmap_mode="r"),
        "train_meta": pd.read_csv(classifier_dir / "train_meta.csv"),
        "val_meta": pd.read_csv(classifier_dir / "val_meta.csv"),
        "test_meta": pd.read_csv(classifier_dir / "test_meta.csv"),
        "metadata": load_json(classifier_dir / "metadata.json"),
    }


def load_classifier_val_slice(bundle: dict, asset_filter=None):
    val_meta = bundle["val_meta"]
    filter_ids = _resolve_filter_ids(asset_filter)
    if filter_ids:
        mask = pd.to_numeric(val_meta["asset_id"], errors="coerce").isin(filter_ids).to_numpy()
    else:
        mask = np.ones(len(val_meta), dtype=bool)
    return (
        np.asarray(bundle["X_val"][mask], dtype=np.float32),
        np.asarray(bundle["y_val"][mask], dtype=np.int8),
    )


def load_classifier_val_slice_with_meta(bundle: dict, asset_filter=None):
    """Same as load_classifier_val_slice but also returns matching val metadata."""
    val_meta = bundle["val_meta"]
    filter_ids = _resolve_filter_ids(asset_filter)
    if filter_ids:
        mask = pd.to_numeric(val_meta["asset_id"], errors="coerce").isin(filter_ids).to_numpy()
    else:
        mask = np.ones(len(val_meta), dtype=bool)
    return (
        np.asarray(bundle["X_val"][mask], dtype=np.float32),
        np.asarray(bundle["y_val"][mask], dtype=np.int8),
        val_meta.loc[mask].reset_index(drop=True),
    )
