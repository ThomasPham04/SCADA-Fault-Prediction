"""
sequence_io.py — training.sequence_io
Data loading helpers: classifier bundles, per-asset autoencoder exports,
global pooled exports, and test-sequence NPZ readers.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from training.sequence_utils import load_json, to_int_list


def list_asset_dirs(autoencoder_root: Path, asset_filter=None):
    filter_ids = set(to_int_list(asset_filter) or [])
    asset_dirs = []
    for asset_dir in sorted(autoencoder_root.glob("asset_*")):
        try:
            asset_id = int(asset_dir.name.replace("asset_", ""))
        except ValueError:
            continue
        if filter_ids and asset_id not in filter_ids:
            continue
        asset_dirs.append(asset_dir)
    return asset_dirs


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


def empty_classifier_bundle() -> dict:
    """Return a zero-row bundle for autoencoder-only runs where no classifier export exists."""
    empty_meta = pd.DataFrame(columns=["asset_id"])
    return {
        "X_train": np.empty((0,), dtype=np.float32),
        "y_train": np.empty((0,), dtype=np.int8),
        "X_val":   np.empty((0,), dtype=np.float32),
        "y_val":   np.empty((0,), dtype=np.int8),
        "X_test":  np.empty((0,), dtype=np.float32),
        "y_test":  np.empty((0,), dtype=np.int8),
        "train_meta": empty_meta.copy(),
        "val_meta":   empty_meta.copy(),
        "test_meta":  empty_meta.copy(),
        "metadata": {},
    }


def load_scaler_if_present(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def load_autoencoder_asset_bundle(asset_dir: Path) -> dict:
    if not asset_dir.exists():
        raise FileNotFoundError(f"Autoencoder asset export not found: {asset_dir}")
    asset_id = int(asset_dir.name.replace("asset_", ""))
    return {
        "asset_id": asset_id,
        "X_train": np.load(asset_dir / "X_train.npy", mmap_mode="r"),
        "X_val_normal": np.load(asset_dir / "X_val.npy", mmap_mode="r"),
        "metadata": load_json(asset_dir / "metadata.json"),
        "test_dir": asset_dir / "test_by_sequence",
        "scaler": load_scaler_if_present(asset_dir / "scaler.pkl"),
    }


def _filter_array_by_asset_meta(X, meta_df: pd.DataFrame, asset_filter=None):
    filter_ids = set(to_int_list(asset_filter) or [])
    if not filter_ids:
        return X, meta_df
    if "asset_id" not in meta_df.columns:
        raise ValueError("Cannot filter global autoencoder data without asset_id metadata.")
    mask = pd.to_numeric(meta_df["asset_id"], errors="coerce").isin(filter_ids).to_numpy()
    return np.asarray(X[mask], dtype=np.float32), meta_df.loc[mask].reset_index(drop=True)


def load_autoencoder_global_bundle(autoencoder_root: Path, asset_filter=None) -> dict:
    """Load pooled autoencoder data, falling back to per-asset exports when needed."""
    filter_ids = set(to_int_list(asset_filter) or [])
    global_dir = autoencoder_root / "global"

    if global_dir.exists():
        X_train = np.load(global_dir / "X_train.npy", mmap_mode="r")
        X_val = np.load(global_dir / "X_val.npy", mmap_mode="r")
        train_meta = pd.read_csv(global_dir / "train_meta.csv")
        val_meta = pd.read_csv(global_dir / "val_meta.csv")
        X_train, train_meta = _filter_array_by_asset_meta(X_train, train_meta, asset_filter)
        X_val, val_meta = _filter_array_by_asset_meta(X_val, val_meta, asset_filter)
        asset_ids = sorted(pd.to_numeric(train_meta["asset_id"], errors="coerce").dropna().astype(int).unique())
        return {
            "scope": "global",
            "source": "global_export",
            "asset_ids": asset_ids,
            "X_train": X_train,
            "X_val_normal": X_val,
            "train_meta": train_meta,
            "val_meta": val_meta,
            "metadata": load_json(global_dir / "metadata.json"),
            "test_dirs": [global_dir / "test_by_sequence"],
            "scalers": load_scaler_if_present(global_dir / "scalers.pkl"),
        }

    asset_dirs = list_asset_dirs(autoencoder_root, asset_filter)
    if not asset_dirs:
        raise FileNotFoundError(f"No autoencoder exports found under: {autoencoder_root}")

    train_parts = []
    val_parts = []
    test_dirs = []
    asset_ids = []
    for asset_dir in asset_dirs:
        bundle = load_autoencoder_asset_bundle(asset_dir)
        train_parts.append(np.asarray(bundle["X_train"], dtype=np.float32))
        val_parts.append(np.asarray(bundle["X_val_normal"], dtype=np.float32))
        test_dirs.append(bundle["test_dir"])
        asset_ids.append(int(bundle["asset_id"]))

    return {
        "scope": "global",
        "source": "pooled_asset_exports",
        "asset_ids": sorted(asset_ids),
        "X_train": np.concatenate(train_parts, axis=0),
        "X_val_normal": np.concatenate(val_parts, axis=0),
        "train_meta": None,
        "val_meta": None,
        "metadata": {
            "scope": "global",
            "source": "pooled_asset_exports",
            "asset_ids": sorted(asset_ids),
            "asset_filter": sorted(filter_ids) if filter_ids else None,
        },
        "test_dirs": test_dirs,
        "scalers": None,
    }


def load_asset_val_classifier_slice(bundle: dict, asset_id: int):
    return load_classifier_val_slice(bundle, asset_filter=[asset_id])


def load_classifier_val_slice(bundle: dict, asset_filter=None):
    val_meta = bundle["val_meta"]
    filter_ids = set(to_int_list(asset_filter) or [])
    if filter_ids:
        mask = pd.to_numeric(val_meta["asset_id"], errors="coerce").isin(filter_ids).to_numpy()
    else:
        mask = np.ones(len(val_meta), dtype=bool)
    return (
        np.asarray(bundle["X_val"][mask], dtype=np.float32),
        np.asarray(bundle["y_val"][mask], dtype=np.int8),
    )


def load_autoencoder_test_sequences(asset_dir: Path, asset_filter=None):
    sequence_rows = []
    test_dir = asset_dir / "test_by_sequence" if (asset_dir / "test_by_sequence").exists() else asset_dir
    filter_ids = set(to_int_list(asset_filter) or [])
    for npz_path in sorted(test_dir.rglob("sequence_*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            label_key = "target_label" if "target_label" in data.files else "y"
            asset_id = int(np.asarray(data["asset_id"]).reshape(-1)[0])
            if filter_ids and asset_id not in filter_ids:
                continue
            sequence_rows.append(
                {
                    "asset_id": asset_id,
                    "sequence_id": int(np.asarray(data["sequence_id"]).reshape(-1)[0]),
                    "X": np.asarray(data["X"], dtype=np.float32),
                    "y": np.asarray(data[label_key], dtype=np.int8),
                }
            )
    return sequence_rows
