"""
sequence_utils.py - training.sequence_utils
General-purpose helpers: seeding, TF cleanup, JSON I/O, model persistence,
and history formatting.
"""

from __future__ import annotations

import gc
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def cleanup_tf() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_int_list(value):
    if value is None:
        return None
    return [int(item) for item in value]


def history_to_frame(history) -> pd.DataFrame:
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "epoch"
    return history_df.reset_index()


def load_best_model(model_path: Path):
    if model_path.exists():
        return tf.keras.models.load_model(model_path, compile=False)
    raise FileNotFoundError(f"Saved model not found: {model_path}")


def save_model_summary(model, path: Path) -> None:
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))
    Path(path).write_text(buf.getvalue(), encoding="utf-8")
