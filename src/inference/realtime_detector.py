"""
realtime_detector.py — inference.realtime_detector
Sliding-window real-time fault detector for a single wind turbine.

At each 10-minute SCADA tick the caller passes a dict of feature values.
The detector maintains one rolling buffer per fault-type detector, scores
each filled window, and returns both a raw alarm and an EWMA drift alarm.

Usage
-----
    from pathlib import Path
    from src.inference.realtime_detector import RealTimeFaultDetector

    detector = RealTimeFaultDetector(
        asset_id=10,
        models_dir=Path("results/fault_type_detectors/window_24h/asset_10"),
        feature_cols=metadata["feature_cols"],   # 21-element list from export metadata
    )

    for row in scada_stream:          # each row is a dict {feature_name: value}
        result = detector.ingest(row)
        if result["ready"]:
            for fault_type, info in result["detectors"].items():
                if info["raw_alarm"] or info["ewma_alarm"]:
                    print(f"ALARM  asset=10  fault={fault_type}  {info}")

Output format per tick (once ready)
-------------------------------------
    {
        "ready": True,
        "asset_id": 10,
        "detectors": {
            "hydraulic": {
                "raw_error": 0.032,
                "raw_alarm": False,
                "ewma": 0.021,
                "ewma_alarm": False,
                "n_windows_scored": 143,
            },
            "gearbox": { ... },
            ...
        }
    }
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from fault_type_config import ASSET_FAULT_TYPES, EWMA_ALPHA, get_feature_indices
from inference.ewma_detector import EWMADriftDetector
from training.sequence_utils import load_json


class _FaultTypeChannel:
    """Internal state for one (asset, fault_type) detector."""

    def __init__(
        self,
        fault_type: str,
        feature_indices: list[int],
        model: tf.keras.Model,
        raw_threshold: float,
        ewma_threshold: float,
        ewma_alpha: float,
        ewma_init: float,
        window_steps: int,
    ) -> None:
        self.fault_type = fault_type
        self.feature_indices = feature_indices
        self.model = model
        self.raw_threshold = raw_threshold
        self.window_steps = window_steps

        self._buffer: deque[list[float]] = deque(maxlen=window_steps)
        self._ewma = EWMADriftDetector(
            alpha=ewma_alpha,
            threshold=ewma_threshold,
            init_value=ewma_init,
        )
        self._n_windows: int = 0
        self._last_raw_error: float | None = None

    def push(self, feature_row: list[float]) -> None:
        """Add one timestep's worth of features (already ordered by feature_indices)."""
        self._buffer.append(feature_row)

    def is_ready(self) -> bool:
        return len(self._buffer) == self.window_steps

    def score(self) -> dict[str, Any]:
        """Score the current window.  Returns empty dict if buffer not full yet."""
        if not self.is_ready():
            return {}

        window = np.array(list(self._buffer), dtype=np.float32)  # (T, K)
        window_batch = window[np.newaxis, :, :]                   # (1, T, K)

        recon = self.model.predict(window_batch, batch_size=1, verbose=0)
        per_ts_l2 = np.sqrt(np.sum(np.square(window_batch - recon), axis=2))  # (1, T)
        raw_error = float(np.mean(per_ts_l2))

        ewma_val = self._ewma.update(raw_error)
        self._last_raw_error = raw_error
        self._n_windows += 1

        return {
            "raw_error": raw_error,
            "raw_alarm": raw_error > self.raw_threshold,
            "ewma": ewma_val,
            "ewma_alarm": self._ewma.alarm(),
            "n_windows_scored": self._n_windows,
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._ewma.reset()
        self._n_windows = 0
        self._last_raw_error = None


class RealTimeFaultDetector:
    """Sliding-window real-time detector for one wind turbine.

    Parameters
    ----------
    asset_id:
        Turbine ID (0, 10, 11, 13, or 21 for Wind Farm A).
    models_dir:
        Directory containing one sub-directory per fault type, each with a
        trained LSTM-AE model and threshold JSON files.  Expected layout::

            models_dir/
              hydraulic/lstm_ae/
                model.keras
                threshold_raw.json   {"threshold": ...}
                threshold_ewma.json  {"threshold": ..., "alpha": ..., "ewma_init_value": ...}
                metrics.json
              gearbox/lstm_ae/
                ...

    feature_cols:
        Full ordered list of feature names from the export metadata.json.
        Used to map fault-type sensor names to column indices.
    model_name:
        Sub-directory name for the model (default: "lstm_ae").
    window_steps:
        Number of timesteps per window (default: 144 = 24 h at 10-min resolution).
    """

    def __init__(
        self,
        asset_id: int,
        models_dir: Path,
        feature_cols: list[str],
        model_name: str = "lstm_ae",
        window_steps: int = 144,
    ) -> None:
        self.asset_id = asset_id
        self.models_dir = Path(models_dir)
        self.feature_cols = feature_cols
        self.window_steps = window_steps

        self._channels: dict[str, _FaultTypeChannel] = {}
        self._load_channels(model_name)

    # ------------------------------------------------------------------
    def _load_channels(self, model_name: str) -> None:
        fault_types = ASSET_FAULT_TYPES.get(self.asset_id, [])
        for fault_type in fault_types:
            model_dir = self.models_dir / fault_type / model_name
            model_path = model_dir / "model.keras"
            raw_thr_path = model_dir / "threshold_raw.json"
            ewma_thr_path = model_dir / "threshold_ewma.json"

            if not model_path.exists():
                print(
                    f"[RealTimeFaultDetector] Model not found for "
                    f"asset={self.asset_id} fault_type={fault_type}: {model_path}. "
                    f"Skipping this channel."
                )
                continue

            model = tf.keras.models.load_model(model_path, compile=False)

            raw_thr_data = load_json(raw_thr_path) if raw_thr_path.exists() else {}
            ewma_thr_data = load_json(ewma_thr_path) if ewma_thr_path.exists() else {}

            raw_threshold = float(raw_thr_data.get("threshold", 0.05))
            ewma_threshold = float(ewma_thr_data.get("threshold", 0.03))
            ewma_alpha = float(ewma_thr_data.get("alpha", EWMA_ALPHA.get(fault_type, 0.005)))
            ewma_init = float(ewma_thr_data.get("ewma_init_value", 0.0))

            feature_indices = get_feature_indices(fault_type, self.feature_cols)
            if not feature_indices:
                print(
                    f"[RealTimeFaultDetector] No feature indices for "
                    f"fault_type={fault_type}. Skipping."
                )
                continue

            self._channels[fault_type] = _FaultTypeChannel(
                fault_type=fault_type,
                feature_indices=feature_indices,
                model=model,
                raw_threshold=raw_threshold,
                ewma_threshold=ewma_threshold,
                ewma_alpha=ewma_alpha,
                ewma_init=ewma_init,
                window_steps=self.window_steps,
            )

        if not self._channels:
            raise RuntimeError(
                f"No fault-type detector channels loaded for asset {self.asset_id}. "
                f"Check that models_dir contains trained model.keras files."
            )

    # ------------------------------------------------------------------
    def ingest(self, new_row: dict[str, float]) -> dict[str, Any]:
        """Process one new 10-minute SCADA row.

        Parameters
        ----------
        new_row:
            Mapping of feature_name → value for all features in feature_cols.
            Missing features default to 0.0.

        Returns
        -------
        dict with keys:
            "ready"     — False until the buffer fills (first 143 ticks)
            "asset_id"  — this turbine's ID
            "detectors" — dict of fault_type → scoring result (only when ready)
        """
        # Build the full feature vector in column order
        full_row = [float(new_row.get(col, 0.0)) for col in self.feature_cols]

        any_ready = False
        detector_results: dict[str, Any] = {}

        for fault_type, channel in self._channels.items():
            # Slice to this channel's features
            channel_row = [full_row[i] for i in channel.feature_indices]
            channel.push(channel_row)

            if channel.is_ready():
                any_ready = True
                detector_results[fault_type] = channel.score()

        return {
            "ready": any_ready,
            "asset_id": self.asset_id,
            "detectors": detector_results,
        }

    def warm_up_steps_remaining(self) -> int:
        """Number of ticks before the first scoring result is available."""
        if not self._channels:
            return self.window_steps
        first = next(iter(self._channels.values()))
        return max(0, self.window_steps - len(first._buffer))

    @property
    def active_fault_types(self) -> list[str]:
        return list(self._channels.keys())

    def reset(self) -> None:
        """Clear all buffers and EWMA state (e.g. when switching to a new event)."""
        for channel in self._channels.values():
            channel.reset()
