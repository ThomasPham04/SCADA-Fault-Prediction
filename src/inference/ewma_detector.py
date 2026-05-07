"""
ewma_detector.py — inference.ewma_detector
Stateful Exponentially Weighted Moving Average detector for real-time drift detection.

Used as the second alarm signal on top of per-window reconstruction error:
  - Raw reconstruction error  → detects sudden anomalies / advanced degradation
  - EWMA of reconstruction error → detects gradual 14-day drift, fires earlier
"""

from __future__ import annotations

import math


class EWMADriftDetector:
    """Stateful EWMA over a stream of scalar scores (e.g. reconstruction errors).

    At each 10-min tick the caller passes the latest window's reconstruction
    error.  The detector maintains an exponentially weighted average and
    signals an alarm when that average crosses ``threshold``.

    Parameters
    ----------
    alpha:
        Smoothing factor in (0, 1).  Smaller → longer memory, reacts more
        slowly but catches gradual drift earlier.  Use ``alpha_for_days()``
        to compute from a desired effective-memory horizon.
    threshold:
        EWMA value above which ``alarm()`` returns True.  Typically calibrated
        as the 95th percentile of EWMA scores computed on normal validation
        windows.
    init_value:
        Starting EWMA value.  Defaults to None, which uses the first observed
        error (standard EWMA initialisation).  Set to the median of normal
        validation errors to avoid a cold-start spike at stream start.
    """

    def __init__(
        self,
        alpha: float,
        threshold: float,
        init_value: float | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.threshold = threshold
        self._init_value = init_value
        self._ewma: float | None = None if init_value is None else float(init_value)
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    def update(self, new_error: float) -> float:
        """Ingest one new reconstruction error and return the updated EWMA."""
        new_error = float(new_error)
        if self._ewma is None:
            self._ewma = new_error
        else:
            self._ewma = self.alpha * new_error + (1.0 - self.alpha) * self._ewma
        self._n_updates += 1
        return self._ewma

    def alarm(self) -> bool:
        """True when the current EWMA exceeds the threshold."""
        if self._ewma is None:
            return False
        return self._ewma > self.threshold

    @property
    def value(self) -> float | None:
        """Current EWMA value, or None before the first update."""
        return self._ewma

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def reset(self) -> None:
        """Reset state to initial conditions (e.g. between test sequences)."""
        self._ewma = None if self._init_value is None else float(self._init_value)
        self._n_updates = 0

    # ------------------------------------------------------------------
    @staticmethod
    def alpha_for_days(days: float, steps_per_day: int = 144) -> float:
        """Compute alpha so the effective memory spans ``days`` × ``steps_per_day`` steps.

        Uses the standard EWMA span formula: alpha = 2 / (N + 1)
        where N is the desired number of lagged steps.
        """
        n = max(1.0, days * steps_per_day)
        return 2.0 / (n + 1.0)

    @staticmethod
    def half_life_to_alpha(half_life_steps: float) -> float:
        """alpha from half-life in steps: alpha = 1 - exp(-ln2 / half_life)."""
        return 1.0 - math.exp(-math.log(2.0) / max(1.0, half_life_steps))

    def __repr__(self) -> str:
        return (
            f"EWMADriftDetector(alpha={self.alpha}, threshold={self.threshold}, "
            f"ewma={self._ewma}, alarm={self.alarm()})"
        )
