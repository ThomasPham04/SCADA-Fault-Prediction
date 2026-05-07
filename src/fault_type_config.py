"""
fault_type_config.py
Fault-type specific sensor subsets, asset-to-fault-type mapping, and EWMA parameters.

All feature names are drawn from the 21 screened features in the current
sequence_exports/window_24h exports.  The fault-type groups choose the sensors
most informative for each failure mode so each per-fault-type LSTM-AE trains on
a narrow, relevant slice instead of the full 21-feature set.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fault-type → feature name mapping
# ---------------------------------------------------------------------------
# Available 21 features (see sequence_exports/window_24h/*/metadata.json):
#   sensor_0_avg, sensor_5_avg_sin, sensor_5_avg_cos, sensor_5_min_sin,
#   sensor_5_min_cos, sensor_5_max_cos, sensor_1_avg_sin, sensor_10_avg,
#   sensor_14_avg, sensor_18_avg, sensor_19_avg, sensor_33_avg, sensor_34_avg,
#   sensor_38_avg, sensor_40_avg, sensor_41_avg, sensor_42_avg_cos, sensor_44,
#   reactive_power_28_min, reactive_power_28_max, wind_speed_3_min

FAULT_TYPE_FEATURE_GROUPS: dict[str, list[str]] = {
    # --- Hydraulic group ---
    # Primary: s41_avg = hydraulic oil temperature
    # Context: RPM, ambient, wind, power to account for operating-point effects
    "hydraulic": [
        "sensor_41_avg",       # hydraulic oil temperature — primary signal
        "sensor_18_avg",       # generator RPM — operational context
        "sensor_0_avg",        # ambient temperature
        "wind_speed_3_min",    # wind speed (load condition)
        "sensor_44",           # active power
    ],

    # --- Gearbox / Gearbox bearings ---
    # s11_avg (gearbox bearing) and s12_avg (gearbox oil) were screened out.
    # Best proxies: generator RPM (gearbox directly drives generator),
    # split ring chamber temp (mechanical proximity), pitch angle (load driver)
    "gearbox": [
        "sensor_18_avg",       # generator RPM — gearbox-to-generator coupling
        "sensor_19_avg",       # split ring chamber temperature
        "sensor_5_avg_sin",    # pitch angle sin
        "sensor_5_avg_cos",    # pitch angle cos
        "sensor_0_avg",        # ambient temperature
        "wind_speed_3_min",    # operational context
        "sensor_44",           # active power / load
    ],

    # --- Generator bearing failure ---
    # s13_avg (DE) was screened out; s14_avg (NDE) is available
    "generator_bearing": [
        "sensor_14_avg",       # generator bearing NDE temperature — primary signal
        "sensor_18_avg",       # generator RPM
        "sensor_0_avg",        # ambient temperature
        "sensor_44",           # active power / load
    ],

    # --- Transformer failure ---
    # s38_avg (L1) and s40_avg (L3) are available; s39_avg (L2) was screened out
    "transformer": [
        "sensor_38_avg",       # HV transformer temperature L1 — primary signal
        "sensor_40_avg",       # HV transformer temperature L3
        "sensor_33_avg",       # grid voltage phase 2
        "sensor_34_avg",       # grid voltage phase 3
        "sensor_44",           # active power (load on transformer)
    ],
}

FAULT_DESCRIPTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hydraulic": ("hydraulic",),
    "gearbox": ("gearbox",),
    "generator_bearing": ("generator bearing",),
    "transformer": ("transformer",),
}

# ---------------------------------------------------------------------------
# Asset → applicable fault types
# Based on event_info.csv for Wind Farm A
# ---------------------------------------------------------------------------
ASSET_FAULT_TYPES: dict[int, list[str]] = {
    0:  ["hydraulic", "generator_bearing"],
    10: ["hydraulic", "generator_bearing", "gearbox"],
    11: ["transformer"],
    13: ["hydraulic"],
    21: ["hydraulic", "gearbox"],
}

# ---------------------------------------------------------------------------
# EWMA smoothing factor per fault type
# alpha = 2 / (N + 1) where N = effective memory in timesteps
# Steps per day = 144  (10-min resolution, stride=6 → ~24 scores/day)
# ---------------------------------------------------------------------------
# Hydraulic / gearbox / transformer: ~3-day effective memory
#   N = 3 × 144 = 432  →  alpha ≈ 0.005
# Generator bearing: ~7-day memory (e40 lasted 32 days, very slow drift)
#   N = 7 × 144 = 1008  →  alpha ≈ 0.002
EWMA_ALPHA: dict[str, float] = {
    "hydraulic":          0.005,
    "gearbox":            0.005,
    "generator_bearing":  0.002,
    "transformer":        0.005,
}

# Assets with only 1 anomaly test event — report results with low-confidence flag
LOW_CONFIDENCE_ASSETS: dict[str, list[int]] = {
    "transformer":        [11],   # single transformer failure event
    "generator_bearing":  [0],    # e0 is one of only 2 generator bearing events
}


def get_feature_indices(fault_type: str, all_feature_cols: list[str]) -> list[int]:
    """Return column indices in all_feature_cols for the given fault type's features.

    Silently skips any fault-type feature not present in all_feature_cols so
    the function is safe when called against different export versions.
    """
    wanted = set(FAULT_TYPE_FEATURE_GROUPS[fault_type])
    return [i for i, col in enumerate(all_feature_cols) if col in wanted]


def is_low_confidence(asset_id: int, fault_type: str) -> bool:
    return asset_id in LOW_CONFIDENCE_ASSETS.get(fault_type, [])


def fault_type_from_description(description: object) -> str | None:
    """Map an event description from event_info.csv to a detector fault type."""
    text = "" if description is None else str(description).strip().lower()
    if not text or text == "nan":
        return None
    for fault_type, keywords in FAULT_DESCRIPTION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return fault_type
    return None
