#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Kaggle Notebook 1
# # Prepare per-asset autoencoder dataset for Wind Farm A
#
# This notebook keeps the same overall flow as your original preprocessing
# notebook, but fixes the main issues:
# - one asset at a time
# - no scaler leakage
# - proper raw-angle engineering
# - train with energy features included
# - status-based window labels instead of labeling the whole event as anomaly

# %%
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# %%
ASSET_ID = 10

# Change this to your Kaggle input dataset path.
RAW_DATASET_DIR = Path("/kaggle/input/scada-wind-farm-a/Wind Farm A")
EVENT_INFO_PATH = RAW_DATASET_DIR / "event_info.csv"
DATASETS_DIR = RAW_DATASET_DIR / "datasets"

OUTPUT_ROOT = Path("/kaggle/working/processed_asset")
OUTPUT_DIR = OUTPUT_ROOT / f"asset_{ASSET_ID}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.85
WINDOW_SIZE = 288
TRAIN_STEP_SIZE = 6
TEST_STEP_SIZE = 6
NORMAL_STATUS = {0, 2}

# %%
event_info = pd.read_csv(EVENT_INFO_PATH, sep=";")
event_info["event_start"] = pd.to_datetime(event_info["event_start"])
event_info["event_end"] = pd.to_datetime(event_info["event_end"])

asset_events = (
    event_info[event_info["asset"] == ASSET_ID]
    .sort_values(["event_start", "event_id"])
    .reset_index(drop=True)
)

print("Selected asset:", ASSET_ID)
print("Number of events:", len(asset_events))
print(asset_events[["event_id", "event_label", "event_description", "event_start", "event_end"]])

# %%
TEMPERATURE_FEATURES = [
    "sensor_6_avg",
    "sensor_8_avg",
    "sensor_10_avg",
    "sensor_13_avg",
    "sensor_14_avg",
    "sensor_37_avg",
    "sensor_39_avg",
    "sensor_41_avg",
]

RPM_FEATURES = [
    "sensor_18_avg",
    "sensor_18_std",
]

ELECTRICAL_FEATURES = [
    "sensor_22_avg",
    "sensor_23_avg",
    "sensor_26_avg",
    "sensor_32_avg",
    "sensor_34_avg",
]

WIND_POWER_FEATURES = [
    "wind_speed_3_avg",
    "wind_speed_3_std",
    "power_29_avg",
    "power_30_avg",
    "power_30_std",
    "sensor_31_avg",
    "sensor_31_std",
]

ENERGY_FEATURES = [
    "sensor_44",
    "sensor_45",
    "sensor_46",
    "sensor_47",
    "sensor_48",
    "sensor_49",
    "sensor_50",
    "sensor_51",
]

PITCH_FEATURES = [
    "sensor_5_std",
]

RAW_ANGLE_FEATURES = [
    "sensor_2_avg",
    "sensor_5_avg",
    "sensor_42_avg",
]

BASE_FEATURE_COLUMNS = (
    TEMPERATURE_FEATURES
    + RPM_FEATURES
    + ELECTRICAL_FEATURES
    + WIND_POWER_FEATURES
    + ENERGY_FEATURES
    + PITCH_FEATURES
)

ENGINEERED_ANGLE_COLUMNS = [
    "sensor_2_avg_sin",
    "sensor_2_avg_cos",
    "sensor_5_avg_sin",
    "sensor_5_avg_cos",
    "sensor_42_avg_sin",
    "sensor_42_avg_cos",
    "yaw_misalignment_abs",
]

# %%
def wrap_angle_deg(series: pd.Series) -> pd.Series:
    return ((series + 180.0) % 360.0) - 180.0


def engineer_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sensor_2_avg" in df.columns:
        relative_direction = wrap_angle_deg(df["sensor_2_avg"].astype(float))
        df["sensor_2_avg_sin"] = np.sin(np.radians(relative_direction))
        df["sensor_2_avg_cos"] = np.cos(np.radians(relative_direction))
        df["yaw_misalignment_abs"] = np.abs(relative_direction)

    if "sensor_5_avg" in df.columns:
        pitch_angle = wrap_angle_deg(df["sensor_5_avg"].astype(float))
        df["sensor_5_avg_sin"] = np.sin(np.radians(pitch_angle))
        df["sensor_5_avg_cos"] = np.cos(np.radians(pitch_angle))

    if "sensor_42_avg" in df.columns:
        nacelle_direction = wrap_angle_deg(df["sensor_42_avg"].astype(float))
        df["sensor_42_avg_sin"] = np.sin(np.radians(nacelle_direction))
        df["sensor_42_avg_cos"] = np.cos(np.radians(nacelle_direction))

    return df
def get_feature_columns(df: pd.DataFrame) -> list:
    ordered_columns = []

    for col in BASE_FEATURE_COLUMNS + ENGINEERED_ANGLE_COLUMNS:
        if col in df.columns:
            ordered_columns.append(col)

    exclude_columns = {
        "time_stamp",
        "asset_id",
        "id",
        "train_test",
        "status_type_id",
        "event_id",
        "event_label",
        "event_label_binary",
        "event_description",
        "status_label",
    }

    return [col for col in ordered_columns if col not in exclude_columns]


def load_event_frame(event_row: pd.Series) -> pd.DataFrame:
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"

    df = pd.read_csv(csv_path, sep=";")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df["event_id"] = event_id
    df["event_label"] = event_row["event_label"]
    df["event_label_binary"] = int(event_row["event_label"] == "anomaly")
    df["event_description"] = event_row["event_description"]
    df["status_label"] = (~df["status_type_id"].isin(NORMAL_STATUS)).astype(int)

    df = engineer_angle_features(df)
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.85) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("time_stamp").reset_index(drop=True)

    if len(df) <= 1:
        return df.copy(), df.iloc[0:0].copy()

    split_idx = int(len(df) * train_ratio)
    split_idx = max(1, min(split_idx, len(df) - 1))

    train_part = df.iloc[:split_idx].copy()
    val_part = df.iloc[split_idx:].copy()
    return train_part, val_part


def fill_missing_by_event(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (
        df.groupby("event_id", group_keys=False)[feature_cols]
        .apply(lambda part: part.ffill().bfill().fillna(0.0))
    )
    return df


def create_sequence_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 288,
    step_size: int = 6,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, meta = [], [], []

    for event_id, group in df.groupby("event_id"):
        group = group.sort_values("time_stamp").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        status_flags = group["status_label"].to_numpy(dtype=np.int32)
        if len(group) < window_size:
            continue

        for start_idx in range(0, len(group) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            end_row = group.iloc[end_idx - 1]

            X.append(values[start_idx:end_idx])
            y.append(int(status_flags[start_idx:end_idx].max()))
            meta.append(
                {
                    "asset_id": int(end_row["asset_id"]),
                    "event_id": int(event_id),
                    "event_label": end_row["event_label"],
                    "event_label_binary": int(end_row["event_label_binary"]),
                    "window_start_time": group.iloc[start_idx]["time_stamp"],
                    "window_end_time": end_row["time_stamp"],
                    "status_window_max": int(status_flags[start_idx:end_idx].max()),
                }
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        pd.DataFrame(meta),
    )

# %%
train_parts = []
val_parts = []
test_parts = []

for _, event_row in asset_events.iterrows():
    event_df = load_event_frame(event_row)

    train_rows = event_df[event_df["train_test"] == "train"].copy()
    prediction_rows = event_df[event_df["train_test"] == "prediction"].copy()

    train_split, val_split = temporal_split(train_rows, train_ratio=TRAIN_RATIO)

    print(
        f"event={int(event_row['event_id'])} "
        f"label={event_row['event_label']} "
        f"train_rows={len(train_split)} "
        f"val_rows={len(val_split)} "
        f"prediction_rows={len(prediction_rows)}"
    )

    if not train_split.empty:
        train_parts.append(train_split)
    if not val_split.empty:
        val_parts.append(val_split)
    if not prediction_rows.empty:
        test_parts.append(prediction_rows)

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

combined_for_schema = pd.concat([train_df, val_df, test_df], ignore_index=True)
feature_cols = get_feature_columns(combined_for_schema)

train_df = fill_missing_by_event(train_df, feature_cols)
val_df = fill_missing_by_event(val_df, feature_cols)
test_df = fill_missing_by_event(test_df, feature_cols)

print("\nNumber of features:", len(feature_cols))
print(feature_cols)

# %%
normal_train_rows = train_df[train_df["status_label"] == 0].copy()

scaler = StandardScaler()
scaler.fit(normal_train_rows[feature_cols].to_numpy(dtype=np.float32))

train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols].to_numpy(dtype=np.float32))
val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

print("Scaler fit rows:", len(normal_train_rows))

# %%
X_train_all, y_train_all, meta_train_all = create_sequence_windows(
    train_df,
    feature_cols=feature_cols,
    window_size=WINDOW_SIZE,
    step_size=TRAIN_STEP_SIZE,
)

X_val_all, y_val_all, meta_val_all = create_sequence_windows(
    val_df,
    feature_cols=feature_cols,
    window_size=WINDOW_SIZE,
    step_size=TRAIN_STEP_SIZE,
)

X_test, y_test, meta_test = create_sequence_windows(
    test_df,
    feature_cols=feature_cols,
    window_size=WINDOW_SIZE,
    step_size=TEST_STEP_SIZE,
)

train_mask = y_train_all == 0
val_mask = y_val_all == 0

X_train = X_train_all[train_mask]
meta_train = meta_train_all.loc[train_mask].reset_index(drop=True)

X_val_fit = X_val_all[val_mask]
meta_val_fit = meta_val_all.loc[val_mask].reset_index(drop=True)

X_val_tune = X_val_all
y_val = y_val_all
meta_val_tune = meta_val_all.reset_index(drop=True)

if len(X_train) == 0:
    raise RuntimeError(
        "No normal training windows were created. "
        "Try a smaller WINDOW_SIZE or choose another asset."
    )

if len(X_val_fit) == 0:
    fallback_count = min(len(X_train), max(32, len(X_train) // 10))
    X_val_fit = X_train[:fallback_count].copy()
    meta_val_fit = meta_train.iloc[:fallback_count].copy().reset_index(drop=True)
    print("\n[WARN] No all-normal validation windows found. Using a fallback slice from X_train for fit validation.")

if len(X_val_tune) == 0:
    fallback_count = min(len(X_train), max(32, len(X_train) // 10))
    X_val_tune = X_train[:fallback_count].copy()
    y_val = np.zeros(fallback_count, dtype=np.int32)
    meta_val_tune = meta_train.iloc[:fallback_count].copy().reset_index(drop=True)
    print("\n[WARN] No validation tuning windows found. Using a fallback slice from X_train.")

if len(X_test) == 0:
    raise RuntimeError(
        "No test windows were created. "
        "Try a smaller WINDOW_SIZE or check the selected asset."
    )

print("X_train:", X_train.shape)
print("X_val_fit:", X_val_fit.shape)
print("X_val_tune:", X_val_tune.shape)
print("X_test:", X_test.shape)

print("\nValidation labels:")
print(pd.Series(y_val).value_counts(dropna=False).sort_index())

print("\nTest labels:")
print(pd.Series(y_test).value_counts(dropna=False).sort_index())

# %%
np.savez_compressed(
    OUTPUT_DIR / "data.npz",
    X_train=X_train,
    X_val_fit=X_val_fit,
    X_val_tune=X_val_tune,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)

meta_train.to_csv(OUTPUT_DIR / "meta_train.csv", index=False)
meta_val_fit.to_csv(OUTPUT_DIR / "meta_val_fit.csv", index=False)
meta_val_tune.to_csv(OUTPUT_DIR / "meta_val_tune.csv", index=False)
meta_test.to_csv(OUTPUT_DIR / "meta_test.csv", index=False)

with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

summary = {
    "asset_id": ASSET_ID,
    "window_size": WINDOW_SIZE,
    "train_step_size": TRAIN_STEP_SIZE,
    "test_step_size": TEST_STEP_SIZE,
    "normal_status": sorted(NORMAL_STATUS),
    "feature_cols": feature_cols,
    "num_train_windows": int(len(X_train)),
    "num_val_fit_windows": int(len(X_val_fit)),
    "num_val_tune_windows": int(len(X_val_tune)),
    "num_test_windows": int(len(X_test)),
}

with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved to:", OUTPUT_DIR)
