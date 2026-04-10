#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Kaggle Notebook
# # Random Forest + SMOTE on Combined Dataset
#
# Notebook này dùng:
# - `combined_dataset.csv`
# - selected features sau khi lọc bằng Spearman correlation
# - custom angle engineering
# - sequence-level split theo `sequence_id`
# - sliding windows rồi flatten cho Random Forest
# - SMOTE chỉ áp dụng trên training windows

# %%
# Nếu Kaggle chưa có imbalanced-learn thì chạy cell này:
# !pip install -q imbalanced-learn

# %%
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Change this path to your Kaggle input dataset path.
COMBINED_DATASET_PATH = Path("/kaggle/input/your-dataset-name/combined_dataset.csv")

OUTPUT_DIR = Path("/kaggle/working/rf_smote_combined")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
TEST_RATIO = 1.0 - TRAIN_RATIO
WINDOW_SIZE = 288
STEP_SIZE = 6

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
def normalize_binary_labels(label_series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(label_series):
        return label_series.astype(int)

    mapped = (
        label_series.astype(str)
        .str.strip()
        .str.lower()
        .map({"normal": 0, "anomaly": 1, "0": 0, "1": 1})
    )

    if mapped.isna().any():
        bad_values = sorted(label_series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported label values found: {bad_values}")

    return mapped.astype(int)


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


def fill_missing_by_group(
    df: pd.DataFrame,
    feature_cols: list,
    group_col: str = "sequence_id",
) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (
        df.groupby(group_col, group_keys=False)[feature_cols]
        .apply(lambda part: part.ffill().bfill().fillna(0.0))
    )
    return df


def create_sequence_windows(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
    window_size: int = 288,
    step_size: int = 6,
    group_col: str = "sequence_id",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, meta = [], [], []

    for sequence_id, group in df.groupby(group_col):
        group = group.sort_values("time_stamp").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group[label_col].astype(np.int32).to_numpy()

        if len(group) < window_size:
            continue

        for start_idx in range(0, len(group) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            end_row = group.iloc[end_idx - 1]

            X.append(values[start_idx:end_idx])
            y.append(int(labels[start_idx:end_idx].max()))
            meta.append(
                {
                    "sequence_id": int(sequence_id),
                    "asset_id": int(end_row["asset_id"]) if "asset_id" in end_row.index else None,
                    "window_label": int(labels[start_idx:end_idx].max()),
                    "window_start_time": group.iloc[start_idx]["time_stamp"],
                    "window_end_time": end_row["time_stamp"],
                }
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        pd.DataFrame(meta),
    )


def flatten_windows(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D window array, got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


def split_sequences(
    df: pd.DataFrame,
    test_ratio: float = 0.30,
    random_state: int = 42,
) -> tuple[list, list, pd.DataFrame]:
    sequence_labels = (
        df.groupby("sequence_id")["label"]
        .max()
        .reset_index()
        .rename(columns={"label": "sequence_label"})
    )

    train_ids, test_ids = train_test_split(
        sequence_labels["sequence_id"],
        test_size=test_ratio,
        random_state=random_state,
        stratify=sequence_labels["sequence_label"],
    )

    return list(train_ids), list(test_ids), sequence_labels


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    class_counts = pd.Series(y_train).value_counts()

    if len(class_counts) < 2:
        raise ValueError("Training windows must contain both normal and anomaly classes.")

    minority_count = int(class_counts.min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least 2 minority windows in the training set.")

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["normal", "anomaly"],
            zero_division=0,
        ),
    }

# %%
df = pd.read_csv(COMBINED_DATASET_PATH)
df["time_stamp"] = pd.to_datetime(df["time_stamp"])
df["label"] = normalize_binary_labels(df["label"])
df = engineer_angle_features(df)

feature_cols = get_feature_columns(df)
df = fill_missing_by_group(df, feature_cols, group_col="sequence_id")

print("Dataset shape:", df.shape)
print("Number of selected features:", len(feature_cols))
print(feature_cols)

# %%
train_sequence_ids, test_sequence_ids, sequence_labels = split_sequences(
    df,
    test_ratio=TEST_RATIO,
    random_state=RANDOM_SEED,
)

train_df = df[df["sequence_id"].isin(train_sequence_ids)].copy()
test_df = df[df["sequence_id"].isin(test_sequence_ids)].copy()

train_sequence_summary = (
    sequence_labels[sequence_labels["sequence_id"].isin(train_sequence_ids)]["sequence_label"]
    .map({0: "normal", 1: "anomaly"})
    .value_counts()
    .sort_index()
)
test_sequence_summary = (
    sequence_labels[sequence_labels["sequence_id"].isin(test_sequence_ids)]["sequence_label"]
    .map({0: "normal", 1: "anomaly"})
    .value_counts()
    .sort_index()
)

print("Train sequence counts:")
print(train_sequence_summary)
print("\nTest sequence counts:")
print(test_sequence_summary)

# %%
X_train_seq, y_train, train_meta = create_sequence_windows(
    train_df,
    feature_cols=feature_cols,
    label_col="label",
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    group_col="sequence_id",
)

X_test_seq, y_test, test_meta = create_sequence_windows(
    test_df,
    feature_cols=feature_cols,
    label_col="label",
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    group_col="sequence_id",
)

print("X_train_seq shape:", X_train_seq.shape)
print("X_test_seq shape :", X_test_seq.shape)
print("\nTrain window labels:")
print(pd.Series(y_train).map({0: "normal", 1: "anomaly"}).value_counts())
print("\nTest window labels:")
print(pd.Series(y_test).map({0: "normal", 1: "anomaly"}).value_counts())

# %%
X_train = flatten_windows(X_train_seq)
X_test = flatten_windows(X_test_seq)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_balanced, y_train_balanced = apply_smote(
    X_train_scaled,
    y_train,
    random_state=RANDOM_SEED,
)

print("Before SMOTE:")
print(pd.Series(y_train).map({0: "normal", 1: "anomaly"}).value_counts())
print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).map({0: "normal", 1: "anomaly"}).value_counts())

# %%
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=None,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

rf_model.fit(X_train_balanced, y_train_balanced)

print("Random Forest training complete.")

# %%
metrics = evaluate_model(rf_model, X_test_scaled, y_test)

print("Metrics:")
print(f"  Accuracy : {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall   : {metrics['recall']:.4f}")
print(f"  F1-score : {metrics['f1']:.4f}")
print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

print("\nConfusion matrix:")
print(np.array(metrics["confusion_matrix"]))

print("\nClassification report:")
print(metrics["classification_report"])

# %%
joblib.dump(rf_model, OUTPUT_DIR / "rf_smote.pkl")
joblib.dump(scaler, OUTPUT_DIR / "rf_smote_scaler.pkl")
joblib.dump(feature_cols, OUTPUT_DIR / "rf_smote_feature_cols.pkl")

pd.DataFrame({"sequence_id": train_sequence_ids}).to_csv(
    OUTPUT_DIR / "train_sequence_ids.csv",
    index=False,
)
pd.DataFrame({"sequence_id": test_sequence_ids}).to_csv(
    OUTPUT_DIR / "test_sequence_ids.csv",
    index=False,
)
train_meta.to_csv(OUTPUT_DIR / "train_window_meta.csv", index=False)
test_meta.to_csv(OUTPUT_DIR / "test_window_meta.csv", index=False)

with open(OUTPUT_DIR / "rf_smote_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Saved artifacts to:", OUTPUT_DIR)
