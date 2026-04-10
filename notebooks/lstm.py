#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install gdown')

import gdown 

# Link thư mục của bạn
url = 'https://drive.google.com/drive/folders/1UPyGiP6ssAe4osBych7G24zrjgeveksz'

# Tải thư mục (sử dụng flag --folder)
get_ipython().system('gdown --id 11G02ZnjvR6F_VMcH9rKSO0yjmYTATKLP')
get_ipython().system('gdown --id 16ESI9OYF_7Fl8WeX3PHD_CaDbPB7Aml0')

import pandas as pd

df_event = pd.read_csv("./event_info.csv", sep=";")
event_id = df_event['event_id']

print(df_event.head())
print("="*40)
print(df_event.columns)
print("="*40)
print(df_event.describe())

# =========================
# SELECTED FEATURES AFTER CORRELATION ANALYSIS
# =========================

TEMPERATURE_FEATURES = [
    'sensor_6_avg',   # Hub controller temp
    'sensor_8_avg',   # VCS choke coils temp
    'sensor_10_avg',  # VCS cooling water temp
    'sensor_13_avg',  # Generator bearing DE
    'sensor_14_avg',  # Generator bearing NDE
    'sensor_37_avg',  # IGBT rotor side inverter L3
    'sensor_39_avg',  # HV transformer L2
    'sensor_41_avg',  # Hydraulic oil temp
]

RPM_FEATURES = [
    'sensor_18_avg',  # Generator RPM
    'sensor_18_std',  # Generator RPM variability
]

ELECTRICAL_FEATURES = [
    'sensor_23_avg',  # Current representative
    'sensor_32_avg',  # Voltage L1
    'sensor_34_avg',  # Voltage L3
    'sensor_26_avg',  # Grid frequency
    'sensor_22_avg',  # Phase displacement
]

WIND_POWER_FEATURES = [
    'wind_speed_3_avg',
    'wind_speed_3_std',
    'power_29_avg',   # Possible active power
    'power_30_avg',   # Actual grid power
    'power_30_std',
    'sensor_31_avg',  # Grid reactive power
    'sensor_31_std',
]

ANGLE_FEATURES = [
    'sensor_2_avg_sin',
    'sensor_2_avg_cos',
    'sensor_5_avg_sin',
    'sensor_5_avg_cos',
    'sensor_5_std',
    'yaw_misalignment_abs',
]

COUNTER_FEATURES = [
    'sensor_44',
    'sensor_48',
    'sensor_50',
    'sensor_51',
]

FEATURE_COLUMNS = (
    TEMPERATURE_FEATURES +
    RPM_FEATURES +
    ELECTRICAL_FEATURES +
    WIND_POWER_FEATURES
)

import numpy as np

def engineer_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in ANGLE_FEATURES:
        if col in df_copy.columns:
            radians = np.radians(df_copy[col])
            df_copy[f"{col}_sin"] = np.sin(radians)
            df_copy[f"{col}_cos"] = np.cos(radians)
            df_copy.drop(col, axis=1, inplace=True)
    return df_copy

def drop_counter_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in COUNTER_FEATURES if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    return df

def get_feature_columns(df: pd.DataFrame) -> list:
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]

    for angle_col in ANGLE_FEATURES:
        sin_col = f"{angle_col}_sin"
        cos_col = f"{angle_col}_cos"

        if sin_col in df.columns:
            feature_cols.append(sin_col)

        if cos_col in df.columns:
            feature_cols.append(cos_col)

    exclude_cols = [
        "time_stamp",
        "train_test",
        "label",
        "sequence_id",
        "status_type_id"
    ]

    exclude_all = exclude_cols + ANGLE_FEATURES + COUNTER_FEATURES

    feature_cols = [
        col for col in feature_cols
        if col not in exclude_all
    ]

    return feature_cols

def preprocess_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    features = df[feature_cols].copy()
    features = features.ffill().bfill()
    features = features.fillna(0)
    return features.values

# import pandas as pd
# anomaly_event_id = [68, 22, 72, 73, 0, 26, 40, 42, 10, 45, 84, 51]
# combined_df = []
# for i in event_id.values:
#     df = pd.read_csv(f"{data_path}/{i}.csv", sep=';')
#     df["sequence_id"] = i
#     df = df.drop(columns=["id"], errors="ignore")
#     df["label"] = 0
#     abnormal_status_condition = ~df["status_type_id"].isin([0, 2])
#     anomaly_prediction_condition = (
#         (df["train_test"] == "prediction")
#         & (i in anomaly_event_id)
#     )
#     anomaly_condition = (
#         abnormal_status_condition
#         | anomaly_prediction_condition
#     )
#     df.loc[anomaly_condition, "label"] = 1
#     combined_df.append(df)
# combined_df = pd.concat(combined_df, ignore_index=True)
# combined_df.to_csv("combined_dataset.csv", index=False)
# print(combined_df["label"].value_counts())

combined_df = pd.read_csv("combined_dataset.csv")
combined_df["time_stamp"] = pd.to_datetime(combined_df["time_stamp"])

# PURE NORMAL training set
df_train = combined_df[
    (combined_df["train_test"] == "train") &
    (combined_df["label"] == 0)
].copy()

# test / prediction set
df_test = combined_df[
    combined_df["train_test"] == "prediction"
].copy()

print("Train labels:")
print(df_train["label"].value_counts())

print("\nTest labels:")
print(df_test["label"].value_counts())

print(
    df_train.groupby("sequence_id")["label"].value_counts()
)

# LSTM INPUT SHAPE:
# (SAMPLE, TIMESTEPS, FEATURES)

import numpy as np
import pandas as pd

def create_sequence_windows(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "ground_truth",
    window_size: int = 24,
    step_size: int = 1,
    group_col: str = "sequence_id",
    time_col: str = "time_stamp"
):
    X, y, meta = [], [], []

    # make sure timestamp is datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # process each original CSV independently
    for seq_id, group in df.groupby(group_col):
        group = group.sort_values(time_col).reset_index(drop=True)

        # fill missing values only within this sequence
        features = group[feature_cols].copy()
        features = features.ffill().bfill().fillna(0)

        labels = group[label_col].values
        timestamps = group[time_col].values

        # sliding window
        for i in range(0, len(group) - window_size + 1, step_size):
            X_window = features.iloc[i:i + window_size].values
            y_label = labels[i + window_size - 1]   # label = last timestep
            end_time = timestamps[i + window_size - 1]

            X.append(X_window)
            y.append(y_label)

            meta.append({
                "sequence_id": seq_id,
                "window_end_time": end_time
            })

    return np.array(X), np.array(y), pd.DataFrame(meta)

from sklearn.preprocessing import MinMaxScaler
feature_cols = get_feature_columns(df_train)

X_train, y_train, meta_train = create_sequence_windows(
    df_train,
    feature_cols=feature_cols,
    label_col="label",
    window_size=48,
    step_size=6
)

X_test, y_test, meta_test = create_sequence_windows(
    df_test,
    feature_cols=feature_cols,
    label_col="label",
    window_size=48
)

X_train = np.nan_to_num(X_train).astype(np.float32)
X_test = np.nan_to_num(X_test).astype(np.float32)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_shape = X_train.shape
X_test_shape = X_test.shape

X_train = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train_shape)

X_test = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test_shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional

timesteps = X_train.shape[1]
n_features = X_train.shape[2]

inputs = Input(shape=(timesteps, n_features))

x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Bidirectional(LSTM(32, return_sequences=True))(x)
x = Bidirectional(LSTM(16, return_sequences=False))(x)

x = RepeatVector(timesteps)(x)

x = Bidirectional(LSTM(16, return_sequences=True))(x)
x = Bidirectional(LSTM(32, return_sequences=True))(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)

outputs = TimeDistributed(Dense(n_features))(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

split_idx = int(len(X_train) * 0.85)

X_val = X_train[split_idx:]
X_train_final = X_train[:split_idx]

history = model.fit(
    X_train_final,
    X_train_final,
    validation_data=(X_val, X_val),
    epochs=40,
    batch_size=128,
    verbose=1
)

model.save('lstm.keras')

import tensorflow as tf

model = tf.keras.models.load_model("/kaggle/input/models/thuyennn/lstm2/keras/default/1/lstm.keras")
model.summary()

reconstructions = model.predict(X_test)

errors = np.mean((X_test - reconstructions)**2, axis=(1, 2))

threshold = np.percentile(errors, 95)

anomalies = errors > threshold

import matplotlib.pyplot as plt
import numpy as np

i = 0  # sample index
t = np.arange(X_test.shape[1])  # trục thời gian

plt.figure(figsize=(12, 5))
plt.plot(t, X_test[i, :, 0], label='Original', linewidth=2)
plt.plot(t, reconstructions[i, :, 0], label='Reconstructed', linewidth=2)
plt.title(f'Sample {i} - Feature 0')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

y_pred = anomalies.astype(int)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

y_pred = (errors > threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, errors)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"AUC ROC   : {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

import matplotlib.pyplot as plt

def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_learning_curves(history)