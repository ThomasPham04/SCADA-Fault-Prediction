#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Kaggle Notebook 2
# # Train and evaluate autoencoders on one asset
#
# This notebook keeps the same style as your training notebook:
# - load one prepared dataset
# - train multiple autoencoders
# - compare metrics
# - save scores, metrics, and plots

# %%
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, GRU, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# %%
ASSET_ID = 10

# If you run notebook 2 right after notebook 1 in the same Kaggle session,
# the first path will work. If you publish notebook 1 output as a Kaggle
# dataset, change the second path to your dataset name.
candidate_dirs = [
    Path("/kaggle/working/processed_asset") / f"asset_{ASSET_ID}",
    Path("/kaggle/input/processed-asset-autoencoder") / f"asset_{ASSET_ID}",
]

PROCESSED_DIR = None
for candidate in candidate_dirs:
    if candidate.exists():
        PROCESSED_DIR = candidate
        break

if PROCESSED_DIR is None:
    raise FileNotFoundError("Could not find processed asset directory. Update candidate_dirs first.")

SAVE_DIR = Path("/kaggle/working/autoencoder_results") / f"asset_{ASSET_ID}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# %%
data = np.load(PROCESSED_DIR / "data.npz")

X_train = data["X_train"]
X_val_fit = data["X_val_fit"]
X_val_tune = data["X_val_tune"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

with open(PROCESSED_DIR / "summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Processed dir:", PROCESSED_DIR)
print("X_train:", X_train.shape)
print("X_val_fit:", X_val_fit.shape)
print("X_val_tune:", X_val_tune.shape)
print("X_test:", X_test.shape)

print("\nValidation labels:")
print(pd.Series(y_val).value_counts(dropna=False).sort_index())

print("\nTest labels:")
print(pd.Series(y_test).value_counts(dropna=False).sort_index())

print("\nFeature count:", len(summary["feature_cols"]))

if len(X_train) == 0:
    raise RuntimeError("X_train is empty. Run notebook 1 again with a different asset or smaller WINDOW_SIZE.")

if len(X_val_fit) == 0:
    raise RuntimeError("X_val_fit is empty. Run notebook 1 again to rebuild the prepared dataset.")

if len(X_val_tune) == 0:
    raise RuntimeError("X_val_tune is empty. Run notebook 1 again to rebuild the prepared dataset.")

if len(X_test) == 0:
    raise RuntimeError("X_test is empty. Run notebook 1 again with a different asset or smaller WINDOW_SIZE.")

# %%
def reconstruction_scores(model, X, batch_size=256):
    X_hat = model.predict(X, batch_size=batch_size, verbose=0)
    scores = np.mean((X - X_hat) ** 2, axis=(1, 2))
    return scores, X_hat


def search_best_threshold(scores, labels):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(labels)) < 2:
        return float(np.percentile(scores, 95)), "fallback_95_percentile"

    candidate_thresholds = np.quantile(scores, np.linspace(0.80, 0.995, 120))

    best_threshold = None
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        preds = (scores > threshold).astype(int)
        current_f1 = f1_score(labels, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold, "best_f1_on_val"


def evaluate_autoencoder(
    model,
    X_val_tune,
    y_val,
    X_test,
    y_test,
    model_name="model",
    save_dir=Path("/kaggle/working"),
):
    val_scores, _ = reconstruction_scores(model, X_val_tune)
    threshold, threshold_source = search_best_threshold(val_scores, y_val)

    test_scores, test_recon = reconstruction_scores(model, X_test)
    y_pred = (test_scores > threshold).astype(int)

    metrics = {
        "model_name": model_name,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_scores))
        if len(np.unique(y_test)) > 1
        else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n===== {model_name.upper()} =====")
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"{key}: {value}")
    print("confusion_matrix:")
    print(np.array(metrics["confusion_matrix"]))

    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / f"{model_name}_val_scores.npy", val_scores)
    np.save(save_dir / f"{model_name}_test_scores.npy", test_scores)
    np.save(save_dir / f"{model_name}_test_recon.npy", test_recon)

    with open(save_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, y_pred, test_scores, test_recon, threshold


def get_callbacks(model_name):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(SAVE_DIR / f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

# %%
def build_gru_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))

    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = Bidirectional(GRU(16, return_sequences=False))(x)

    x = RepeatVector(timesteps)(x)
    x = Bidirectional(GRU(16, return_sequences=True))(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)

    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs, name="gru_autoencoder")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def build_cnn_lstm_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))

    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)

    x = RepeatVector(timesteps)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(32, activation="relu"))(x)
    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs, name="cnn_lstm_autoencoder")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model

# %%
timesteps = X_train.shape[1]
n_features = X_train.shape[2]

gru_model = build_gru_autoencoder(timesteps, n_features)
gru_model.summary()

history_gru = gru_model.fit(
    X_train,
    X_train,
    validation_data=(X_val_fit, X_val_fit),
    epochs=40,
    batch_size=128,
    callbacks=get_callbacks("gru_autoencoder"),
    verbose=1,
)

# %%
gru_metrics, gru_y_pred, gru_test_scores, gru_test_recon, gru_threshold = evaluate_autoencoder(
    gru_model,
    X_val_tune=X_val_tune,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    model_name="gru_autoencoder",
    save_dir=SAVE_DIR,
)

# %%
cnn_lstm_model = build_cnn_lstm_autoencoder(timesteps, n_features)
cnn_lstm_model.summary()

history_cnn_lstm = cnn_lstm_model.fit(
    X_train,
    X_train,
    validation_data=(X_val_fit, X_val_fit),
    epochs=40,
    batch_size=128,
    callbacks=get_callbacks("cnn_lstm_autoencoder"),
    verbose=1,
)

# %%
cnn_lstm_metrics, cnn_lstm_y_pred, cnn_lstm_test_scores, cnn_lstm_test_recon, cnn_lstm_threshold = evaluate_autoencoder(
    cnn_lstm_model,
    X_val_tune=X_val_tune,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    model_name="cnn_lstm_autoencoder",
    save_dir=SAVE_DIR,
)

# %%
comparison = pd.DataFrame(
    [
        {
            "model": "GRU Autoencoder",
            "threshold": gru_metrics["threshold"],
            "accuracy": gru_metrics["accuracy"],
            "precision": gru_metrics["precision"],
            "recall": gru_metrics["recall"],
            "f1": gru_metrics["f1"],
            "roc_auc": gru_metrics["roc_auc"],
        },
        {
            "model": "CNN-LSTM Autoencoder",
            "threshold": cnn_lstm_metrics["threshold"],
            "accuracy": cnn_lstm_metrics["accuracy"],
            "precision": cnn_lstm_metrics["precision"],
            "recall": cnn_lstm_metrics["recall"],
            "f1": cnn_lstm_metrics["f1"],
            "roc_auc": cnn_lstm_metrics["roc_auc"],
        },
    ]
)

comparison.to_csv(SAVE_DIR / "comparison.csv", index=False)
comparison

# %%
def plot_history(history, title="Training History"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_history(history_gru, "GRU Autoencoder Loss")
plot_history(history_cnn_lstm, "CNN-LSTM Autoencoder Loss")

# %%
def plot_score_distribution(scores, labels, threshold, title="Score Distribution"):
    plt.figure(figsize=(9, 5))
    plt.hist(scores[labels == 0], bins=80, alpha=0.6, label="Normal")
    plt.hist(scores[labels == 1], bins=80, alpha=0.6, label="Anomaly")
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
    plt.title(title)
    plt.xlabel("Reconstruction score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_score_distribution(gru_test_scores, y_test, gru_threshold, "GRU Score Distribution")
plot_score_distribution(cnn_lstm_test_scores, y_test, cnn_lstm_threshold, "CNN-LSTM Score Distribution")

# %%
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, int(cm[row, col]), ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(
    np.array(gru_metrics["confusion_matrix"]),
    "GRU Confusion Matrix",
)
plot_confusion_matrix(
    np.array(cnn_lstm_metrics["confusion_matrix"]),
    "CNN-LSTM Confusion Matrix",
)
