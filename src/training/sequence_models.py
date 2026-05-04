"""
sequence_models.py — training.sequence_models
Keras model builders and training callbacks for classifiers and autoencoders.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers


def build_classifier_model(
    model_name: str,
    input_shape: tuple,
    learning_rate: float = 1e-3,
    dropout_rate: float | None = None,
    l2_strength: float = 0.0,
):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if dropout_rate is not None and not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1).")
    if l2_strength < 0:
        raise ValueError("l2_strength must be non-negative.")

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    regularizer = regularizers.l2(l2_strength) if l2_strength > 0 else None
    recurrent_dropout = 0.25 if dropout_rate is None else dropout_rate
    conv_dropout = 0.20 if dropout_rate is None else dropout_rate

    if model_name == "lstm":
        x = layers.LSTM(
            96,
            return_sequences=True,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(inputs)
        x = layers.Dropout(recurrent_dropout)(x)
        x = layers.LSTM(
            48,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    elif model_name == "gru":
        x = layers.GRU(
            96,
            return_sequences=True,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(inputs)
        x = layers.Dropout(recurrent_dropout)(x)
        x = layers.GRU(
            48,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    elif model_name == "cnn_lstm":
        x = layers.Conv1D(
            64,
            5,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(conv_dropout)(x)
        x = layers.Conv1D(
            64,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(
            64,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    elif model_name == "cnn_gru":
        x = layers.Conv1D(
            64,
            5,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(conv_dropout)(x)
        x = layers.Conv1D(
            64,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.GRU(
            64,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    else:
        raise ValueError(f"Unsupported classifier model: {model_name}")

    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def build_autoencoder_model(
    model_name: str,
    input_shape: tuple,
    encoder_units: int | None = None,
    bottleneck_units: int | None = None,
    learning_rate: float = 1e-3,
    noise_stddev: float = 0.0,
):
    time_steps, feature_count = input_shape

    # Paper defaults for dense_ae (Wind Farm A: 44→25→4→25→44)
    if encoder_units is None:
        encoder_units = 25 if model_name == "dense_ae" else 96
    if bottleneck_units is None:
        bottleneck_units = 4 if model_name == "dense_ae" else 48

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    x = layers.GaussianNoise(noise_stddev)(inputs) if noise_stddev > 0.0 else inputs

    if model_name == "lstm_ae":
        x = layers.LSTM(encoder_units, return_sequences=True)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.LSTM(bottleneck_units, return_sequences=False)(x)
        x = layers.RepeatVector(time_steps)(x)
        x = layers.LSTM(bottleneck_units, return_sequences=True)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.LSTM(encoder_units, return_sequences=True)(x)
    elif model_name == "gru_ae":
        x = layers.GRU(encoder_units, return_sequences=True)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.GRU(bottleneck_units, return_sequences=False)(x)
        x = layers.RepeatVector(time_steps)(x)
        x = layers.GRU(bottleneck_units, return_sequences=True)(x)
        x = layers.Dropout(0.20)(x)
        x = layers.GRU(encoder_units, return_sequences=True)(x)
    elif model_name == "dense_ae":
        # Pointwise MLP: each timestep processed independently (no temporal dependencies).
        # Paper architecture for Wind Farm A: input→25→4→25→output
        x = layers.TimeDistributed(layers.Dense(encoder_units, activation="relu"))(x)
        x = layers.TimeDistributed(layers.Dense(bottleneck_units, activation="relu"))(x)
        x = layers.TimeDistributed(layers.Dense(encoder_units, activation="relu"))(x)
    else:
        raise ValueError(f"Unsupported autoencoder model: {model_name!r}")

    outputs = layers.TimeDistributed(layers.Dense(feature_count))(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_threshold_nn(
    input_dim: int,
    hidden_units: int = 23,
    learning_rate: float = 1e-3,
):
    """Small MLP regression: input features → expected L2-norm reconstruction error."""
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(1),
        ],
        name="threshold_nn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def threshold_nn_callbacks(model_path: Path, patience: int = 3):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
    ]


def classifier_callbacks(model_path: Path):
    return [
        callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]


def autoencoder_callbacks(model_path: Path, patience: int = 3):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
    ]
