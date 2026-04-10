"""
RF_SMOTE.py
=============================================================
Train a Random Forest classifier on the combined dataset with:
  - sequence-level stratified train/test split
  - selected Spearman-filtered features
  - custom angle engineering
  - sliding windows flattened for Random Forest input
  - SMOTE applied only on the training windows
=============================================================
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
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

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:
    raise ImportError(
        "imblearn is required for RF_SMOTE.py. Install it with "
        "'pip install imbalanced-learn'."
    ) from exc


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, SEQUENCE_LENGTH, STRIDE
from data_pipeline.preprocessing.feature_engineering import (
    create_sequence_windows,
    engineer_selected_angle_features,
    fill_missing_by_group,
    flatten_windows,
    get_selected_feature_columns,
    # normalize_binary_labels,
)
from models.architectures.random_forest import RandomForestModel
from src.data_pipeline.preprocessing.feature_engineering import FeatureEngineer

SPLIT_RATIO = 0.70
TEST_RATIO = 1.0 - SPLIT_RATIO
WINDOW_SIZE = SEQUENCE_LENGTH
STEP_SIZE = STRIDE
COMBINED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "combined_dataset.csv")
MODELS_OUT_DIR = os.path.join(MODELS_DIR, "rf_smote")


def split_sequences(
    df: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
) -> Tuple[List[int], List[int], pd.DataFrame]:
    """
    Split sequence_id values in a stratified way so each whole sequence
    stays entirely in train or test.
    """
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


def load_preprocessed_combined_dataset(csv_path: str = COMBINED_DATASET_PATH) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the combined dataset and apply the selected preprocessing flow.
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("combined_dataset.csv must contain a 'label' column.")
    if "sequence_id" not in df.columns:
        raise ValueError("combined_dataset.csv must contain a 'sequence_id' column.")

    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    # df["label"] = normalize_binary_labels(df["label"])
    df = engineer_selected_angle_features(df)

    feature_cols = get_selected_feature_columns(df)
    if not feature_cols:
        raise ValueError("No selected feature columns were found in combined_dataset.csv.")

    df = fill_missing_by_group(df, feature_cols, group_col="sequence_id")
    return df, feature_cols


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance the training windows with SMOTE."""
    class_counts = pd.Series(y_train).value_counts()

    if len(class_counts) < 2:
        raise ValueError("Training windows must contain both normal and anomaly classes.")

    minority_count = int(class_counts.min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least 2 minority windows in the training set.")

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute evaluation metrics on held-out windows."""
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


def main() -> None:
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)

    df, feature_cols = load_preprocessed_combined_dataset(COMBINED_DATASET_PATH)
    train_sequence_ids, test_sequence_ids, sequence_labels = split_sequences(df)

    train_df = df[df["sequence_id"].isin(train_sequence_ids)].copy()
    test_df = df[df["sequence_id"].isin(test_sequence_ids)].copy()

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

    if len(X_train_seq) == 0:
        raise ValueError("No training windows were created. Check WINDOW_SIZE and sequence lengths.")
    if len(X_test_seq) == 0:
        raise ValueError("No test windows were created. Check WINDOW_SIZE and sequence lengths.")

    X_train_raw = flatten_windows(X_train_seq)
    X_test_raw = flatten_windows(X_test_seq)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

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

    print("=" * 70)
    print("RANDOM FOREST + SMOTE (COMBINED DATASET)")
    print("=" * 70)
    print(f"Dataset: {COMBINED_DATASET_PATH}")
    print(f"Selected features: {len(feature_cols)}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Step size: {STEP_SIZE}")
    print("\nTrain sequence counts:")
    print(train_sequence_summary)
    print("\nTest sequence counts:")
    print(test_sequence_summary)
    print("\nTrain window counts before SMOTE:")
    print(pd.Series(y_train).map({0: "normal", 1: "anomaly"}).value_counts())
    print("\nTrain window counts after SMOTE:")
    print(pd.Series(y_train_balanced).map({0: "normal", 1: "anomaly"}).value_counts())
    print("\nTest window counts:")
    print(pd.Series(y_test).map({0: "normal", 1: "anomaly"}).value_counts())

    model = RandomForestModel(class_weight=None).build()
    model.fit(X_train_balanced, y_train_balanced)

    metrics = evaluate_model(model, X_test_scaled, y_test)

    print("\nMetrics:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:")
    print(metrics["classification_report"])

    joblib.dump(model, os.path.join(MODELS_OUT_DIR, "rf_smote.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_OUT_DIR, "rf_smote_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_OUT_DIR, "rf_smote_feature_cols.pkl"))

    pd.DataFrame({"sequence_id": train_sequence_ids}).to_csv(
        os.path.join(MODELS_OUT_DIR, "train_sequence_ids.csv"),
        index=False,
    )
    pd.DataFrame({"sequence_id": test_sequence_ids}).to_csv(
        os.path.join(MODELS_OUT_DIR, "test_sequence_ids.csv"),
        index=False,
    )
    train_meta.to_csv(os.path.join(MODELS_OUT_DIR, "train_window_meta.csv"), index=False)
    test_meta.to_csv(os.path.join(MODELS_OUT_DIR, "test_window_meta.csv"), index=False)

    with open(os.path.join(MODELS_OUT_DIR, "rf_smote_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to: {MODELS_OUT_DIR}")


if __name__ == "__main__":
    main()
