"""
Kaggle-ready sequence preparation for SCADA fault detection.

Edit the settings block below, place the required CSV files next to the
notebook/script, and run this file. The script will:
1. Search for the best window sizes with a small GRU probe model.
2. Export classifier-ready global datasets.
3. Export per-asset autoencoder datasets.
"""

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks, layers


# ============================================================================
# SETTINGS
# ============================================================================

DATA_FILE = "df_final.csv"
FEATURE_FILE = "final_features.csv"
OUTPUT_DIR = "sequence_exports"
WINDOW_CANDIDATES_HOURS = [6, 12, 24, 36, 48, 72]
TOP_K_WINDOWS = 2
STRIDE_STEPS = 6
VAL_RATIO = 0.15
NORMAL_STATUSES = [0, 2]
RANDOM_SEED = 42
PROBE_EPOCHS = 8
PROBE_BATCH_SIZE = 256


TIME_RESOLUTION_MINUTES = 10
EXPECTED_FEATURE_COUNT = 17
PROBE_MAX_TRAIN_WINDOWS = 60000


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def window_steps_from_hours(window_hours: int) -> int:
    return int((window_hours * 60) / TIME_RESOLUTION_MINUTES)


def load_inputs():
    data_path = Path(DATA_FILE)
    feature_path = Path(FEATURE_FILE)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = pd.read_csv(data_path)
    feature_df = pd.read_csv(feature_path)

    if feature_df.empty:
        raise ValueError("Feature file is empty.")

    feature_col_name = "final_feature" if "final_feature" in feature_df.columns else feature_df.columns[0]
    feature_cols = (
        feature_df[feature_col_name]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_cols)} in {FEATURE_FILE}."
        )

    return df, feature_cols


def prepare_base_dataframe(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    required_cols = {"time_stamp", "asset_id", "train_test", "sequence_id"}
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing selected features in dataset: {missing_features}")

    if "label" not in df.columns and "status_type_id" not in df.columns:
        raise ValueError("Dataset must contain either 'label' or 'status_type_id'.")

    prepared = df.copy()
    prepared["time_stamp"] = pd.to_datetime(prepared["time_stamp"], errors="coerce")
    if prepared["time_stamp"].isna().any():
        bad_count = int(prepared["time_stamp"].isna().sum())
        raise ValueError(f"Found {bad_count} invalid time_stamp values.")

    prepared["train_test"] = prepared["train_test"].astype(str).str.strip().str.lower()
    valid_train_test = {"train", "prediction"}
    invalid_values = sorted(set(prepared["train_test"]) - valid_train_test)
    if invalid_values:
        raise ValueError(f"Unexpected train_test values: {invalid_values}")

    for col in feature_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    if "status_type_id" in prepared.columns:
        prepared["status_type_id"] = (
            pd.to_numeric(prepared["status_type_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
    else:
        prepared["status_type_id"] = -1

    if "label" in prepared.columns:
        prepared["label"] = (
            pd.to_numeric(prepared["label"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        prepared["label"] = (prepared["label"] > 0).astype(np.int8)
    else:
        prepared["label"] = (~prepared["status_type_id"].isin(NORMAL_STATUSES)).astype(np.int8)

    prepared = prepared.sort_values(
        ["asset_id", "sequence_id", "time_stamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    return prepared


def empty_split_frame(columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns=columns + ["data_split"])


def split_train_val_test_by_sequence(df: pd.DataFrame):
    train_parts = []
    val_parts = []
    test_parts = []

    for (_, _), group in df.groupby(["asset_id", "sequence_id"], sort=False):
        group = group.sort_values("time_stamp", kind="mergesort")
        group_train = group[group["train_test"] == "train"].copy()
        group_test = group[group["train_test"] == "prediction"].copy()

        if not group_train.empty:
            if len(group_train) == 1:
                split_idx = 1
            else:
                split_idx = int(len(group_train) * (1.0 - VAL_RATIO))

            fit_part = group_train.iloc[:split_idx].copy()
            val_part = group_train.iloc[split_idx:].copy()

            fit_part["data_split"] = "train"
            val_part["data_split"] = "val"

            train_parts.append(fit_part)
            
            if not val_part.empty:
                val_parts.append(val_part)

        if not group_test.empty:
            group_test["data_split"] = "test"
            test_parts.append(group_test)

    base_columns = df.columns.tolist()
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty_split_frame(base_columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else empty_split_frame(base_columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty_split_frame(base_columns)

    return train_df, val_df, test_df


def fill_feature_gaps(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    filled = df.sort_values(
        ["asset_id", "sequence_id", "time_stamp"],
        kind="mergesort",
    ).copy()

    filled[feature_cols] = (
        filled.groupby(["asset_id", "sequence_id"], group_keys=False)[feature_cols]
        .apply(lambda part: part.ffill().bfill().fillna(0.0))
    )
    filled[feature_cols] = filled[feature_cols].astype(np.float32)
    return filled


def fit_asset_scalers(train_df: pd.DataFrame, feature_cols: list):
    prepared_train = fill_feature_gaps(train_df, feature_cols)
    scalers = {}

    for asset_id, asset_rows in prepared_train.groupby("asset_id", sort=False):
        scaler = MinMaxScaler()
        scaler.fit(asset_rows[feature_cols].to_numpy(dtype=np.float32))
        scalers[asset_id] = scaler

    return scalers


def transform_by_asset(df: pd.DataFrame, feature_cols: list, scalers: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    transformed = fill_feature_gaps(df, feature_cols)

    for asset_id, asset_rows in transformed.groupby("asset_id", sort=False):
        if asset_id not in scalers:
            raise KeyError(f"No scaler fitted for asset_id={asset_id}")

        transformed_values = scalers[asset_id].transform(
            asset_rows[feature_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        transformed.loc[asset_rows.index, feature_cols] = transformed_values

    return transformed


def build_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window_steps: int,
    stride_steps: int,
    group_cols=("asset_id", "sequence_id"),
    split_name: str = "",
):
    feature_count = len(feature_cols)
    meta_columns = [
        "split",
        "asset_id",
        "sequence_id",
        "start_time",
        "end_time",
        "last_label",
        "last_status_type_id",
        "window_steps",
    ]

    if df.empty:
        empty_x = np.empty((0, window_steps, feature_count), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int8)
        return empty_x, empty_y, pd.DataFrame(columns=meta_columns)

    X_list = []
    y_list = []
    meta_rows = []

    for _, group in df.groupby(list(group_cols), sort=False):
        group = group.sort_values("time_stamp", kind="mergesort").reset_index(drop=True)
        if len(group) < window_steps:
            continue

        features = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=np.int8)
        statuses = group["status_type_id"].to_numpy(dtype=int)
        timestamps = group["time_stamp"].astype(str).to_numpy()

        for start in range(0, len(group) - window_steps + 1, stride_steps):
            end = start + window_steps
            window_x = features[start:end]
            last_label = int(labels[end - 1])
            last_status = int(statuses[end - 1])

            if last_label != int(group.iloc[end - 1]["label"]):
                raise ValueError("Window label does not match the last row label.")

            X_list.append(window_x)
            y_list.append(last_label)
            meta_rows.append(
                {
                    "split": split_name,
                    "asset_id": group.iloc[end - 1]["asset_id"],
                    "sequence_id": group.iloc[end - 1]["sequence_id"],
                    "start_time": timestamps[start],
                    "end_time": timestamps[end - 1],
                    "last_label": last_label,
                    "last_status_type_id": last_status,
                    "window_steps": window_steps,
                }
            )

    if not X_list:
        empty_x = np.empty((0, window_steps, feature_count), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int8)
        return empty_x, empty_y, pd.DataFrame(columns=meta_columns)

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int8)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def extract_contiguous_normal_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["run_key"] = pd.Series(dtype=str)
        return out

    expected_gap = pd.Timedelta(minutes=TIME_RESOLUTION_MINUTES)
    run_frames = []

    for (_, _), group in df.groupby(["asset_id", "sequence_id"], sort=False):
        group = group.sort_values("time_stamp", kind="mergesort").copy()
        normal_mask = group["label"].eq(0)
        if not normal_mask.any():
            continue

        prev_normal = normal_mask.shift(fill_value=False)
        gap_ok = group["time_stamp"].diff().eq(expected_gap)
        start_new_run = normal_mask & (~prev_normal | ~gap_ok)
        run_ids = start_new_run.cumsum()

        normal_rows = group.loc[normal_mask].copy()
        normal_rows["run_key"] = [
            f"{asset}_{sequence}_{int(run_id)}"
            for asset, sequence, run_id in zip(
                normal_rows["asset_id"],
                normal_rows["sequence_id"],
                run_ids.loc[normal_mask].to_numpy(),
            )
        ]
        run_frames.append(normal_rows)

    if not run_frames:
        out = df.iloc[0:0].copy()
        out["run_key"] = pd.Series(dtype=str)
        return out

    return pd.concat(run_frames, ignore_index=True)


def compute_class_weights(y: np.ndarray) -> dict:
    counts = np.bincount(y.astype(int), minlength=2)
    present = counts > 0
    if present.sum() < 2:
        return {0: 1.0, 1: 1.0}

    total = counts.sum()
    class_weights = {}
    for class_id, class_count in enumerate(counts):
        if class_count > 0:
            class_weights[class_id] = float(total / (present.sum() * class_count))
    return class_weights


def sample_probe_training_windows(X: np.ndarray, y: np.ndarray):
    if len(X) <= PROBE_MAX_TRAIN_WINDOWS:
        return X, y

    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = []

    unique_classes, class_counts = np.unique(y, return_counts=True)
    for class_id, class_count in zip(unique_classes, class_counts):
        class_indices = np.where(y == class_id)[0]
        target_count = max(1, int(round(PROBE_MAX_TRAIN_WINDOWS * (class_count / len(y)))))
        target_count = min(target_count, len(class_indices))
        chosen = rng.choice(class_indices, size=target_count, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    if len(selected_indices) > PROBE_MAX_TRAIN_WINDOWS:
        selected_indices = rng.choice(selected_indices, size=PROBE_MAX_TRAIN_WINDOWS, replace=False)

    selected_indices = np.sort(selected_indices)
    return X[selected_indices], y[selected_indices]


def build_probe_model(input_shape):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.GRU(32),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_event_f1(meta_df: pd.DataFrame, scores: np.ndarray):
    if meta_df.empty or len(scores) == 0:
        return 0.0, 0.5, 0

    event_df = meta_df[["asset_id", "sequence_id", "last_label"]].copy()
    event_df["score"] = scores
    event_df = (
        event_df.groupby(["asset_id", "sequence_id"], as_index=False)
        .agg(true_label=("last_label", "max"), max_score=("score", "max"))
    )

    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (event_df["max_score"] >= threshold).astype(int)
        current_f1 = float(f1_score(event_df["true_label"], predictions, zero_division=0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(round(threshold, 2))

    return best_f1, best_threshold, int(len(event_df))


def search_best_windows(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    results = []

    print("\nWindow size search")
    print("-" * 60)

    for window_hours in WINDOW_CANDIDATES_HOURS:
        window_steps = window_steps_from_hours(window_hours)
        print(f"Testing {window_hours}h window ({window_steps} steps)")

        X_train, y_train, _ = build_windows(
            train_df,
            feature_cols,
            window_steps,
            STRIDE_STEPS,
            group_cols=("asset_id", "sequence_id"),
            split_name="train",
        )
        X_val, y_val, val_meta = build_windows(
            val_df,
            feature_cols,
            window_steps,
            STRIDE_STEPS,
            group_cols=("asset_id", "sequence_id"),
            split_name="val",
        )

        usable = True
        best_threshold = np.nan
        pr_auc = np.nan
        roc_auc = np.nan
        event_f1 = np.nan
        sampled_train = len(X_train)
        val_events = 0
        note = ""

        if len(X_train) == 0 or len(X_val) == 0:
            usable = False
            note = "not_enough_windows"
        elif len(np.unique(y_train)) < 2:
            usable = False
            note = "train_has_one_class"
        elif len(np.unique(y_val)) < 2:
            usable = False
            note = "val_has_one_class"

        if usable:
            X_probe, y_probe = sample_probe_training_windows(X_train, y_train)
            sampled_train = len(X_probe)

            tf.keras.backend.clear_session()
            set_random_seed(RANDOM_SEED)
            model = build_probe_model((window_steps, len(feature_cols)))
            early_stop = callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
                verbose=0,
            )

            model.fit(
                X_probe,
                y_probe,
                validation_data=(X_val, y_val),
                epochs=PROBE_EPOCHS,
                batch_size=PROBE_BATCH_SIZE,
                class_weight=compute_class_weights(y_probe),
                callbacks=[early_stop],
                verbose=0,
            )

            val_scores = model.predict(X_val, batch_size=PROBE_BATCH_SIZE, verbose=0).reshape(-1)
            pr_auc = safe_pr_auc(y_val, val_scores)
            roc_auc = safe_roc_auc(y_val, val_scores)
            event_f1, best_threshold, val_events = compute_event_f1(val_meta, val_scores)

            del model, X_probe, y_probe, val_scores
            tf.keras.backend.clear_session()
            gc.collect()

        results.append(
            {
                "window_hours": int(window_hours),
                "window_steps": int(window_steps),
                "train_windows": int(len(X_train)),
                "probe_train_windows": int(sampled_train),
                "val_windows": int(len(X_val)),
                "val_events": int(val_events),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "event_f1": event_f1,
                "best_threshold": best_threshold,
                "usable": bool(usable),
                "note": note,
            }
        )

        del X_train, y_train, X_val, y_val, val_meta
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["event_f1", "pr_auc", "roc_auc"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    print(results_df[["window_hours", "event_f1", "pr_auc", "roc_auc", "usable", "note"]])
    return results_df


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_metadata(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def save_scalers(scalers: dict, scaler_dir: Path) -> None:
    scaler_dir.mkdir(parents=True, exist_ok=True)
    for asset_id, scaler in scalers.items():
        joblib.dump(scaler, scaler_dir / f"asset_{asset_id}.pkl")


def export_classifier_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    window_hours: int,
    output_dir: Path,
    scalers: dict,
) -> dict:
    classifier_dir = output_dir / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)

    window_steps = window_steps_from_hours(window_hours)
    X_train, y_train, train_meta = build_windows(
        train_df,
        feature_cols,
        window_steps,
        STRIDE_STEPS,
        group_cols=("asset_id", "sequence_id"),
        split_name="train",
    )
    X_val, y_val, val_meta = build_windows(
        val_df,
        feature_cols,
        window_steps,
        STRIDE_STEPS,
        group_cols=("asset_id", "sequence_id"),
        split_name="val",
    )
    X_test, y_test, test_meta = build_windows(
        test_df,
        feature_cols,
        window_steps,
        STRIDE_STEPS,
        group_cols=("asset_id", "sequence_id"),
        split_name="test",
    )

    if len(y_train) and not np.array_equal(y_train, train_meta["last_label"].to_numpy(dtype=np.int8)):
        raise ValueError("Classifier train labels do not match train metadata labels.")
    if len(y_val) and not np.array_equal(y_val, val_meta["last_label"].to_numpy(dtype=np.int8)):
        raise ValueError("Classifier val labels do not match val metadata labels.")
    if len(y_test) and not np.array_equal(y_test, test_meta["last_label"].to_numpy(dtype=np.int8)):
        raise ValueError("Classifier test labels do not match test metadata labels.")

    np.save(classifier_dir / "X_train.npy", X_train)
    np.save(classifier_dir / "y_train.npy", y_train)
    np.save(classifier_dir / "X_val.npy", X_val)
    np.save(classifier_dir / "y_val.npy", y_val)
    np.save(classifier_dir / "X_test.npy", X_test)
    np.save(classifier_dir / "y_test.npy", y_test)

    train_meta.to_csv(classifier_dir / "train_meta.csv", index=False)
    val_meta.to_csv(classifier_dir / "val_meta.csv", index=False)
    test_meta.to_csv(classifier_dir / "test_meta.csv", index=False)

    save_scalers(scalers, classifier_dir / "scalers")
    save_metadata(
        classifier_dir / "metadata.json",
        {
            "window_hours": int(window_hours),
            "window_steps": int(window_steps),
            "stride_steps": int(STRIDE_STEPS),
            "feature_cols": feature_cols,
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
            "test_shape": list(X_test.shape),
        },
    )

    return {
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
    }


def export_autoencoder_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    window_hours: int,
    output_dir: Path,
    scalers: dict,
) -> dict:
    autoencoder_dir = output_dir / "autoencoder"
    autoencoder_dir.mkdir(parents=True, exist_ok=True)

    window_steps = window_steps_from_hours(window_hours)
    normal_train_df = extract_contiguous_normal_runs(train_df)
    normal_val_df = extract_contiguous_normal_runs(val_df)

    summary = {}

    for asset_id in scalers.keys():
        asset_dir = autoencoder_dir / f"asset_{asset_id}"
        asset_dir.mkdir(parents=True, exist_ok=True)

        asset_train = normal_train_df[normal_train_df["asset_id"] == asset_id].copy()
        asset_val = normal_val_df[normal_val_df["asset_id"] == asset_id].copy()
        asset_test = test_df[test_df["asset_id"] == asset_id].copy()

        X_train, y_train, _ = build_windows(
            asset_train,
            feature_cols,
            window_steps,
            STRIDE_STEPS,
            group_cols=("asset_id", "run_key"),
            split_name="train_normal",
        )
        X_val, y_val, _ = build_windows(
            asset_val,
            feature_cols,
            window_steps,
            STRIDE_STEPS,
            group_cols=("asset_id", "run_key"),
            split_name="val_normal",
        )

        if len(y_train) and int(y_train.max()) != 0:
            raise ValueError(f"Autoencoder train windows for asset {asset_id} contain anomalous labels.")
        if len(y_val) and int(y_val.max()) != 0:
            raise ValueError(f"Autoencoder val windows for asset {asset_id} contain anomalous labels.")

        np.save(asset_dir / "X_train.npy", X_train)
        np.save(asset_dir / "X_val.npy", X_val)
        joblib.dump(scalers[asset_id], asset_dir / "scaler.pkl")

        test_by_sequence_dir = asset_dir / "test_by_sequence"
        test_by_sequence_dir.mkdir(parents=True, exist_ok=True)

        saved_test_sequences = 0
        for sequence_id, sequence_rows in asset_test.groupby("sequence_id", sort=False):
            X_seq, y_seq, meta_seq = build_windows(
                sequence_rows,
                feature_cols,
                window_steps,
                STRIDE_STEPS,
                group_cols=("asset_id", "sequence_id"),
                split_name="test",
            )
            if len(X_seq) == 0:
                continue

            np.savez_compressed(
                test_by_sequence_dir / f"sequence_{sequence_id}.npz",
                X=X_seq,
                y=y_seq,
                end_time=meta_seq["end_time"].to_numpy(),
                last_label=meta_seq["last_label"].to_numpy(dtype=np.int8),
                asset_id=np.asarray([asset_id] * len(X_seq)),
                sequence_id=np.asarray([sequence_id] * len(X_seq)),
            )
            saved_test_sequences += 1

        save_metadata(
            asset_dir / "metadata.json",
            {
                "asset_id": str(asset_id),
                "window_hours": int(window_hours),
                "window_steps": int(window_steps),
                "stride_steps": int(STRIDE_STEPS),
                "feature_cols": feature_cols,
                "train_shape": list(X_train.shape),
                "val_shape": list(X_val.shape),
                "saved_test_sequences": int(saved_test_sequences),
            },
        )

        summary[str(asset_id)] = {
            "train_windows": int(len(X_train)),
            "val_windows": int(len(X_val)),
            "test_sequences": int(saved_test_sequences),
        }

    return summary


def main() -> None:
    set_random_seed(RANDOM_SEED)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_inputs()
    prepared_df = prepare_base_dataframe(df, feature_cols)

    print("=" * 70)
    print("Kaggle Sequence Preparation")
    print("=" * 70)
    print(f"Loaded rows      : {len(prepared_df):,}")
    print(f"Selected features: {len(feature_cols)}")
    print(f"Assets           : {prepared_df['asset_id'].nunique()}")
    print(f"Sequences        : {prepared_df['sequence_id'].nunique()}")

    train_df, val_df, test_df = split_train_val_test_by_sequence(prepared_df)
    scalers = fit_asset_scalers(train_df, feature_cols)

    train_df_sc = transform_by_asset(train_df, feature_cols, scalers)
    val_df_sc = transform_by_asset(val_df, feature_cols, scalers)
    test_df_sc = transform_by_asset(test_df, feature_cols, scalers)

    results_df = search_best_windows(train_df_sc, val_df_sc, feature_cols)
    results_df.to_csv(output_root / "window_search_results.csv", index=False)

    usable_results = results_df[results_df["usable"]].copy()
    if usable_results.empty:
        raise RuntimeError("Window search did not find any usable candidate.")

    best_window_rows = usable_results.head(TOP_K_WINDOWS)
    best_windows = best_window_rows["window_hours"].astype(int).tolist()

    save_metadata(
        output_root / "best_windows.json",
        {
            "best_windows_hours": best_windows,
            "top_k": int(TOP_K_WINDOWS),
            "ranked_results": best_window_rows.to_dict(orient="records"),
        },
    )

    export_summaries = []
    for window_hours in best_windows:
        print(f"\nExporting datasets for {window_hours}h window")
        window_dir = output_root / f"window_{window_hours}h"
        classifier_summary = export_classifier_data(
            train_df_sc,
            val_df_sc,
            test_df_sc,
            feature_cols,
            window_hours,
            window_dir,
            scalers,
        )
        autoencoder_summary = export_autoencoder_data(
            train_df_sc,
            val_df_sc,
            test_df_sc,
            feature_cols,
            window_hours,
            window_dir,
            scalers,
        )

        export_summaries.append(
            {
                "window_hours": int(window_hours),
                "classifier": classifier_summary,
                "autoencoder": autoencoder_summary,
            }
        )

    save_metadata(output_root / "export_summary.json", {"exports": export_summaries})

    print("\n" + "=" * 70)
    print("Preparation complete")
    print("=" * 70)
    print(f"Loaded rows      : {len(prepared_df):,}")
    print(f"Number of features: {len(feature_cols)}")
    print("Ranked windows:")
    print(results_df[["window_hours", "event_f1", "pr_auc", "roc_auc", "usable"]].to_string(index=False))
    print(f"Chosen top {len(best_windows)} windows: {best_windows}")
    print(f"Saved outputs to : {output_root}")


if __name__ == "__main__":
    main()
