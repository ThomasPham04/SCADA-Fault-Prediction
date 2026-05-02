"""
AssetNormalizer — data_pipeline.preprocessing.normalizer
MinMaxScaler normalization fitted exclusively on each asset's training data.
"""

import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import STRIDE
from data_pipeline.loaders.sequence_maker import SequenceMaker


class AssetNormalizer:
    """
    Normalizes SCADA sequence data using a MinMaxScaler fitted only on
    training data — ensuring the model never sees anomalous patterns during
    normalization.

    Args:
        output_dir: Root directory where fitted scalers are saved.
        seq_len: Sequence window length used when sequencing test events.
        scaler_type: Type of scaling ('standard' or 'minmax').
    """

    def __init__(self, output_dir: str, seq_len: int, scaler_type: str = "standard") -> None:
        self.output_dir = output_dir
        self.seq_len    = seq_len
        self.scaler_type = scaler_type.lower()
        if self.scaler_type not in ("standard", "minmax"):
            raise ValueError("scaler_type must be 'standard' or 'minmax'")

    def normalize_asset(
        self,
        asset_id: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        test_data_dict: dict,
    ) -> tuple:
        """
        Normalize data for a single asset using the specified scaler fit only on that
        asset's training sequences.

        Each asset gets its own scaler (scaler_asset_{id}.pkl) — ensuring the
        normal-behaviour baseline is specific to that turbine.

        Args:
            asset_id: Turbine/asset ID (used for scaler filename).
            X_train: Train sequences (n_seq, window, features).
            X_val: Val sequences (n_seq, window, features).
            y_train: Train targets (n_seq, features).
            y_val: Val targets (n_seq, features).
            test_data_dict: {event_id: {'features': np.ndarray, ...}} from process_asset_test().

        Returns:
            Tuple of (X_train_sc, X_val_sc, y_train_sc, y_val_sc, test_data_scaled_dict).
        """
        n_train, _seq_len, n_features = X_train.shape
        n_val = X_val.shape[0]

        if self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
            
        X_train_sc = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, _seq_len, n_features)
        X_val_sc   = scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, _seq_len, n_features)
        y_train_sc = scaler.transform(y_train)
        y_val_sc   = scaler.transform(y_val)

        seq_maker = SequenceMaker(window_size=self.seq_len, stride=STRIDE)
        test_data_scaled = {}
        for event_id, data in test_data_dict.items():
            X_test, y_test = seq_maker.create_sequences(data["features"])
            if len(X_test) == 0:
                continue
            X_test_sc = scaler.transform(X_test.reshape(-1, n_features)).reshape(-1, self.seq_len, n_features)
            y_test_sc = scaler.transform(y_test)

            seq_ts = []
            if "time_stamps" in data:
                ts = data["time_stamps"]
                stride = seq_maker.stride
                seq_ts = [ts[i * stride + self.seq_len] for i in range(len(X_test))]

            # Label each window by its last timestep: window i is a fault window
            # if and only if the last row it covers (i*stride + seq_len - 1) is
            # labeled as a fault row. This mirrors the causal interpretation —
            # the window predicts the state at the moment it ends.
            n_seqs = len(X_test)
            stride = seq_maker.stride
            if "row_labels" in data:
                row_labels = np.asarray(data["row_labels"])
                last_row_indices = np.arange(n_seqs) * stride + self.seq_len - 1
                seq_labels = row_labels[last_row_indices].astype(np.int8)
            else:
                seq_labels = np.zeros(n_seqs, dtype=np.int8)

            test_data_scaled[event_id] = {
                "X": X_test_sc, "y": y_test_sc,
                "seq_labels":  seq_labels,
                "time_stamps": seq_ts,
                "label":      data["label"],
                "event_start": data["event_start"],
                "event_end":   data["event_end"],
                "asset_id":    data.get("asset_id", asset_id),
            }

        os.makedirs(self.output_dir, exist_ok=True)
        scaler_path = os.path.join(self.output_dir, f"scaler_asset_{asset_id}.pkl")
        joblib.dump(scaler, scaler_path)
        # Also save as scaler.pkl so the training loader can find it
        joblib.dump(scaler, os.path.join(self.output_dir, "scaler.pkl"))
        print(f"  Asset {asset_id}: scaler saved → {scaler_path}")

        return X_train_sc, X_val_sc, y_train_sc, y_val_sc, test_data_scaled


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases
# ---------------------------------------------------------------------------

def normalize_data(
    X_train, X_val, y_train, y_val,
    test_data_dict: dict,
    output_dir: str,
    scaler_type: str = "standard",
) -> tuple:
    """Legacy alias — wraps AssetNormalizer.normalize_data()."""
    norm = AssetNormalizer(output_dir=output_dir, seq_len=X_train.shape[1], scaler_type=scaler_type)
    return norm.normalize_asset(0, X_train, X_val, y_train, y_val, test_data_dict)


def normalize_asset(
    asset_id: int,
    X_train, X_val, y_train, y_val,
    test_data_dict: dict,
    output_dir: str,
    seq_len: int,
    scaler_type: str = "standard",
) -> tuple:
    """Legacy alias — wraps AssetNormalizer.normalize_asset()."""
    norm = AssetNormalizer(output_dir=output_dir, seq_len=seq_len, scaler_type=scaler_type)
    return norm.normalize_asset(asset_id, X_train, X_val, y_train, y_val, test_data_dict)
