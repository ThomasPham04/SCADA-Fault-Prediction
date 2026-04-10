"""
01_train_autoencoder_batch.py
=============================================================
Phase 1: Trains 22 AutoEncoder models (one per event) on 
perfectly clean 'normal' data using Temporal Split (85/15).
Saves the models, scalers, and training MSE arrays to disk.
=============================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    MODELS_DIR, WIND_FARM_A_DIR, WIND_FARM_A_DATASETS,
    BATCH_SIZE, EPOCHS, RANDOM_SEED
)
from data_pipeline.preprocessing.splitter import DataSplitter, temporal_split_train_val
from data_pipeline.preprocessing.feature_engineering import FeatureEngineer
from models.architectures.autodecoder import build_autodecoder_model
from training.callbacks.early_stopping import get_lstm_callbacks
from sklearn.preprocessing import MinMaxScaler

MODELS_OUT_DIR = os.path.join(MODELS_DIR, "autoencoder_batch")

def set_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    
    event_info = pd.read_csv(os.path.join(WIND_FARM_A_DIR, "event_info.csv"), sep=";")
    label_map = dict(zip(event_info["event_id"].astype(str), event_info["event_label"]))

    csvs = sorted(glob.glob(os.path.join(WIND_FARM_A_DATASETS, "*.csv")))
    
    print("="*60)
    print("PHASE 1: TRAINING AUTOENCODER (22 EVENTS)")
    print("="*60)

    splitter = DataSplitter(datasets_dir="", event_info=pd.DataFrame())
    fe = FeatureEngineer()

    for csv in csvs:
        eid = os.path.splitext(os.path.basename(csv))[0]
        true_label = label_map.get(eid, "unknown")
        
        df = pd.read_csv(csv, sep=None, engine="python")
        
        # 1. Train Prep (Normal Only)
        train_df, _ = splitter.split_train_val_autoencoder(df)
        if len(train_df) == 0:
            print(f"[{eid}] SKIP: No normal training data.")
            continue
            
        train_df = fe.engineer_angle_features(train_df)
        train_df = fe.drop_counter_features(train_df)
        feat_cols = fe.get_feature_columns(train_df)
        X_train_full = fe.preprocess_features(train_df, feat_cols).astype("float32")
        
        # 2. Validation Split (15%) for True Early Stopping
        X_train, X_val = temporal_split_train_val(X_train_full, val_ratio=0.15)
        
        # 3. Scale & Save Scaler
        scaler = MinMaxScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)
        
        scaler_path = os.path.join(MODELS_OUT_DIR, f"ae_scaler_{eid}.pkl")
        joblib.dump(scaler, scaler_path)
        
        # 4. Train Model
        set_seeds()
        model_path = os.path.join(MODELS_OUT_DIR, f"ae_{eid}.keras")
        model, _, _ = build_autodecoder_model(X_train_sc.shape[1], 16)
        
        print(f"[{eid:<2}] Training '{true_label}' data -> {len(X_train_sc)} rows...")
        model.fit(
            X_train_sc, X_train_sc,
            validation_data=(X_val_sc, X_val_sc),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=get_lstm_callbacks(model_path), verbose=0
        )
        
        # 5. Save Train MSE Array (for ultra-fast threshold tuning later)
        # Load best model to compute mse
        best_model = tf.keras.models.load_model(model_path, compile=False)
        mse_train = np.mean((X_train_sc - best_model.predict(X_train_sc, verbose=0))**2, axis=1)
        np.save(os.path.join(MODELS_OUT_DIR, f"ae_msetrain_{eid}.npy"), mse_train)
        
        # Save feature definition locally so the tuner knows the exact columns
        joblib.dump(feat_cols, os.path.join(MODELS_OUT_DIR, f"ae_featcols_{eid}.pkl"))
        
    print("\nPhase 1 Complete! Scalers, Models, and MSE Arrays saved to:", MODELS_OUT_DIR)

if __name__ == "__main__":
    main()
