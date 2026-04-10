"""
train_autoencoder_global.py
=============================================================
Trains a SINGLE "Global" AutoEncoder model using the normal 
training data from ALL 22 events concatenated together.
Saves 1 Model, 1 Scaler, and 1 Threshold Array.
=============================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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

MODELS_OUT_DIR = os.path.join(MODELS_DIR, "autoencoder_global")

def set_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    
    csvs = sorted(glob.glob(os.path.join(WIND_FARM_A_DATASETS, "*.csv")))
    splitter = DataSplitter(datasets_dir="", event_info=pd.DataFrame())
    fe = FeatureEngineer()

    print("="*60)
    print("PHASE 1 (GLOBAL): GOM DATA NORMAL CỦA TẤT CẢ 22 EVENTS")
    print("="*60)

    # 1. Collect all Train Data
    all_train_arrays = []
    global_feat_cols = None
    
    for csv in csvs:
        eid = os.path.splitext(os.path.basename(csv))[0]
        df = pd.read_csv(csv, sep=None, engine="python")
        
        train_df, _ = splitter.split_train_val_autoencoder(df)
        if len(train_df) == 0:
            continue
            
        train_df = fe.engineer_angle_features(train_df)
        train_df = fe.drop_counter_features(train_df)
        feat_cols = fe.get_feature_columns(train_df)
        
        if global_feat_cols is None:
            global_feat_cols = feat_cols
            
        X_train_event = fe.preprocess_features(train_df, global_feat_cols).astype("float32")
        all_train_arrays.append(X_train_event)
        print(f"  + Load {len(X_train_event)} rows từ Event {eid}")

    # Gom vào 1 mảng khổng lồ
    X_train_full = np.vstack(all_train_arrays)
    print(f"\n=> TỔNG CỘNG DATA NORMAL GLOBAL: {len(X_train_full)} rows x {X_train_full.shape[1]} features")

    # 2. Split (85/15)
    # Vì là mảng khổng lồ nối tiếp, ta xáo trộn index để Validation dàn trải đều các event
    # Cải tiến: Shuffle the global array
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(X_train_full))
    split_idx = int(0.85 * len(X_train_full))
    
    X_train = X_train_full[indices[:split_idx]]
    X_val   = X_train_full[indices[split_idx:]]
    
    # 3. Scale chung (1 Global Scaler)
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    
    scaler_path = os.path.join(MODELS_OUT_DIR, "ae_global_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(global_feat_cols, os.path.join(MODELS_OUT_DIR, "ae_global_featcols.pkl"))

    # 4. Train MỘT Model Duy Nhất
    print("\n=> BẮT ĐẦU TRAIN MỘT MODEL TRÊN TẤT CẢ DỮ LIỆU...")
    set_seeds()
    model_path = os.path.join(MODELS_OUT_DIR, "ae_global.keras")
    model, _, _ = build_autodecoder_model(X_train_sc.shape[1], 16)
    
    model.fit(
        X_train_sc, X_train_sc,
        validation_data=(X_val_sc, X_val_sc),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=get_lstm_callbacks(model_path), verbose=1
    )
    
    # 5. Lưu mảng MSE Train để tìm Threshold sau này
    best_model = tf.keras.models.load_model(model_path, compile=False)
    mse_train = np.mean((X_train_sc - best_model.predict(X_train_sc, verbose=0, batch_size=256))**2, axis=1)
    np.save(os.path.join(MODELS_OUT_DIR, "ae_global_msetrain.npy"), mse_train)
    
    print(f"\n✅ HOÀN TẤT! Mô hình Global đã được lưu tại: {MODELS_OUT_DIR}")

if __name__ == "__main__":
    main()
