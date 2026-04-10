"""
plot_mse_timelines.py
=============================================================
Phase 3: Visualization of MSE Distribution over Time.
Generates timeline plots for the test splits of all 22 events.
X-axis: Date (time_stamp)
Y-axis: Reconstruction Error (MSE)
Points above the threshold are marked entirely in RED (Anomaly),
points below the threshold are BLUE (Normal).
=============================================================
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    MODELS_DIR, RESULTS_DIR, WIND_FARM_A_DIR, WIND_FARM_A_DATASETS
)
from data_pipeline.preprocessing.feature_engineering import FeatureEngineer

MODELS_OUT_DIR = os.path.join(MODELS_DIR, "autoencoder_batch")
RESULTS_OUT    = os.path.join(RESULTS_DIR, "autoencoder_batch", "mse_timelines")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=90.0, help="Percentile threshold on train MSE")
    args = parser.parse_args()

    os.makedirs(RESULTS_OUT, exist_ok=True)
    
    event_info = pd.read_csv(os.path.join(WIND_FARM_A_DIR, "event_info.csv"), sep=";")
    label_map = dict(zip(event_info["event_id"].astype(str), event_info["event_label"]))

    csvs = sorted(glob.glob(os.path.join(WIND_FARM_A_DATASETS, "*.csv")))
    fe = FeatureEngineer()

    print("="*70)
    print(f"PHASE 3: PLOTTING MSE TIMELINES (Threshold = {args.threshold}th Pct)")
    print("="*70)

    for csv in csvs:
        eid = os.path.splitext(os.path.basename(csv))[0]
        true_label = label_map.get(eid, "unknown")
        
        # Paths
        model_path = os.path.join(MODELS_OUT_DIR, f"ae_{eid}.keras")
        scaler_path = os.path.join(MODELS_OUT_DIR, f"ae_scaler_{eid}.pkl")
        mse_train_path = os.path.join(MODELS_OUT_DIR, f"ae_msetrain_{eid}.npy")
        feat_cols_path = os.path.join(MODELS_OUT_DIR, f"ae_featcols_{eid}.pkl")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, mse_train_path, feat_cols_path]):
            print(f"[{eid}] SKIP: Missing trained assets.")
            continue
            
        # 1. Load artifacts
        scaler = joblib.load(scaler_path)
        mse_train = np.load(mse_train_path)
        feat_cols = joblib.load(feat_cols_path)
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # 2. Get exact threshold
        threshold_val = float(np.percentile(mse_train, args.threshold))
        
        # 3. Load & Predict
        df = pd.read_csv(csv, sep=None, engine="python")
        eval_df = df[df["train_test"] == "prediction"].copy()
        
        if len(eval_df) == 0:
            print(f"[{eid}] SKIP: No evaluation data.")
            continue
            
        # Convert time_stamp for X axis
        time_stamp = pd.to_datetime(eval_df["time_stamp"])
            
        # Preprocess
        eval_processed = fe.engineer_angle_features(eval_df)
        eval_processed = fe.drop_counter_features(eval_processed)
        X_eval = scaler.transform(fe.preprocess_features(eval_processed, feat_cols).astype("float32"))
        
        # Generate MSE
        mse_eval = np.mean((X_eval - model.predict(X_eval, verbose=0))**2, axis=1)
        
        # Identify outliers based on threshold
        outliers = mse_eval > threshold_val
        
        # ---------------------------------------------------------
        # 4. Draw Plot
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        
        # Scatter blue dots where MSE <= threshold (normals)
        normals = ~outliers
        if np.any(normals):
            ax.scatter(time_stamp[normals], mse_eval[normals], color="#2196F3", s=10, alpha=0.7, label="Normal MSE")
        
        # Scatter red dots where MSE > threshold
        if np.any(outliers):
            ax.scatter(time_stamp[outliers], mse_eval[outliers], color="#E91E63", s=15, zorder=5, label="Anomaly MSE (Outlier)")
            
        # Draw Threshold line
        ax.axhline(threshold_val, color="#FF9800", linestyle="--", linewidth=2, label=f"Threshold ({args.threshold}th pct)")
        
        # Formatting
        ax.set_title(f"Event [{eid}]  |  True Label: {true_label.upper()}", color="white", fontsize=14, pad=15)
        ax.set_ylabel("Reconstruction Error (MSE)", color="white", fontsize=11)
        ax.set_xlabel("Time (Date)", color="white", fontsize=11)
        
        # Handle X-axis date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white")
        plt.setp(ax.get_yticklabels(), color="white")
        
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
            
        ax.legend(facecolor="#1e1e2e", edgecolor="#444", labelcolor="white", loc="upper left")
        ax.grid(True, color="#333", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        
        # 5. Save
        out_file = os.path.join(RESULTS_OUT, f"mse_plot_{eid}_th{args.threshold}.png")
        fig.savefig(out_file, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        
        print(f"[{eid:<2}] Plotted ({true_label}) -> {out_file}")

    print(f"\nAll plots saved to: {RESULTS_OUT}")

if __name__ == "__main__":
    main()
