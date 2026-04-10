"""
tune_autoencoder_global.py
=============================================================
Evaluates the SINGLE "Global" model against all 22 test events.
Generates the overall Confusion Matrix under a global threshold.
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
import tensorflow as tf

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (MODELS_DIR, RESULTS_DIR, WIND_FARM_A_DIR, WIND_FARM_A_DATASETS)
from data_pipeline.preprocessing.feature_engineering import FeatureEngineer

MODELS_OUT_DIR = os.path.join(MODELS_DIR, "autoencoder_global")
RESULTS_OUT    = os.path.join(RESULTS_DIR, "autoencoder_global")

def plot_confusion_matrix(tp, fp, fn, tn, acc, prec, rec, f1, threshold_pct, anomaly_ratio, out_path):
    cm = np.array([[tp, fn], [fp, tn]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor("#0f1117")

    ax = axes[0]
    ax.set_facecolor("#0f1117")
    im = ax.imshow(cm, cmap="YlOrRd", vmin=0, vmax=max(cm.max(), 1))

    labels = [["True Positive\n(TP)", "False Negative\n(FN)"],
              ["False Positive\n(FP)", "True Negative\n(TN)"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            col = "white" if val > cm.max() * 0.5 else "#1a1a1a"
            ax.text(j, i, f"{labels[i][j]}\n{val}", ha="center", va="center", color=col, fontsize=13, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nAnomaly", "Predicted\nNormal"], color="white", fontsize=11)
    ax.set_yticklabels(["Actual\nAnomaly", "Actual\nNormal"], color="white", fontsize=11)
    ax.set_title(f"GLOBAL AutoEncoder CM (Threshold: {threshold_pct}th Pct | Anomaly Ratio: {anomaly_ratio})", color="white", fontsize=12)
    ax.tick_params(colors="white")

    ax2 = axes[1]
    ax2.set_facecolor("#0f1117")
    bars = ax2.barh(["Accuracy", "Precision", "Recall", "F1"], [acc, prec, rec, f1], color=["#4CAF50", "#2196F3", "#FF9800", "#E91E63"])
    ax2.set_xlim(0, 1.15)
    ax2.set_title("Metrics", color="white")
    ax2.tick_params(colors="white")
    for bar, val in zip(bars, [acc, prec, rec, f1]):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=90.0, help="Percentile threshold on train MSE")
    parser.add_argument("--anomaly_ratio", type=float, default=0.15, help="Tolerance ratio for entire event")
    args = parser.parse_args()

    os.makedirs(RESULTS_OUT, exist_ok=True)
    
    event_info = pd.read_csv(os.path.join(WIND_FARM_A_DIR, "event_info.csv"), sep=";")
    label_map = dict(zip(event_info["event_id"].astype(str), event_info["event_label"]))

    csvs = sorted(glob.glob(os.path.join(WIND_FARM_A_DATASETS, "*.csv")))
    fe = FeatureEngineer()

    print("="*60)
    print(f"PHASE 2 (GLOBAL): TUNING & EVAL (Threshold = {args.threshold}th Pct)")
    print("="*60)

    # 1. Load GLOBAL artifacts
    model_path = os.path.join(MODELS_OUT_DIR, "ae_global.keras")
    scaler_path = os.path.join(MODELS_OUT_DIR, "ae_global_scaler.pkl")
    mse_train_path = os.path.join(MODELS_OUT_DIR, "ae_global_msetrain.npy")
    feat_cols_path = os.path.join(MODELS_OUT_DIR, "ae_global_featcols.pkl")
    
    if not os.path.exists(model_path):
        print("SKIP: Missing GLOBAL trained assets. Run train_autoencoder_global.py first.")
        sys.exit(1)
        
    scaler = joblib.load(scaler_path)
    mse_train = np.load(mse_train_path)
    feat_cols = joblib.load(feat_cols_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # 2. Get cutting line based on GLOBAL MSE Array
    threshold_val = float(np.percentile(mse_train, args.threshold))
    
    TP = FP = TN = FN = 0

    for csv in csvs:
        eid = os.path.splitext(os.path.basename(csv))[0]
        true_label = label_map.get(eid, "unknown")
        
        # 3. Predict on unseen eval data
        df = pd.read_csv(csv, sep=None, engine="python")
        eval_df = df[df["train_test"] == "prediction"].copy()
        if len(eval_df) == 0:
            continue
            
        eval_df = fe.engineer_angle_features(eval_df)
        eval_df = fe.drop_counter_features(eval_df)
        X_eval = scaler.transform(fe.preprocess_features(eval_df, feat_cols).astype("float32"))
        
        mse_eval = np.mean((X_eval - model.predict(X_eval, verbose=0, batch_size=256))**2, axis=1)
        anomaly_ratio = float(np.mean(mse_eval > threshold_val))
        pred_label = "anomaly" if anomaly_ratio > args.anomaly_ratio else "normal"
        
        if true_label == "anomaly" and pred_label == "anomaly": outcome="TP"; TP+=1
        elif true_label == "anomaly" and pred_label == "normal": outcome="FN"; FN+=1
        elif true_label == "normal" and pred_label == "anomaly": outcome="FP"; FP+=1
        else: outcome="TN"; TN+=1
            
        print(f"[{eid:<2}] Actual: {true_label:7s} | Pred: {pred_label:7s} → {outcome} | Outlier: {anomaly_ratio:.1%}")
        
    tot = TP+FP+TN+FN
    acc = (TP+TN)/tot if tot > 0 else 0
    prec = TP/(TP+FP) if TP+FP > 0 else 0
    rec = TP/(TP+FN) if TP+FN > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
    
    print("\nGLOBAL SUMMARY:")
    print(f"TP={TP} FN={FN} FP={FP} TN={TN}")
    print(f"Acc={acc:.1%} Prec={prec:.1%} Rec={rec:.1%} F1={f1:.1%}")
    
    cm_path = os.path.join(RESULTS_OUT, f"tuning_matrix_GLOBAL_th{args.threshold}_ar{args.anomaly_ratio}.png")
    plot_confusion_matrix(TP, FP, FN, TN, acc, prec, rec, f1, args.threshold, args.anomaly_ratio, cm_path)
    print(f"Saved: {cm_path}")

if __name__ == "__main__":
    main()
