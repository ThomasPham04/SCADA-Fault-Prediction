# SCADA Fault Prediction

Early fault detection on wind turbine SCADA data using supervised sequence classifiers. Built on the EDP/CARE Wind Farm A dataset (5 turbines, ~86 features, 10-minute resolution).

Each model takes a sliding input window of SCADA readings and predicts whether a fault will occur within a configurable future horizon:

```
X[t-W+1 : t]  →  binary fault label
```

Supported models: `lstm`, `gru`, `cnn_lstm`, `cnn_gru`

---

## Setup

Requires Python 3.10+ and TensorFlow.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Experiments

All commands go through `src/main.py`.

### 1. Build sequences from raw CARE event CSVs

```powershell
python src/main.py prepare-care --window-hours 24
```

This builds a combined CSV from raw event files and exports classifier-ready `.npy` sequences. Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--farm-dir DIR` | config | Wind farm root with `event_info.csv` |
| `--window-hours H [H ...]` | config | Input window size(s) in hours |
| `--prediction-horizon-hours H` | config | How far ahead to predict |
| `--label-mode future_horizon\|input_window` | `future_horizon` | Window label contract |
| `--validation-source train_tail\|prediction` | `train_tail` | Val split strategy |

### 2. Build sequences from an existing combined CSV

```powershell
python src/main.py prepare `
    --csv "Dataset/processed/combined.csv" `
    --feature-file "results/final_features.csv" `
    --window-hours 24
```

### 3. Train classifiers

```powershell
# All models
python src/main.py train-sequences --windows 24

# Specific model(s)
python src/main.py train-sequences --windows 24 --model cnn_gru
python src/main.py train-sequences --windows 24 --model lstm gru cnn_lstm cnn_gru
```

---

## Typical End-to-End Run

```powershell
python src/main.py prepare-care --window-hours 24
python src/main.py train-sequences --windows 24 --model cnn_gru
```

---

## Output

Sequence exports land in `Dataset/processed/sequence_exports/window_24h/classifier/`.

Training results land in `results/sequence_training_results/window_24h/classifier/`:

```
model_comparison.csv          ← cross-model summary
<model_name>/
  metrics.json                ← accuracy, precision, recall, F1, ROC-AUC, PR-AUC
  model.keras
  history.csv / loss_history.png
  threshold_sweep_val.csv / threshold_sweep_val.png
  confusion_matrix_test.png
  pr_curve_test.png / roc_curve_test.png
  test_predictions.csv / test_event_predictions.csv
```

Start evaluation from `results/sequence_training_results/run_summary.json` and drill into per-model `metrics.json` files.
