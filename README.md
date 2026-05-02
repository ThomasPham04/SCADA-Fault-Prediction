# SCADA Fault Prediction

This repository contains a Python workflow for early fault prediction on wind
turbine SCADA time-series data. The pipeline covers everything from raw CARE
event files through sequence exports to trained classifiers and autoencoders.

1. Build a combined CSV from raw CARE per-event files (or supply your own).
2. Prepare fixed-length sequence exports from the combined CSV.
3. Train supervised sequence classifiers and per-asset sequence autoencoders.
4. Save metrics, predictions, plots, and comparison summaries for evaluation.

This README is a project guide, not a thesis outline. Report writing notes and
long-form research discussion belong under `reports/`.

## Project Scope

The modelling task is binary anomaly detection on SCADA time-series windows:

```text
X[t-W+1 : t]  →  fault_label(t)
```

A window is labelled positive if and only if its **last timestep** falls inside
a known fault interval (as defined by `event_start_id` → `event_end_id` in
`event_info.csv`). This mirrors the causal interpretation — the window predicts
the machine state at the moment it ends.

The default version of the project uses:

- SCADA sampling interval: 10 minutes
- Input window: 24 hours (144 time steps)
- Stride: 6 steps (1 hour)
- Window boundary: do not cross `asset_id + sequence_id`
- Main data format: one combined CSV with event boundaries preserved

The raw CARE to Compare data contains event-level SCADA files for Wind Farm A,
Wind Farm B, and Wind Farm C. Wind Farm A is the initial experimental scope.

## Repository Layout

```text
scada-fault-prediction/
├── configs/
│   └── problem_v1.yaml
├── Dataset/
│   ├── raw/
│   └── processed/
├── experiments/
├── notebooks/
├── reports/
├── src/
│   ├── data_pipeline/
│   │   ├── loaders/
│   │   │   ├── event_loader.py
│   │   │   └── sequence_maker.py
│   │   └── preprocessing/
│   │       ├── build_combined_csv.py      ← CARE events → combined CSV
│   │       ├── combined_sequence_pipeline.py
│   │       ├── feature_engineering.py
│   │       ├── ground_truth.py
│   │       ├── normalizer.py
│   │       └── splitter.py
│   ├── training/
│   │   ├── sequence_model_trainer.py      ← orchestration
│   │   ├── sequence_experiments.py        ← experiment runners
│   │   ├── sequence_io.py                 ← data loading helpers
│   │   ├── sequence_metrics.py            ← evaluation & threshold sweep
│   │   ├── sequence_models.py             ← Keras model builders & callbacks
│   │   ├── sequence_plots.py              ← plots & prediction frames
│   │   ├── sequence_utils.py              ← seeding, JSON I/O, TF cleanup
│   │   └── sequence_model_utils.py        ← backward-compat re-export shim
│   ├── config.py
│   ├── main.py
│   └── problem_config.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

Key source files:

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point (`prepare-care`, `prepare`, `train-sequences`) |
| `src/data_pipeline/preprocessing/build_combined_csv.py` | Convert raw CARE event CSVs to a single combined CSV |
| `src/data_pipeline/preprocessing/combined_sequence_pipeline.py` | Prepare classifier and autoencoder sequence exports |
| `src/training/sequence_model_trainer.py` | Orchestrate training across windows and model types |
| `src/training/sequence_experiments.py` | `run_classifier_experiment`, `run_autoencoder_experiment`, `run_global_autoencoder_experiment` |
| `src/training/sequence_io.py` | Load classifier bundles, asset/global autoencoder exports, test NPZ files |
| `src/training/sequence_metrics.py` | Threshold sweep, evaluation metrics, comparison frame builders |
| `src/training/sequence_models.py` | Keras model builders and training callbacks |
| `src/training/sequence_plots.py` | Loss history, PR/ROC curves, confusion matrix, prediction frames |
| `src/training/sequence_utils.py` | Random seeding, TF cleanup, JSON I/O, model persistence |

## Setup

Create a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project requires Python 3.10 or newer. TensorFlow is required for training.

## Input Data

### CARE to Compare dataset (recommended)

Use `prepare-care` (see below). It reads the raw per-event CSVs and
`event_info.csv` directly from the wind farm directory and handles everything.

### Pre-built combined CSV

The `prepare` command expects a CSV with these required columns:

- `time_stamp`
- `asset_id`
- `train_test`
- `sequence_id`
- `label` or `status_type_id`

Feature columns are selected automatically from numeric non-metadata columns,
or can be supplied through a feature list CSV (use the `final_feature` column
when present, otherwise the first column):

```text
results/results/final_features.csv
```

## CLI Reference

### `prepare-care` — raw CARE files → combined CSV → sequence exports

Reads raw CARE per-event CSVs and `event_info.csv` for a wind farm, builds a
combined CSV, then prepares all sequence exports in one step.

```powershell
python src/main.py prepare-care
```

Use a custom farm directory or output path:

```powershell
python src/main.py prepare-care `
    --farm-dir "Dataset\raw\Wind Farm A" `
    --combined-csv-output "Dataset\processed\Wind Farm A\combined.csv" `
    --feature-file "results\final_features.csv" `
    --window-hours 24
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--farm-dir DIR` | `WIND_FARM_A_DIR` from config | Wind farm root containing `event_info.csv` |
| `--datasets-dir DIR` | `<farm-dir>/datasets` | Sub-directory with per-event CSVs |
| `--combined-csv-output PATH` | `Dataset/processed/<farm-name>/combined.csv` | Where to save the combined CSV |
| `--feature-file PATH` | auto-detect | Feature list CSV |
| `--sequence-output-dir DIR` | `Dataset/processed/sequence_exports` | Root for sequence exports |
| `--window-hours H [H ...]` | window search | Export these window sizes directly |
| `--window-candidates-hours H [H ...]` | 12 24 48 | Candidate sizes for window search |
| `--top-k-windows N` | 1 | Export the best N windows after search |
| `--skip-window-search` | off | Skip search, export first candidate only |
| `--combined-scaler minmax\|standard` | `minmax` | Scaler for exports |
| `--validation-source train_tail\|prediction` | `train_tail` | Validation split strategy |
| `--prediction-val-ratio FLOAT` | 0.5 | Fraction of each prediction segment used for validation |
| `--seed INT` | 42 | Random seed |

---

### `prepare` — pre-built combined CSV → sequence exports

Use this when you already have a combined CSV.

```powershell
python src/main.py prepare --csv "Dataset\processed\combined.csv" `
    --feature-file "results\final_features.csv" `
    --window-hours 24
```

With a prediction-source validation split (recommended for threshold selection):

```powershell
python src/main.py prepare `
    --csv "Dataset\processed\combined.csv" `
    --feature-file "results\final_features.csv" `
    --window-hours 24 `
    --validation-source prediction `
    --prediction-val-ratio 0.5 `
    --sequence-output-dir "Dataset\processed\sequence_exports_prediction_val"
```

Accepts the same `--window-hours`, `--combined-scaler`, `--validation-source`,
and related flags as `prepare-care`. `--csv` is required.

Export output layout:

```text
Dataset/processed/sequence_exports/
├── best_windows.json
├── export_summary.json
├── window_search_results.csv
└── window_24h/
    ├── classifier/
    │   ├── X_train.npy / y_train.npy
    │   ├── X_val.npy   / y_val.npy
    │   ├── X_test.npy  / y_test.npy
    │   ├── train_meta.csv / val_meta.csv / test_meta.csv
    │   └── metadata.json
    └── autoencoder/
        └── asset_<id>/
            ├── X_train.npy
            ├── X_val.npy
            ├── scaler.pkl
            ├── metadata.json
            └── test_by_sequence/
                └── sequence_<event_id>.npz
```

---

### `train-sequences` — sequence exports → trained models

```powershell
python src/main.py train-sequences --windows 24
```

Train only specific models (auto-routes to classifier or autoencoder branch):

```powershell
python src/main.py train-sequences --windows 24 --model lstm gru
python src/main.py train-sequences --windows 24 --model lstm_ae gru_ae
python src/main.py train-sequences --windows 24 --model lstm gru_ae
```

Skip one branch entirely:

```powershell
python src/main.py train-sequences --windows 24 --skip-autoencoders
python src/main.py train-sequences --windows 24 --skip-classifiers
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--exports-dir DIR` | `Dataset/processed/sequence_exports` | Root with `window_<H>h/` sub-folders |
| `--results-dir DIR` | `results/sequence_training_results` | Output root |
| `--windows H [H ...]` | `24` | Window sizes to train |
| `--model MODEL [...]` | all | Train only these models; overrides branch flags |
| `--classifier-models ...` | lstm gru cnn_lstm cnn_gru | Supervised classifiers |
| `--classifier-epochs N` | 25 | Max epochs |
| `--classifier-batch-size N` | 256 | Batch size |
| `--classifier-learning-rate FLOAT` | 0.001 | Adam learning rate |
| `--classifier-dropout FLOAT` | arch default | Override all classifier dropout layers |
| `--classifier-l2 FLOAT` | 0.0 | L2 regularization on Conv/RNN/Dense kernels |
| `--autoencoder-models ...` | lstm_ae gru_ae | Autoencoder architectures |
| `--autoencoder-epochs N` | 30 | Max epochs |
| `--autoencoder-batch-size N` | 128 | Batch size |
| `--autoencoder-scope per_asset\|global\|both` | `per_asset` | Training scope |
| `--assets ID [ID ...]` | all | Restrict autoencoder training to specific assets |
| `--overwrite` | off | Retrain even when `metrics.json` already exists |
| `--no-save-predictions` | off | Skip verbose prediction CSV outputs |
| `--seed INT` | 42 | Random seed |

Training output layout:

```text
results/sequence_training_results/
├── run_summary.json
└── window_24h/
    ├── window_summary.json
    ├── classifier/
    │   ├── model_comparison.csv
    │   └── <model_name>/
    │       ├── metrics.json
    │       ├── model.keras
    │       ├── history.csv / loss_history.png
    │       ├── threshold_sweep_val.csv / .png
    │       ├── confusion_matrix_test.png
    │       ├── pr_curve_test.png / roc_curve_test.png
    │       ├── test_metrics_bar.png
    │       ├── val_predictions.csv
    │       └── test_predictions.csv
    └── autoencoder/
        ├── model_comparison.csv
        ├── global/
        │   └── <model_name>/       (same layout as classifier above)
        └── asset_<id>/
            └── <model_name>/
                ├── metrics.json
                ├── model.keras
                ├── threshold.json
                ├── val_scores.npy
                └── test_window_predictions.csv
```

## Typical Workflow

```powershell
# Step 1 — build combined CSV from raw CARE files and prepare exports
python src/main.py prepare-care --window-hours 24

# Step 2 — train autoencoders only (fast first pass)
python src/main.py train-sequences --windows 24 --skip-classifiers

# Step 3 — add classifiers
python src/main.py train-sequences --windows 24 --skip-autoencoders
```

Or in two commands when you already have a combined CSV:

```powershell
python src/main.py prepare --csv "Dataset\processed\combined.csv" --window-hours 24
python src/main.py train-sequences --windows 24
```

## Models

### Supervised classifiers (global)

| Model | Architecture |
|-------|-------------|
| `lstm` | 2-layer stacked LSTM → Dense |
| `gru` | 2-layer stacked GRU → Dense |
| `cnn_lstm` | 2× Conv1D + MaxPool → LSTM → Dense |
| `cnn_gru` | 2× Conv1D + MaxPool → GRU → Dense |

All classifiers use binary cross-entropy loss with class-weighted training,
PR-AUC early stopping, and a validation threshold sweep to select the operating
point before test evaluation.

### Sequence autoencoders (per-asset or global)

| Model | Architecture |
|-------|-------------|
| `lstm_ae` | LSTM encoder-decoder with RepeatVector |
| `gru_ae` | GRU encoder-decoder with RepeatVector |

Autoencoders are trained exclusively on normal (label = 0) windows from the
training split. At evaluation time, reconstruction MSE is compared against a
threshold selected from the validation set.

**Threshold selection:** when labeled validation data contains fault examples,
the threshold is chosen by F1-sweeping reconstruction quantiles. When no fault
examples are present (common with the `train_tail` validation source), the
threshold falls back to the 99th percentile of normal-window reconstruction
errors.

### Labeling convention

A window `[t-W+1 : t]` carries label `1` if and only if the row at index `t`
(the last row of the window) is marked as a fault row in the ground-truth table.
This is the *last-timestep* convention — label reflects the state at the moment
the window ends, not a future prediction horizon.

## Evaluation Outputs

Start with the summary files before drilling into individual model folders:

```text
results/sequence_training_results/run_summary.json
results/sequence_training_results/window_24h/window_summary.json
results/sequence_training_results/window_24h/classifier/model_comparison.csv
results/sequence_training_results/window_24h/autoencoder/model_comparison.csv
```

Each model folder's `metrics.json` contains accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, confusion matrix, threshold, and data-shape metadata.

## Development Notes

- Keep windows inside `asset_id + sequence_id` groups. Crossing these
  boundaries leaks information across turbines or events.
- Fit scalers only on the training split. Per-asset scalers are saved to
  `scaler_asset_<id>.pkl` and re-used for validation and test transforms.
- Do not use metadata or target columns as model inputs.
- `src/training/sequence_model_utils.py` is now a backward-compat re-export
  shim. New code should import directly from the focused sub-modules
  (`sequence_utils`, `sequence_io`, `sequence_models`, `sequence_metrics`,
  `sequence_plots`, `sequence_experiments`).
- `configs/problem_v1.yaml` documents the locked problem setting. If using
  `src/problem_config.py` as a strict preflight, ensure duration values are
  numeric and consistent with step counts.

## Reference Documents

- `Dataset/README.md`: dataset structure and field meanings.
- `reports/problem_definition.md`: project objective, labels, boundaries, and
  evaluation criteria.
- `reports/`: capstone notes, research summaries, and longer analysis.
