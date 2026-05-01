# SCADA Fault Prediction

This repository contains a Python workflow for early fault prediction on wind
turbine SCADA time-series data. The current runnable project focuses on a
combined-sequence pipeline:

1. Prepare fixed-length sequence exports from one combined CSV.
2. Train supervised sequence classifiers and per-asset sequence autoencoders.
3. Save metrics, predictions, plots, and comparison summaries for evaluation.

This README is a project guide, not a thesis outline. Report writing notes and
long-form research discussion belong under `reports/`.

## Project Scope

The modelling task is binary future anomaly prediction:

```text
X[t-W+1:t] -> P(any anomaly appears in the next H steps)
```

The default version of the project uses:

- SCADA sampling interval: 10 minutes
- Input window: 24 hours, or 144 time steps
- Prediction horizon: 12 hours, or 72 time steps
- Stride: 6 steps, or 1 hour
- Window boundary: do not cross `asset_id + sequence_id`
- Main data format: one combined CSV with event boundaries preserved

The raw CARE to Compare data contains event-level SCADA files for Wind Farm A,
Wind Farm B, and Wind Farm C. The current training path is designed for a
processed combined CSV, with Wind Farm A as the initial experimental scope.

## Repository Layout

```text
scada-fault-prediction/
|-- configs/
|   `-- problem_v1.yaml
|-- Dataset/
|   |-- raw/
|   `-- processed/
|-- experiments/
|-- notebooks/
|-- reports/
|-- src/
|   |-- data_pipeline/
|   |-- evaluation/
|   |-- inference/
|   |-- models/
|   |-- training/
|   |-- config.py
|   |-- main.py
|   `-- problem_config.py
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

Important files:

- `src/main.py`: main CLI entrypoint.
- `src/data_pipeline/preprocessing/combined_sequence_pipeline.py`: prepares
  classifier and autoencoder sequence exports from a combined CSV.
- `src/training/sequence_model_trainer.py`: trains sequence classifiers and
  autoencoders from exported windows.
- `src/training/sequence_model_utils.py`: model builders, metric helpers,
  threshold search, plots, and saved prediction outputs.
- `configs/problem_v1.yaml`: machine-readable problem definition.
- `reports/problem_definition.md`: readable project problem statement.

## Setup

Create a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project requires Python 3.10 or newer. TensorFlow is required for the
sequence models.

## Input Data

The active `prepare` command expects a combined CSV with these required columns:

- `time_stamp`
- `asset_id`
- `train_test`
- `sequence_id`
- `label` or `status_type_id`

Feature columns can be selected automatically from numeric non-metadata columns,
or supplied through a feature list CSV:

```text
results/results/final_features.csv
```

If a feature file is provided, the pipeline uses the `final_feature` column when
present; otherwise it uses the first column of the feature file.

## Run Pipeline

### 1. Prepare Sequence Exports

```powershell
$csv = "D:\path\to\combined_dataset.csv"
$features = "results\results\final_features.csv"
python src/main.py prepare --csv $csv --feature-file $features --window-hours 24
```

For the current combined Wind Farm A experiment, use a prediction-like
validation split so threshold selection sees data similar to the final test
segment:

```powershell
$csv = "D:\Final Project\C2C_Windturbine\df_final.csv"
$features = "D:\Final Project\C2C_Windturbine\final_features.csv"
$exports = "Dataset\processed\sequence_exports_prediction_val"
python src/main.py prepare --csv $csv --feature-file $features --window-hours 24 --validation-source prediction --prediction-val-ratio 0.5 --sequence-output-dir $exports
```

Default output:

```text
Dataset/processed/sequence_exports/
|-- best_windows.json
|-- export_summary.json
|-- window_search_results.csv
`-- window_24h/
    |-- classifier/
    |   |-- X_train.npy
    |   |-- y_train.npy
    |   |-- X_val.npy
    |   |-- y_val.npy
    |   |-- X_test.npy
    |   |-- y_test.npy
    |   |-- train_meta.csv
    |   |-- val_meta.csv
    |   |-- test_meta.csv
    |   |-- metadata.json
    |   `-- scalers/
    `-- autoencoder/
        `-- asset_<id>/
```

Useful prepare options:

```powershell
python src/main.py prepare --help
```

Common options:

- `--window-hours 24`: export a specific window size.
- `--window-candidates-hours 12 24 48`: test several candidate windows.
- `--top-k-windows 2`: export the best N windows after window search.
- `--combined-scaler minmax`: use `minmax` or `standard` scaling.
- `--expected-feature-count 17`: fail if the feature list has the wrong size.
- `--validation-source train_tail|prediction`: choose whether validation comes
  from the legacy tail of each train segment or from the first part of each
  prediction segment.
- `--prediction-val-ratio 0.5`: when using `--validation-source prediction`,
  send this fraction of each prediction segment to validation.
- `--sequence-output-dir <DIR>`: write exports to a custom location.

### 2. Train Sequence Models

```powershell
python src/main.py train-sequences --windows 24
```

Default output:

```text
results/sequence_training_results/
|-- run_summary.json
`-- window_24h/
    |-- window_summary.json
    |-- classifier/
    |   |-- model_comparison.csv
    |   `-- <model_name>/
    |       |-- metrics.json
    |       |-- training_history.csv
    |       |-- test_predictions.csv
    |       |-- val_predictions.csv
    |       `-- test_metrics_bar.png
    `-- autoencoder/
        |-- model_comparison.csv
        `-- asset_<id>/
```

Train only selected models:

```powershell
python src/main.py train-sequences --windows 24 --model lstm gru
python src/main.py train-sequences --windows 24 --model lstm_ae gru_ae
```

Skip one branch:

```powershell
python src/main.py train-sequences --windows 24 --skip-autoencoders
python src/main.py train-sequences --windows 24 --skip-classifiers
```

Useful training options:

```powershell
python src/main.py train-sequences --help
```

Common options:

- `--exports-dir <DIR>`: read sequence exports from a custom folder.
- `--results-dir <DIR>`: write training results to a custom folder.
- `--classifier-models lstm gru cnn_lstm cnn_gru`: choose classifiers.
- `--classifier-learning-rate 0.0003`: set the classifier Adam learning rate.
- `--classifier-dropout 0.35`: override classifier dropout layers.
- `--classifier-l2 0.0001`: add L2 regularization to classifier kernels.
- `--autoencoder-models lstm_ae gru_ae`: choose autoencoder architectures.
- `--assets 10 11`: restrict autoencoder training to specific assets.
- `--overwrite`: retrain even when metrics already exist.
- `--no-save-predictions`: skip verbose prediction CSV outputs.

## Models

The supervised classifier branch trains global sequence models:

- `lstm`
- `gru`
- `cnn_lstm`
- `cnn_gru`

The autoencoder branch trains per-asset reconstruction models:

- `lstm_ae`
- `gru_ae`

Classifier labels are future-horizon labels: a window is positive when at least
one anomalous row appears after the input window and inside the prediction
horizon. Autoencoders are fit on normal windows and evaluated by reconstruction
score thresholding.

## Evaluation Outputs

The training step writes both detailed model folders and summary files. The main
files to inspect first are:

- `results/sequence_training_results/run_summary.json`
- `results/sequence_training_results/window_24h/window_summary.json`
- `results/sequence_training_results/window_24h/classifier/model_comparison.csv`
- `results/sequence_training_results/window_24h/autoencoder/model_comparison.csv`
- Each model folder's `metrics.json`

Reported metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrices, threshold sweeps, and saved prediction tables where enabled.

## Development Notes

- Keep windows inside `asset_id + sequence_id` groups. Crossing these boundaries
  can leak information across turbines or events.
- Fit scalers only on the training split. The combined preparation code uses
  per-asset scalers fitted from training rows.
- Do not use metadata or target columns as model inputs.
- The visible CLI currently supports `prepare` and `train-sequences`.
  Older per-asset and experiment scripts may still exist for reference, but they
  are not the main project entrypoint.
- `configs/problem_v1.yaml` is intended to document the locked problem setting.
  If using `src/problem_config.py` as a strict preflight, make sure duration
  values are numeric and consistent with the step counts.

## Reference Documents

- `Dataset/README.md`: dataset structure and field meanings.
- `reports/problem_definition.md`: project objective, labels, boundaries, and
  evaluation criteria.
- `reports/`: capstone notes, research summaries, and longer analysis.
