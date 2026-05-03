# SCADA Fault Prediction Method Catalog

This document lists the reusable methods, classes, scripts, and workflows that
already exist in this project. Use it before adding new code so duplicate data
loading, feature engineering, sequence export, model training, evaluation, or
plotting logic is not reimplemented.

The import examples assume either:

- the project is installed in editable mode with `pip install -e .`, or
- commands are run from the repository with `src/` on `PYTHONPATH`.

Most project modules import each other without the `src.` prefix, for example
`from data_pipeline.preprocessing.combined_sequence_pipeline import CombinedSequencePipeline`.

## Main Workflow

The current runnable workflow is centered on a combined sequence CSV and the
`src/main.py` CLI.

| Stage | CLI | Reusable API | Main output |
| --- | --- | --- | --- |
| Build combined CSV from raw CARE event files | `python src/main.py prepare-care` | `data_pipeline.preprocessing.build_combined_csv.CAREToCombinedCSV` | `Dataset/processed/<farm>/combined.csv` |
| Prepare sequence exports from an existing combined CSV | `python src/main.py prepare --csv <combined.csv>` | `data_pipeline.preprocessing.combined_sequence_pipeline.CombinedSequencePipeline` | `Dataset/processed/sequence_exports/window_<H>h/` |
| Train sequence classifiers and autoencoders | `python src/main.py train-sequences --windows 24` | `training.sequence_model_trainer.SequenceModelTrainer` | `results/sequence_training_results/window_<H>h/` |
| Run combined CSV EDA | `python src/main.py eda --csv <combined.csv>` | `data_pipeline.eda.EDAReport` | `results/eda/<name>/` |

The supported `train-sequences` model names are:

- Classifiers: `lstm`, `gru`, `cnn_lstm`, `cnn_gru`
- Autoencoders: `lstm_ae`, `gru_ae`

Use `--model` to train a specific subset. The CLI automatically routes names to
the classifier or autoencoder branch.

```powershell
python src/main.py train-sequences --windows 24 --model cnn_gru
python src/main.py train-sequences --windows 24 --model lstm_ae gru_ae --skip-classifiers
python src/main.py train-sequences --windows 24 --autoencoder-scope both
```

## Data Contracts

### Combined CSV

`CombinedSequencePipeline` expects one CSV with event boundaries preserved. The
minimum required columns are:

- `time_stamp`
- `asset_id`
- `sequence_id`
- `train_test`
- `label` or `status_type_id`

Feature columns are either read from `--feature-file` or inferred as numeric
non-metadata columns. Metadata columns excluded from features include:

```text
time_stamp, asset_id, train_test, status_type_id, sequence_id, label, data_split
```

### Sequence Labels

The active combined-sequence path labels each window using the future horizon:

```text
target_label = max(label rows from t+1 through t+H)
```

This is implemented in `CombinedSequencePipeline.build_windows`. It is different
from some older per-asset helpers that label a window by the last row inside the
input window.

### Sequence Export Layout

Default export root:

```text
Dataset/processed/sequence_exports/
  split_summary.json
  window_search_results.csv
  best_windows.json
  export_summary.json
  window_<H>h/
    classifier/
      X_train.npy, y_train.npy
      X_val.npy, y_val.npy
      X_test.npy, y_test.npy
      train_meta.csv, val_meta.csv, test_meta.csv
      metadata.json
      scalers/asset_<id>.pkl
    autoencoder/
      global/
        X_train.npy, X_val.npy
        train_meta.csv, val_meta.csv
        scalers.pkl
        test_by_sequence/asset_<id>/sequence_<id>.npz
      asset_<id>/
        X_train.npy, X_val.npy
        scaler.pkl
        metadata.json
        test_by_sequence/sequence_<id>.npz
```

## Import Cheat Sheet

### CARE Data to Combined CSV

```python
from data_pipeline.preprocessing.build_combined_csv import CAREToCombinedCSV

builder = CAREToCombinedCSV(
    farm_dir="Dataset/raw/Wind Farm A",
    datasets_dir="Dataset/raw/Wind Farm A/datasets",
)
combined_path = builder.build("Dataset/processed/Wind Farm A/combined.csv")
```

### Combined CSV to Window Exports

```python
from data_pipeline.preprocessing.combined_sequence_pipeline import CombinedSequencePipeline

pipeline = CombinedSequencePipeline(
    csv_path="Dataset/processed/Wind Farm A/combined.csv",
    feature_file="results/feature_screening_combined_csv/final_selected_features.csv",
    selected_windows_hours=[24],
    validation_source="train_tail",
    scaler_type="minmax",
)
summary = pipeline.run()
```

### Train Sequence Models

```python
from training.sequence_model_trainer import SequenceModelTrainer, SequenceTrainingConfig

config = SequenceTrainingConfig(
    windows=[24],
    classifier_models=["cnn_gru"],
    autoencoder_models=["lstm_ae", "gru_ae"],
    autoencoder_scope="per_asset",
)
run_summary = SequenceModelTrainer(config).run()
```

### Run EDA

```python
from data_pipeline.eda import EDAReport

summary = EDAReport(
    csv_path="Dataset/processed/Wind Farm A/combined.csv",
    output_dir="results/eda/Wind Farm A",
    feature_file="results/feature_screening_combined_csv/final_selected_features.csv",
    select_features=True,
).run()
```

### Run CNN-GRU Inference

```python
from inference.utils.cnn_gru import run_cnn_gru_inference_from_dataframe

outputs = run_cnn_gru_inference_from_dataframe(
    df=input_df,
    model_path="results/sequence_training_results/window_24h/classifier/cnn_gru/model.keras",
    threshold_path="results/sequence_training_results/window_24h/classifier/cnn_gru/metrics.json",
    feature_file="results/feature_screening_combined_csv/final_selected_features.csv",
    scaler_dir="Dataset/processed/sequence_exports/window_24h/classifier/scalers",
    window_hours=24,
)

window_predictions = outputs["window_predictions"]
event_predictions = outputs["event_predictions"]
```

## Module Catalog

### CLI and Configuration

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `src/main.py` | CLI orchestration for `prepare-care`, `prepare`, `train-sequences`, and `eda`. | Run as a script. Programmatic code should import the underlying classes listed below. |
| `src/config.py` | Project paths, feature groups, constants, legacy model hyperparameters, and `ensure_dirs()`. | `from config import PROCESSED_DATA_DIR, RESULTS_DIR, TIME_RESOLUTION` |
| `src/problem_config.py` | Typed YAML config loader with dataclasses for problem, dataset, time-series, label, split, and metric settings. | `from problem_config import load_problem_config` |
| `configs/problem_v1.yaml` | Locked problem-definition YAML. | Used by `problem_config.py` and attempted by `CombinedSequencePipeline` defaults. |

Status note: `configs/problem_v1.yaml` currently has `input_window_hours: 24d`.
`CombinedSequencePipeline` catches config-load errors and falls back to 24 hours,
72 horizon steps, and 6 stride steps. Fix the YAML value before relying directly
on `load_problem_config()`.

### Data Loading

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `data_pipeline.loaders.event_loader` | `EventLoader` loads CARE `event_info.csv` and per-event `<event_id>.csv` files. Also exposes `load_event_info()` and `load_event_data()` aliases. | `from data_pipeline.loaders import load_event_info, load_event_data` |
| `data_pipeline.loaders.sequence_maker` | `SequenceMaker` creates sliding-window `(X, y)` sequences and probe-only `X` sequences. | `from data_pipeline.loaders import create_sequences, create_probe_sequences` |
| `data_pipeline.loaders.tabular_loader` | `TabularLoader` flattens sequence NPZ exports or computes statistical window features for Random Forest and XGBoost. | `from data_pipeline.loaders.tabular_loader import TabularLoader` |
| `data_pipeline.utils.io` | Small CSV utilities: label ratio print, concat CSVs, save CSV, and data info printout. | `from data_pipeline.utils.io import check_ratio, read_and_concat_csv` |

### Preprocessing and Feature Methods

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `data_pipeline.preprocessing.build_combined_csv` | `CAREToCombinedCSV` converts raw CARE event files into one combined CSV with engineered features, `asset_id`, `sequence_id`, and row labels. | `from data_pipeline.preprocessing.build_combined_csv import CAREToCombinedCSV` |
| `data_pipeline.preprocessing.combined_sequence_pipeline` | `CombinedSequencePipeline` validates combined CSV schema, splits train/val/test by sequence, fills gaps, fits per-asset scalers, builds future-horizon windows, searches windows, and exports classifier/autoencoder datasets. | `from data_pipeline.preprocessing.combined_sequence_pipeline import CombinedSequencePipeline` |
| `data_pipeline.preprocessing.feature_engineering` | `FeatureEngineer` handles angle sin/cos encoding, yaw misalignment, feature column selection, missing fill, and backward-compatible helper aliases. | `from data_pipeline.preprocessing import FeatureEngineer` is not exported; use `from data_pipeline.preprocessing.feature_engineering import FeatureEngineer`. |
| `data_pipeline.preprocessing.ground_truth` | `GroundTruth` creates row-level fault labels from `event_start_id` and `event_end_id`. Provides `make_labels`, `make_normal_index`, and `add_label_column`. | `from data_pipeline.preprocessing import GroundTruth, make_labels` |
| `data_pipeline.preprocessing.normalizer` | `AssetNormalizer` fits per-asset scalers on train data and transforms train/val/test sequences. Used mainly by the older per-asset pipeline. | `from data_pipeline.preprocessing import normalize_asset` |
| `data_pipeline.preprocessing.splitter` | `DataSplitter` implements older CARE-style global/per-asset event processing, normal-row filtering, temporal split, and train/test event grouping. | `from data_pipeline.preprocessing import process_all_events_train, temporal_split_train_val` |
| `data_pipeline.preprocessing.feature_screening` | `FeatureScreening` runs point-biserial/Spearman screening across CSV files, removes redundant features, and writes plots/tables. | `from data_pipeline.preprocessing.feature_screening import FeatureScreening` |
| `data_pipeline.preprocessing.combine_data` | Hard-coded utility script that concatenates CSV files into `og_combined_data.csv`. | Do not import for new reusable code; prefer `CAREToCombinedCSV` or `read_and_concat_csv()`. |

Compatibility note: `FeatureEngineer.drop_counter_features()` is retained for
older callers but currently returns a copy without dropping additional columns.
Feature exclusion is handled through explicit metadata/drop-column lists.

### EDA and Feature Selection

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `data_pipeline.eda` | `EDAReport` generates missing-value tables, feature stats, Spearman label correlation, per-asset summaries, label-balance plots, time-series overview, feature distributions, and optional EDA-based feature selection. | `from data_pipeline.eda import EDAReport` |
| `analysis.combined_csv_eda` | Paper/report-oriented EDA script for a combined CSV. Writes profile, missing values, numeric summary, distribution tests, Spearman outputs, and plots. | Run as a script if you need report artifacts. |
| `analysis.check_label_imbalance` | Event-level and row-level class imbalance checks across raw wind farm folders. | Run as a script. |
| `analysis.analyze_data` | Quick DataFrame diagnostic print helper. | `from analysis.analyze_data import data_info` |
| `data_pipeline.validation.feature_selector` | Probe LSTM autoencoder feature selection with permutation importance, reconstruction sensitivity, and group ablation. | `from data_pipeline.validation import model_based_feature_selection` |
| `training.scripts.run_feature_screening` | CLI wrapper around `FeatureScreening`. | Run as `python -m training.scripts.run_feature_screening`. |
| `training.experiments.feature_screening_sweep` | Grid sweep over feature-screening thresholds with run summaries and feature-frequency outputs. | Run as an experiment script. |

### Sequence Model Training

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `training.sequence_model_trainer` | `SequenceTrainingConfig` and `SequenceModelTrainer` orchestrate training per exported window. Handles classifier and autoencoder branches, comparisons, summaries, and result folders. | `from training.sequence_model_trainer import SequenceModelTrainer, SequenceTrainingConfig` |
| `training.sequence_experiments` | Runners for a single classifier, per-asset autoencoder, and global pooled autoencoder. Handles fit, threshold selection, metrics, predictions, and plots. | `from training.sequence_experiments import run_classifier_experiment` |
| `training.sequence_models` | Keras builders and callbacks for `lstm`, `gru`, `cnn_lstm`, `cnn_gru`, `lstm_ae`, and `gru_ae`. | `from training.sequence_models import build_classifier_model, build_autoencoder_model` |
| `training.sequence_io` | Loads classifier bundles, autoencoder asset/global bundles, validation slices, test NPZ sequences, and scaler files. | `from training.sequence_io import load_classifier_bundle` |
| `training.sequence_metrics` | Class weights, PR/ROC helpers, threshold sweep, best-threshold selection, comparison DataFrames, and autoencoder architecture summaries. | `from training.sequence_metrics import evaluate_at_threshold, sweep_thresholds` |
| `training.sequence_plots` | Saves history, threshold, confusion matrix, metric bar, PR/ROC plots, and prediction DataFrames. | `from training.sequence_plots import build_prediction_frame` |
| `training.sequence_utils` | Random seeding, TensorFlow cleanup, JSON I/O, history formatting, best-model loading, model summary saving, and reconstruction scores. | `from training.sequence_utils import save_json, reconstruction_scores` |
| `training.sequence_model_utils` | Backward-compatible re-export shim for older imports. | New code should import from the focused modules above. |
| `training.scripts.train_sequence_models` | Older CLI wrapper for `SequenceModelTrainer`. Does not expose every newer `src/main.py train-sequences` flag. | Prefer `python src/main.py train-sequences` for current runs. |

### Legacy or Secondary Training Paths

These modules are still useful, but they belong to the older per-asset or
tree-model workflow rather than the current combined-sequence CLI path.

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `training.scripts.prepare_per_asset` | `PerAssetPipeline` prepares per-turbine arrays and test NPZ files from raw CARE events. Also has `run_from_csv()` for CSV-based preparation. | `from training.scripts.prepare_per_asset import PerAssetPipeline` |
| `training.trainer` | `AutoEncoderTrainer`, `LSTMTrainer`, and `TreeTrainer` for older autoencoder/LSTM/tree model training flows. | Import only when maintaining legacy outputs. |
| `training.scripts.train_lstm` | Thin script for older per-asset LSTM training. | Prefer current `train-sequences` unless reproducing old runs. |
| `training.scripts.train_random_forest` | Thin script for older Random Forest training. | Use with `TreeTrainer`/`TabularLoader` outputs. |
| `training.scripts.train_xgboost` | Thin script for older XGBoost training. | Use with `TreeTrainer`/`TabularLoader` outputs. |
| `training.callbacks.early_stopping` | Callback builders for older autoencoder, LSTM, and generic Keras training. | `from training.callbacks.early_stopping import get_lstm_callbacks` |
| `training.experiments.threshold_tuning` | Hybrid IQR + PR threshold tuning for older event-level MAE autoencoder outputs. | Run after legacy autoencoder training. |

### Model Architectures

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `models.architectures.lstm` | `LSTMModel` next-timestep prediction model and `build_lstm_model()` alias. | `from models.architectures import build_lstm_model` |
| `models.architectures.autodecoder` | Dense autoencoder/encoder/decoder builder and `build_autodecoder_model()` alias. | `from models.architectures import build_autodecoder_model` |
| `models.architectures.random_forest` | Random Forest classifier factory with balanced class weights. | `from models.architectures import build_random_forest_model` |
| `models.architectures.xgboost_model` | XGBoost classifier factory with lazy `xgboost` import. | `from models.architectures import build_xgboost_model` |

For new sequence classifier or sequence autoencoder experiments, prefer
`training.sequence_models` because it matches the active exported sequence data
contract.

### Evaluation, Metrics, and Plotting

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `evaluation.evaluator` | `LSTMEvaluator`, `TreeEvaluator`, and `AutoEncoderEvaluator` for older global/per-asset evaluations. | `from evaluation.evaluator import LSTMEvaluator, TreeEvaluator` |
| `evaluation.evaluate_lstm` | Thin entrypoint around `LSTMEvaluator`. | Run as a script for legacy LSTM evaluation. |
| `evaluation.evaluate_tree` | Thin entrypoint around `TreeEvaluator`. | Run as a script for legacy tree evaluation. |
| `evaluation.compare_models` | Older model comparison framework using event labels and saved model predictions. | Run as a script for legacy comparisons. |
| `evaluation.plot_metrics` | Small helpers for smoothing MAE and plotting asset metrics. | `from evaluation.plot_metrics import plot_asset_metrics` |
| `utils.metrics` | `MetricsCalculator` for binary metrics and F1 threshold search. | `from utils.metrics import MetricsCalculator` |
| `utils.visualization` | `Visualizer` for older training/evaluation plots and comparison charts. | `from utils.visualization import Visualizer` |
| `analysis.plot_lstm_confusion_matrix` | Script for older LSTM confusion-matrix plot. | Run as a script. |
| `analysis.plot_naive_confusion_matrix` | Script for older naive baseline confusion-matrix plot. | Run as a script. |
| `analysis.plot_mae_per_event` | Script/helper for event-level MAE plots. | Run as a script or import `compute_event_mae`. |

### Inference

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `inference.utils.cnn_gru` | CNN-GRU inference helpers for loading feature lists, models, thresholds, scalers, preparing DataFrames, building windows, running predictions, and aggregating event-level predictions. | `from inference.utils.cnn_gru import run_cnn_gru_inference_from_dataframe` |

This inference module is model-specific to CNN-GRU. For another classifier,
reuse the preparation helpers and swap the model-loading/prediction function.

### Sourcing

| Module | Already implemented | Reuse/import |
| --- | --- | --- |
| `data_pipeline.sourcing.downloaders.CTCompare_data` | Small KaggleHub downloader for `azizkasimov/wind-turbine-scada-data-for-early-fault-detection`. | Run manually when downloading data. It requires network access and `kagglehub`. |

## Package Exports

Some packages expose convenience imports:

```python
from data_pipeline.loaders import create_sequences, load_event_info
from data_pipeline.preprocessing import GroundTruth, make_labels, normalize_asset
from data_pipeline.validation import model_based_feature_selection
from models.architectures import build_lstm_model, build_random_forest_model
```

Many other modules are not re-exported from `__init__.py`. Import those directly
from their file-level module path, as shown in the catalog.

## Do Not Duplicate These Methods

Before writing new code, check these existing implementations:

- Raw CARE event loading: `EventLoader`
- CARE events to combined CSV: `CAREToCombinedCSV`
- Angle sin/cos engineering: `FeatureEngineer`
- Row-level labels from event intervals: `GroundTruth`
- Combined CSV split/scale/window export: `CombinedSequencePipeline`
- EDA report and optional EDA feature selection: `EDAReport`
- Cross-dataset statistical feature screening: `FeatureScreening`
- Sequence training orchestration: `SequenceModelTrainer`
- Keras sequence architectures: `training.sequence_models`
- Threshold sweeps and binary metrics: `training.sequence_metrics`
- Training plots and prediction CSV construction: `training.sequence_plots`
- CNN-GRU inference from DataFrame or array: `inference.utils.cnn_gru`

## When to Use Which Path

Use the current combined-sequence path when:

- the input data is already one combined CSV, or can be built into one;
- you need `asset_id + sequence_id` boundaries preserved;
- you want sequence classifiers and sequence autoencoders trained from the same
  exported window contract;
- you want current result folders under `results/sequence_training_results`.

Use the legacy per-asset path only when:

- reproducing older autoencoder/LSTM/tree experiments;
- consuming older `Dataset/processed/Wind Farm A/per_asset/` or `global/`
  NPZ layouts;
- maintaining older evaluator scripts.

## Known Status Notes

- `src/training/sequence_model_utils.py` is only a compatibility shim. New code
  should import from `sequence_io`, `sequence_metrics`, `sequence_models`,
  `sequence_plots`, `sequence_utils`, or `sequence_experiments` directly.
- `src/data_pipeline/preprocessing/combine_data.py` is hard-coded to a local
  Wind Farm A path. Do not use it as a reusable project API.
- `problem_config.py` is useful, but the current YAML value
  `input_window_hours: 24d` must be corrected before direct config loading can
  be treated as reliable.
- The combined-sequence workflow currently evaluates at window level. Some
  older files mention event-level metrics, lead time, or false alarms, but those
  are not the primary outputs of `train-sequences`.
