# Classifier Model Building Report

**Project:** SCADA Fault Prediction  
**Report scope:** How the repository builds and trains the supervised classifier models using the final 21-feature set  
**Main workflow:** `prepare -> train-sequences`  
**Primary feature file:** `results/feature_screening_per_event/final_21_features.csv`  
**Date:** May 2026

---

## 1. Purpose of This Report

This document explains how the classifier models are built in the repository. It is intended as a source document for the model-building chapter of the capstone report. The focus is the supervised sequence-classification pipeline using the final 21-feature modeling contract, not the autoencoder path.

The classifier task is formulated as early fault prediction from multivariate SCADA time-series windows. Each input sample is a fixed-length window of historical SCADA measurements containing 21 selected SCADA features. The model outputs one probability score:

```text
score = probability that a fault appears in the future prediction horizon
```

The current classifier models are:

```text
lstm
gru
cnn_lstm
cnn_gru
```

These models are trained from sequence exports created by the repository, not directly from raw SCADA event CSV files. In the final capstone report, the classifier method should be described around the 21-feature export only.

---

## 2. Position in Chapter 4

In the capstone report, this content fits under the model-building chapter after data splitting and feature reduction:

```text
Chapter 4. Model Building and Proposed Classifier
4.1 Overall training pipeline
4.2 Sequence-window representation
4.3 Classifier architectures
4.4 Training configuration
4.5 Threshold selection and evaluation design
4.6 Saved artifacts and reproducibility
```

Recommended figures and tables:

| Report item | Where to place it | Content |
|---|---|---|
| Figure 4.1 | Start of model-building section | End-to-end classifier pipeline: combined CSV -> selected features -> windows -> classifier -> threshold -> evaluation |
| Table 4.1 | Sequence-window representation | Window length, stride, prediction horizon, feature count, scaler |
| Table 4.2 | Dataset split summary | Train, validation, and test shapes plus class counts |
| Table 4.3 | Model architecture comparison | LSTM, GRU, CNN-LSTM, CNN-GRU layer designs |
| Table 4.4 | Training hyperparameters | Epochs, batch size, optimizer, loss, callbacks, class weighting |
| Figure 4.2 | Architecture section | Diagram of CNN-RNN hybrid classifier |
| Figure 4.3 | Threshold section | Validation threshold sweep plot |
| Figure 4.4 | Evaluation section | Test confusion matrix |
| Figure 4.5 | Evaluation section | PR curve and ROC curve |

---

## 3. Repository Entry Points

The active classifier workflow is exposed through `src/main.py`.

### 3.1 Prepare Sequence Exports

The first stage converts the combined SCADA table into classifier-ready NumPy arrays:

```powershell
python src/main.py prepare `
  --csv Dataset\processed\combined_dataset.csv `
  --feature-file results\feature_screening_per_event\final_21_features.csv `
  --expected-feature-count 21 `
  --window-hours 24 `
  --skip-autoencoder-export
```

The `prepare` command uses:

```text
src/data_pipeline/preprocessing/combined_sequence_pipeline.py
```

Its role is to:

1. Load the combined CSV.
2. Load the selected 21-feature list.
3. Remove metadata and target columns from the model input.
4. Sort rows by `asset_id`, `sequence_id`, and `time_stamp`.
5. Split data into train, validation, and test sections.
6. Fit scalers on training data only.
7. Transform each asset with its own scaler.
8. Build sliding windows without crossing event or asset boundaries.
9. Save classifier arrays under `Dataset/processed/sequence_exports/window_24h/classifier/`.

### 3.2 Train Classifier Models

The second stage trains the classifier models from the exported arrays:

```powershell
python src/main.py train-sequences `
  --windows 24 `
  --model lstm gru cnn_lstm cnn_gru `
  --skip-autoencoders
```

The `train-sequences` command uses:

```text
src/training/sequence_model_trainer.py
src/training/sequence_experiments.py
src/training/sequence_models.py
src/training/sequence_metrics.py
```

If `--model` is omitted, `train-sequences` defaults to all classifier models unless `--skip-classifiers` is used.

---

## 4. Input Data and 21-Feature Contract

The classifier pipeline starts from:

```text
Dataset/processed/combined_dataset.csv
```

The combined CSV must include these columns:

```text
time_stamp
asset_id
train_test
sequence_id
label
```

The classifier pipeline is intentionally built around the final 21-feature file:

```text
results/feature_screening_per_event/final_21_features.csv
```

This file is the feature contract for the classifier. It means every classifier input window has this shape:

```text
[window_steps, 21]
```

For the 24-hour experiment:

```text
[144, 21]
```

The selected features are:

| Group | Features |
|---|---|
| Sensor averages | `sensor_0_avg`, `sensor_10_avg`, `sensor_14_avg`, `sensor_18_avg`, `sensor_19_avg`, `sensor_33_avg`, `sensor_34_avg`, `sensor_38_avg`, `sensor_40_avg`, `sensor_41_avg`, `sensor_44` |
| Angle encodings | `sensor_5_avg_sin`, `sensor_5_avg_cos`, `sensor_5_min_sin`, `sensor_5_min_cos`, `sensor_5_max_cos`, `sensor_1_avg_sin`, `sensor_42_avg_cos` |
| Reactive power | `reactive_power_28_min`, `reactive_power_28_max` |
| Wind speed | `wind_speed_3_min` |

The 21 features are the result of the project feature-screening stage. They are used to keep the classifier input compact and reduce redundant SCADA channels before deep-learning training. The report should not describe the classifier as using all 89 candidate features or the 33 EDA exploratory features. Those are earlier analysis stages; the classifier section should use the 21-feature contract.

The following columns are never used as model input features:

```text
time_stamp
asset_id
train_test
status_type_id
sequence_id
label
data_split
```

This prevents metadata leakage and label leakage.

---

## 5. Sequence-Window Representation

The classifier input is a three-dimensional tensor:

```text
X: [num_windows, window_steps, 21]
y: [num_windows]
```

For the current 24-hour export:

| Setting | Value |
|---|---:|
| Window length | 24 hours |
| SCADA sampling interval | 10 minutes |
| Window steps | 144 |
| Stride | 6 steps, equal to 1 hour |
| Prediction horizon | 72 steps, equal to 12 hours |
| Scaler | MinMaxScaler |
| Validation source | `train_tail` |
| Feature count | 21 |

The current classifier export metadata reports:

| Split | Shape | Negative windows | Positive windows |
|---|---:|---:|---:|
| Train | `[161590, 144, 21]` | 144399 | 17191 |
| Validation | `[27874, 144, 21]` | 24507 | 3367 |
| Test | `[7654, 144, 21]` | 3990 | 3664 |

This representation keeps the temporal order inside each sample. The model does not receive isolated rows; it receives 24 hours of previous SCADA behavior, where each timestamp is represented by the same 21 selected features.

---

## 6. Label Definition

The repository currently uses future-horizon labeling:

```text
label_definition = future_horizon_any_positive_after_input_window
```

For an input window:

```text
X[t-W+1 : t]
```

and a future horizon:

```text
[t+1 : t+H]
```

the target is:

```text
y = 1 if at least one fault label appears in the future horizon
y = 0 if no fault label appears in the future horizon
```

With the current 24-hour setup:

```text
W = 144 steps = 24 hours
H = 72 steps = 12 hours
```

Therefore, the classifier is an early-warning model. It predicts whether a fault appears in the next 12 hours after the 24-hour input window.

This is different from a same-timestamp detector. If the project wants detection at the final timestamp of the window, the repository must be run with `--label-mode detection` or `--label-mode last_timestamp`.

---

## 7. Splitting and Leakage Control

The pipeline includes several controls to reduce data leakage:

1. Rows are sorted by `asset_id`, `sequence_id`, and `time_stamp`.
2. Windows are built inside each `(asset_id, sequence_id)` group.
3. A window is never allowed to cross an event boundary or turbine boundary.
4. The scaler is fit on training rows only.
5. Scaling is performed per asset, not as one global scaler for all turbines.
6. Validation is created from the tail of each training segment when `validation_source = train_tail`.
7. Test data comes from the `prediction` split.
8. The threshold is selected on validation data, then evaluated on test data.

This matters because sliding windows overlap heavily. If windows were randomly split, near-identical windows could appear in both train and test, producing overly optimistic evaluation.

---

## 8. Classifier Architectures

All classifier architectures are implemented in:

```text
src/training/sequence_models.py
```

The builder function is:

```text
build_classifier_model(...)
```

All models share the same output design:

```text
Dense(1, activation="sigmoid")
```

The sigmoid output is interpreted as the probability of a future fault.

### 8.1 LSTM Classifier

```text
Input [144, 21]
-> LSTM(96, return_sequences=True)
-> Dropout
-> LSTM(48)
-> Dropout
-> Dense(32, relu)
-> Dense(1, sigmoid)
```

The LSTM model is a recurrent baseline. It is designed to learn temporal dependencies in SCADA behavior over the 24-hour input window.

### 8.2 GRU Classifier

```text
Input [144, 21]
-> GRU(96, return_sequences=True)
-> Dropout
-> GRU(48)
-> Dropout
-> Dense(32, relu)
-> Dense(1, sigmoid)
```

The GRU model has a similar purpose to LSTM but uses a simpler recurrent unit. It is included as a recurrent baseline with fewer gating mechanisms than LSTM.

### 8.3 CNN-LSTM Classifier

```text
Input [144, 21]
-> Conv1D(64, kernel_size=5, padding="same", relu)
-> MaxPooling1D(pool_size=2)
-> Dropout
-> Conv1D(64, kernel_size=3, padding="same", relu)
-> MaxPooling1D(pool_size=2)
-> LSTM(64)
-> Dropout
-> Dense(32, relu)
-> Dense(1, sigmoid)
```

The CNN-LSTM model first extracts local temporal patterns using one-dimensional convolution. The LSTM layer then models the longer temporal dependency after the sequence has been compressed by pooling.

### 8.4 CNN-GRU Classifier

```text
Input [144, 21]
-> Conv1D(64, kernel_size=5, padding="same", relu)
-> MaxPooling1D(pool_size=2)
-> Dropout
-> Conv1D(64, kernel_size=3, padding="same", relu)
-> MaxPooling1D(pool_size=2)
-> GRU(64)
-> Dropout
-> Dense(32, relu)
-> Dense(1, sigmoid)
```

The CNN-GRU model has the same local-pattern extraction stage as CNN-LSTM, but replaces the LSTM layer with GRU. This makes it a hybrid classifier with lower recurrent complexity.

### 8.5 Architecture Comparison Table

| Model | Local pattern extractor | Temporal layer | Final classifier |
|---|---|---|---|
| `lstm` | None | LSTM(96) + LSTM(48) | Dense(32) + sigmoid |
| `gru` | None | GRU(96) + GRU(48) | Dense(32) + sigmoid |
| `cnn_lstm` | Conv1D(64, k=5) + Conv1D(64, k=3) | LSTM(64) | Dense(32) + sigmoid |
| `cnn_gru` | Conv1D(64, k=5) + Conv1D(64, k=3) | GRU(64) | Dense(32) + sigmoid |

---

## 9. Training Configuration

The training loop is implemented in:

```text
src/training/sequence_experiments.py
```

The default classifier training settings from `src/main.py` are:

| Hyperparameter | Default value |
|---|---:|
| Epochs | 25 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Binary cross-entropy |
| Recurrent dropout | 0.25 if not overridden |
| Convolution dropout | 0.20 if not overridden |
| L2 regularization | 0.0 by default |
| Training metrics | PR-AUC, ROC-AUC |
| Random seed | 42 |

The compile step is:

```text
optimizer = Adam(learning_rate)
loss = binary_crossentropy
metrics = PR-AUC and ROC-AUC
```

The model is trained with class weights:

```text
class_weight = compute_class_weights(y_train)
```

This is necessary because the classifier windows are imbalanced, especially in the training split.

### 9.1 Training Callbacks

The classifier callbacks are defined in:

```text
classifier_callbacks(...)
```

They include:

| Callback | Monitor | Purpose |
|---|---|---|
| EarlyStopping | `val_pr_auc` | Stop training when validation PR-AUC stops improving |
| ReduceLROnPlateau | `val_pr_auc` | Reduce learning rate when validation PR-AUC plateaus |
| ModelCheckpoint | `val_pr_auc` | Save the best model only |

Early stopping uses:

```text
patience = 4
mode = max
restore_best_weights = True
```

The learning-rate scheduler uses:

```text
factor = 0.5
patience = 2
min_lr = 1e-5
```

The checkpoint is saved as:

```text
model.keras
```

---

## 10. Threshold Selection

The model outputs a probability score, but a binary alarm requires a threshold. The repository does not assume a fixed threshold of `0.5`. Instead, it performs a validation sweep.

The threshold grid for classifiers is:

```text
0.10, 0.15, 0.20, ..., 0.90
```

If validation metadata contains `asset_id` and `sequence_id`, and validation contains positive windows, the repository uses event-level threshold selection:

```text
threshold_source = event_level_f1_sweep
```

Otherwise, it uses window-level threshold selection:

```text
threshold_source = validation_f1_sweep
```

The best threshold is selected by sorting:

```text
1. highest F1
2. highest precision
3. highest recall
4. lowest threshold
```

The selected threshold is then fixed and applied to the test set.

---

## 11. Evaluation Design

The repository evaluates classifier performance at two levels.

### 11.1 Window-Level Evaluation

Window-level evaluation treats every sliding window as one prediction:

```text
y_pred = 1 if score >= selected_threshold
y_pred = 0 otherwise
```

The saved metrics include:

```text
accuracy
precision
recall
f1
PR-AUC
ROC-AUC
confusion matrix
```

### 11.2 Event-Level Evaluation

Event-level evaluation groups windows by:

```text
asset_id, sequence_id
```

The event score is:

```text
event_score = max(window_scores inside the event)
```

The event is predicted positive if:

```text
event_score >= selected_threshold
```

This is useful for reporting because the real operational question is often whether the event was detected at least once, not whether every positive window was detected.

However, the report must clearly separate window-level metrics from event-level metrics. Event-level F1 can be higher than window-level F1 because max aggregation makes the event easier to detect.

---

## 12. Saved Training Artifacts

Classifier outputs are saved under:

```text
results/sequence_training_results/window_24h/classifier/<model_name>/
```

Each model folder can contain:

| Artifact | Meaning |
|---|---|
| `model.keras` | Best saved Keras model |
| `model_summary.txt` | Layer-by-layer architecture summary |
| `metrics.json` | Validation, test, and event-level metrics |
| `history.csv` | Epoch-level training history |
| `loss_history.png` | Training and validation loss curve |
| `threshold_sweep_val.csv` | Validation threshold sweep table |
| `threshold_sweep_val.png` | Validation threshold sweep plot |
| `confusion_matrix_test.png` | Test confusion matrix |
| `test_metrics_bar.png` | Test metric summary |
| `pr_curve_test.png` | Test precision-recall curve |
| `roc_curve_test.png` | Test ROC curve |
| `val_predictions.csv` | Validation window scores and predictions |
| `test_predictions.csv` | Test window scores and predictions |
| `test_event_predictions.csv` | Test event-level scores and predictions |

The global classifier comparison table is:

```text
results/sequence_training_results/window_24h/classifier/model_comparison.csv
```

---

## 13. Current Artifact Consistency Note

The final report should focus on the current 21-feature export:

```text
Dataset/processed/sequence_exports/window_24h/classifier/metadata.json
```

Some older saved model metrics under:

```text
results/sequence_training_results/window_24h/classifier/*/metrics.json
```

report training shapes with 17 features:

```text
[161590, 144, 17]
```

Those 17-feature artifacts should be treated as legacy runs, not as the final classifier evidence. For the final capstone result, report only runs whose `metrics.json` shape matches the 21-feature contract:

```text
train_shape = [*, 144, 21]
val_shape   = [*, 144, 21]
test_shape  = [*, 144, 21]
```

If the current result folders still show `[*, 144, 17]`, retrain the classifiers from the current 21-feature export before copying performance numbers into the capstone report.

The safest wording for the capstone report is:

```text
The repository builds the classifier models using the 24-hour, 21-feature sequence-window pipeline.
Final quantitative results are reported only from training artifacts whose input shape matches the declared 21-feature contract.
```

---

## 14. Suggested Capstone Figures and Tables

### 14.1 Pipeline Figure

Place this figure at the start of the classifier section:

```text
combined_dataset.csv
-> final_21_features.csv
-> per-asset train/val/test split
-> per-asset scaler fit on train
-> 24h sliding windows with shape [144, 21]
-> LSTM / GRU / CNN-LSTM / CNN-GRU
-> validation threshold sweep
-> test window-level and event-level evaluation
```

### 14.2 Architecture Figure

For the hybrid model figure, show:

```text
Input SCADA window
-> Conv1D local feature extraction
-> MaxPooling temporal compression
-> LSTM or GRU temporal modeling
-> Dense classifier
-> Sigmoid probability
```

### 14.3 Results Figures

Use these saved figures for each reported model:

```text
threshold_sweep_val.png
confusion_matrix_test.png
pr_curve_test.png
roc_curve_test.png
loss_history.png
```

The most useful visual order is:

1. Loss history to show training behavior.
2. Threshold sweep to justify the selected decision threshold.
3. Confusion matrix to explain false positives and false negatives.
4. PR curve because the task is imbalanced.
5. ROC curve as a secondary ranking metric.

---

## 15. Report-Ready Summary

The repository builds the classifier model as a supervised sequence-classification system using the final 21-feature SCADA subset. The combined SCADA dataset is first reduced to this feature contract, split by temporal logic, scaled using training data only, and transformed into 24-hour sliding windows of shape `[144, 21]`. Each window is labeled according to whether a fault appears in the following 12-hour prediction horizon. Four neural classifiers are supported: LSTM, GRU, CNN-LSTM, and CNN-GRU. The recurrent models act as sequence baselines, while the CNN-RNN hybrids combine local temporal pattern extraction with longer temporal dependency modeling.

Training uses binary cross-entropy, Adam optimization, class weighting for imbalance, dropout regularization, early stopping, learning-rate reduction, and checkpointing based on validation PR-AUC. After training, the model threshold is selected on validation data through an F1-based sweep, then fixed for test evaluation. Results are saved as model files, metrics, prediction CSVs, threshold plots, confusion matrices, PR curves, and ROC curves. This makes the classifier pipeline reproducible and suitable for reporting as the main model-building workflow of the project.

---

## 16. Source Files Used

The report is based on these repository files and artifacts:

```text
src/main.py
src/data_pipeline/preprocessing/combined_sequence_pipeline.py
src/training/sequence_model_trainer.py
src/training/sequence_experiments.py
src/training/sequence_models.py
src/training/sequence_metrics.py
Dataset/processed/sequence_exports/window_24h/classifier/metadata.json
results/sequence_training_results/window_24h/classifier/*/metrics.json
```
