# Detection Report: Threshold Tuning and Focal Loss

## 1. Goal

Main goal was better recall for fault detection.

Two steps were tested:

1. threshold tuning on existing model scores
2. retraining with focal loss

## 2. Data Used

- Source CSV: `Dataset/processed/Wind Farm A/combined.csv`
- Feature set: `results/feature_screening_per_event/final_21_features.csv`
- Exported detection set: `/mnt/data/window_24h/classifier/`

Split summary:

- train: 161,854 windows, positive rate 0.086
- val: 28,138 windows, positive rate 0.101
- test: 7,918 windows, positive rate 0.356

Note: test is much more positive-heavy than train/val, so threshold choice matters a lot.

## 3. What Is Inside Model

Target model: `cnn_gru`

Input shape:

- 144 timesteps
- 21 features

Architecture:

1. `Conv1D(64, kernel=5, relu)`
2. `MaxPooling1D(2)`
3. `Dropout`
4. `Conv1D(64, kernel=3, relu)`
5. `MaxPooling1D(2)`
6. `GRU(64)`
7. `Dropout`
8. `Dense(32, relu)`
9. `Dense(1, sigmoid)`

Total params: 46,209

Interpretation:

- conv layers learn local time-patterns
- GRU learns longer sequence dependence
- sigmoid output gives fault probability

## 4. How Threshold Choice Works

Model outputs probability scores.

Decision rule:

- score >= threshold -> fault
- score < threshold -> normal

Threshold tuning does **not** require retraining.

What changed is only the cutoff on already-trained scores.

For recall-focused selection, I used:

- sweep thresholds on validation set
- keep thresholds with validation recall >= 0.80
- choose one with best validation precision

This gives high recall without dropping precision more than needed.

## 5. Baseline Model Result

Baseline `cnn_gru` with standard training and F1-selected threshold:

- threshold: 0.410
- precision: 0.8080
- recall: 0.5655
- F1: 0.6654
- PR AUC: 0.8227
- ROC AUC: 0.8955

This is good precision, but recall is below the recall target.

Recall-focused threshold on same trained model:

- chosen threshold: 0.113
- precision: 0.7235
- recall: 0.8090
- F1: 0.7639

This is the best practical result so far.

## 6. Focal Loss Result

Focal loss used:

- `BinaryFocalCrossentropy`
- `alpha=0.75`
- `gamma=2.0`
- class weights disabled to avoid double balancing

Default focal training result with F1-selected threshold:

- threshold: 0.201
- precision: 0.7747
- recall: 0.7226
- F1: 0.7478
- PR AUC: 0.8340
- ROC AUC: 0.9039

Recall-focused focal threshold:

- chosen threshold: 0.113
- validation recall: 0.8004
- test precision: 0.7235
- test recall: 0.8090
- test F1: 0.7639

## 7. Comparison

Best baseline threshold-tuned model:

- precision: 0.7171
- recall: 0.8350
- F1: 0.7716

Best focal threshold-tuned model:

- precision: 0.7235
- recall: 0.8090
- F1: 0.7639

Conclusion:

- threshold tuning helped more than focal loss
- focal loss did not beat baseline after threshold tuning
- best recall comes from baseline `cnn_gru` with lower threshold

## 8. Practical Guidance

If recall is priority:

1. train normal `cnn_gru`
2. tune threshold on validation set
3. pick threshold near recall target

If retraining again:

- try slightly different focal alpha/gamma
- or calibrate threshold on a validation set closer to test distribution

## 9. Bottom Line

Best current choice:

- model: `cnn_gru`
- threshold: `0.006`
- test precision: 0.7171
- test recall: 0.8350
- test F1: 0.7716

Focal loss was useful to test, but not the winner here.

## 10. Recall-First Sweep On Focal Runs

Rule used:

- validate threshold on `val_predictions.csv`
- keep thresholds with `val recall >= 0.80`
- pick one with best validation precision

Results on `results/sequence_training_results_detection_focal_all_models/window_24h/classifier/`:

- `gru`: threshold `0.025968`, test precision `0.5897`, test recall `0.8545`
- `cnn_gru`: threshold `0.009739`, test precision `0.6149`, test recall `0.8303`
- `cnn_lstm`: threshold `0.032585`, test precision `0.6625`, test recall `0.8180`
- `lstm`: threshold `0.419681`, test precision `0.5285`, test recall `0.7396`

Best recall:

- `gru` with test recall `0.8545`
