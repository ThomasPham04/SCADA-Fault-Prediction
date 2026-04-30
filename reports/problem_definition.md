# Problem Definition - CRISP-DM Business Understanding

## 1. Project Context

This project studies early fault prediction for wind turbines using Supervisory Control and Data Acquisition (SCADA) time-series data. The dataset is organized into three wind farm folders: `Wind Farm A`, `Wind Farm B`, and `Wind Farm C`. Each wind farm contains event-level time-series files in `datasets/*.csv`, an `event_info.csv` metadata table, and a `feature_description.csv` table describing the SCADA variables.

Each CSV file represents one event time series from one wind turbine. An event is labelled as either normal or anomalous. The operational objective is to use recent SCADA measurements to estimate whether the turbine is likely to enter an anomalous or faulty state within a future prediction horizon.

This document locks the first version of the business and machine learning problem definition. It is intended to support the Business Understanding phase of CRISP-DM and to provide a stable basis for data preparation, modelling, and evaluation.

## 2. Business Objective

The business objective is to support early detection of wind turbine faults from routinely collected SCADA data. A useful model should identify signals that precede abnormal operation or failure, so that maintenance teams can inspect the turbine earlier, reduce unplanned downtime, and improve the reliability of wind farm operation.

The desired operational output is an early warning score. For each turbine and time point, the system should estimate the probability that an anomaly or fault will occur within a fixed future horizon. This score can later be converted into alarms using a threshold selected according to operational priorities such as high recall, acceptable false alarm rate, and sufficient lead time for maintenance response.

## 3. Machine Learning Objective

The main machine learning task is supervised binary future anomaly prediction.

Given a fixed-length SCADA history window from one turbine event, the model predicts the probability that an anomaly or fault will occur in the next prediction horizon.

The locked mathematical definition is:

```text
X[t-W+1:t] -> P(y[t,H] = 1)

y[t,H] = 1 iff any timestamp in [t+1, t+H] has anomaly label 1.
```

Where:

- `X[t-W+1:t]` is the input SCADA window ending at timestamp `t`.
- `W` is the number of historical input steps.
- `H` is the number of future steps in the prediction horizon.
- `y[t,H]` is the binary target for the future horizon.
- `y[t,H] = 1` indicates that at least one anomalous timestamp occurs after the input window and within the horizon.
- `y[t,H] = 0` indicates that no anomalous timestamp occurs within the horizon.

This formulation is different from ordinary anomaly detection on the current window. The model is not merely asked whether the current window already contains an anomaly. Instead, it is asked whether the next horizon contains an anomaly, using only information available up to time `t`.

## 4. Input and Output Definition

### 4.1 Input

The model input is a multivariate SCADA time-series window:

```text
X[t-W+1:t] in R^(W x F)
```

Where:

- `W = 144` time steps.
- The SCADA sampling interval is 10 minutes.
- Therefore, `W = 144` corresponds to 24 hours of history.
- `F` is the number of selected SCADA input features after feature selection and preprocessing.

The initial stride is:

```text
stride = 6 steps = 1 hour
```

This means that consecutive training examples are generated every hour rather than every 10 minutes, reducing redundancy while preserving temporal resolution.

### 4.2 Output

The model output is a scalar probability:

```text
P(y[t,H] = 1)
```

The prediction horizon is:

```text
H = 72 steps = 12 hours
```

The output can be interpreted as the estimated probability that the turbine will enter an anomalous or faulty state at least once during the next 12 hours.

## 5. Label Definition

The binary label for each generated window is defined from future timestamps, not from the input window itself.

For a window ending at timestamp `t`, the future horizon is:

```text
[t+1, t+H]
```

The target is:

```text
y[t,H] = 1 if any timestamp in [t+1, t+H] has anomaly label 1
y[t,H] = 0 otherwise
```

This label definition enforces an early prediction setting. It prevents the model from using the target anomaly timestamp inside the input window when the goal is to forecast a future anomaly.

In the raw dataset, event-level labels are available through `event_info.csv`. For time-step-level modelling, the data preparation phase must ensure that an appropriate row-level anomaly label exists before generating future-horizon labels. In the combined processed dataset, the row-level label may be represented by a `label` column. If the combined table uses `sequence_id` to identify event files, then `sequence_id` is treated as the event identifier for window generation.

## 6. Scope Version 1

The initial experimental scope is intentionally limited.

### 6.1 Included Data

Version 1 uses only:

```text
Wind Farm A
```

Wind Farm B and Wind Farm C are excluded from the initial modelling scope. They may be used later for cross-farm generalization, transfer learning, or external validation after the Wind Farm A pipeline is stable.

### 6.2 Pooling Strategy

The initial pooling strategy is vertical pooling within the same wind farm.

This means that windows from multiple turbines and multiple event files in Wind Farm A may be stacked into one supervised training dataset, provided that every window is generated within a valid turbine-event boundary.

The model is therefore allowed to learn from all Wind Farm A assets, but not from other wind farms in version 1.

### 6.3 Window Boundary Rule

Window generation must be grouped by:

```text
asset_id + event_id
```

If the processed combined table uses `sequence_id` as the event-file identifier, then the practical grouping key is:

```text
asset_id + sequence_id
```

No input window or future label horizon may cross:

- from one turbine to another,
- from one event file to another,
- from one wind farm to another,
- from training data into validation or test data,
- or from prediction data into unavailable future rows.

This rule is essential because events from the same turbine may overlap in calendar time or may represent different event contexts. Sorting only by `asset_id` is not sufficient and can cause leakage across event files.

## 7. Assumptions

The first version of the problem definition relies on the following assumptions:

1. The SCADA sampling interval is approximately 10 minutes.
2. Each CSV file represents one coherent event time series for one wind turbine.
3. `asset_id` identifies the turbine.
4. `event_id` or `sequence_id` identifies the event file.
5. A row-level binary anomaly label can be produced before future-horizon window labelling.
6. The model receives only SCADA measurements available up to timestamp `t`.
7. The future label is generated only from timestamps after `t`.
8. Wind Farm A is sufficiently representative for the initial capstone experiment.
9. Vertical pooling inside Wind Farm A is acceptable because all pooled windows come from the same farm and are kept within valid event boundaries.
10. Model evaluation should be reported at both window level and event level because these levels answer different questions.

## 8. Risks and Mitigation

### 8.1 Data Leakage

Data leakage is the most important methodological risk.

Potential leakage sources include:

- generating windows across different event files,
- generating windows across different turbines,
- fitting scalers on validation or test data,
- using future timestamps inside the input window,
- using `status_type_id` or any directly target-derived column as an input feature,
- splitting windows randomly after window generation when adjacent overlapping windows share most of their values,
- selecting thresholds on the test set,
- or tuning features using information from held-out test events.

Mitigation rules:

- Generate windows only inside each `asset_id + event_id` group.
- Fit preprocessing scalers only on training rows.
- Generate labels from `[t+1, t+H]`, never from the input interval.
- Keep metadata, target columns, and leakage-prone operational state columns out of model inputs unless explicitly justified.
- Use temporal or event-level validation/test splits.
- Select thresholds using validation data only.

### 8.2 Class Imbalance

Fault and anomaly prediction is expected to be imbalanced. Normal windows are likely to be more frequent than future-anomaly windows.

Risks include:

- high accuracy with poor anomaly recall,
- weak precision due to excessive false alarms,
- unstable threshold selection when validation anomalies are rare,
- and overfitting to a small number of anomalous event types.

Mitigation options include:

- reporting PR-AUC in addition to ROC-AUC,
- using class weights or balanced sampling during supervised training,
- evaluating event-level recall and false alarms,
- comparing against simple baselines,
- and reporting class counts for each split.

### 8.3 Event-Level Dependence

Windows from the same event are not independent. A model may appear strong at window level because many windows come from the same event context.

Mitigation:

- Report event-level metrics in addition to window-level metrics.
- Aggregate window scores into event-level decisions.
- Include lead time and false alarm duration in the evaluation.

### 8.4 Generalization Risk

Version 1 uses only Wind Farm A. Results may not generalize to Wind Farm B or Wind Farm C because turbines, sensors, feature dimensions, operating conditions, and anonymization schemes may differ.

Mitigation:

- State clearly that Wind Farm B and Wind Farm C are out of scope for version 1.
- Treat cross-farm validation as future work.

## 9. Evaluation Metrics

### 9.1 Window-Level Metrics

Window-level evaluation measures whether individual generated windows are classified correctly.

Primary window-level metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Because the dataset may be imbalanced, PR-AUC and recall are especially important. Accuracy should not be interpreted alone.

### 9.2 Event-Level Metrics

Event-level evaluation measures whether the model provides useful alarms for complete turbine events.

Primary event-level metrics:

- Event precision
- Event recall
- Event F1-score
- Lead time
- False alarm count
- False alarm duration

Event-level metrics are important for operational interpretation. A model that detects an anomalous event early with a manageable number of false alarms is more useful than a model that only performs well on isolated windows.

## 10. Success Criteria

The version 1 experiment is considered successful if it satisfies both methodological and predictive criteria.

### 10.1 Methodological Success Criteria

The pipeline must:

- use only Wind Farm A for the initial scope,
- generate windows strictly within `asset_id + event_id` boundaries,
- use `W = 144`, `H = 72`, and `stride = 6`,
- fit preprocessing transformations only on training data,
- exclude target and metadata columns from model inputs,
- report class balance for train, validation, and test splits,
- report both window-level and event-level metrics,
- and save the locked configuration used for the experiment.

### 10.2 Predictive Success Criteria

The model should:

- achieve better event-level F1-score than simple baseline methods,
- achieve useful event recall for anomalous events,
- provide positive lead time before the future anomaly horizon,
- maintain a false alarm count and duration that are explainable in the final report,
- and show PR-AUC improvement over a class-frequency baseline.

For thesis reporting, the final conclusion should not rely on accuracy alone. The main evidence should combine event recall, event F1-score, PR-AUC, lead time, and false alarm analysis.

## 11. Locked Version 1 Parameters

The locked parameters for version 1 are:

| Item | Value |
| --- | --- |
| Main task | Supervised binary future anomaly/fault prediction |
| Initial wind farm | Wind Farm A |
| Pooling strategy | Vertical pooling within Wind Farm A |
| Sampling interval | 10 minutes |
| Input window `W` | 144 steps |
| Input history duration | 24 hours |
| Prediction horizon `H` | 72 steps |
| Prediction horizon duration | 12 hours |
| Stride | 6 steps |
| Stride duration | 1 hour |
| Window grouping | `asset_id + event_id` |
| Practical processed grouping | `asset_id + sequence_id` when `sequence_id` stores event identity |
| Model output | `P(y[t,H] = 1)` |
| Positive label | At least one future anomalous timestamp in `[t+1, t+H]` |

The machine-readable version of this configuration is stored in `configs/problem_v1.yaml`.
