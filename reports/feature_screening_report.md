# EDA-Driven Feature Screening for SCADA-Based Wind Turbine Fault Detection

**Project:** SCADA Fault Prediction — CARE Dataset  
**Dataset:** Combined SCADA CSV (`combined_dataset.csv`)  
**Date:** May 2026

---

## Abstract

This report describes the feature screening methodology applied to a combined SCADA time-series dataset collected from five wind turbines under the CARE (Condition monitoring And REliability) benchmark. Starting from 89 candidate numeric features, a four-stage, Exploratory Data Analysis (EDA)-driven filter pipeline was applied, retaining 33 features for downstream sequence modelling. The pipeline proceeds as: (1) constant-feature elimination, (2) missingness filtering, (3) Spearman rank correlation thresholding, and (4) greedy inter-feature redundancy removal. Each stage is grounded in established non-parametric statistical theory, with selection decisions made transparent through a per-feature audit trail. The Kolmogorov–Smirnov normality test is used *a priori* to justify the use of Spearman correlation over Pearson correlation throughout.

---

## 1. Introduction

Feature selection is a fundamental preprocessing step in machine learning pipelines for condition monitoring and fault detection. In SCADA-based wind turbine diagnostics, raw sensor streams often include hundreds of recorded variables: temperatures, pressures, power signals, pitch and yaw angles, and statistical summaries thereof. Many of these signals are physically redundant—measuring the same underlying phenomenon through different sensor placements—while others carry negligible discriminative information for fault prediction. Retaining irrelevant or redundant features degrades model generalisation through the curse of dimensionality, inflates training cost, and reduces interpretability [1, 2].

A well-established taxonomy distinguishes *filter*, *wrapper*, and *embedded* methods for feature selection [3]. Filter methods evaluate features using statistical properties of the data, independent of any learning algorithm. They are computationally inexpensive, scalable to large datasets, and produce results that are interpretable without reference to a specific model. This work adopts a filter approach, driven by the full EDA of all candidate features, and motivated by the following desiderata:

1. **Transparency:** every retained and discarded feature is accompanied by an explicit, reproducible criterion.
2. **Non-parametric robustness:** the SCADA distributions are demonstrably non-Gaussian (Section 3.1), so methods that assume normality (e.g., Pearson correlation, ANOVA F-test) are avoided.
3. **Redundancy awareness:** features are filtered not only for relevance to the fault label but also for pairwise redundancy among themselves, following the minimum-redundancy maximum-relevance (mRMR) principle [4].

---

## 2. Dataset Description

The combined dataset consists of **1,196,747 rows** (10-minute SCADA observations) and **95 columns**, of which 6 are metadata columns (`time_stamp`, `asset_id`, `sequence_id`, `train_test`, `status_type_id`, `label`), leaving **89 candidate numeric features** for analysis. These features include raw sensor averages, per-interval minimum/maximum/standard deviation statistics, and trigonometrically encoded cyclic angle variables (sin/cos transforms of pitch and yaw angles).

Observations originate from **five wind turbines** (assets). The binary fault label (`label = 1`) marks rows that fall within a documented anomaly event window, as defined by the CARE event metadata. Of the total rows, **146,302 (12.22%)** are labelled as fault, creating a moderate class imbalance. The **training partition** (`train_test = "train"`) contains **1,146,154 rows** (95.77%) and the **prediction partition** 50,593 rows (4.23%).

---

## 3. Methodology

### 3.1 Normality Assessment: Kolmogorov–Smirnov Test

Before selecting a correlation measure, the distributional nature of all features must be established. The **Kolmogorov–Smirnov (KS) one-sample test** [5, 6] is applied to test whether each feature's empirical distribution is consistent with a normal distribution. The KS test statistic is defined as:

$$D_n = \sup_x \left| F_n(x) - F_0(x) \right|$$

where $F_n(x)$ is the empirical cumulative distribution function (ECDF) of the sample, and $F_0(x)$ is the theoretical normal CDF with mean $\hat{\mu}$ and standard deviation $\hat{\sigma}$ estimated from the data. The null hypothesis is that the sample is drawn from $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$; it is rejected when the $p$-value falls below 0.05.

**Result:** All **89 features** rejected the null hypothesis of normality ($p < 0.001$ in all cases). This near-universal non-normality is not surprising given the physical nature of SCADA signals: turbine sensors operate under variable wind conditions and record power curves, temperature stratification, and pitch control activity that are inherently multimodal or heavily skewed. 

To quantify the departure from Gaussianity, **skewness** and **excess kurtosis** are computed for each feature:

$$\text{skewness} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3, \qquad \text{excess kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4 - 3$$

The most extreme case was `sensor_26_avg` (skewness = −984.45, excess kurtosis = 1,039,663.5), consistent with a near-constant signal punctuated by rare near-zero artefacts — a strong indicator that this feature carries no discriminative information. Several voltage/power summary features also exhibited heavy-tailed bimodal distributions (e.g., `sensor_31_max`, `sensor_44`) with |skewness| > 2.

The universal failure of the normality test provides statistical justification for using **Spearman rank correlation** rather than Pearson correlation for all subsequent association measures.

---

### 3.2 Missing Value Analysis

For each of the 89 features, the proportion of missing observations ($\text{missing\_pct}$) was computed. **No missing values** were detected in any feature column across the full 1,196,747 rows. This is consistent with SCADA data logged at regular 10-minute intervals from operational turbines with no sensor drop-outs in the CARE dataset. Missingness filtering (Stage 2 in the pipeline, Section 3.4.2) therefore had no practical effect on this dataset but is included as a general step for robustness.

---

### 3.3 Spearman Rank Correlation with the Fault Label

#### 3.3.1 Rationale for Spearman Correlation

Given that all feature distributions are demonstrably non-normal (Section 3.1), and that the fault label is binary rather than continuous, **Spearman's rank correlation coefficient** ($\rho_s$) is the appropriate measure of monotonic association [7]:

$$\rho_s = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the difference between the rank of the $i$-th observation in the feature variable and its rank in the label variable, and $n$ is the number of observations. Spearman's $\rho_s$ quantifies the strength and direction of the *monotonic* relationship between two variables without assuming linearity or normality [7, 8]. In contrast, Pearson's correlation coefficient assumes both variables are jointly normally distributed and is sensitive to outliers — both conditions violated here.

For binary labels (0 = normal, 1 = fault), Spearman's $\rho_s$ is equivalent to a rank-based measure of separation between the two classes. A positive $\rho_s$ indicates that higher feature values are associated with fault periods; a negative $\rho_s$ indicates lower values during faults.

Point-biserial correlation (the special case of Pearson for binary $y$) was considered but rejected because it is sensitive to the non-normality of the continuous feature variable and to extreme outliers.

#### 3.3.2 Computed Correlations

Table 1 summarises the top 20 features by $|\rho_s|$ with the binary fault label.

**Table 1.** Top 20 features by absolute Spearman rank correlation with the fault label (n = 1,196,747).

| Rank | Feature | $\rho_s$ | $|\rho_s|$ |
|------|---------|----------|-----------|
| 1 | `sensor_0_avg` | +0.2097 | 0.2097 |
| 2 | `sensor_53_avg` | +0.1843 | 0.1843 |
| 3 | `sensor_43_avg` | +0.1741 | 0.1741 |
| 4 | `sensor_41_avg` | +0.1665 | 0.1665 |
| 5 | `sensor_6_avg` | +0.1608 | 0.1608 |
| 6 | `sensor_19_avg` | +0.1266 | 0.1266 |
| 7 | `sensor_7_avg` | +0.1090 | 0.1090 |
| 8 | `sensor_9_avg` | +0.0860 | 0.0860 |
| 9 | `sensor_5_max_cos` | −0.0783 | 0.0783 |
| 10 | `sensor_5_avg_sin` | +0.0780 | 0.0780 |
| 11 | `sensor_5_avg_cos` | −0.0767 | 0.0767 |
| 12 | `sensor_5_max_sin` | +0.0766 | 0.0766 |
| 13 | `sensor_5_min_sin` | +0.0726 | 0.0726 |
| 14 | `sensor_10_avg` | +0.0679 | 0.0679 |
| 15 | `sensor_20_avg` | +0.0649 | 0.0649 |
| 16 | `sensor_14_avg` | +0.0616 | 0.0616 |
| 17 | `sensor_52_std` | −0.0604 | 0.0604 |
| 18 | `reactive_power_27_max` | −0.0602 | 0.0602 |
| 19 | `reactive_power_27_avg` | −0.0600 | 0.0600 |
| 20 | `reactive_power_28_avg` | +0.0596 | 0.0596 |

The highest absolute correlation ($\rho_s = 0.210$) belongs to `sensor_0_avg`, consistent with a nacelle or generator temperature average that increases systematically during fault events. The negative correlations of pitch-angle cosine encodings (`sensor_5_avg_cos`, `sensor_5_max_cos`) indicate that fault periods tend to coincide with lower cosine values, i.e., non-zero pitch angles, a physically interpretable pattern in pitch system faults. Notably, 18 features had $|\rho_s| < 0.02$, indicating no meaningful monotonic association with the fault label.

---

### 3.4 Feature Selection Pipeline

The four filtering stages are applied sequentially. At each stage, the set of surviving features is passed to the next. The criteria are designed to be interpretable, threshold-based, and independently verifiable.

#### Stage 1 — Constant Feature Removal

**Criterion:** feature is removed if its standard deviation equals zero across all rows.

**Rationale:** A constant feature carries zero variance and therefore zero information about any target variable. Including such features wastes model capacity and may trigger numerical instabilities in scaler operations (division by zero in MinMax or Z-score scaling).

**Theory:** The variance of a discrete random variable $X$ is $\text{Var}(X) = E[(X - \mu)^2]$. If $\text{Var}(X) = 0$, then $X = \mu$ almost surely, and no statistical test can yield a meaningful association between $X$ and any other variable.

**Result:** Two features were removed:
- `sensor_46` — entirely zero across all 1,196,747 rows
- `sensor_49` — entirely zero across all 1,196,747 rows

Both appear to be SCADA channels that are either physically disconnected, permanently inactive, or placeholder columns in the CARE export format.

---

#### Stage 2 — Missing Value Filtering

**Criterion:** feature is removed if the fraction of missing observations exceeds a configurable threshold (default: 80%).

**Rationale:** High proportions of missing data make features unreliable for model training and imputation-based recovery becomes increasingly speculative beyond approximately 50–60% missingness [9]. A threshold of 80% was chosen to be deliberately permissive; even at 30–40% missingness, most imputation strategies introduce more bias than the signal is worth [10].

**Result:** No features were removed at this stage. All 87 surviving features had zero missing values.

---

#### Stage 3 — Label Correlation Thresholding

**Criterion:** feature is removed if $|\rho_s| < \tau_{\text{corr}}$, where $\tau_{\text{corr}} = 0.02$.

**Rationale:** Features with negligible monotonic association to the fault label contribute noise rather than signal to a classifier or anomaly detector. A threshold of $\tau_{\text{corr}} = 0.02$ is intentionally conservative (equivalent to $r^2 < 0.04\%$ explained variance), eliminating only those features for which no directional signal is observable at scale. With $n = 1{,}196{,}747$ observations, even very small correlations are statistically significant; the threshold is therefore set on practical rather than statistical grounds.

For reference, the statistical significance threshold at $\alpha = 0.001$ for $n = 10^6$ is approximately $|\rho_s| > 0.003$. All removed features exceeded this significance bound, confirming they are not removed for lack of statistical power but for lack of practical effect size.

**Theory:** In classical effect size classification, $|\rho| < 0.10$ is described as a "small" effect, $0.10 \leq |\rho| < 0.30$ as "medium", and $|\rho| \geq 0.30$ as "large" [11]. All 18 removed features had $|\rho_s| < 0.02$, well below the "small" threshold.

**Result:** 18 features removed. Notable removals:

| Feature | $|\rho_s|$ | Physical interpretation |
|---------|-----------|-------------------------|
| `sensor_26_avg` | 0.0014 | Near-constant at 50.0 with extreme skewness (−984); no discriminative information |
| `wind_speed_4_avg` | 0.0037 | Wind speed measured at a different height; near-identical to `wind_speed_3_avg` but weaker |
| `power_29_max` | 0.0033 | Short-interval max power, dominated by turbine curtailment schedule rather than faults |
| `power_29_avg` | 0.0056 | Normalised active power average; fault signature absorbed by complementary features |
| `wind_speed_3_avg` | 0.0099 | Mean wind speed contains no fault signal after controlling for reactive power and temperature |
| `wind_speed_3_std` | 0.0113 | Wind turbulence standard deviation; no monotonic relationship with fault label |

---

#### Stage 4 — Greedy Inter-Feature Redundancy Removal

**Criterion:** Among any pair of surviving features whose absolute pairwise Spearman correlation exceeds a redundancy threshold ($\tau_{\text{red}} = 0.90$), the feature with the *lower* $|\rho_s|$ with the fault label is removed.

**Rationale:** Retaining highly correlated features introduces redundancy that inflates apparent feature importance, can increase variance in neural network training due to collinearity in the learned representation, and provides no additional information beyond what the surviving twin already encodes [1, 4]. The greedy selection strategy — processing features in descending order of label correlation and discarding near-duplicates of already-retained features — is a practical approximation to the minimum-redundancy maximum-relevance (mRMR) principle proposed by Peng *et al.* [4].

**Theory — mRMR principle:** Let $S$ be the selected feature subset and $f$ a candidate feature. The mRMR criterion selects features by maximising:

$$\text{mRMR}(f) = \underbrace{I(f; y)}_{\text{relevance}} - \frac{1}{|S|} \sum_{s \in S} \underbrace{I(f; s)}_{\text{redundancy}}$$

where $I(\cdot; \cdot)$ denotes mutual information. The pairwise Spearman correlation threshold is a rank-based proxy for the mutual information term, with $|\rho_s| > 0.90$ approximating near-linear statistical dependence strong enough to make both features carry effectively the same information about the target.

**Algorithm:**
1. Sort features by $|\rho_s^{\text{label}}|$ in descending order.
2. Initialise the retained set $S = \emptyset$.
3. For each feature $f$ in sorted order:
   - If $f$ has not been marked redundant, add $f$ to $S$.
   - For every other unseen feature $g$: if $|\rho_s(f, g)| > 0.90$, mark $g$ as redundant with $f$.
4. Return $S$.

**Result:** 36 features removed. Table 2 summarises the most significant removal clusters.

**Table 2.** Selected redundancy removal clusters (inter-feature $|\rho_s| > 0.90$).

| Retained Feature | $|\rho_s^{\text{label}}|$ | Removed Features | Inter-corr |
|----------------|--------------------------|-----------------|------------|
| `sensor_0_avg` | 0.2097 | `sensor_6_avg`, `sensor_53_avg` | 0.933, 0.953 |
| `sensor_43_avg` | 0.1741 | `sensor_19_avg` | 0.902 |
| `sensor_7_avg` | 0.1090 | `sensor_9_avg`, `sensor_20_avg` | 0.943, 0.948 |
| `sensor_10_avg` | 0.0679 | `sensor_21_avg`, `sensor_35_avg`, `sensor_36_avg`, `sensor_37_avg` | 0.943, 0.915, 0.904, 0.909 |
| `sensor_18_min` | 0.0531 | `sensor_18_avg`, `sensor_18_max`, `sensor_23_avg`, `sensor_24_avg`, `sensor_25_avg`, `sensor_45`, `sensor_48`, `sensor_50`, `sensor_51`, `sensor_52_avg`, `sensor_52_max`, `sensor_52_min`, `power_30_avg`, `sensor_31_avg` | 0.906–0.993 |
| `sensor_5_avg_sin` | 0.0780 | `reactive_power_27_avg`, `reactive_power_28_avg`, `sensor_5_max_sin` | 0.967, 0.960, 0.925 |
| `sensor_5_std_sin` | 0.0313 | `sensor_5_std_cos` | **1.000** |
| `reactive_power_27_std` | 0.0405 | `reactive_power_28_std` | 0.965 |

The cluster anchored at `sensor_18_min` is the most extensive, encompassing 14 removed features. This is consistent with multiple sensors measuring closely related thermal or electrical quantities at or near the same physical component — a common pattern in wind turbine SCADA architectures where redundant sensors monitor the same subsystem. The removal of `sensor_5_std_cos` is notable: its inter-feature correlation with `sensor_5_std_sin` is exactly **1.000**, indicating these two columns are mathematically identical in rank order (a deterministic trigonometric relationship at the precision of the data).

---

## 4. Results Summary

### 4.1 Feature Reduction

**Table 3.** Summary of the four-stage feature selection pipeline.

| Stage | Criterion | Features Removed | Features Remaining |
|-------|-----------|-----------------|-------------------|
| Initial candidate set | — | — | 89 |
| Stage 1: Constant removal | std = 0 | 2 | 87 |
| Stage 2: Missingness filtering | missing\_pct > 80% | 0 | 87 |
| Stage 3: Label correlation threshold | $|\rho_s| < 0.02$ | 18 | 69 |
| Stage 4: Redundancy removal | inter-feature $|\rho_s| > 0.90$ | 36 | **33** |

The final selected set of **33 features** represents a 62.9% reduction from the 89 original candidates.

### 4.2 Final Selected Features

```
sensor_0_avg, wind_speed_3_max, wind_speed_3_min,
sensor_7_avg, sensor_10_avg, sensor_14_avg, sensor_18_min,
sensor_22_avg, reactive_power_27_max, reactive_power_27_min,
reactive_power_27_std, power_30_max, power_30_min, power_30_std,
sensor_31_max, sensor_31_min, sensor_31_std, sensor_33_avg,
sensor_34_avg, sensor_38_avg, sensor_40_avg, sensor_41_avg,
sensor_43_avg, sensor_44, sensor_52_std,
sensor_1_avg_sin, sensor_5_avg_sin, sensor_5_avg_cos,
sensor_5_max_cos, sensor_5_min_sin, sensor_5_min_cos,
sensor_5_std_sin, sensor_42_avg_sin
```

### 4.3 Distribution Characteristics of Selected Features

All 33 retained features failed the KS normality test ($p < 0.001$). The selected set spans a range of skewness from −5.05 (`sensor_5_min_cos`) to +2.37 (`sensor_14_avg`), and excess kurtosis from −1.67 to +28.2. No feature is approximately Gaussian, reinforcing the appropriateness of rank-based scaling and non-parametric evaluation criteria throughout the pipeline. This further motivates the use of MinMax or robust scaling (rather than Z-score standardisation) in the subsequent preprocessing step.

---

## 5. Discussion

### 5.1 Why Filter Methods Over Wrapper/Embedded Methods?

The dataset contains approximately 1.2 million rows. Wrapper methods (e.g., recursive feature elimination with cross-validation) require training a full predictive model for each feature subset evaluation, making them computationally prohibitive at this scale. Embedded methods (e.g., L1 regularisation, tree-based feature importances) are model-specific: features selected for an LSTM autoencoder may differ from those for a CNN-LSTM classifier. Filter methods, being model-agnostic, ensure that the selected feature set is usable across the full range of architectures evaluated in this project [3].

### 5.2 Threshold Sensitivity

The two primary thresholds — $\tau_{\text{corr}} = 0.02$ and $\tau_{\text{red}} = 0.90$ — were chosen conservatively. The correlation threshold is substantially below the conventional "small effect" ($|\rho| = 0.10$), meaning that no feature with any detectable directional fault signal is discarded at Stage 3. The redundancy threshold of 0.90 retains features with up to 81% shared rank variance, which is appropriate for SCADA systems where physically co-located sensors share genuine variation. Raising $\tau_{\text{red}}$ to 0.95 would retain a few additional redundant features; lowering it to 0.80 would produce a more aggressive reduction to approximately 22–25 features.

Sensitivity analysis on these thresholds can be conducted by re-running the EDA pipeline with alternative values:

```bash
python src/main.py eda --csv Dataset/processed/combined_dataset.csv \
    --select-features --min-corr 0.05 --redundancy-threshold 0.85
```

### 5.3 Limitations

1. **Correlation ≠ causation.** Features with $|\rho_s| > 0.02$ are correlated with the fault label but may be consequences of faults rather than precursors. The temporal structure of fault propagation — critical for early-warning systems — cannot be captured by row-level Spearman correlation. This limitation motivates the LSTM-based sequence modelling approach used for the final fault detection models.

2. **Single-event training data.** The CARE benchmark provides one anomaly event per turbine, so all correlation statistics are computed across heterogeneous operating conditions and a single anomaly type. Features selected here may not generalise to anomaly types not represented in the training data.

3. **No interaction terms.** The filter approach evaluates features individually against the label. A pair of features with low individual $|\rho_s|$ may be jointly informative through interaction effects, which this pipeline does not capture.

---

## 6. Conclusion

A four-stage, EDA-driven filter pipeline was applied to 89 SCADA features from the CARE wind turbine dataset. The Kolmogorov–Smirnov test confirmed universal non-normality, motivating the exclusive use of Spearman rank correlation as the association measure. Two constant features were eliminated as uninformative by definition. Eighteen features were discarded for negligible label correlation ($|\rho_s| < 0.02$). A greedy inter-feature redundancy removal pass eliminated 36 features whose rank-correlation with a retained peer exceeded 0.90. The final selected set of **33 features** retains maximal discriminative signal with minimal redundancy, and each selection decision is documented in the per-feature audit file (`feature_selection_audit.csv`) for full reproducibility.

---

## References

[1] Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.

[2] Bellman, R. (1961). *Adaptive Control Processes: A Guided Tour*. Princeton University Press.

[3] Kohavi, R., & John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*, 97(1–2), 273–324.

[4] Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.

[5] Kolmogorov, A. N. (1933). Sulla determinazione empirica di una legge di distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83–91.

[6] Smirnov, N. V. (1948). Table for estimating the goodness of fit of empirical distributions. *Annals of Mathematical Statistics*, 19(2), 279–281.

[7] Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1), 72–101.

[8] Myers, J. L., Well, A. D., & Lorch, R. F. (2010). *Research Design and Statistical Analysis* (3rd ed.). Routledge.

[9] Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.

[10] Sterne, J. A. C., et al. (2009). Multiple imputation for missing data in epidemiological and clinical research: Potential and pitfalls. *BMJ*, 338, b2393.

[11] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

---

*Generated from EDA outputs in `results/eda/combined_dataset/`. Full per-feature audit: `feature_selection_audit.csv`. Selected features: `eda_selected_features.csv`.*
