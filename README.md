# Semiconductor Chip Defect Detection Using Machine Learning

An end-to-end Machine Learning pipeline built to predict whether a semiconductor chip is defective based on manufacturing process telemetry data. This project successfully tackles extreme class imbalance to catch critical failures before they leave the fabrication floor.

## 📊 Dataset Overview
The project utilizes the **UCI SECOM dataset**, containing telemetry configurations from a semiconductor manufacturing process:
* **Total Samples:** 1,567 chips
* **Total Raw Features:** 592 sensor channels
* **Class Distribution:** * ` -1 ` (Pass / Normal): **1,463 chips**
  * `  1 ` (Fail / Defect): **104 chips** (~6.6% defect rate)

---

## The Challenge: Extreme Imbalance
Standard machine learning models completely fail on this dataset. Because 93.4% of the data belongs to the passing class, naive models naturally cheat by predicting `-1` for every single chip. While this yields a high overall accuracy, it results in a **0% Recall for defects**, allowing 100% of broken chips to slip through—a catastrophic scenario in hardware manufacturing.

---

##  The Solution Pipeline

### 1. Advanced Feature Selection & Noise Reduction
With 592 original features and extensive missing data, a strict noise filter was established:
* **Null Value Filtering:** Dropped columns containing more than 50% missing data.
* **Imputation:** Applied median imputation to fill remaining missing values securely.
* **Variance Thresholding:** Discarded flat-lined features with near-zero variance (`threshold = 0.01`).
* **Multicollinearity Elimination:** Evaluated a correlation matrix and stripped highly correlated feature pairs (`r > 0.85`) to streamline the dataset shape down to **171 clean features**.

### 2. Imbalance-Aware Modeling
To break through the 0% recall barrier, the final architecture relies on a two-pronged adjustment Strategy:
* **Balanced Sample Weighting:** Used a `RandomForestClassifier` initialized with `class_weight="balanced_subsample"`, forcing every individual decision tree to penalize misclassified defects more severely.
* **Probability Threshold Shifting:** Instead of using the standard 50% classification boundary, we extract raw probabilities via `.predict_proba()` and drop the alert threshold to **25%**. If the system is even 25% suspicious of a flaw, it aggressively flags the chip.

---

##  Production Performance Results

Compared to standard linear and non-linear baseline models, the final optimized Random Forest architecture fundamentally changes the predictive output:

| Model | Normal Class (-1) Precision | Defect Class (1) Recall | Overall Test Accuracy |
| :--- | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 93% | 14% | 82% |
| Support Vector Machine (RBF) | 94% | 14% | 93% |
| **Optimized Random Forest (Final)** | **97%** | **67%** | **78%** |

### Final Optimized Confusion Matrix
```text
               Predicted Pass (-1)    Predicted Defect (1)
Actual Pass           230                     63
Actual Defect           7                     14
