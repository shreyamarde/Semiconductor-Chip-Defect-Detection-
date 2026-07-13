# Semiconductor Chip Defect Detection & Predictive Yield-to-Demand Optimization

An end-to-end Machine Learning pipeline built to predict whether a semiconductor chip is defective based on manufacturing process telemetry data. This project successfully tackles extreme class imbalance to catch critical failures before they leave the fabrication floor, and integrates a dynamic yield-compensation engine that updates production quotas based on real-time global industry risk factors.

## 📊 Dataset Overview
The project utilizes the UCI SECOM dataset, containing telemetry configurations from a semiconductor manufacturing process:
* **Total Samples:** 1,567 chips
* **Total Raw Features:** 592 sensor channels
* **Class Distribution:** 
  * `-1` (Pass / Normal): 1,463 chips
  * `1` (Fail / Defect): 104 chips (~6.6% defect rate)

### The Challenge: Extreme Imbalance
Standard machine learning models completely fail on this dataset. Because 93.4% of the data belongs to the passing class, naive models naturally cheat by predicting `-1` for every single chip. While this yields a high overall accuracy, it results in a 0% Recall for defects, allowing 100% of broken chips to slip through—a catastrophic scenario in hardware manufacturing.

---

## 🛠️ The Solution Pipeline

### 1. Advanced Feature Selection & Noise Reduction
With 592 original features and extensive missing data, a strict noise filter was established:
* **Null Value Filtering:** Dropped columns containing more than 50% missing data.
* **Imputation:** Applied median imputation to fill remaining missing values securely.
* **Variance Thresholding:** Discarded flat-lined features with near-zero variance (`threshold = 0.01`).
* **Multicollinearity Elimination:** Evaluated a correlation matrix and stripped highly correlated feature pairs ($r > 0.85$) to streamline the dataset shape down to 171 clean features.

### 2. Imbalance-Aware Modeling
To break through the 0% recall barrier, the final architecture relies on a two-pronged adjustment strategy:
* **Balanced Sample Weighting:** Used a `RandomForestClassifier` initialized with `class_weight="balanced_subsample"`, forcing every individual decision tree to penalize misclassified defects more severely.
* **Probability Threshold Shifting:** Instead of using the standard 50% classification boundary, we extract raw probabilities via `.predict_proba()` and drop the alert threshold to 25%. If the system is even 25% suspicious of a flaw, it aggressively flags the chip.

---

## 📈 Production Performance Results
Compared to standard linear and non-linear baseline models, the final optimized Random Forest architecture fundamentally changes the predictive output:

| Model | Normal Class (-1) Precision | Defect Class (1) Recall | Overall Test Accuracy |
| :--- | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 93% | 33% | 80% |
| Support Vector Machine (RBF) | 94% | 10% | 93% |
| **Optimized Random Forest (Final)** | **97%** | **67%** | **78%** |

🌐 Dynamic Systems Integration: Macro News & Yield OperationsThis architecture bridges the gap between historical telemetry and real-world 2026 industrial operations via two active, production-ready custom cells:

1. Real-Time Macro News ContextThe pipeline executes an automated live query using the NewsAPI to grab real-time global chip testing updates, foundry trends, and supply chain updates. If current tech headlines detect elevated industry validation hurdles or sub-3nm packaging anomalies, the system flags the production environment as high-risk.
2. Predictive Yield Compensation EngineRather than relying on static, traditional yield formulas, the final production phase feeds model metrics and live news indicators into an automated manufacturing buffer calculator.

The Problem: At our optimal 67% recall, an undetected defect escape rate of $2.23\%$ still slips past (False Negatives). Furthermore, if live headlines signal active foundry complications, factory risk compounds.
The Solution: The pipeline dynamically shifts a Safety Overstock Buffer. For example, to guarantee a client order of exactly 100,000 flawless chips, the system automatically interprets current sensor performance alongside live news signals to instruct the fab floor to execute exactly 110,195 wafer 


 FACTORY COMPENSATION REPORT (INTEGRATING LIVE NEWS)
 Target Quota Ordered by Client:   100,000 units
 Baseline Fabricator Pure Yield:   93.31%
 Adjusted Defect Escape Margin:    2.56% (News Adjusted)
 Calculated Operational Net Yield: 90.75%
------------------------------------------------------------
🚀 PRODUCTION ORDER REQUEST TO FAB:  110,195 wafer starts
📦 Required Safety Overstock Buffer: 10,195 units

### Final Optimized Confusion Matrix
```text
               Predicted Pass (-1)    Predicted Defect (1)
Actual Pass           230                     63
Actual Defect           7                     14 
