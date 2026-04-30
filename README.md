# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using imbalanced-learning techniques, threshold tuning, and model comparison under realistic fraud-detection conditions.

## Objective

The goal of this project is to build a fraud detection pipeline that can identify rare fraudulent transactions while balancing:

- high fraud recall
- manageable false positives
- reliable evaluation on highly imbalanced data

## Dataset

- Total transactions: **284,807**
- Total features: **31**
- Fraud cases: **492**
- Fraud rate: **0.1727%**
- Duplicate rows found: **1081**

### Data Notes
- The dataset contains anonymized PCA-transformed features: `V1` to `V28`
- Additional fields include:
  - `Time`
  - `Amount`
  - `Class`
- `Class = 1` indicates fraud
- `Class = 0` indicates non-fraud

## Project Workflow

### 1. Data Loading and Cleaning
- Loaded the credit card transaction dataset
- Checked schema, missing values, and duplicates
- Removed missing values
- Inspected class imbalance and data structure

### 2. Exploratory Data Analysis
- Analyzed fraud vs non-fraud class distribution
- Visualized:
  - class imbalance
  - transaction time vs fraud
  - amount vs fraud
  - fraud density using time and log-transformed amount

### 3. Feature Engineering
- Derived time-based features from `Time`
- Created:
  - `hour_of_day`
  - `sin_hour`
  - `cos_hour`
  - `log1p_amount`
- Preserved raw `Amount` for later cost-based interpretation
- Dropped raw `Time` from modeling features

### 4. Data Splitting
Used stratified train-test split to preserve fraud ratio.

- Train fraud cases: **394**
- Test fraud cases: **98**

### 5. Preprocessing
- Applied skewness checks on training data
- Used **PowerTransformer (Yeo-Johnson)** to reduce skew where needed
- Ensured preprocessing was fit only on training data to avoid leakage

### 6. Model Building
Built and compared models under two settings:

#### Imbalanced Training
Models trained on original class distribution.

#### Balanced Training
Used sampling only inside cross-validation folds:
- Random Oversampling
- SMOTE
- ADASYN

### 7. Evaluation Strategy
- Used **Stratified K-Fold Cross-Validation**
- Prioritized:
  - **ROC-AUC**
  - **PR-AUC**
- Selected classification threshold using:
  - **Youden’s J statistic**
  - based on **out-of-fold predictions**

This avoids relying on the default `0.5` threshold, which is often unsuitable for fraud detection.

## Models Explored

- Logistic Regression
- Random Forest
- XGBoost
- Other tuned classifiers under imbalanced and balanced settings

## Best Models

- **Best Imbalanced Model:** `XGBoost`
- **Best Balanced Model:** `SMOTE + RandomForest`
- **Final Selected Model:** `SMOTE + RandomForest`

## Final Results

### Best Balanced Model: SMOTE + RandomForest

#### Train (OOF)
- ROC-AUC: **0.9857**
- Precision: **0.1080**
- Recall: **0.9162**
- F1-Score: **0.1933**

#### Test
- ROC-AUC: **0.9865**
- PR-AUC: **0.8656**
- Precision: **0.106**
- Recall: **0.908**
- F1-Score: **0.190**

### Test Confusion Matrix
- **TN = 56113**
- **FP = 751**
- **FN = 9**
- **TP = 89**

## Key Insights

- The dataset is extremely imbalanced, so accuracy alone is misleading
- High recall is critical because missed fraud is costly
- Balanced training with **SMOTE + RandomForest** gave the best practical result
- Threshold tuning significantly improved fraud detection decisions
- The final model captures most fraud cases while keeping false positives at a manageable level

## Challenges

### 1. Extreme Class Imbalance
Fraud cases are very rare, making model learning difficult.

### 2. Risk of Data Leakage
Oversampling and preprocessing must not be applied before splitting or outside CV folds.

### 3. Threshold Selection
A default threshold of `0.5` is not suitable for fraud detection.

### 4. Precision vs Recall Trade-off
Improving recall can increase false alerts and investigation workload.

### 5. Training Cost
Multiple samplers, models, and hyperparameter searches increase runtime.

## Mitigations Applied

- Stratified split and Stratified K-Fold
- Leakage-safe preprocessing
- Resampling only inside CV folds
- ROC-AUC and PR-AUC based model selection
- Youden’s J based threshold tuning
- Result persistence to reduce retraining cost

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost

## Conclusion

This project built an end-to-end credit card fraud detection pipeline using robust evaluation, imbalance handling, and threshold optimization. Among all approaches tested, **SMOTE + RandomForest** delivered the best overall balance between fraud detection and false positive control, making it the most practical model for real-world fraud screening.

## Future Improvements

- Add cost-sensitive learning
- Use anomaly detection methods for rare-event modeling
- Calibrate predicted probabilities
- Optimize threshold using business cost instead of only statistical criteria
- Deploy as a real-time fraud scoring API
