# Fraud Detection (Imbalanced Data)
**Oasis Infobyte SIP · Data Analytics · Level 2 · Task 3**

A machine learning pipeline that detects fraudulent card transactions where fraud is extremely rare (about 0.17% in the real dataset), with class imbalance as the central challenge.

## Dataset
Kaggle **Credit Card Fraud Detection** (284,807 transactions, 492 frauds). Download `creditcard.csv` from Kaggle and place it in this folder. It is not committed because of its size.

## What the notebook covers
- Class imbalance analysis (fraud percentage)
- EDA: amount distribution for fraud vs. non-fraud, and time-of-day analysis
- Why accuracy is misleading on imbalanced data (an "always legitimate" baseline)
- Imbalance handling: `class_weight='balanced'` (SMOTE optional)
- Stratified train/test split
- Models: Logistic Regression and Random Forest
- Evaluation: precision, recall, F1, AUC-ROC, PR-AUC, confusion matrices
- Recall vs. precision trade-off discussion
- Feature importance and coefficients
- Scalability discussion for 1 million transactions per hour

## How to run
```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb   # Run All
```
Running the notebook regenerates every chart into `images/`.

## Charts (`images/`)
| File | Shows |
|---|---|
| 01_class_distribution.png | Fraud vs. legitimate counts |
| 02_amount_distribution.png | Amount by class |
| 03_time_of_day_analysis.png | Volume and fraud rate by hour |
| 04_confusion_matrices.png | Both models |
| 05_roc_curve.png | ROC / AUC |
| 06_precision_recall_curve.png | PR curve |
| 07_model_comparison.png | Metric comparison |
| 08_feature_importance.png | RF importances and LR coefficients |

## Results
Fill in from your run on the real dataset: best model, recall, precision, F1, AUC-ROC.

> **Note:** The PNGs currently in `images/` were generated from a *synthetic stand-in* dataset, because the notebook falls back to one when `creditcard.csv` is missing. Replace them by re-running on the real data before submitting.
