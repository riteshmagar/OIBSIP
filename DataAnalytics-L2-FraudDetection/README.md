# OIBSIP — Data Analytics Level 2 Task 3: Fraud Detection

## Objective
Build a machine-learning pipeline to flag fraudulent financial transactions in a highly imbalanced dataset.

## Dataset
`transactions_demo.csv` contains **12,000 synthetic transactions** created for a reproducible portfolio demonstration. It is not real financial data. Fraud rate in this generated dataset: **1.04%**.

For a real analysis, download Kaggle's **Credit Card Fraud Detection** dataset (284,807 transactions; 492 fraud cases as described in the task card) and adapt the columns. Do not claim synthetic results are real-world findings.

## Included
- `Fraud_Detection_Analysis.ipynb` — complete, commented analysis
- `transactions_demo.csv` — sample demo dataset
- `charts/` — PNG visualizations (class imbalance, amount distribution, hour trends, confusion matrices, ROC, model metrics)
- `requirements.txt` — dependencies

## Models
- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced'`)

## Evaluation
Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curve. Accuracy is not used alone because the target class is highly imbalanced.

## Run locally
```bash
python -m pip install -r requirements.txt
jupyter notebook Fraud_Detection_Analysis.ipynb
```
Keep `transactions_demo.csv` in the same directory as the notebook.

## Important metric note
Recall captures the share of fraud cases found; precision captures the share of alerts that are truly fraud. Threshold selection is a business trade-off and should be tuned on validation data.

## GitHub folder
`OIBSIP/DataAnalytics-L2-Task3-FraudDetection/`

## Internship submission checklist (from task guide)
- Push source code, README, and output/screenshots to the `OIBSIP` repository.
- Record a demo video with a 2-second title card: Full Name + Track + Task Title.
- Post/link demo on LinkedIn and tag Oasis Infobyte with `#oasisinfobyte`.
- Peer-review at least two cohort demo videos with meaningful comments.
- Submit the OIBSIP repo link through the assigned submission form.
