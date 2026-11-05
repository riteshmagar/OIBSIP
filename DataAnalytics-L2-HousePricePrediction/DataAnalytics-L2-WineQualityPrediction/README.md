# Wine Quality Prediction

**Oasis Infobyte SIP · Data Analytics · Level 2 · Task 2**

Predicts wine quality from physicochemical properties (acidity, sugar, density, alcohol, etc.) and compares three classifiers: **Random Forest**, **SGD Classifier** and **Support Vector Classifier (SVC)**.

## Dataset
Wine Quality dataset, free from either source:
- UCI: https://archive.ics.uci.edu/dataset/186/wine+quality
- Kaggle: search "Wine Quality" (e.g. `WineQT.csv`)

Download the CSV, place it in this folder, and set `DATA_PATH` in the notebook (default: `winequality-red.csv`). The loader auto-detects `;` or `,` separators.

## What the notebook covers
1. Data loading, structure check, missing values, duplicates, class distribution
2. EDA: feature distributions, correlation heatmap, box plots by quality
3. Class imbalance discussion
4. Feature engineering: quality binned into 3 classes (low ≤5, medium 6, high ≥7), with justification
5. Stratified 80/20 train/test split
6. Models: Random Forest, SGD, SVC (scaling via pipeline; `class_weight="balanced"`)
7. Evaluation: accuracy, classification report, confusion matrix for each model
8. Random Forest feature importance chart
9. Side-by-side comparison table and chart
10. Conclusion on the most suitable model for deployment

## Run it
```bash
pip install -r requirements.txt
jupyter notebook wine_quality_prediction.ipynb
```

## Results
Add your final table here after running the notebook (accuracy, macro F1, weighted F1 per model) and a screenshot of the comparison chart.

## Tech stack
Python, pandas, numpy, scikit-learn, seaborn, matplotlib, Jupyter Notebook
