# 🏠 Predicting House Prices with Linear Regression

**Oasis Infobyte SIP · Data Analytics · Level 2 · Task 1**

A complete, end-to-end regression project: exploring a housing dataset, preparing it for modelling, training a Linear Regression model, and interpreting which features drive house prices.

---

## 📌 Objective
Build and evaluate a Linear Regression model that predicts house prices from features such as area, number of rooms, amenities and location-related flags.

## 📂 Dataset
- **Name:** Housing Prices Dataset (`Housing.csv`), ~545 rows × 13 columns
- **Source:** Kaggle (search "Housing Prices Dataset" / "Housing.csv")
- **Target:** `price`
- **Features:** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`
- **Note:** The dataset has no explicit *location* or *age* column. `prefarea` and `mainroad` serve as location proxies.

> The dataset file is not committed. Download it from Kaggle and place it at `data/Housing.csv`.

## 🧰 Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## ▶️ How to Run
```bash
git clone https://github.com/<your-username>/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-HousePricePrediction
pip install -r requirements.txt
# place Housing.csv inside the data/ folder
jupyter notebook house_price_prediction.ipynb
```
Then run **Kernel → Restart & Run All**.

## 🔬 Methodology
| Step | What was done |
|---|---|
| 1. EDA | Shape, dtypes, null/duplicate checks, descriptive statistics, price distribution, feature plots |
| 2. Feature selection | Reasoned discussion of expected predictors (area, bathrooms, amenities, location proxies) |
| 3. Cleaning & encoding | Median/mode imputation (if needed), duplicate removal, outlier check (IQR, retained), yes/no → 1/0, One-Hot Encoding for `furnishingstatus` |
| 4. Correlation | Heatmap + ranking of features by correlation with price |
| 5. Split | 80/20 train/test (`random_state=42`) |
| 6. Model | `sklearn.linear_model.LinearRegression` |
| 7. Evaluation | MSE, RMSE, MAE, R² on train and test sets |
| 8. Diagnostics | Actual vs. predicted scatter, residual plot, residual histogram |
| 9. Interpretation | Raw and standardised coefficients |
| 10. Bonus | Linear vs. Ridge vs. Lasso comparison |

## 📊 Results
> Fill this table with the numbers from your notebook output.

| Metric | Train | Test |
|---|---|---|
| MSE | | |
| RMSE | | |
| R² | | |

**Top price drivers:** _(from the standardised coefficient chart)_

**Key insights:**
1. _…_
2. _…_
3. _…_

## 🖼️ Screenshots
Add images to the `screenshots/` folder and reference them here:
```
![Correlation heatmap](screenshots/heatmap.png)
![Actual vs Predicted](screenshots/actual_vs_predicted.png)
![Residuals](screenshots/residuals.png)
```

## ⚠️ Limitations
- Small dataset; results may vary with a different split.
- Linear Regression assumes linear relationships; coefficients show association, not causation.
- Correlated features (multicollinearity) can distort individual coefficients.
- Prices are right-skewed; modelling `log(price)` may improve fit.

## 🚀 Future Work
Log-transform the target, add interaction terms, and try Random Forest / Gradient Boosting.

## 📁 Structure
```
OIBSIP/
└── DataAnalytics-L2-HousePricePrediction/
    ├── house_price_prediction.ipynb
    ├── README.md
    ├── requirements.txt
    ├── data/            # put Housing.csv here
    └── screenshots/
```

---
*Submitted as part of the Oasis Infobyte Student Internship Program (#oasisinfobyte).*
