# Unveiling the Android App Market (Google Play Store Analysis)
**Oasis Infobyte SIP · Data Analytics · Level 2 · Task 4**

End-to-end analysis of the Google Play Store ecosystem: cleaning messy real-world data, exploring categories, ratings, size/installs and pricing, and running sentiment analysis on user reviews.

## Datasets
Kaggle **Google Play Store Apps**. Download both files and place them in this folder (not committed):
- `googleplaystore.csv`
- `googleplaystore_user_reviews.csv`

## What the notebook covers
- Loading the apps and reviews datasets separately
- Cleaning: `Installs` ("10,000+"), `Price` ("$4.99"), `Size` ("19M"/"201k"), `Reviews` to numeric; nulls and duplicates handled
- Category analysis (most saturated categories)
- Ratings distribution and average rating by category
- Size vs. installs scatter plot and correlation
- Paid vs. free, price distribution, revenue estimate by category
- Sentiment analysis of reviews with VADER (TextBlob fallback)
- Sentiment by category
- Optional interactive Plotly chart (saved to `images/` as HTML)
- Conclusion with 3 data-driven insights for a developer

## How to run
```bash
pip install -r requirements.txt
jupyter notebook playstore_analysis.ipynb   # Run All
```
Running the notebook regenerates all charts into `images/`.

## Charts (`images/`)
| File | Shows |
|---|---|
| 01_category_distribution.png | Apps per category |
| 02_ratings_analysis.png | Rating distribution and by category |
| 03_size_vs_installs.png | Size vs. installs |
| 04_pricing_analysis.png | Free/paid, price distribution, revenue estimate |
| 05_sentiment_analysis.png | Overall and per-category sentiment |

## Key insights
Fill in from your run on the real data.

> **Note:** The PNGs currently in `images/` were generated from a *synthetic stand-in* dataset (the notebook falls back to one when the CSVs are missing), and a simple keyword scorer instead of VADER. Re-run with the real files and VADER installed before submitting.
