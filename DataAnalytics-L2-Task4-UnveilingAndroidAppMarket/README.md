# OIBSIP Data Analytics — Level 2 Task 4
## Unveiling the Android App Market (Google Play Store Analysis)

### Tech Stack
Python, pandas, NumPy, matplotlib, seaborn, TextBlob, Jupyter Notebook, Plotly (optional).

### Dataset
Download Google Play Store Apps + User Reviews from:
https://www.kaggle.com/datasets/lava18/google-play-store-apps

Put the full files into `data/` as:
- `googleplaystore.csv`
- `googleplaystore_user_reviews.csv`

Included demo CSVs are small illustrative examples only. Notebook automatically uses full datasets when available.

### Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open `Unveiling_Android_App_Market.ipynb` and Run All.

### Features
- Data inspection, null/duplicate handling and type cleaning
- Category saturation charts
- Rating distribution and average rating by category
- App size vs installs and correlation
- Free/paid distribution, price stats and approximate revenue by category
- TextBlob sentiment labels and sentiment by category
- Optional Plotly chart
- Three insight prompts and limitations

**Note:** Never present demo-data outputs as actual Play Store findings. Run with full Kaggle data and add measured numeric insights.