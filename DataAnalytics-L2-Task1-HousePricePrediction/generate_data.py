"""Generates a synthetic house-price dataset (price in Rs. lakh) with realistic features and some missing values.
To use a real dataset (e.g. Kaggle 'Housing Prices' / 'House Prices: Advanced Regression Techniques'), save it as
house_prices.csv with a numeric 'Price' column and adapt the feature lists in the notebook."""
import numpy as np, pandas as pd
rng = np.random.default_rng(5)
N = 1500
area = np.clip(rng.lognormal(7.2, 0.35, N), 450, 5000).round(0)
beds = np.clip(np.round(area/650 + rng.normal(0, .7, N)), 1, 6).astype(int)
baths = np.clip(np.round(beds*0.7 + rng.normal(0, .6, N)), 1, 5).astype(int)
age = rng.integers(0, 46, N)
parking = rng.choice([0,1,2,3], N, p=[.2,.4,.3,.1])
loc = rng.choice(["City Center","Riverside","Suburb","Outskirts","Industrial Area"], N, p=[.15,.2,.35,.2,.1])
furn = rng.choice(["Unfurnished","Semi-Furnished","Fully-Furnished"], N, p=[.35,.4,.25])
road = rng.choice(["Yes","No"], N, p=[.55,.45])
loc_eff = {"City Center":28,"Riverside":14,"Suburb":0,"Outskirts":-10,"Industrial Area":-18}
furn_eff = {"Unfurnished":0,"Semi-Furnished":3,"Fully-Furnished":7}
price = (0.045*area + np.array([loc_eff[l] for l in loc]) + 4*baths + 2*beds - 0.55*age + 3*parking
         + np.array([furn_eff[f] for f in furn]) + np.where(road=="Yes",5,0) + rng.normal(0, 7, N))
price = np.clip(price, 15, None)  # no unrealistic negative prices
df = pd.DataFrame({"Area_sqft":area,"Bedrooms":beds,"Bathrooms":baths,"Age_years":age,"Parking":parking,
                   "Location":loc,"Furnishing":furn,"Main_Road":road,"Price":price.round(2)})
for col,p in [("Age_years",.05),("Bathrooms",.03),("Furnishing",.04),("Location",.02)]:
    df.loc[df.sample(frac=p, random_state=sum(map(ord,col))).index, col] = np.nan
df.to_csv("house_prices.csv", index=False); print(df.shape); print(df.isnull().sum().to_dict()); print(df.Price.describe().round(1).to_dict())
