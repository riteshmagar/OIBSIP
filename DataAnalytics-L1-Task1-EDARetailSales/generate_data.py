"""Generates a synthetic retail sales dataset (schema mirrors the Kaggle 'Retail Sales Dataset').
Replace retail_sales_dataset.csv with a real Kaggle file if you prefer - the notebook adapts."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 1000
products = {
 "Beauty": [("Face Cream",35),("Lipstick",25),("Shampoo",15),("Perfume",120)],
 "Clothing": [("T-Shirt",20),("Jeans",60),("Jacket",150),("Sneakers",90)],
 "Electronics": [("Headphones",80),("Smartwatch",250),("Phone Case",12),("Power Bank",40)],
}
cat_p = {"Beauty":0.30,"Clothing":0.34,"Electronics":0.36}
dates = pd.date_range("2023-01-01","2023-12-31")
w = np.array([1.0+0.35*(d.month in (11,12))+0.15*(d.month==5)+0.1*(d.dayofweek>=5) for d in dates]); w/=w.sum()
rows=[]
for i in range(1,N+1):
    cat = rng.choice(list(cat_p), p=list(cat_p.values()))
    name, price = products[cat][rng.integers(0,4)]
    gender = rng.choice(["Male","Female"], p=[0.49,0.51])
    age = int(np.clip(rng.normal(41,13),18,64))
    qty = int(rng.choice([1,2,3,4],p=[0.25,0.25,0.25,0.25]))
    rows.append([i, rng.choice(dates,p=w), f"CUST{rng.integers(1,901):03d}", gender, age, cat, name, qty, price, qty*price])
df = pd.DataFrame(rows, columns=["Transaction ID","Date","Customer ID","Gender","Age","Product Category","Product Name","Quantity","Price per Unit","Total Amount"])
df["Date"]=pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df.sort_values("Date").reset_index(drop=True).to_csv("retail_sales_dataset.csv",index=False)
print(df.shape)
