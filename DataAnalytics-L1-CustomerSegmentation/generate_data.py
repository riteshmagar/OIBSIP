"""Generates a synthetic e-commerce transaction dataset (schema mirrors the UCI 'Online Retail' dataset).
Replace online_retail.csv with the real UCI/Kaggle file if you prefer - the notebook uses the same columns."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
products = [("P%03d"%i, n, p) for i,(n,p) in enumerate([
 ("Ceramic Mug",4.5),("Scented Candle",8.0),("Notebook Set",6.5),("Desk Lamp",24.0),("Throw Pillow",12.0),
 ("Wall Clock",18.0),("Photo Frame",9.5),("Tea Towel Pack",5.0),("Storage Basket",15.0),("Water Bottle",11.0),
 ("Plant Pot",7.5),("Wooden Tray",13.0),("Glass Jar Set",10.0),("Table Runner",14.0),("Gift Wrap Roll",3.5),
 ("Door Mat",16.0),("Kitchen Timer",6.0),("Tote Bag",8.5),("Wine Glasses (6)",22.0),("Bath Towel",19.0)])]
countries = ["United Kingdom"]*8 + ["Germany","France","Spain","Netherlands"]
# archetype: (share, n_orders range, recency-days range, basket size mean)
arch = {"champion":(0.15,(18,40),(1,30),9), "loyal":(0.25,(8,18),(10,90),6),
        "at_risk":(0.25,(5,12),(150,300),5), "new_low":(0.35,(1,3),(5,330),3)}
end = pd.Timestamp("2023-12-31"); start = pd.Timestamp("2023-01-01")
rows=[]; inv=10000; cid=12000
for name,(share,(lo,hi),(rlo,rhi),bs) in arch.items():
    for _ in range(int(500*share)):
        cid+=1; country=rng.choice(countries)
        n=int(rng.integers(lo,hi+1)); last=end-pd.Timedelta(days=int(rng.integers(rlo,rhi+1)))
        span=max((last-start).days,1)
        days=np.sort(rng.integers(0,span+1,size=n)); days[-1]=span
        for d in days:
            inv+=1; ts=start+pd.Timedelta(days=int(d),hours=int(rng.integers(8,19)),minutes=int(rng.integers(0,60)))
            for k in rng.choice(len(products),size=max(1,int(rng.poisson(bs))),replace=True):
                code,desc,price=products[k]
                q=int(rng.choice([1,2,3,4,6,12],p=[.3,.25,.2,.1,.1,.05]))
                rows.append([f"{inv}",code,desc,q,ts,round(price*rng.uniform(.95,1.05),2),cid,country])
df=pd.DataFrame(rows,columns=["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"])
# realistic messiness: cancellations, missing customer ids, duplicates
c=df.sample(frac=0.02,random_state=1).copy(); c["InvoiceNo"]="C"+c["InvoiceNo"]; c["Quantity"]=-c["Quantity"]
g=df.sample(frac=0.04,random_state=2).copy(); g["CustomerID"]=np.nan
df.loc[g.index,"CustomerID"]=np.nan
d=df.sample(frac=0.01,random_state=3)
df=pd.concat([df,c,d]).sort_values("InvoiceDate").reset_index(drop=True)
df["InvoiceDate"]=df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
df.to_csv("online_retail.csv",index=False); print(df.shape, df.CustomerID.nunique())
