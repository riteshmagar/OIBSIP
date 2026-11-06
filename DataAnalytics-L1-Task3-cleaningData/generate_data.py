"""Generates a deliberately MESSY customer dataset for the data-cleaning task.
Problems injected: missing values, duplicates, inconsistent categories/formats, wrong dtypes,
invalid values and outliers. To use a real messy dataset (e.g. from Kaggle), save it as
messy_customer_data.csv and adapt the column names in the notebook."""
import numpy as np, pandas as pd
rng = np.random.default_rng(11)
N = 1200
first = ["Aarav","Vivaan","Aditya","Sai","Arjun","Ishaan","Rohan","Kabir","Ananya","Diya","Isha","Meera","Priya","Riya","Saanvi","Neha","Karan","Pooja","Rahul","Sneha"]
last = ["Sharma","Patel","Singh","Magar","Joshi","Verma","Gupta","Kulkarni","Deshmukh","Rao","Nair","Iyer","Khan","Mehta","Shah"]
cities = {"Mumbai":["Mumbai","mumbai","MUMBAI","Bombay","Mumbai "],"Delhi":["Delhi","delhi","New Delhi","DELHI"],
          "Bangalore":["Bangalore","bangalore","Bengaluru","BANGALORE"],"Pune":["Pune","pune","PUNE "],
          "Nagpur":["Nagpur","nagpur","NAGPUR"],"Chennai":["Chennai","chennai","CHENNAI "]}
gm = {"Male":["Male","male","M","MALE","m"," Male"],"Female":["Female","female","F","FEMALE","f","Fem"]}
rows=[]
for i in range(1,N+1):
    fn,ln=rng.choice(first),rng.choice(last)
    g=rng.choice(["Male","Female"]); city=rng.choice(list(cities))
    age=int(np.clip(rng.normal(38,12),18,70)); inc=float(np.round(rng.lognormal(10.9,0.45),0))
    d=pd.Timestamp("2020-01-01")+pd.Timedelta(days=int(rng.integers(0,1460)))
    f=rng.choice(["iso","dmy","long"],p=[.5,.25,.25])
    ds={"iso":d.strftime("%Y-%m-%d"),"dmy":d.strftime("%d/%m/%Y"),"long":d.strftime("%b %d, %Y")}[f]
    inc_s=rng.choice([f"{inc:.0f}",f"{inc:,.0f}",f"Rs. {inc:,.0f}"],p=[.6,.25,.15])
    mail=f"{fn}.{ln}{rng.integers(1,99)}@{rng.choice(['gmail.com','yahoo.com','outlook.com'])}".lower()
    if rng.random()<.3: mail=mail.upper()
    rows.append({"Customer_ID":f"C{i:04d}","Name":f"{fn} {ln}","Gender":rng.choice(gm[g]),"Age":age,
        "Join_Date":ds,"City":rng.choice(cities[city]),"Income":inc_s,
        "Purchase_Amount":round(float(rng.lognormal(6.0,0.8)),2),"Rating":int(rng.choice([1,2,3,4,5],p=[.05,.1,.2,.35,.3])),
        "Email":mail,"Is_Member":rng.choice(["Yes","yes","Y","True","1"] if rng.random()<.5 else ["No","no","N","False","0"])})
df=pd.DataFrame(rows).astype(object)
def blank(col,p):
    idx=df.sample(frac=p,random_state=sum(map(ord,col))).index; df.loc[idx,col]=np.nan
for col,p in [("Gender",.04),("Age",.05),("Join_Date",.03),("City",.03),("Income",.06),("Purchase_Amount",.04),("Rating",.07),("Email",.05),("Is_Member",.02)]: blank(col,p)
# invalid / outlier values
pos=rng.choice(df.index,40,replace=False)
for j,v in zip(pos[:6],[150,200,-5,999,0,121]): df.loc[j,"Age"]=v
for j in pos[6:9]: df.loc[j,"Age"]="N/A"
for j,v in zip(pos[9:17],["5000000","9999999","8,500,000","Rs. 7,200,000","6000000","9500000","4800000","Rs. 12,000,000"]): df.loc[j,"Income"]=v
for j,v in zip(pos[17:20],["-45000","-1200","-80000"]): df.loc[j,"Income"]=v
for j,v in zip(pos[20:24],[0,11,99,-1]): df.loc[j,"Rating"]=v
for j,v in zip(pos[24:32],["john.doe#mail.com","abc@","no-email","priya@@gmail.com","rahul.com","sneha@.com","none","xyz@gmail"]): df.loc[j,"Email"]=v
for j,v in zip(pos[32:36],[48000.0,52000.5,61000.0,75000.0]): df.loc[j,"Purchase_Amount"]=v
# duplicates: exact copies + near-duplicates (case/whitespace differences, same Customer_ID)
exact=df.sample(60,random_state=5)
near=df.sample(40,random_state=6).copy()
near["Name"]=near["Name"].str.upper(); near["City"]=near["City"].astype(str).str.lower().replace("nan",np.nan)
near["Customer_ID"]=near["Customer_ID"].str.lower()
df=pd.concat([df,exact,near]).sample(frac=1,random_state=9).reset_index(drop=True)
df.to_csv("messy_customer_data.csv",index=False); print(df.shape)
