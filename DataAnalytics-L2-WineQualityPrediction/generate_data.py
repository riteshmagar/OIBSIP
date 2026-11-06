"""Generates a synthetic red-wine quality dataset with the same columns, typical ranges, correlations and
quality distribution (scores 3-8, heavily imbalanced) as the UCI / Kaggle 'Wine Quality' (red) dataset.
To use the real data, download winequality-red.csv (UCI: archive.ics.uci.edu/dataset/186/wine+quality, or Kaggle)
and save it as winequality.csv (the file uses ';' as separator - the notebook reads either)."""
import numpy as np, pandas as pd
from scipy.stats import norm
rng = np.random.default_rng(3)
N = 1599
cols = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
        "total sulfur dioxide","density","pH","sulphates","alcohol"]
C = np.eye(11)
def setc(a,b,v): C[cols.index(a),cols.index(b)] = C[cols.index(b),cols.index(a)] = v
for a,b,v in [("fixed acidity","citric acid",.67),("fixed acidity","density",.67),("fixed acidity","pH",-.68),
              ("free sulfur dioxide","total sulfur dioxide",.67),("alcohol","density",-.5),("volatile acidity","citric acid",-.55),
              ("citric acid","pH",-.54),("density","residual sugar",.36),("chlorides","sulphates",.37),("fixed acidity","volatile acidity",-.26),
              ("alcohol","chlorides",-.22),("citric acid","sulphates",.31),("density","pH",-.34)]: setc(a,b,v)
w,V = np.linalg.eigh(C); C = V @ np.diag(np.clip(w,1e-3,None)) @ V.T; d = np.sqrt(np.diag(C)); C = C/np.outer(d,d)
Z = rng.multivariate_normal(np.zeros(11), C, N)
U = norm.cdf(Z)
def q(u,lo,hi,skew=0): return lo + (hi-lo)*u**(1+skew)
X = pd.DataFrame({
 "fixed acidity": np.clip(8.3 + 1.7*Z[:,0] + 0.25*Z[:,0]**2, 4.6, 15.9),
 "volatile acidity": np.clip(0.53 + 0.18*Z[:,1], 0.12, 1.58),
 "citric acid": np.clip(0.27 + 0.19*Z[:,2], 0.0, 1.0),
 "residual sugar": np.clip(np.exp(0.75 + 0.45*Z[:,3]), 0.9, 15.5),
 "chlorides": np.clip(np.exp(-2.5 + 0.4*Z[:,4]), 0.012, 0.61),
 "free sulfur dioxide": np.clip(np.exp(2.55 + 0.6*Z[:,5]), 1, 72),
 "total sulfur dioxide": np.clip(np.exp(3.6 + 0.65*Z[:,6]), 6, 289),
 "density": np.clip(0.9967 + 0.0019*Z[:,7], 0.990, 1.004),
 "pH": np.clip(3.31 + 0.154*Z[:,8], 2.74, 4.01),
 "sulphates": np.clip(np.exp(-0.48 + 0.24*Z[:,9]), 0.33, 2.0),
 "alcohol": np.clip(10.4 + 0.95*Z[:,10] + 0.25*np.abs(Z[:,10]), 8.4, 14.9),
})
z = (X - X.mean())/X.std()
lat = (0.85*z["alcohol"] - 0.60*z["volatile acidity"] + 0.40*z["sulphates"] - 0.30*z["total sulfur dioxide"]
       + 0.15*z["citric acid"] - 0.10*z["chlorides"] + rng.normal(0, 1.0, N))
cum = np.cumsum([10,53,681,638,199,18])/N   # UCI red wine counts for quality 3..8
cuts = np.quantile(lat, cum[:-1])
X["quality"] = 3 + np.searchsorted(cuts, lat)
X.round({"fixed acidity":1,"volatile acidity":3,"citric acid":2,"residual sugar":1,"chlorides":3,"free sulfur dioxide":0,
         "total sulfur dioxide":0,"density":5,"pH":2,"sulphates":2,"alcohol":1}).to_csv("winequality.csv", index=False)
print(X.shape, X.quality.value_counts().sort_index().to_dict())
