"""Step 5: tender simulation, 600 tenders (Section 3.6.3, Tables 4.6-4.7). Seed 2026.
Uses the XGBoost model if present, otherwise falls back to the best available model."""
import numpy as np, pandas as pd, joblib
from common import *
import topsis as T
df = pd.read_csv("../data/synthetic_procurement.csv"); sp = joblib.load("../models/split.joblib")
models = joblib.load("../models/all_models.joblib")
name = "XGBoost (proposed)" if "XGBoost (proposed)" in models else "Gradient Boosting"
model = models[name]; print("Model used:",name)
te = df.iloc[sp["ite"]].reset_index(drop=True); Xte = te[FEATURES]
P = model.predict_proba(Xte); te["Sexp"] = P[:,2]+.5*P[:,1]
te["topsis"] = T.topsis_scores(Xte,ref=sp["Xtr"])
te["lcc_lakh"] = te.lcc*te.benchmark_lakh; te["acq_lakh"] = te.acq_cost*te.benchmark_lakh
rng = np.random.default_rng(2026); out=[]
for cat in CATEGORIES:
    pool = te.index[te.category==cat].values
    for _ in range(100):
        idx = rng.choice(pool,5,replace=False); t = te.loc[idx]; best = t.suitability.max()
        picks = {"Lowest quotation":t.acq_cost.idxmin(),"TOPSIS (equal wts.)":t.topsis.idxmax(),
                 "Proposed framework":t.Sexp.idxmax()}
        for rule,i in picks.items():
            r = te.loc[i]
            out.append(dict(rule=rule,acq=r.acq_lakh,lcc=r.lcc_lakh,down=r.downtime,
              regret=best-r.suitability,best=int(i==t.suitability.idxmax()),
              notrec=int(r.label==0),high=int(r.label==2),full=int(r.regulatory==2),noncomp=int(r.regulatory==0),
              tender=len(out)//3))
o = pd.DataFrame(out); o.to_csv("../data/tender_results.csv",index=False)
ci = lambda s: 1.96*s.std()/np.sqrt(len(s))
print("\nTable 4.6 (mean over 600 tenders)")
print(o.groupby("rule").agg(acq=("acq","mean"),lcc=("lcc","mean"),down=("down","mean"),regret=("regret","mean"),
      best=("best","mean"),notrec=("notrec","mean"),high=("high","mean"),full=("full","mean"),noncomp=("noncomp","mean")).round(3).T)
print("\nTable 4.7 paired differences: proposed minus baseline (mean +/- 95% CI)")
pv = lambda col: o.pivot(index="tender",columns="rule",values=col)
for col in ["acq","lcc","down","regret"]:
    p = pv(col)
    for b in ["Lowest quotation","TOPSIS (equal wts.)"]:
        d = p["Proposed framework"]-p[b]; print(f"{col:7s} vs {b:20s}: {d.mean():+.2f} +/- {ci(d):.2f}")
