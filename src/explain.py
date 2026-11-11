"""Step 4: TreeSHAP explanations + four quality checks (Sections 3.5.5, 3.6.2, 4.4-4.7).
Needs: pip install xgboost shap. Produces Table 4.3, Figs 4.6-4.13."""
import numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from common import *

df = pd.read_csv("../data/synthetic_procurement.csv")
sp = joblib.load("../models/split.joblib"); xgb = joblib.load("../models/xgb_model.joblib")
Xtr,Xte,ytr,yte = sp["Xtr"],sp["Xte"],sp["ytr"],sp["yte"]
names = [NICE[f] for f in FEATURES]

def shap_3d(model,X):
    """Return (explainer, array n x features x classes) for any SHAP version."""
    ex = shap.TreeExplainer(model); sv = ex.shap_values(X)
    sv = np.stack(sv,axis=-1) if isinstance(sv,list) else np.asarray(sv)
    return ex, sv
ex, sv = shap_3d(xgb,Xte)
base = np.atleast_1d(ex.expected_value)

# ---- Global importance (Table 4.3, Fig 4.6)
by_class = np.abs(sv).mean(0)                       # features x classes
total = by_class.sum(1); share = total/total.sum()
t43 = pd.DataFrame(by_class,index=names,columns=["Not Rec.","Cond. Rec.","Highly Rec."])
t43.insert(0,"Share of total",share); t43 = t43.sort_values("Share of total",ascending=False)
t43.round(3).to_csv("../data/table_4_3.csv"); print(t43.round(3))
o = np.argsort(total)
plt.figure(figsize=(8,5)); left=np.zeros(len(o))
for c,col in zip([2,1,0],["#2979d8","#1db07a","#ec6a37"]):
    plt.barh(np.array(names)[o],by_class[o,c],left=left,color=col,label=CLASSES[c]); left+=by_class[o,c]
plt.xlabel("Mean |SHAP value| (log-odds), summed over classes"); plt.legend(); plt.tight_layout()
plt.savefig("../figures/fig4_6_global_importance.png",dpi=200); plt.close()

# ---- Beeswarm (Fig 4.7) and dependence (Fig 4.8) for Highly Recommended
shap.summary_plot(sv[:,:,2],Xte.values,feature_names=names,show=False)
plt.title("SHAP impact on log-odds of 'Highly Recommended'"); plt.tight_layout()
plt.savefig("../figures/fig4_7_beeswarm.png",dpi=200,bbox_inches="tight"); plt.close()
fig,ax = plt.subplots(1,3,figsize=(13,3.6))
for a,f in zip(ax,["clin_match","lcc","downtime"]):
    i = FEATURES.index(f); a.scatter(Xte[f],sv[:,i,2],s=6,alpha=.5)
    a.axhline(0,c="grey",lw=.6); a.set_xlabel(NICE[f]); a.set_ylabel("SHAP ('Highly Recommended')")
plt.tight_layout(); plt.savefig("../figures/fig4_8_dependence.png",dpi=200); plt.close()

# ---- Local explanations (Figs 4.9-4.11): one confident, correct record per class
proba = xgb.predict_proba(Xte); pred = proba.argmax(1)
def waterfall(k,cls,fname,title):
    e = shap.Explanation(values=sv[k,:,cls],base_values=base[cls],data=Xte.iloc[k].values,feature_names=names)
    shap.plots.waterfall(e,show=False); plt.title(title); plt.savefig(fname,dpi=200,bbox_inches="tight"); plt.close()
for cls,tag in zip([0,1,2],["A","B","C"]):
    cand = np.where((yte==cls)&(pred==cls)&(proba[:,cls]>.9))[0]
    k = cand[0] if len(cand) else np.where((yte==cls)&(pred==cls))[0][0]
    print(f"Candidate {tag}: test row {k}, {CLASSES[cls]}, p={proba[k,cls]:.3f}")
    waterfall(k,cls,f"../figures/fig_local_{tag}.png",f"Candidate {tag}: {CLASSES[cls]}")
    if cls==1: waterfall(k,2,"../figures/fig4_10_contrastive_B.png","Candidate B: contrastive view (towards Highly Rec.)")

# ---- Quality checks
raw = xgb.predict(Xte,output_margin=True)
print("Additivity max error:",np.abs(base[None,:]+sv.sum(1)-raw).max())
pi = permutation_importance(xgb,Xte,yte,n_repeats=10,random_state=SEED,scoring="f1_macro").importances_mean
gen = np.array([WEIGHTS[f] for f in FEATURES])
print("Spearman SHAP vs permutation:",spearmanr(total,pi)[0]," vs generating weights:",spearmanr(total,gen))
# Stability: retrain on seeds 1-5, Kendall tau vs reference ranking
X,y = df[FEATURES],df["label"].values; taus=[]
for s in range(1,6):
    a,b,c,d = train_test_split(X,y,test_size=.25,stratify=y,random_state=s)
    m = XGBClassifier(objective="multi:softprob",random_state=SEED,**{k:v for k,v in xgb.get_params().items()
        if k in ("n_estimators","learning_rate","max_depth")}).fit(a,c)
    _,sv_s = shap_3d(m,b); taus.append(kendalltau(np.abs(sv_s).mean(0).sum(1),total)[0])
print("Kendall tau per seed:",np.round(taus,3),"mean",np.mean(taus))
# Faithfulness: deletion test
rng = np.random.default_rng(SEED); means = Xtr.mean().values; res=[]
sv_pred = np.take_along_axis(sv,pred[:,None,None].repeat(len(FEATURES),1),axis=2)[:,:,0]
for k in (1,2,3,5):
    Xa,Xr = Xte.values.copy(),Xte.values.copy()
    for r in range(len(Xte)):
        Xa[r,np.argsort(-np.abs(sv_pred[r]))[:k]] = means[np.argsort(-np.abs(sv_pred[r]))[:k]]
        j = rng.choice(len(FEATURES),k,replace=False); Xr[r,j] = means[j]
    p0 = proba[np.arange(len(Xte)),pred]
    da = (p0-xgb.predict_proba(pd.DataFrame(Xa,columns=FEATURES))[np.arange(len(Xte)),pred]).mean()
    dr = (p0-xgb.predict_proba(pd.DataFrame(Xr,columns=FEATURES))[np.arange(len(Xte)),pred]).mean()
    res.append((k,da,dr,da/dr)); print(f"k={k}: top-SHAP {da:.3f}  random {dr:.3f}  ratio {da/dr:.1f}x")
pd.DataFrame(res,columns=["k","top_shap","random","ratio"]).to_csv("../data/table_4_5.csv",index=False)
