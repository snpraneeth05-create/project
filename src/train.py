"""Step 3: split, grid-search, evaluate (Tables 3.6, 4.1, 4.2; Figs 4.3-4.5)."""
import numpy as np, pandas as pd, joblib, json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
from common import *
import topsis as T
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

df = pd.read_csv("../data/synthetic_procurement.csv")
X, y = df[FEATURES], df["label"].values
Xtr,Xte,ytr,yte,itr,ite = train_test_split(X,y,np.arange(len(df)),test_size=.25,
                                           stratify=y,random_state=SEED)
cv = StratifiedKFold(5,shuffle=True,random_state=SEED)

grids = {
 "Logistic Regression": (make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),
                         {"logisticregression__C":[.1,1,10]}),
 "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                   {"max_depth":[4,6,8],"min_samples_leaf":[5,10,20]}),
 "Random Forest": (RandomForestClassifier(random_state=SEED,n_jobs=-1),
                   {"n_estimators":[200,400],"max_depth":[8,12,None],"min_samples_leaf":[1,3]}),
 "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                   {"n_estimators":[150,300],"learning_rate":[.05,.1],"max_depth":[2,3]}),
}
if XGBClassifier:
    grids["XGBoost (proposed)"] = (XGBClassifier(objective="multi:softprob",random_state=SEED,
                   eval_metric="mlogloss",n_jobs=-1),
                   {"n_estimators":[150,300],"learning_rate":[.05,.1],"max_depth":[2,3,4]})
else:
    print("!! xgboost not installed - XGBoost row skipped. Run `pip install xgboost` and re-run.")

rows, fitted = [], {}
def metrics(name, pred, proba=None, cvf1=None, cvsd=None):
    p,r,f,_ = precision_recall_fscore_support(yte,pred,average="macro",zero_division=0)
    rows.append(dict(Method=name,CV_macroF1=cvf1,CV_sd=cvsd,Accuracy=accuracy_score(yte,pred),
        Precision=p,Recall=r,MacroF1=f,
        AUC=None if proba is None else roc_auc_score(yte,proba,multi_class="ovr",average="macro")))

# TOPSIS baselines
s_tr_eq = T.topsis_scores(Xtr); s_te_eq = T.topsis_scores(Xte,ref=Xtr)
metrics("TOPSIS (equal weights)", T.classify(s_tr_eq,ytr,s_te_eq))
s_tr_ex = T.topsis_scores(Xtr,WEIGHTS); s_te_ex = T.topsis_scores(Xte,WEIGHTS,ref=Xtr)
metrics("TOPSIS (expert weights)", T.classify(s_tr_ex,ytr,s_te_ex))
pred_topsis_eq = T.classify(s_tr_eq,ytr,s_te_eq)

for name,(est,grid) in grids.items():
    gs = GridSearchCV(est,grid,cv=cv,scoring="f1_macro",n_jobs=-1).fit(Xtr,ytr)
    best = gs.best_estimator_; fitted[name]=best
    i = gs.best_index_
    print(name, gs.best_params_)
    metrics(name,best.predict(Xte),best.predict_proba(Xte),
            gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i])

res = pd.DataFrame(rows); res.to_csv("../data/table_4_1_results.csv",index=False)
print(res.round(3).to_string(index=False))

dep = fitted.get("XGBoost (proposed)")
if dep is not None:
    pred = dep.predict(Xte)
    print(classification_report(yte,pred,target_names=CLASSES,digits=3))
    print("XGBoost confusion:\n",confusion_matrix(yte,pred))
    print("TOPSIS(eq) confusion:\n",confusion_matrix(yte,pred_topsis_eq),
          "\nsevere errors TOPSIS:",((yte==0)&(pred_topsis_eq==2)).sum()+((yte==2)&(pred_topsis_eq==0)).sum())
    joblib.dump(dep,"../models/xgb_model.joblib")
joblib.dump(dict(Xtr=Xtr,Xte=Xte,ytr=ytr,yte=yte,itr=itr,ite=ite),"../models/split.joblib")
joblib.dump(fitted,"../models/all_models.joblib")
