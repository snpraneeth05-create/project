"""TOPSIS baseline (Section 3.5.4). Utilisation treated as a benefit criterion."""
import numpy as np
from common import FEATURES, WEIGHTS
BENEFIT = {"clin_match","patient_vol","utilisation","vendor_rel","service","regulatory"}
COST    = {"acq_cost","downtime","lcc","install","energy"}

def topsis_scores(X, weights=None, ref=None):
    """X: DataFrame with FEATURES. weights: dict (default equal 1/11).
    ref: DataFrame used for normalisation norms (keeps train/test consistent)."""
    ref = X if ref is None else ref
    w = np.array([1/len(FEATURES) if weights is None else weights[f] for f in FEATURES])
    A = X[FEATURES].values.astype(float); R = ref[FEATURES].values.astype(float)
    V = A/np.sqrt((R**2).sum(0))*w
    Vr = R/np.sqrt((R**2).sum(0))*w
    ben = np.array([f in BENEFIT for f in FEATURES])
    ideal = np.where(ben,Vr.max(0),Vr.min(0)); anti = np.where(ben,Vr.min(0),Vr.max(0))
    dp = np.sqrt(((V-ideal)**2).sum(1)); dm = np.sqrt(((V-anti)**2).sum(1))
    return dm/(dp+dm)

def classify(train_scores, ytr, test_scores):
    """Cut closeness at the training class proportions."""
    p = np.bincount(ytr,minlength=3)/len(ytr)
    c1,c2 = np.quantile(train_scores,[p[0],p[0]+p[1]])
    return np.where(test_scores<c1,0,np.where(test_scores<c2,1,2))
