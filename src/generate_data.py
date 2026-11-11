"""Steps 1-2: synthetic dataset (3,000 records) and labels. Seed 42.
Dependencies follow Section 3.4.2: vendor -> service & downtime; volume -> utilisation;
LCC grows with acquisition cost, downtime, energy and installation complexity."""
import numpy as np, pandas as pd
from common import *

def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    cat = np.concatenate([[k]*n for k,(p,n) in CATEGORIES.items()])
    N = len(cat); assert N==3000
    bench = np.array([CATEGORIES[c][0] for c in cat])
    z = lambda: rng.normal(size=N)

    zv = z()                                             # vendor reliability
    vendor = np.clip(7.0+1.54*zv,1,10)
    zacq = .14*zv+np.sqrt(1-.14**2)*z()
    acq = np.clip(1.01+.19*zacq,.6,1.7)
    clin = np.clip(71.8+17.0*z(),10,100)
    zvol = z(); vol = np.clip(np.exp(np.log(405)+.60*zvol),48,2000)
    zu = .68*zvol+np.sqrt(1-.68**2)*z()
    util = np.clip(60.4+10.96*zu,20,98)
    zs = .70*zv+.02*zacq+np.sqrt(1-.70**2-.02**2)*z()
    service = np.clip(75.3+13.8*zs,20,100)
    zd = -.14*zv-.12*zs*0+np.sqrt(1-.14**2)*z()
    down = np.clip(13.2+9.3*zd,1,40)
    zi = z(); install = np.clip(4.95+2.0*zi,1,10)
    ze = z(); energy = np.clip(1.03+.26*ze,.5,2)
    zd_c=(down-13.2)/9.3
    raw = .72*zacq+.61*zd_c+.11*zi+.14*ze+np.sqrt(.078)*z()
    lcc = np.clip(1.92+.50*raw/raw.std(),.72,4.08)
    reg = rng.choice([0,1,2],N,p=[.06,.16,.78])

    df = pd.DataFrame(dict(category=cat,benchmark_lakh=bench,acq_cost=acq,clin_match=clin,
        patient_vol=vol,utilisation=util,downtime=down,lcc=lcc,vendor_rel=vendor,
        service=service,install=install,energy=energy,regulatory=reg))
    return df, rng

def label(df, rng):
    S = suitability(df, rng.normal(0,3,len(df)))
    q28,q70 = np.quantile(S,[.28,.70])               # lowest 28 %, next 42 %, top 30 %
    y = np.where(S<q28,0,np.where(S<q70,1,2))
    y[df.regulatory.values==0] = 0                   # hard regulatory gate
    return S,y

if __name__=="__main__":
    df,rng = generate(); S,y = label(df,rng)
    df["suitability"]=S; df["label"]=y
    df.to_csv("../data/synthetic_procurement.csv",index=False)
    print(df.groupby("category",sort=False).size().to_dict()); print("labels",np.bincount(y),"(report: 924 1190 886)")
