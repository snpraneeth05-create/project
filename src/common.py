"""Shared definitions taken directly from the report (Tables 3.2, 3.3, 3.5)."""
import numpy as np

SEED = 42
FEATURES = ["acq_cost","clin_match","patient_vol","utilisation","downtime",
            "lcc","vendor_rel","service","install","energy","regulatory"]
NICE = {"acq_cost":"Acquisition cost index","clin_match":"Clinical match score",
        "patient_vol":"Patient volume","utilisation":"Utilization rate",
        "downtime":"Maintenance downtime","lcc":"Life-cycle cost index",
        "vendor_rel":"Vendor reliability","service":"Service availability",
        "install":"Installation complexity","energy":"Energy consumption index",
        "regulatory":"Regulatory compliance"}
CLASSES = ["Not Recommended","Conditionally Recommended","Highly Recommended"]

# Table 3.3: category -> (benchmark price INR lakh, records)
CATEGORIES = {"Ventilator":(15.0,557),"Patient monitor":(4.5,579),
              "Ultrasound system":(22.0,503),"Digital X-ray":(45.0,354),
              "Infusion pump":(1.2,584),"Defibrillator":(6.0,423)}
# Table 3.5 (expert weights)
WEIGHTS = {"clin_match":.19,"lcc":.14,"vendor_rel":.11,"service":.10,"downtime":.10,
           "acq_cost":.09,"utilisation":.06,"patient_vol":.05,"install":.05,
           "energy":.04,"regulatory":.07}

def desirability(df):
    """Map each criterion to a 0-1 desirability value c_i (Section 3.4.3)."""
    lin = lambda x,lo,hi: np.clip((x-lo)/(hi-lo),0,1)
    return {
     "acq_cost":1-lin(df.acq_cost,.6,1.7),
     "clin_match":lin(df.clin_match,0,100),
     "patient_vol":lin(df.patient_vol,48,2000),
     "utilisation":np.exp(-((df.utilisation-75)/20)**2),   # best near 75 %
     "downtime":1-lin(df.downtime,1,40),
     "lcc":1-lin(df.lcc,.7,4.1),
     "vendor_rel":lin(df.vendor_rel,1,10),
     "service":lin(df.service,20,100),
     "install":1-lin(df.install,1,10),
     "energy":1-lin(df.energy,.5,2),
     "regulatory":df.regulatory/2.0}

def suitability(df, noise=0.0):
    """S = 100*sum(w_i c_i) + 8(c_clin*c_vend - 0.35) - 15 max(0, .5 - c_clin) + eps"""
    c = desirability(df)
    S = 100*sum(WEIGHTS[k]*c[k] for k in WEIGHTS)
    S = S + 8*(c["clin_match"]*c["vendor_rel"]-0.35) - 15*np.maximum(0,.5-c["clin_match"])
    return S + noise
