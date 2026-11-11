"""Step 6: decision-support dashboard (FR1-FR7). Run: streamlit run app.py"""
import streamlit as st, pandas as pd, numpy as np, joblib, shap, json, datetime, os
import matplotlib.pyplot as plt
from common import *
st.set_page_config(page_title="XAI Medical Equipment Procurement",layout="wide")
model = joblib.load("../models/xgb_model.joblib"); MODEL_VERSION="xgb_model.joblib"
names=[NICE[f] for f in FEATURES]; LOG="../logs/audit_trail.csv"
st.title("Explainable AI Medical Equipment Procurement")
st.caption("Decision support only: the committee keeps the final decision. Trained on a synthetic dataset.")
up = st.file_uploader("FR1: upload candidates (CSV with the 11 criteria columns)",type="csv")
df = pd.read_csv(up) if up else pd.read_csv("../data/synthetic_procurement.csv").sample(5,random_state=1)[FEATURES]
st.dataframe(df[FEATURES])
P = model.predict_proba(df[FEATURES]); cls = P.argmax(1)
df["Recommendation"]=[CLASSES[c] for c in cls]; df["Expected suitability"]=P[:,2]+.5*P[:,1]  # FR2, FR3
for j,c in enumerate(CLASSES): df[f"P({c})"]=P[:,j]
rank = df.sort_values("Expected suitability",ascending=False)
st.subheader("Ranking (FR2, FR3)"); st.dataframe(rank.drop(columns=FEATURES).round(3))
ex = shap.TreeExplainer(model); sv = ex.shap_values(df[FEATURES])
sv = np.stack(sv,-1) if isinstance(sv,list) else np.asarray(sv); base=np.atleast_1d(ex.expected_value)
k = st.selectbox("Candidate",list(rank.index))
i = list(df.index).index(k)
c1,c2 = st.columns(2)
with c1:
    st.markdown("**FR4: local explanation**")
    shap.plots.waterfall(shap.Explanation(values=sv[i,:,cls[i]],base_values=base[cls[i]],
        data=df[FEATURES].iloc[i].values,feature_names=names),show=False); st.pyplot(plt.gcf()); plt.close()
with c2:
    if cls[i]<2:
        st.markdown("**FR6: why not higher (towards Highly Recommended)**")
        shap.plots.waterfall(shap.Explanation(values=sv[i,:,2],base_values=base[2],
            data=df[FEATURES].iloc[i].values,feature_names=names),show=False); st.pyplot(plt.gcf()); plt.close()
st.markdown("**FR5: global importance**")
imp = pd.Series(np.abs(sv).mean(0).sum(1),index=names).sort_values(); st.bar_chart(imp)
st.subheader("FR7: audit trail")
decision = st.radio("Committee decision",["Accept recommendation","Override"]); reason = st.text_input("Reason (required for override)")
if st.button("Save decision"):
    rec = dict(time=datetime.datetime.now().isoformat(),model=MODEL_VERSION,candidate=k,
        inputs=json.dumps(df[FEATURES].iloc[i].to_dict()),recommendation=df.Recommendation.iloc[i],
        probabilities=json.dumps(P[i].round(4).tolist()),shap=json.dumps(sv[i,:,cls[i]].round(4).tolist()),
        decision=decision,reason=reason)
    pd.DataFrame([rec]).to_csv(LOG,mode="a",header=not os.path.exists(LOG),index=False); st.success("Saved to audit trail")
