
"""Step 6: Decision-support dashboard (FR1-FR7).
Run locally: python -m streamlit run app.py
"""

import sys
import json
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

# Ensure common.py can be imported from src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import *

# Project folders and files
MODEL_PATH = BASE_DIR / "models" / "xgb_model.joblib"
DATA_PATH = BASE_DIR / "data" / "synthetic_procurement.csv"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "audit_trail.csv"

# Create logs directory if it does not exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Page configuration and model loading
# --------------------------------------------------

st.set_page_config(
    page_title="XAI Medical Equipment Procurement",
    layout="wide"
)

st.title("Explainable AI Medical Equipment Procurement")

st.caption(
    "Decision support only: the committee keeps the final decision. "
    "Trained on a synthetic dataset."
)

# Check that the required files exist
if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Procurement dataset not found: {DATA_PATH}")
    st.stop()

# Load trained model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))
MODEL_VERSION = MODEL_PATH.name

# --------------------------------------------------
# 3. Upload or load procurement candidates (FR1)
# --------------------------------------------------

names = [NICE[f] for f in FEATURES]

up = st.file_uploader(
    "FR1: Upload candidates (CSV with the 11 criteria columns)",
    type="csv"
)

if up is not None:
    df = pd.read_csv(up)
else:
    df = pd.read_csv(DATA_PATH).sample(5, random_state=1)[FEATURES]

# Validate required columns
missing_columns = [col for col in FEATURES if col not in df.columns]

if missing_columns:
    st.error(f"Missing required CSV columns: {missing_columns}")
    st.stop()

df = df.copy()

st.dataframe(df[FEATURES])

# --------------------------------------------------
# 4. Prediction and ranking (FR2, FR3)
# --------------------------------------------------

P = model.predict_proba(df[FEATURES])
cls = P.argmax(1)

df["Recommendation"] = [CLASSES[c] for c in cls]

df["Expected suitability"] = P[:, 2] + 0.5 * P[:, 1]

for j, c in enumerate(CLASSES):
    df[f"P({c})"] = P[:, j]

rank = df.sort_values(
    "Expected suitability",
    ascending=False
)

st.subheader("Ranking (FR2, FR3)")

st.dataframe(
    rank.drop(columns=FEATURES).round(3)
)

# --------------------------------------------------
# 5. SHAP explanations (FR4, FR5, FR6)
# --------------------------------------------------

ex = shap.TreeExplainer(model)
sv = ex.shap_values(df[FEATURES])

sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)

base = np.atleast_1d(ex.expected_value)

k = st.selectbox(
    "Candidate",
    list(rank.index)
)

i = list(df.index).index(k)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**FR4: Local explanation**")

    shap.plots.waterfall(
        shap.Explanation(
            values=sv[i, :, cls[i]],
            base_values=base[cls[i]],
            data=df[FEATURES].iloc[i].values,
            feature_names=names
        ),
        show=False
    )

    st.pyplot(plt.gcf())
    plt.close()

with c2:
    if cls[i] < 2:
        st.markdown(
            "**FR6: Why not higher (towards Highly Recommended)**"
        )

        shap.plots.waterfall(
            shap.Explanation(
                values=sv[i, :, 2],
                base_values=base[2],
                data=df[FEATURES].iloc[i].values,
                feature_names=names
            ),
            show=False
        )

        st.pyplot(plt.gcf())
        plt.close()

st.markdown("**FR5: Global importance**")

imp = pd.Series(
    np.abs(sv).mean(0).sum(1),
    index=names
).sort_values()

st.bar_chart(imp)

# --------------------------------------------------
# 6. Committee decision and audit trail (FR7)
# --------------------------------------------------

st.subheader("FR7: Audit trail")

decision = st.radio(
    "Committee decision",
    ["Accept recommendation", "Override"]
)

reason = st.text_input("Reason (required for override)")

if st.button("Save decision"):

    if decision == "Override" and not reason.strip():
        st.warning("Please provide a reason for overriding the recommendation.")

    else:
        rec = dict(
            time=datetime.datetime.now().isoformat(),
            model=MODEL_VERSION,
            candidate=k,
            inputs=json.dumps(df[FEATURES].iloc[i].to_dict()),
            recommendation=df.Recommendation.iloc[i],
            probabilities=json.dumps(P[i].round(4).tolist()),
            shap=json.dumps(sv[i, :, cls[i]].round(4).tolist()),
            decision=decision,
            reason=reason
        )

        pd.DataFrame([rec]).to_csv(
            LOG_PATH,
            mode="a",
            header=not LOG_PATH.exists(),
            index=False
        )

        st.success(f"Decision saved to {LOG_PATH}")

