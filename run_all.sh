#!/usr/bin/env bash
set -e
pip install -r requirements.txt
cd src
python generate_data.py   # Step 1-2: dataset + labels
python train.py           # Step 3: models, Table 4.1, 4.2, confusion matrices
python explain.py         # Step 4: SHAP, Tables 4.3 / 4.5, Figs 4.6-4.13, quality checks
python simulate.py        # Step 5: 600-tender simulation, Tables 4.6 / 4.7
streamlit run app.py      # Step 6: dashboard
