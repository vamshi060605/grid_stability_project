# Grid Stability Fault Detection System

This system provides a real-time monitoring and diagnostic solution for electrical power grid stability. It uses machine learning to detect potential instability and fault conditions within the grid, such as line outages or load surges. By combining predictive modeling with explainable AI (SHAP), the system identifies the physical drivers of each anomaly and provides human-readable explanations. It is designed specifically for grid operators to bridge the gap between complex model outputs and actionable decision support. The core novelty of this project is the SHAP-to-action pipeline, which maps feature importance directly to corrective operator recommendations.

## Project Structure
  data/          — training dataset
  simulation/    — IEEE 14-bus grid simulator (pandapower)
  features/      — physics-based feature engineering
  models/        — Random Forest and XGBoost classifiers
  xai/           — SHAP explainability and recommendation engine
  dashboard/     — Streamlit web dashboard
  docs/          — architecture, demo script, viva Q&A
  tests/         — 58 automated unit tests
  reports/       — evaluation outputs and PDF report
  scripts/       — offline utility scripts

## Technologies Used
  - Python 3.x
  - Streamlit
  - scikit-learn
  - XGBoost
  - SHAP
  - pandapower
  - Plotly
  - scipy
  - pandas

## Setup Instructions
  1. python -m venv venv
  2. venv\Scripts\activate
  3. pip install -r requirements.txt
  4. python models/train.py                      (skip if models/saved/ already exists)
  5. python scripts/compute_shap_weights.py      (skip if shap_weights.json already exists)

## Running the Dashboard
  cd grid_stability
  venv\Scripts\python.exe -m streamlit run dashboard\app.py
  Open browser at http://localhost:8501

## How to Use
  1. On load — the dashboard shows a stable baseline grid. The "Live Monitor" tab displays the current system state with healthy parameters.
  2. Inject a fault — use the sidebar in the "Live Monitor" tab to click any fault injection button (e.g., "Line Outage"). The system will immediately update its prediction to UNSTABLE.
  3. Read the explanation — navigate to the "Fault Explainer" tab. Here, the system provides an AI-driven diagnosis of the fault's cause and specific corrective actions to restore stability.

## Running Tests
  venv\Scripts\python.exe -m pytest tests\ -v

## Academic Context
  - B.Tech CSE Final Year Project, SRM Institute of Science and Technology
  - Trained on UCI Electrical Grid Stability dataset (9,999 samples), evaluated against IEEE 14-bus network simulation
