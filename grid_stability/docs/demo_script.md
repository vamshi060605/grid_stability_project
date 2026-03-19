# Demo Script — Viva Day (~4 minutes)

---

## Step 1 — Open Dashboard (30 seconds)

**Do:** Run `cd grid_stability && streamlit run dashboard/app.py`  
**Show:** Landing state — all 3 tabs visible, sidebar sliders at defaults  
**Say:** "This is a real-time fault detection dashboard for power grid stability.
The system combines a machine learning classifier with SHAP explainability and a
corrective recommendation engine — making the model's decisions operationally actionable."

---

## Step 2 — Show Stable State (30 seconds)

**Do:** Leave all sliders at default values  
**Show:** Green STABLE prediction, low unstable probability gauge  
**Say:** "With nominal grid parameters, the model correctly predicts stable operation.
Notice the physics pre-filter is also running in the background — if VSI drops below
0.85 or thermal stress exceeds 0.95, the physics rules take priority over the ML model."

---

## Step 3 — Inject Line Outage (60 seconds)

**Do:** Click the **"Line Outage"** button in the sidebar  
**Show:** Sliders auto-populate, prediction flips to UNSTABLE in red  
**Say:** "I've just injected a line outage fault into the pandapower simulation.
The model has flipped to UNSTABLE. Watch the SHAP values update in Panel 2 —
they show which features drove this prediction.  
The recommendation card here gives an **operationally actionable explanation**:
not just 'this feature was important', but the specific fault cause and what the
operator should do to correct it. This is the core novelty of the project."

---

## Step 4 — Show SHAP Explanation (45 seconds)

**Do:** Click the **Fault Explainer** tab  
**Show:** SHAP horizontal bar chart, red bars for positive contributions  
**Say:** "SHAP — SHapley Additive exPlanations — computes each feature's exact
contribution to this specific prediction using cooperative game theory. Red bars
push toward instability, blue bars toward stability.
The top driver here is [feature name], with a SHAP contribution of [value]. Our
rule engine maps this to a specific fault cause and corrective action."

---

## Step 5 — Model Comparison (30 seconds)

**Do:** Click the **Model Comparison** tab  
**Show:** RF vs XGBoost metrics side-by-side, AUC-ROC curves  
**Say:** "Both Random Forest and XGBoost achieve above [X]% accuracy on the
UCI test set. We use Random Forest as the primary model because it's natively
compatible with TreeExplainer for exact SHAP values, and its feature importances
provide an independent validation of the SHAP rankings."

---

## Step 6 — Trigger Physics Override (30 seconds)

**Do:** Drag the VSI slider below 0.85  
**Show:** Yellow warning banner at top of page — "Physics rule override — ML bypassed"  
**Say:** "When VSI drops below 0.85, the IEEE 1159 threshold for undervoltage,
our physics pre-filter fires and bypasses the ML model entirely.
Physics rules handle clear violations — the ML model handles the ambiguous middle ground
where multiple moderate signals combine to predict instability."

---

**Total: ~4 minutes**

---

# Crash Recovery Plan

| Failure | Recovery |
|---------|----------|
| Streamlit won't start | `pip install streamlit && streamlit run dashboard/app.py` |
| Model file missing | `cd grid_stability && python models/train.py` (~2 min) |
| Pandapower crashes on fault injection | Use pre-saved simulation CSV or skip to slider demo |
| SHAP too slow | Set `SHAP_CONFIDENCE_LOW = 2.0` in config.yaml to skip SHAP |
| Browser won't open | Manually go to `http://localhost:8501` |
| Port already in use | `streamlit run dashboard/app.py --server.port 8502` |
