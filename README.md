# Grid Stability Fault Detection with SHAP-Driven Recommendation Engine

> Explainability-Driven Operator Decision Support for Power Grid Fault Detection

---

## Architecture

```
UCI Dataset (9,999 samples)          Pandapower IEEE 14-bus
         │                                    │
         ▼                                    ▼
  feature_engineering.py          grid_simulator.py
  (physics features + scaling)    (fault injection)
         │
         ▼
   models/train.py
   (Random Forest + XGBoost)
         │
         ▼
   xai/shap_explainer.py
   (TreeExplainer → top features)
         │
         ▼
   xai/recommendation_engine.py   ← CORE NOVELTY
   (SHAP → cause + corrective action)
         │
         ▼
   dashboard/app.py (Streamlit)
```

---

## Novelty

Existing XAI work explains *which* features drove a prediction but stops there.
This project implements a **SHAP-to-action pipeline**: after every unstable prediction,
SHAP identifies the top contributing features and a domain-knowledge rule engine maps
each feature–direction pair to a specific fault cause and corrective action with severity
labelling — an **operationally actionable explanation**.

---

## Installation

```bash
# Create environment
conda create -n grid_stability python=3.10
conda activate grid_stability

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandapower, shap, streamlit; print('All packages OK')"
```

> ⚠️ **Pandapower + NumPy note:** pandapower 2.14.x requires numpy < 2.0.
> The pinned requirements.txt uses numpy==1.26.4 to avoid conflicts.

---

## Data

Download the UCI Electrical Grid Stability Simulated Data:
- URL: https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data
- Paper: Arzamasov et al., 2018
- Save as: `data/grid_stability.csv` (9,999 rows, 14 columns)

---

## How to Run

### 1. Train models
```bash
cd grid_stability
python models/train.py
```

### 2. Run evaluation suite
```bash
python models/evaluate.py
# Plots saved to evaluation/outputs/
```

### 3. Launch dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 4. Run full pipeline
```bash
python main.py
```

### 5. Run tests
```bash
pytest tests/ -v
```

### 6. Generate PDF report
```bash
python reports/generate_summary.py
# Saved to reports/project_results.pdf
```

---

## Project Structure

```
grid_stability/
├── config.yaml                  # All hyperparameters and thresholds
├── requirements.txt
├── README.md
├── main.py                      # Full pipeline entry point
├── data/
│   ├── grid_stability.csv       # UCI dataset (download separately)
│   ├── audit.py                 # Data quality checks
│   └── outputs/
├── simulation/
│   └── grid_simulator.py        # Pandapower fault injection
├── features/
│   └── feature_engineering.py  # Physics features + scaling
├── models/
│   ├── train.py                 # RF + XGBoost training
│   ├── evaluate.py              # Full evaluation suite
│   └── saved/                   # .pkl model files
├── xai/
│   ├── shap_explainer.py        # SHAP TreeExplainer wrapper
│   └── recommendation_engine.py # SHAP → action (CORE NOVELTY)
├── dashboard/
│   └── app.py                   # 3-panel Streamlit dashboard
├── evaluation/
│   ├── full_report.py
│   ├── robustness_test.py
│   ├── recommendation_validation.py
│   └── outputs/
├── reports/
│   └── generate_summary.py      # 6-page PDF
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_recommendation_engine.py
│   └── test_physics_check.py
├── utils/
│   └── thresholds.py            # Centralised constants
├── docs/
│   ├── architecture.md
│   ├── viva_qa.md
│   ├── demo_script.md
│   └── pre_demo_checklist.md
├── demo/
│   └── static_demo.html         # Offline backup demo
└── logs/
    └── project.log
```

---

## Screenshots

*[Add screenshots of dashboard panels here after first run]*

---

## References

- Arzamasov, V., Böhm, K., & Jochem, P. (2018). *Towards Concise Models of Grid Stability.*
  IEEE International Conference on Communications, Control, and Computing Technologies for
  Smart Grids (SmartGridComm).
  UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data

- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.*
  NeurIPS 2017. https://arxiv.org/abs/1705.07874
