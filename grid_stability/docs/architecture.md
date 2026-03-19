# System Architecture

## Overview

This project implements an end-to-end fault detection and operator decision support
system for power grids. A pandapower IEEE 14-bus simulation generates labelled fault
scenarios. A Random Forest and XGBoost classifier trained on UCI grid stability data
predict instability. After every unstable prediction, SHAP values identify the top
contributing features, and a rule engine maps them to human-readable fault causes and
corrective actions — bridging explainable AI with operational decision support.

---

## Module Dependency Diagram

```
config.yaml + utils/thresholds.py
         │
         ▼
simulation/grid_simulator.py  ──────────────────────────────────────────┐
         │                                                               │
         ▼                                                               │
data/grid_stability.csv (UCI) ──► features/feature_engineering.py       │
                                          │                             │
                                          ▼                             │
                                  models/train.py                       │
                                          │                             │
                                          ▼                             │
                                  models/evaluate.py                    │
                                          │                             │
                                          ▼                             │
                                  xai/shap_explainer.py ◄───────────────┘
                                          │
                                          ▼
                              xai/recommendation_engine.py  ← CORE NOVELTY
                                          │
                                          ▼
                               dashboard/app.py (Streamlit)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    evaluation/      reports/        tests/
                   full_report.py  generate_summary  pytest suite
```

---

## Data Flow

```
Pandapower          UCI CSV
case14 network         │
     │                 │
     ▼                 ▼
grid_simulator  →  feature_engineering  →  StandardScaler (train-only fit)
     │                 │
     │         ┌───────┴────────┐
     │         ▼                ▼
     │    RandomForest      XGBoost
     │         │                │
     │         └───────┬────────┘
     │                 ▼
     │          predict_proba()
     │                 │
     │        ┌────────┴─────────┐
     │        ▼                  ▼
     │   physics_check        SHAP
     │   (pre-filter)     TreeExplainer
     │        │                  │
     │        │          top_2_features
     │        │                  │
     │        └────────┬─────────┘
     │                 ▼
     │       recommendation_engine
     │       (SHAP → cause + action)
     │                 │
     └────────────────►▼
                  Streamlit Dashboard
```

---

## RECOMMENDATION_RULES

| Feature | Direction | Cause | Action | Severity |
|---------|-----------|-------|--------|----------|
| VSI | low | Voltage collapse risk | Reduce load or switch feeder | CRITICAL |
| RoCoV | high | Rapid voltage fluctuation | Activate voltage regulator | HIGH |
| fault_impedance | low | High impedance fault | Inspect feeder for insulation failure | HIGH |
| thermal_stress | high | Thermal overload | Redistribute load or check transformer | MEDIUM |
| tau1 | high | Participant response too slow | Review demand response contract | MEDIUM |

---

## Physics Pre-filter Rules

These run **before** ML inference and bypass the model when triggered:

| Condition | Threshold | Level | Response |
|-----------|-----------|-------|----------|
| VSI < 0.85 | IEEE 1159 | CRITICAL | Override — bypass ML |
| VSI > 1.10 | IEEE 1159 | WARNING | Override — bypass ML |
| thermal_stress > 0.95 | IEC 60076 | CRITICAL | Override — bypass ML |

---

## Known Limitations

- The SHAP-to-action rule table covers 5 core features; edge cases fall back to a generic
  "manual inspection" recommendation.
- Pandapower simulations use the IEEE 14-bus test network — a simplified academic model,
  not a real utility grid topology.
- The UCI dataset is simulation-derived (not measured field data), which may not capture
  all real-world fault signatures.
- The scaler is fitted on UCI training data; simulation data uses proxy mappings which may
  introduce feature distribution mismatch at the dashboard level.
- SHAP TreeExplainer can be memory-intensive on large forests; the system falls back to
  RF feature importances when memory errors occur.

---

## Future Work

- Replace rule-based recommendation engine with a learned policy trained on operator
  incident logs, enabling data-driven corrective guidance.
- Integrate real PMU (Phasor Measurement Unit) streams via a Kafka topic to replace
  simulated inputs with live grid telemetry.
- Extend the pandapower network from IEEE 14-bus to IEEE 118-bus for higher realism
  and more diverse fault injection scenarios.
