import sys
import os
import joblib
from pathlib import Path

sys.path.insert(0, r"d:\grid_stability_project\grid_stability")
from dashboard.app import inject_fault, _build_feature_vector

model = joblib.load('models/saved/random_forest.pkl')
scaler = joblib.load('models/saved/scaler.pkl')
feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None

faults = ["line_outage", "load_surge", "generator_trip", "high_impedance"]
print("=== FAULT CONFIDENCE TEST ===")
for f in faults:
    inputs = inject_fault(f)
    print(f"\nFault: {f} -> Inputs: {inputs}")
    if inputs:
        X_df = _build_feature_vector(
            inputs['vsi'], inputs['rocov'], inputs['thermal'], inputs['fault_imp'],
            scaler, feature_names
        )
        prob = model.predict_proba(X_df)[0][1]
        print(f"Confidence (Unstable): {prob:.4f}")
