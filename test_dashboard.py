import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, r"d:\grid_stability_project\grid_stability")
import features.feature_engineering as fe
import joblib

model = joblib.load(r"d:\grid_stability_project\grid_stability\models\saved\random_forest.pkl")
scaler = joblib.load(r"d:\grid_stability_project\grid_stability\models\saved\scaler.pkl")

df = fe.load_uci_dataset()
X_train, X_test, y_train, y_test = fe.prepare_train_test(df, fit_scaler=False)

# Find a stable row
stable_rows = X_train[y_train == 0]
if len(stable_rows) > 0:
    stable_row = stable_rows.iloc[[0]]
    print("Found stable row from dataset:")
    print(stable_row.to_dict('records')[0])
    
    # Predict
    prob = model.predict_proba(stable_row)[0][1]
    print(f"Model predict prob (unstable): {prob}")
else:
    print("No stable rows found!")

# Try SHAP on this row
import shap
print(f"SHAP version: {shap.__version__}")
try:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    raw = explainer.shap_values(stable_row, check_additivity=False)
    print("SHAP SUCCESS!")
except Exception as e:
    import traceback
    print("SHAP ERROR:")
    traceback.print_exc()

# Also let's check what app.py was doing
base_stable = {
    "tau1": 4.39, "tau2": 4.35, "tau3": 4.39, "tau4": 4.38,
    "p1": 3.74, "p2": -1.25, "p3": -1.25, "p4": -1.24,
    "g1": 0.45, "g2": 0.45, "g3": 0.44, "g4": 0.45,
}
row = np.zeros((1, 12))
for i, key in enumerate(base_stable.keys()):
    row[0, i] = base_stable[key]

# app.py scales it!
scaled = scaler.transform(row)
X_app = pd.DataFrame(scaled, columns=X_train.columns)
print("app.py base_stable features:")
print(X_app.to_dict('records')[0])
prob_app = model.predict_proba(X_app)[0][1]
print(f"app.py base_stable prob (unstable): {prob_app}")
