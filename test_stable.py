import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, r"d:\grid_stability_project\grid_stability")
import features.feature_engineering as fe

model = joblib.load(r"d:\grid_stability_project\grid_stability\models\saved\random_forest.pkl")
scaler = joblib.load(r"d:\grid_stability_project\grid_stability\models\saved\scaler.pkl")

df = fe.load_uci_dataset()
raw_df = df.drop(columns=['stab', 'stabf']).copy()

# Find row that yields the lowest unstable prob
probs = model.predict_proba(scaler.transform(raw_df))[:, 1]
best_idx = np.argmin(probs)
print(f"Best unstable prob: {probs[best_idx]}")
print(f"Raw features for best row:")
best_raw = raw_df.iloc[best_idx].to_dict()
print(best_raw)

# Let's test SHAP without check_additivity=False to see if it raises AdditivityCheckError
import shap
X_instance = pd.DataFrame(scaler.transform(raw_df.iloc[[best_idx]]), columns=raw_df.columns)
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
try:
    explainer.shap_values(X_instance)
    print("SHAP values succeeded WITH additivity check!")
except Exception as e:
    print(f"SHAP values failed WITH additivity check. Error: {type(e).__name__}: {e}")

try:
    explainer.shap_values(X_instance, check_additivity=False)
    print("SHAP values succeeded WITHOUT additivity check!")
except Exception as e:
    print("SHAP failed again.")
