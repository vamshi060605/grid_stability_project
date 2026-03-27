"""
scripts/compute_shap_weights.py

Offline script: computes global SHAP feature importance, normalizes weights,
and saves to models/saved/shap_weights.json. Run ONCE before dashboard startup.

Usage:
    python scripts/compute_shap_weights.py
"""
import sys
import json
import logging
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import shap

from features.feature_engineering import load_uci_dataset, prepare_train_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_DIR = _ROOT / "models" / "saved"
RF_PATH = _MODEL_DIR / "random_forest.pkl"
OUTPUT_PATH = _MODEL_DIR / "shap_weights.json"


def main():
    # 1. Load trained Random Forest
    if not RF_PATH.exists():
        logger.error("Model not found at %s. Run models/train.py first.", RF_PATH)
        sys.exit(1)

    rf = joblib.load(RF_PATH)
    logger.info("Loaded RF model from %s", RF_PATH)

    # 2. Load X_test from the feature pipeline
    df = load_uci_dataset()
    _, X_test, _, _ = prepare_train_test(df, fit_scaler=False)
    logger.info("Loaded test data. Shape: %s", X_test.shape)

    # 3. Compute global SHAP feature importance (mean |SHAP| per feature)
    logger.info("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test.iloc[:300])  # Use first 300 for efficiency

    # Handle binary classifier output
    if isinstance(shap_values, list):
        sv = np.abs(shap_values[1])  # Unstable class
    elif shap_values.ndim == 3:
        sv = np.abs(shap_values[:, :, 1])
    else:
        sv = np.abs(shap_values)

    raw_importance = sv.mean(axis=0)
    feature_names = X_test.columns.tolist()

    # 4. Normalize so weights sum to 1.0
    total = raw_importance.sum()
    if total == 0:
        logger.error("All SHAP values are zero — cannot compute weights.")
        sys.exit(1)

    normalized = raw_importance / total

    # 5. Save to JSON
    weights = {name: round(float(w), 6) for name, w in zip(feature_names, normalized)}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    # 6. Print confirmation table
    print(f"\n{'Feature':<20} {'Raw SHAP':>12} {'Normalized Weight':>18}")
    print("─" * 52)
    for name, raw, norm in zip(feature_names, raw_importance, normalized):
        print(f"{name:<20} {raw:>12.6f} {norm:>18.6f}")
    print("─" * 52)
    print(f"{'TOTAL':<20} {total:>12.6f} {sum(normalized):>18.6f}")
    print(f"\nSaved {len(weights)} features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
