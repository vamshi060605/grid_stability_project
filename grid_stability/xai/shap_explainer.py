"""
Module 4 — xai/shap_explainer.py

SHAP-based explanation layer for the trained Random Forest model.
Graceful fallback to RF feature importances on SHAP failure.
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from features.feature_engineering import load_uci_dataset, prepare_train_test

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_MODEL_DIR = _ROOT / "models" / "saved"
_XAI_DIR = _ROOT / "xai"

RF_PATH = _MODEL_DIR / "random_forest.pkl"


def _load_rf_model():
    """Load saved Random Forest model. Raises if missing."""
    if not RF_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {RF_PATH}. Run models/train.py first."
        )
    return joblib.load(RF_PATH)


def explain_prediction(
    X_instance: pd.DataFrame,
    training_means: Optional[pd.Series] = None,
    model=None,
) -> dict:
    """
    Generate SHAP explanation for a single prediction instance.

    Args:
        X_instance: single-row DataFrame with feature values.
        training_means: Series of training feature means for direction labelling.
        model: optional pre-loaded RF model (loaded from disk if None).

    Returns:
        Dict with keys:
          - shap_values: raw SHAP values array
          - top_2_features: list of (feature_name, shap_value, direction)
          - force_plot_html: SHAP force plot as HTML string (or None)
          - fallback: bool — True if RF importances used instead of SHAP
    """
    if model is None:
        model = _load_rf_model()

    result = {
        "shap_values": None,
        "top_2_features": [],
        "force_plot_html": None,
        "fallback": False,
    }

    try:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_instance)

        # shap_values is a list [class0_vals, class1_vals] for binary classifiers
        if isinstance(raw, list):
            sv = raw[1]  # Unstable class SHAP values
        else:
            sv = raw

        sv_1d = sv[0] if sv.ndim == 2 else sv
        result["shap_values"] = sv_1d

        feature_names = X_instance.columns.tolist()
        feature_values = X_instance.iloc[0].values

        # Determine direction relative to training mean
        if training_means is None:
            directions = ["high" if v > 0 else "low" for v in feature_values]
        else:
            directions = [
                "high" if feature_values[i] > training_means.get(feature_names[i], 0) else "low"
                for i in range(len(feature_names))
            ]

        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sv_1d))[::-1]
        top_2 = [
            (feature_names[i], float(sv_1d[i]), directions[i])
            for i in sorted_idx[:2]
        ]
        result["top_2_features"] = top_2

        # Generate HTML force plot
        try:
            shap.initjs()
            expected_val = explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = expected_val[1]
            force = shap.force_plot(expected_val, sv_1d, X_instance.iloc[0])
            result["force_plot_html"] = shap.save_html(None, force)
        except Exception as plot_exc:  # pylint: disable=broad-except
            logger.debug("Force plot generation skipped: %s", plot_exc)

        logger.info("SHAP explanation generated for instance.")

    except MemoryError:
        logger.warning("SHAP MemoryError — falling back to RF feature importances")
        result = _fallback_explanation(model, X_instance, training_means)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("SHAP explanation failed: %s — falling back to RF importances", exc)
        result = _fallback_explanation(model, X_instance, training_means)

    return result


def _fallback_explanation(model, X_instance: pd.DataFrame, training_means=None) -> dict:
    """
    Fallback explanation using RF feature importances when SHAP is unavailable.

    Args:
        model: trained RF model.
        X_instance: single-row DataFrame.
        training_means: optional training means for direction.

    Returns:
        Explanation dict with fallback=True.
    """
    feature_names = X_instance.columns.tolist()
    importances = model.feature_importances_
    feature_values = X_instance.iloc[0].values

    if training_means is None:
        directions = ["high" if v > 0 else "low" for v in feature_values]
    else:
        directions = [
            "high" if feature_values[i] > training_means.get(feature_names[i], 0) else "low"
            for i in range(len(feature_names))
        ]

    sorted_idx = np.argsort(importances)[::-1]
    top_2 = [
        (feature_names[i], float(importances[i]), directions[i])
        for i in sorted_idx[:2]
    ]

    return {
        "shap_values": importances,
        "top_2_features": top_2,
        "force_plot_html": None,
        "fallback": True,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SHAP Explainer Standalone Run ===")

    model = _load_rf_model()
    df = load_uci_dataset()
    X_train, X_test, y_train, y_test = prepare_train_test(df, fit_scaler=False)
    training_means = X_train.mean()

    instance = X_test.iloc[[0]]
    explanation = explain_prediction(instance, training_means=training_means, model=model)

    print(f"\nTop 2 features:")
    for feat, val, direction in explanation["top_2_features"]:
        print(f"  {feat}: SHAP={val:.4f}  direction={direction}")
    print(f"Fallback used: {explanation['fallback']}")
