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

import json

RF_PATH = _MODEL_DIR / "random_forest.pkl"
_MEANS_PATH = _MODEL_DIR / "training_means.json"

TRAINING_MEANS = {}
if _MEANS_PATH.exists():
    with open(_MEANS_PATH, "r") as _f:
        TRAINING_MEANS = json.load(_f)


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
    feature_boost: Optional[dict] = None,
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

    feature_names = X_instance.columns.tolist()

    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        raw = explainer.shap_values(X_instance, check_additivity=False)

        # shap_values is a list [class0_vals, class1_vals] for binary classifiers in older versions
        if isinstance(raw, list):
            sv = raw[1]  # Unstable class SHAP values
            sv_1d = sv[0] if getattr(sv, 'ndim', 1) == 2 else sv
        elif getattr(raw, 'ndim', 0) == 3:
            # SHAP 0.45+ returns 3D array: (samples, features, classes)
            sv_1d = raw[0, :, 1]
        else:
            sv = raw
            sv_1d = sv[0] if getattr(sv, 'ndim', 1) == 2 else sv
        
        # Apply boosting if specified
        if feature_boost:
            for feat, multiplier in feature_boost.items():
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    sv_1d[idx] *= multiplier

        result["shap_values"] = sv_1d

        feature_names = X_instance.columns.tolist()
        
        expected_features = list(TRAINING_MEANS.keys()) if TRAINING_MEANS else feature_names
        assert feature_names == expected_features, (
            f"Feature mismatch: got {feature_names}"
            f" expected {expected_features}"
        )
        
        feature_values = X_instance.iloc[0].values

        if not TRAINING_MEANS:
            directions = ["high" if v > 0 else "low" for v in feature_values]
        else:
            directions = [
                "high" if feature_values[i] > TRAINING_MEANS.get(feature_names[i], 0) else "low"
                for i in range(len(feature_names))
            ]

        # Show SHAP as contribution percentage
        # ONLY consider positive values as drivers of instability
        pos_vals = np.maximum(sv_1d, 0)
        total = pos_vals.sum()
        pcts = (pos_vals / total * 100).round(1) if total > 0 else np.zeros_like(pos_vals)

        # Sort by positive SHAP value
        sorted_idx = np.argsort(pos_vals)[::-1]
        top_2 = [
            (feature_names[i], float(sv_1d[i]), directions[i], float(pcts[i]))
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
            force_plot_html = shap.save_html(None, force)
            result["force_plot_html"] = force_plot_html

            # Save force plot to disk
            if force_plot_html:
                import os
                from datetime import datetime as _dt
                _plot_dir = str(_ROOT / "reports" / "shap_plots")
                os.makedirs(_plot_dir, exist_ok=True)
                _ts = _dt.now().strftime("%Y%m%d_%H%M%S_%f")
                _plot_path = os.path.join(_plot_dir, f"force_plot_{_ts}.html")
                with open(_plot_path, "w", encoding="utf-8") as _fp:
                    _fp.write(force_plot_html)
                logger.info("Force plot saved to %s", _plot_path)
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

    if not TRAINING_MEANS:
        directions = ["high" if v > 0 else "low" for v in feature_values]
    else:
        directions = [
            "high" if feature_values[i] > TRAINING_MEANS.get(feature_names[i], 0) else "low"
            for i in range(len(feature_names))
        ]

    abs_vals = np.abs(importances)
    total = abs_vals.sum()
    pcts = (abs_vals / total * 100).round(1) if total > 0 else np.zeros_like(abs_vals)

    sorted_idx = np.argsort(importances)[::-1]
    top_2 = [
        (feature_names[i], float(importances[i]), directions[i], float(pcts[i]))
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
