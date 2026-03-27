"""
stability_metrics.py

Derived metrics for grid stability assessment.
Contains Stability Margin Score and Fault Severity Index (FSI).
"""


def compute_stability_margin(vsi, rocov, thermal_stress, rocov_max=1.0):
    """
    Stability Margin Score = (0.5 × VSI) + (0.3 × (1 - RoCoV_norm)) + (0.2 × (1 - thermal_stress))

    Where:
        RoCoV_norm = min(RoCoV / rocov_max, 1.0)

    Weights:
        w1 = 0.5 (voltage stability dominates)
        w2 = 0.3 (rate of change of voltage)
        w3 = 0.2 (thermal loading)
        Sum = 1.0

    Args:
        vsi: Voltage Stability Index (per-unit bus voltage).
        rocov: Rate of Change of Voltage (absolute delta).
        thermal_stress: Thermal loading fraction (0–1).
        rocov_max: Maximum expected RoCoV for normalization (from config.yaml).

    Returns:
        float clamped to [0.0, 1.0]. Higher = more stable.
    """
    rocov_norm = min(rocov / rocov_max, 1.0) if rocov_max > 0 else 0.0
    score = (0.5 * vsi) + (0.3 * (1.0 - rocov_norm)) + (0.2 * (1.0 - thermal_stress))
    return float(max(0.0, min(1.0, score)))


def compute_fsi(feature_vector: dict, shap_weights: dict, feature_bounds: dict) -> float:
    """
    Fault Severity Index = Sum(w_i × f_i_normalized)

    Where:
        f_i_normalized = (f_i - f_min) / (f_max - f_min), clipped to [0, 1]
        w_i = normalized global SHAP importance for feature i

    Weight source:
        Computed offline by scripts/compute_shap_weights.py,
        saved to models/saved/shap_weights.json.

    Args:
        feature_vector: dict of {feature_name: raw_value}.
        shap_weights: dict of {feature_name: weight}. Must sum to 1.0.
        feature_bounds: dict of {feature_name: {"min": float, "max": float}}.

    Returns:
        float clamped to [0.0, 1.0]. Higher = more severe.
    """
    fsi = 0.0
    for feat, weight in shap_weights.items():
        val = feature_vector.get(feat, 0.0)
        bounds = feature_bounds.get(feat, {"min": 0.0, "max": 1.0})
        f_range = bounds["max"] - bounds["min"]
        f_norm = (val - bounds["min"]) / f_range if f_range > 0 else 0.0
        f_norm = max(0.0, min(1.0, f_norm))
        fsi += weight * f_norm
    return float(max(0.0, min(1.0, fsi)))
