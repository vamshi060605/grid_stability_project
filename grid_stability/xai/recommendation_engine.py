"""
Module 5 — xai/recommendation_engine.py

THE CORE NOVELTY: SHAP-to-action pipeline.

After every unstable prediction, SHAP identifies the top contributing features
and this rule engine maps them to human-readable fault causes and corrective
actions. This bridges the gap between XAI and operator decision support.
"""
import logging
from pathlib import Path

from utils.thresholds import SHAP_CONFIDENCE_LOW, SHAP_CONFIDENCE_HIGH

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).parent.parent / "logs" / "project.log"


# ── Rule table — maps (feature_name, direction) → operator guidance ──────────

RECOMMENDATION_RULES: dict = {
    ("VSI", "low"): {
        "cause": "Voltage collapse risk — bus voltage below safe margin",
        "action": "Reduce active load at affected bus or switch to alternate feeder",
        "severity": "CRITICAL",
    },
    ("RoCoV", "high"): {
        "cause": "Rapid voltage fluctuation — likely renewable intermittency",
        "action": "Activate voltage regulator or reduce renewable injection rate",
        "severity": "HIGH",
    },
    ("fault_impedance", "low"): {
        "cause": "High impedance fault on line",
        "action": "Inspect feeder for partial contact or insulation failure",
        "severity": "HIGH",
    },
    ("thermal_stress", "high"): {
        "cause": "Thermal overload on transmission element",
        "action": "Redistribute load or check transformer rating",
        "severity": "MEDIUM",
    },
    ("tau1", "low"): {
        "cause": "Voltage collapse risk — bus voltage critically below nominal",
        "action": "Switch to alternate feeder or reduce active load",
        "severity": "CRITICAL"
    },
    ("tau2", "low"): {
        "cause": "Voltage collapse risk at node 2 — voltage below safe margin",
        "action": "Switch to alternate feeder or reduce active load at node 2",
        "severity": "CRITICAL"
    },
    ("p1", "high"): {
        "cause": "Excess power injection — load surge detected",
        "action": "Reduce generation at node 1 or shed load",
        "severity": "MEDIUM"
    },
    ("p2", "high"): {
        "cause": "Excess power injection at producer node 2",
        "action": "Reduce generation at node 2 or redistribute",
        "severity": "MEDIUM"
    },
    ("g1", "high"): {
        "cause": "Rapid voltage fluctuation — generator response aggressive",
        "action": "Activate voltage regulator or reduce injection rate",
        "severity": "HIGH"
    },
    ("g2", "high"): {
        "cause": "Aggressive gamma response causing oscillation at node 2",
        "action": "Activate voltage regulator or reduce injection rate",
        "severity": "HIGH"
    },
    ("g1", "low"): {
        "cause": "High impedance fault — abnormal line resistance detected",
        "action": "Inspect feeder for insulation failure or partial contact",
        "severity": "HIGH"
    },
    ("g2", "low"): {
        "cause": "High impedance fault on line 2",
        "action": "Inspect feeder for partial contact or insulation failure",
        "severity": "HIGH"
    },
    ("tau1", "high"): {
        "cause": "Participant response time too slow",
        "action": "Review demand response contract at this node",
        "severity": "MEDIUM",
    },
    ("tau2", "high"): {
        "cause": "Secondary participant response delay — node 2 slow to react",
        "action": "Review demand-response contract at node 2",
        "severity": "HIGH"
    },
    ("tau3", "high"): {
        "cause": "Tertiary participant response delay — node 3 slow to react", 
        "action": "Review demand-response contract at node 3",
        "severity": "MEDIUM"
    },
    ("tau4", "high"): {
        "cause": "Quaternary participant response delay detected",
        "action": "Inspect control signal at node 4",
        "severity": "MEDIUM"
    },
    ("p3", "high"): {
        "cause": "Excess power injection at producer node 3",
        "action": "Reduce generation at node 3 or redistribute load",
        "severity": "MEDIUM"
    },
}

_FALLBACK_RULE = {
    "cause": "Anomalous feature combination detected",
    "action": "Manual inspection recommended",
    "severity": "HIGH",
}

_UNCERTAIN_SENTINEL = "UNCERTAIN"


def _confidence_note(confidence: float, show_action: bool) -> str:
    """
    Return human-readable confidence label.

    Args:
        confidence: model predict_proba score (0–1).
        show_action: whether corrective action should be displayed.

    Returns:
        Confidence note string.
    """
    if confidence >= SHAP_CONFIDENCE_HIGH:
        return "High"
    elif confidence >= SHAP_CONFIDENCE_LOW:
        return "Medium"
    else:
        return "Low confidence — manual verification advised"


def generate_recommendation(
    top_features: list,
    feature_values: dict,
    training_means: dict,
    confidence: float,
) -> list:
    """
    Generate operator recommendations from SHAP top features.

    This is the academic novelty: maps XAI output to actionable operator guidance.

    Args:
        top_features: list of (feature_name, shap_value, old_direction, pct) from shap_explainer.
        feature_values: dict of {feature_name: raw_value} for the current instance.
        training_means: dict of feature training means for dynamic direction thresholds.
        confidence: model predict_proba score for the unstable class (0–1).

    Returns:
        List of recommendation dicts.
    """
    if abs(confidence - 0.5) < 0.02:
        logger.info("Model uncertainty detected (confidence=%.3f) — returning UNCERTAIN", confidence)
        return [{"state": _UNCERTAIN_SENTINEL, "confidence": confidence}]

    show_action = confidence >= SHAP_CONFIDENCE_LOW
    recommendations = []

    for feature_name, shap_value, _, pct in top_features:
        feat_val = feature_values.get(feature_name, 0.0)
        mean_val = training_means.get(feature_name, 0.5)
        
        direction = "high" if feat_val > mean_val else "low"
        rule_key = (feature_name, direction)
        rule = RECOMMENDATION_RULES.get(rule_key)

        if rule is None:
            logger.warning(
                f"No rule for ({feature_name}, {direction})"
                f" — feat_val={feat_val:.3f}"
                f" mean={mean_val:.3f}"
            )
            rule = _FALLBACK_RULE
            
        logger.debug(
            f"Recommendation: feat={feature_name} "
            f"val={feat_val:.3f} mean={mean_val:.3f} "
            f"direction={direction} rule={rule_key}"
        )

        rec = {
            "feature": feature_name,
            "shap_contribution": round(float(shap_value), 4),
            "contribution_pct": pct,
            "cause": rule["cause"],
            "action": rule["action"] if show_action else "Confidence too low — verify manually",
            "severity": rule["severity"],
            "confidence_note": _confidence_note(confidence, show_action),
            "confidence_pct": round(confidence * 100, 1),
        }
        recommendations.append(rec)

    return recommendations


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_LOG_PATH),
        ],
    )
    logger.info("=== Recommendation Engine Test — all 5 rules + fallback ===\n")

    test_cases = [
        # (feature, shap_value, direction, feature_value, confidence, description)
        ([("VSI", -0.42, "low")],             {"VSI": 0.72},            0.91, "VSI low — voltage collapse"),
        ([("RoCoV", 0.38, "high")],           {"RoCoV": 0.88},          0.80, "RoCoV high — rapid fluctuation"),
        ([("fault_impedance", -0.29, "low")], {"fault_impedance": 0.3}, 0.86, "Fault impedance low — HIF"),
        ([("thermal_stress", 0.33, "high")],  {"thermal_stress": 0.98}, 0.75, "Thermal stress high — overload"),
        ([("tau1", 0.21, "high")],            {"tau1": 1.8},            0.88, "Tau1 high — slow response"),
        ([("unknown_feat", 0.15, "low")],     {"unknown_feat": 0.4},    0.83, "Unknown feature — fallback"),
        ([("VSI", -0.31, "low")],             {"VSI": 0.83},            0.50, "Confidence 0.5 — uncertain"),
    ]

    for top_features, feat_vals, conf, description in test_cases:
        print(f"\n{'─'*60}")
        print(f"CASE: {description}  (confidence={conf})")
        recs = generate_recommendation(top_features, feat_vals, conf)
        for r in recs:
            if r.get("state") == _UNCERTAIN_SENTINEL:
                print("  → STATE: UNCERTAIN — consult physics indicators directly")
            else:
                print(f"  Feature:     {r['feature']}")
                print(f"  SHAP:        {r['shap_contribution']}")
                print(f"  Cause:       {r['cause']}")
                print(f"  Action:      {r['action']}")
                print(f"  Severity:    {r['severity']}")
                print(f"  Confidence:  {r['confidence_note']} ({r['confidence_pct']}%)")
