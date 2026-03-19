"""
tests/test_recommendation_engine.py

Unit tests for recommendation engine rule firing,
fallback logic, confidence notes, and uncertainty state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from xai.recommendation_engine import (
    generate_recommendation,
    RECOMMENDATION_RULES,
    _UNCERTAIN_SENTINEL,
)


def _gen(feature, direction, confidence=0.90, shap_val=0.35):
    """Helper: call generate_recommendation with a single feature."""
    top_features = [(feature, shap_val, direction)]
    feature_values = {feature: 0.5}
    return generate_recommendation(top_features, feature_values, confidence)


class TestRuleFiring:
    def test_vsi_low_fires_critical(self):
        recs = _gen("VSI", "low")
        assert recs[0]["severity"] == "CRITICAL"

    def test_rocov_high_fires_high(self):
        recs = _gen("RoCoV", "high")
        assert recs[0]["severity"] == "HIGH"

    def test_fault_impedance_low_fires_high(self):
        recs = _gen("fault_impedance", "low")
        assert recs[0]["severity"] == "HIGH"

    def test_thermal_stress_high_fires_medium(self):
        recs = _gen("thermal_stress", "high")
        assert recs[0]["severity"] == "MEDIUM"

    def test_tau1_high_fires_medium(self):
        recs = _gen("tau1", "high")
        assert recs[0]["severity"] == "MEDIUM"

    def test_rule_includes_cause_and_action(self):
        recs = _gen("VSI", "low")
        assert "cause" in recs[0]
        assert "action" in recs[0]
        assert len(recs[0]["cause"]) > 5
        assert len(recs[0]["action"]) > 5


class TestFallback:
    def test_fallback_fires_for_unknown_pair(self):
        recs = _gen("unknown_feature_xyz", "high")
        assert recs[0]["cause"] == "Anomalous feature combination detected"
        assert recs[0]["action"] == "Manual inspection recommended"
        assert recs[0]["severity"] == "HIGH"

    def test_fallback_fires_for_wrong_direction(self):
        """VSI high is not in rules — should fall back."""
        recs = _gen("VSI", "high")
        assert recs[0]["cause"] == "Anomalous feature combination detected"


class TestConfidenceNotes:
    def test_high_confidence_note(self):
        recs = _gen("VSI", "low", confidence=0.92)
        assert recs[0]["confidence_note"] == "High"

    def test_medium_confidence_note(self):
        recs = _gen("VSI", "low", confidence=0.77)
        assert recs[0]["confidence_note"] == "Medium"

    def test_low_confidence_note(self):
        recs = _gen("VSI", "low", confidence=0.60)
        assert "Low confidence" in recs[0]["confidence_note"]

    def test_low_confidence_hides_action(self):
        """Below SHAP_CONFIDENCE_LOW, corrective action should be withheld."""
        recs = _gen("VSI", "low", confidence=0.60)
        assert "verify manually" in recs[0]["action"].lower() or \
               "confidence too low" in recs[0]["action"].lower()


class TestUncertaintyState:
    def test_uncertainty_at_exactly_half(self):
        recs = _gen("VSI", "low", confidence=0.50)
        assert len(recs) == 1
        assert recs[0].get("state") == _UNCERTAIN_SENTINEL

    def test_uncertainty_near_half(self):
        """Within 0.02 of 0.5 should also trigger UNCERTAIN."""
        recs = _gen("VSI", "low", confidence=0.505)
        assert recs[0].get("state") == _UNCERTAIN_SENTINEL

    def test_no_uncertainty_above_threshold(self):
        recs = _gen("VSI", "low", confidence=0.55)
        assert recs[0].get("state") != _UNCERTAIN_SENTINEL


class TestRecommendationStructure:
    def test_all_required_keys_present(self):
        required = {"feature", "shap_contribution", "cause", "action", "severity",
                    "confidence_note", "confidence_pct"}
        recs = _gen("thermal_stress", "high", confidence=0.88)
        assert required.issubset(set(recs[0].keys()))

    def test_shap_contribution_type(self):
        recs = _gen("RoCoV", "high", shap_val=0.42)
        assert isinstance(recs[0]["shap_contribution"], float)

    def test_multiple_features_produce_multiple_recs(self):
        top_features = [("VSI", -0.42, "low"), ("RoCoV", 0.38, "high")]
        recs = generate_recommendation(top_features, {"VSI": 0.7, "RoCoV": 0.9}, 0.91)
        assert len(recs) == 2
