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


# ── Operator Feedback Tests ──────────────────────────────────────────────────


import json
import shutil

import joblib
import numpy as np
import pandas as pd

from models.train import update_model_with_feedback, _load_feedback_log, _save_feedback_log
from utils.thresholds import FEEDBACK_BUFFER_MAX


def _setup_feedback_env(tmp_path):
    """Copy model, scaler, and create empty feedback log in tmp_path."""
    _ROOT = Path(__file__).parent.parent
    model_src = _ROOT / "models" / "saved" / "random_forest.pkl"
    scaler_src = _ROOT / "models" / "saved" / "scaler.pkl"

    model_dst = tmp_path / "random_forest.pkl"
    scaler_dst = tmp_path / "scaler.pkl"
    log_dst = tmp_path / "feedback_log.json"

    shutil.copy(model_src, model_dst)
    shutil.copy(scaler_src, scaler_dst)

    # Create empty feedback log
    empty_log = {
        "samples": [],
        "total_confirmations": 0,
        "total_false_alarms": 0,
        "last_retrain_accuracy": None,
    }
    log_dst.write_text(json.dumps(empty_log, indent=2), encoding="utf-8")

    # Build a sample UCI-space feature instance (unscaled)
    feature_names = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4",
                     "g1", "g2", "g3", "g4"]
    sample_vals = [5.74, 5.76, 5.74, 5.74, 3.76, -1.25, -1.25, -1.26,
                   0.57, 0.57, 0.57, 0.57]
    X_instance = pd.DataFrame([sample_vals], columns=feature_names)

    return model_dst, scaler_dst, log_dst, X_instance


class TestOperatorFeedback:
    def test_fix_it_maintains_accuracy(self, tmp_path):
        """Fix It with correct unstable instance should maintain or improve accuracy."""
        model_dst, scaler_dst, log_dst, X_instance = _setup_feedback_env(tmp_path)
        result = update_model_with_feedback(
            model_path=model_dst,
            X_instance=X_instance,
            confirmed_label=1,
            scaler_path=scaler_dst,
            feedback_log_path=log_dst,
        )
        # Reinforcing correct prediction should not significantly hurt accuracy
        assert result["new_accuracy"] >= result["old_accuracy"] - 0.02

    def test_false_alarm_adjusts_model(self, tmp_path):
        """False Alarm with incorrect prediction should adjust model without error."""
        model_dst, scaler_dst, log_dst, X_instance = _setup_feedback_env(tmp_path)
        result = update_model_with_feedback(
            model_path=model_dst,
            X_instance=X_instance,
            confirmed_label=0,
            scaler_path=scaler_dst,
            feedback_log_path=log_dst,
        )
        assert "new_accuracy" in result
        assert "delta" in result
        assert result["samples_added"] == 1

    def test_feedback_buffer_max_size(self, tmp_path):
        """Feedback buffer should respect FEEDBACK_BUFFER_MAX (200)."""
        model_dst, scaler_dst, log_dst, X_instance = _setup_feedback_env(tmp_path)

        # Pre-fill log with 210 dummy samples
        log = _load_feedback_log(log_dst)
        for i in range(210):
            log["samples"].append({
                "features": {f"f{j}": float(j) for j in range(12)},
                "label": 1,
                "verdict": "CONFIRMED",
                "timestamp": f"2026-01-01T00:00:{i:02d}",
                "weight": 10,
            })
        _save_feedback_log(log_dst, log)

        # One more feedback call should trigger trim
        result = update_model_with_feedback(
            model_path=model_dst,
            X_instance=X_instance,
            confirmed_label=1,
            scaler_path=scaler_dst,
            feedback_log_path=log_dst,
        )
        log_after = _load_feedback_log(log_dst)
        assert len(log_after["samples"]) <= FEEDBACK_BUFFER_MAX

    def test_feedback_log_correctly_updated(self, tmp_path):
        """feedback_log.json should have correct structure after one call."""
        model_dst, scaler_dst, log_dst, X_instance = _setup_feedback_env(tmp_path)
        update_model_with_feedback(
            model_path=model_dst,
            X_instance=X_instance,
            confirmed_label=1,
            scaler_path=scaler_dst,
            feedback_log_path=log_dst,
        )
        log = json.loads(log_dst.read_text(encoding="utf-8"))

        assert log["total_confirmations"] == 1
        assert log["total_false_alarms"] == 0
        assert log["last_retrain_accuracy"] is not None
        assert len(log["samples"]) == 1

        sample = log["samples"][0]
        assert "features" in sample
        assert "label" in sample
        assert sample["label"] == 1
        assert sample["verdict"] == "CONFIRMED"
        assert "timestamp" in sample
        assert sample["weight"] == 10
