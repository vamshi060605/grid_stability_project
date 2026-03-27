"""
tests/test_stability_metrics.py

Unit tests for Stability Margin Score and Fault Severity Index.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import pytest

from stability_metrics import compute_stability_margin, compute_fsi


# ── Tests for compute_stability_margin ───────────────────────────────────────

class TestStabilityMargin:
    def test_all_safe_values_near_one(self):
        """Maximum safe: VSI=1.0, RoCoV=0.0, thermal=0.0 → score ≈ 1.0."""
        score = compute_stability_margin(vsi=1.0, rocov=0.0, thermal_stress=0.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_all_worst_case_zero(self):
        """Worst case: VSI=0, RoCoV=rocov_max, thermal=1.0 → score = 0.0."""
        score = compute_stability_margin(vsi=0.0, rocov=1.0, thermal_stress=1.0, rocov_max=1.0)
        assert score == 0.0

    def test_midpoint_inputs_near_half(self):
        """Midpoint (VSI=0.5, RoCoV=half_max, thermal=0.5) → score ≈ 0.5."""
        score = compute_stability_margin(vsi=0.5, rocov=0.025, thermal_stress=0.5, rocov_max=0.05)
        assert abs(score - 0.5) < 0.05

    def test_rocov_exceeding_max_clamped(self):
        """RoCoV > rocov_max → RoCoV_norm clamped at 1.0, no exception."""
        score = compute_stability_margin(vsi=1.0, rocov=10.0, thermal_stress=0.0, rocov_max=0.05)
        assert 0.0 <= score <= 1.0

    def test_output_always_clamped(self):
        """100 random inputs → output always in [0.0, 1.0]."""
        rng = random.Random(42)
        for _ in range(100):
            vsi = rng.uniform(-0.5, 1.5)
            rocov = rng.uniform(0.0, 2.0)
            thermal = rng.uniform(-0.5, 1.5)
            score = compute_stability_margin(vsi, rocov, thermal, rocov_max=0.05)
            assert 0.0 <= score <= 1.0


# ── Tests for compute_fsi ────────────────────────────────────────────────────

_BOUNDS = {
    "VSI": {"min": 0.0, "max": 1.5},
    "RoCoV": {"min": 0.0, "max": 0.05},
    "thermal_stress": {"min": 0.0, "max": 1.0},
}

class TestFSI:
    def test_all_features_at_max(self):
        """All features at max bounds → FSI = 1.0."""
        weights = {"VSI": 0.4, "RoCoV": 0.3, "thermal_stress": 0.3}
        vector = {"VSI": 1.5, "RoCoV": 0.05, "thermal_stress": 1.0}
        fsi = compute_fsi(vector, weights, _BOUNDS)
        assert fsi == pytest.approx(1.0, abs=0.001)

    def test_all_features_at_min(self):
        """All features at min bounds → FSI = 0.0."""
        weights = {"VSI": 0.4, "RoCoV": 0.3, "thermal_stress": 0.3}
        vector = {"VSI": 0.0, "RoCoV": 0.0, "thermal_stress": 0.0}
        fsi = compute_fsi(vector, weights, _BOUNDS)
        assert fsi == 0.0

    def test_single_feature_midpoint(self):
        """Single feature weight=1.0 at midpoint → FSI = 0.5."""
        weights = {"thermal_stress": 1.0}
        vector = {"thermal_stress": 0.5}
        fsi = compute_fsi(vector, weights, _BOUNDS)
        assert fsi == pytest.approx(0.5, abs=0.001)

    def test_feature_outside_bounds_clamped(self):
        """Feature value outside bounds → clamped, no exception."""
        weights = {"VSI": 1.0}
        vector = {"VSI": 10.0}  # Way above max=1.5
        fsi = compute_fsi(vector, weights, _BOUNDS)
        assert 0.0 <= fsi <= 1.0

    def test_empty_weights_zero(self):
        """Empty shap_weights dict → FSI = 0.0."""
        fsi = compute_fsi({"VSI": 0.9}, {}, _BOUNDS)
        assert fsi == 0.0
