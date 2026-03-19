"""
tests/test_feature_engineering.py

Unit tests for physics feature computations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import compute_physics_features


def _make_df(v_pu, i_pu=None, loading_pct=None):
    """Helper to build minimal simulation DataFrame."""
    n = len(v_pu)
    return pd.DataFrame({
        "v_pu": v_pu,
        "i_pu": i_pu if i_pu is not None else [0.5] * n,
        "loading_pct": loading_pct if loading_pct is not None else [50.0] * n,
        "fault_type": ["normal"] * n,
        "label": [0] * n,
    })


class TestVSI:
    def test_vsi_nominal(self):
        """VSI should equal v_pu / 1.0."""
        df = _make_df([1.0, 0.95, 1.05])
        result = compute_physics_features(df)
        expected = [1.0, 0.95, 1.05]
        # Drop first row (RoCoV NaN), check remaining
        assert list(result["VSI"]) == pytest.approx(expected[1:], rel=1e-5)

    def test_vsi_below_critical(self):
        """VSI below 0.85 should still compute correctly — not filtered here."""
        df = _make_df([0.8, 0.75, 0.80])
        result = compute_physics_features(df)
        assert all(result["VSI"] < 0.85)

    def test_vsi_values_are_positive(self):
        """VSI must always be positive."""
        df = _make_df([0.9, 1.0, 1.1, 1.05])
        result = compute_physics_features(df)
        assert (result["VSI"] > 0).all()


class TestRoCoV:
    def test_rocov_constant_voltage_is_zero(self):
        """RoCoV should be zero for constant voltage."""
        df = _make_df([1.0] * 5)
        result = compute_physics_features(df)
        assert np.allclose(result["RoCoV"].values, 0.0, atol=1e-12)

    def test_rocov_first_row_dropped(self):
        """First row must be dropped (NaN RoCoV), not filled."""
        df = _make_df([0.9, 1.0, 1.1, 0.95])
        result = compute_physics_features(df)
        # Original length 4 → after drop first row = 3
        assert len(result) == 3
        assert result["RoCoV"].isna().sum() == 0

    def test_rocov_detects_change(self):
        """RoCoV should be non-zero when voltage changes."""
        df = _make_df([1.0, 0.8, 1.2, 1.0])
        result = compute_physics_features(df)
        assert (result["RoCoV"] > 0).any()

    def test_rocov_symmetry(self):
        """RoCoV is abs(diff) — symmetric for up and down changes."""
        df_up = _make_df([1.0, 1.2, 1.0])
        df_dn = _make_df([1.0, 0.8, 1.0])
        r_up = compute_physics_features(df_up)["RoCoV"].values
        r_dn = compute_physics_features(df_dn)["RoCoV"].values
        assert r_up == pytest.approx(r_dn, rel=1e-5)


class TestFaultImpedance:
    def test_fault_impedance_ohms_law(self):
        """fault_impedance = v_pu / (i_pu + 1e-9)."""
        df = _make_df([1.0, 1.0, 1.0], i_pu=[2.0, 2.0, 2.0])
        result = compute_physics_features(df)
        expected = 1.0 / (2.0 + 1e-9)
        assert np.allclose(result["fault_impedance"].values, expected, rtol=1e-4)

    def test_fault_impedance_zero_current(self):
        """Zero current should not cause division error (epsilon guard)."""
        df = _make_df([1.0, 1.0, 1.0], i_pu=[0.0, 0.0, 0.0])
        result = compute_physics_features(df)
        # Value should be finite (large but not inf/nan)
        assert result["fault_impedance"].isfinite().all() if hasattr(
            result["fault_impedance"], "isfinite"
        ) else np.isfinite(result["fault_impedance"].values).all()

    def test_fault_impedance_positive(self):
        """Impedance must be non-negative."""
        df = _make_df([0.9, 1.0, 1.1, 0.8], i_pu=[0.1, 0.5, 1.0, 0.3])
        result = compute_physics_features(df)
        assert (result["fault_impedance"] >= 0).all()


class TestThermalStress:
    def test_thermal_stress_scaling(self):
        """thermal_stress = loading_pct / 100.0."""
        df = _make_df([1.0, 1.0, 1.0], loading_pct=[50.0, 95.0, 100.0])
        result = compute_physics_features(df)
        assert list(result["thermal_stress"]) == pytest.approx([0.95, 1.0], rel=1e-5)

    def test_thermal_stress_range(self):
        """thermal_stress should be in [0, 1] for valid loading_pct."""
        df = _make_df([1.0] * 6, loading_pct=[0.0, 25.0, 50.0, 75.0, 95.0, 100.0])
        result = compute_physics_features(df)
        assert (result["thermal_stress"] >= 0).all()
        assert (result["thermal_stress"] <= 1.0).all()
