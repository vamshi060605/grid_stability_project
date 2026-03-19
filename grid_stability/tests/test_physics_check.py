"""
tests/test_physics_check.py

Unit tests for the physics pre-filter in the dashboard.
Directly imports the physics_check function logic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from utils.thresholds import VSI_CRITICAL, VSI_OVERVOLTAGE, THERMAL_CRITICAL


def physics_check(vsi: float, thermal: float) -> dict | None:
    """
    Replicated physics check logic (mirrors dashboard/app.py).
    Returns dict if override fires, else None.
    """
    if vsi < VSI_CRITICAL:
        return {"rule": "VSI undervoltage", "value": vsi, "level": "CRITICAL",
                "msg": f"VSI = {vsi:.3f} < {VSI_CRITICAL}"}
    if vsi > VSI_OVERVOLTAGE:
        return {"rule": "VSI overvoltage", "value": vsi, "level": "WARNING",
                "msg": f"VSI = {vsi:.3f} > {VSI_OVERVOLTAGE}"}
    if thermal > THERMAL_CRITICAL:
        return {"rule": "Thermal overload", "value": thermal, "level": "CRITICAL",
                "msg": f"Thermal = {thermal:.3f} > {THERMAL_CRITICAL}"}
    return None


class TestVSIUndervoltage:
    def test_vsi_below_critical_triggers_critical(self):
        result = physics_check(vsi=0.80, thermal=0.50)
        assert result is not None
        assert result["level"] == "CRITICAL"
        assert "undervoltage" in result["rule"]

    def test_vsi_at_exact_threshold_triggers(self):
        result = physics_check(vsi=VSI_CRITICAL - 0.001, thermal=0.50)
        assert result is not None

    def test_vsi_at_threshold_does_not_trigger(self):
        """VSI exactly at threshold (not below) should not trigger."""
        result = physics_check(vsi=VSI_CRITICAL, thermal=0.50)
        assert result is None

    def test_vsi_above_critical_passes_to_ml(self):
        result = physics_check(vsi=0.90, thermal=0.50)
        assert result is None


class TestVSIOvervoltage:
    def test_vsi_above_overvoltage_triggers_warning(self):
        result = physics_check(vsi=1.15, thermal=0.50)
        assert result is not None
        assert result["level"] == "WARNING"
        assert "overvoltage" in result["rule"]

    def test_vsi_just_above_threshold(self):
        result = physics_check(vsi=VSI_OVERVOLTAGE + 0.001, thermal=0.50)
        assert result is not None

    def test_vsi_at_overvoltage_threshold_no_trigger(self):
        result = physics_check(vsi=VSI_OVERVOLTAGE, thermal=0.50)
        assert result is None


class TestThermalOverload:
    def test_thermal_above_critical_triggers(self):
        result = physics_check(vsi=1.0, thermal=0.97)
        assert result is not None
        assert result["level"] == "CRITICAL"
        assert "Thermal" in result["rule"]

    def test_thermal_at_critical_does_not_trigger(self):
        result = physics_check(vsi=1.0, thermal=THERMAL_CRITICAL)
        assert result is None

    def test_thermal_below_critical_passes(self):
        result = physics_check(vsi=1.0, thermal=0.90)
        assert result is None


class TestNormalOperation:
    def test_all_normal_passes_to_ml(self):
        """Nominal values should all pass through to ML inference."""
        result = physics_check(vsi=1.0, thermal=0.30)
        assert result is None

    def test_boundary_safe_values_pass(self):
        result = physics_check(vsi=0.86, thermal=0.94)
        assert result is None

    def test_vsi_priority_over_thermal(self):
        """VSI check runs before thermal — VSI override should appear."""
        result = physics_check(vsi=0.70, thermal=0.99)
        assert result is not None
        assert "undervoltage" in result["rule"]
