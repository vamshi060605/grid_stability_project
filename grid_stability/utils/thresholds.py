"""
Centralised threshold constants for the grid stability project.
All values are loaded from config.yaml — no magic numbers anywhere.
"""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)

_t = _cfg["thresholds"]
_m = _cfg["model"]
_s = _cfg["simulation"]
_d = _cfg["dashboard"]

# ── Voltage thresholds (IEEE 1159 / grid operator standards) ──────────────────
VSI_CRITICAL: float = _t["vsi_critical"]       # IEEE 1159 undervoltage threshold
VSI_OVERVOLTAGE: float = _t["vsi_overvoltage"]  # IEEE 1159 overvoltage threshold

# ── Thermal threshold ─────────────────────────────────────────────────────────
THERMAL_CRITICAL: float = _t["thermal_critical"]  # IEC 60076 transformer rating limit

# ── SHAP / confidence thresholds ─────────────────────────────────────────────
SHAP_CONFIDENCE_LOW: float = _t["shap_confidence_low"]   # Below this → low-conf note
SHAP_CONFIDENCE_HIGH: float = 0.85                        # Above this → high-conf note

# ── Robustness test ───────────────────────────────────────────────────────────
ROBUSTNESS_MAX_DROP: float = _t["robustness_max_drop"]  # Max tolerated accuracy drop

# ── Simulation ────────────────────────────────────────────────────────────────
N_SAMPLES: int = _s["n_samples"]
CONVERGENCE_THRESHOLD: float = _s["convergence_threshold"]

# ── Model ─────────────────────────────────────────────────────────────────────
RANDOM_STATE: int = _m["random_state"]
TEST_SIZE: float = _m["test_size"]
N_ESTIMATORS: int = _m["n_estimators"]

# ── Dashboard ─────────────────────────────────────────────────────────────────
HISTORY_LENGTH: int = _d["history_length"]
AUTO_REFRESH_SECONDS: int = _d["auto_refresh_seconds"]

# ── Noise levels for robustness tests ────────────────────────────────────────
ROBUSTNESS_SIGMAS: list = [0.01, 0.05, 0.10, 0.20]
