"""
Module 1 — simulation/grid_simulator.py

Generates labelled fault scenarios using pandapower case14 network.
Fault types: normal, line_outage, load_surge, generator_trip, high_impedance.
"""
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from utils.thresholds import N_SAMPLES, CONVERGENCE_THRESHOLD, RANDOM_STATE

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "project.log"),
    ],
)
logger = logging.getLogger(__name__)

FAULT_TYPES = ["normal", "line_outage", "load_surge", "generator_trip", "high_impedance"]


def _build_network() -> pp.pandapowerNet:
    """Return a fresh copy of the IEEE 14-bus test case."""
    return pn.case14()


def _run_power_flow(net: pp.pandapowerNet) -> bool:
    """
    Run AC power flow. Returns True on convergence, False otherwise.

    Args:
        net: pandapower network object (modified in-place).

    Returns:
        True if power flow converged successfully.
    """
    try:
        pp.runpp(net, verbose=False, numba=False)
        return net.converged
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Power flow exception: %s", exc)
        return False


def _inject_fault(net: pp.pandapowerNet, fault_type: str, rng: random.Random) -> None:
    """
    Modify *net* in-place to simulate the given fault type.

    Args:
        net: pandapower network (modified in-place).
        fault_type: one of FAULT_TYPES.
        rng: seeded random.Random instance for reproducibility.
    """
    if fault_type == "normal":
        pass  # No modification — baseline stable scenario

    elif fault_type == "line_outage":
        # Disconnect a random line (avoid isolating buses with only one connection)
        if len(net.line) > 0:
            line_idx = rng.randint(0, len(net.line) - 1)
            net.line.at[line_idx, "in_service"] = False

    elif fault_type == "load_surge":
        # Spike active and reactive load by 150–200% on all load buses
        factor = rng.uniform(1.5, 2.0)
        net.load["p_mw"] *= factor
        net.load["q_mvar"] *= factor

    elif fault_type == "generator_trip":
        # Remove one generator (skip the slack bus to avoid non-convergence)
        non_slack_gens = net.gen[net.gen.index.isin(net.gen.index)].index.tolist()
        if non_slack_gens:
            gen_idx = rng.choice(non_slack_gens)
            net.gen.at[gen_idx, "in_service"] = False

    elif fault_type == "high_impedance":
        # Inject high-impedance load at a random PQ bus
        bus_idx = rng.randint(0, len(net.bus) - 1)
        pp.create_load(net, bus=bus_idx, p_mw=0.05, q_mvar=0.02)


def _extract_features(net: pp.pandapowerNet, fault_type: str) -> Optional[dict]:
    """
    Extract scalar features from a converged power flow result.

    Args:
        net: converged pandapower network.
        fault_type: fault label for this simulation.

    Returns:
        Dict of features, or None if extraction fails.
    """
    try:
        vm_pu_mean = float(net.res_bus["vm_pu"].mean())
        loading_pct_mean = float(net.res_line["loading_percent"].mean()) if len(net.res_line) > 0 else 0.0
        # Derived current in per-unit from line results
        i_pu_mean = float((net.res_line["i_ka"] / (net.line["max_i_ka"] + 1e-9)).mean()) if len(net.res_line) > 0 else 0.0

        label = 0 if fault_type == "normal" else 1

        return {
            "v_pu": vm_pu_mean,
            "i_pu": i_pu_mean,
            "loading_pct": loading_pct_mean,
            "fault_type": fault_type,
            "label": label,
        }
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Feature extraction failed: %s", exc)
        return None


def run_simulations(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Run *n_samples* fault simulations and return a labelled DataFrame.

    Args:
        n_samples: total number of simulation runs.
        random_state: seed for reproducibility.

    Returns:
        DataFrame with columns [v_pu, i_pu, loading_pct, fault_type, label].
    """
    rng = random.Random(random_state)
    np.random.seed(random_state)

    records: list[dict] = []
    stats = {"total_runs": 0, "converged": 0, "diverged": 0}

    for i in range(n_samples):
        fault_type = FAULT_TYPES[i % len(FAULT_TYPES)]  # Round-robin ensures all types
        net = _build_network()
        _inject_fault(net, fault_type, rng)

        stats["total_runs"] += 1
        converged = _run_power_flow(net)

        if not converged:
            stats["diverged"] += 1
            logger.debug("Diverged: sample=%d fault=%s", i, fault_type)
            continue

        stats["converged"] += 1
        row = _extract_features(net, fault_type)
        if row is not None:
            records.append(row)

    rate = stats["converged"] / max(stats["total_runs"], 1)
    logger.info(
        "Simulation complete — total=%d converged=%d diverged=%d rate=%.2f",
        stats["total_runs"], stats["converged"], stats["diverged"], rate,
    )

    if rate < CONVERGENCE_THRESHOLD:
        logger.warning("Convergence rate %.2f below threshold %.2f", rate, CONVERGENCE_THRESHOLD)

    df = pd.DataFrame(records)
    return df


def run_single_fault(fault_type: str, random_state: int = RANDOM_STATE) -> Optional[pd.DataFrame]:
    """
    Run a single simulation for a specific fault type. Used by dashboard buttons.

    Args:
        fault_type: one of FAULT_TYPES.
        random_state: seed for reproducibility.

    Returns:
        Single-row DataFrame or None on divergence.
    """
    rng = random.Random(random_state)
    net = _build_network()
    _inject_fault(net, fault_type, rng)

    if not _run_power_flow(net):
        logger.warning("Single fault simulation diverged: fault_type=%s", fault_type)
        return None

    row = _extract_features(net, fault_type)
    if row is None:
        return None

    return pd.DataFrame([row])


if __name__ == "__main__":
    logger.info("Running grid simulator standalone...")
    df = run_simulations()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Fault type distribution:\n{df['fault_type'].value_counts()}")
    print(df.head())
