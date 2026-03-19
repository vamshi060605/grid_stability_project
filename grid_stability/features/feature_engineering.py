"""
Module 2 — features/feature_engineering.py

Computes physics-derived features from simulation output and UCI dataset.
Implements strict leakage prevention: scaler is fit ONLY on training split.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from utils.thresholds import TEST_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_DATA_DIR = _ROOT / "data"
_MODEL_DIR = _ROOT / "models" / "saved"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

UCI_PATH = _DATA_DIR / "grid_stability.csv"
SCALER_PATH = _MODEL_DIR / "scaler.pkl"


# ── Physics feature computation ───────────────────────────────────────────────

def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-derived features to raw simulation DataFrame.

    VSI     = v_pu / 1.0  (Voltage Stability Index, nominal = 1.0 pu)
    fault_impedance = v_pu / (i_pu + 1e-9)  (Ohm's Law, avoids div-by-zero)
    RoCoV   = abs(delta_v / delta_t)         (Rate of Change of Voltage)
    thermal_stress = loading_pct / 100.0

    Args:
        df: raw simulation DataFrame with [v_pu, i_pu, loading_pct, label].

    Returns:
        DataFrame with new physics columns; first row dropped (RoCoV NaN).
    """
    df = df.copy()

    df["VSI"] = df["v_pu"] / 1.0
    df["fault_impedance"] = df["v_pu"] / (df["i_pu"] + 1e-9)
    df["RoCoV"] = df["v_pu"].diff().abs()  # delta_t = 1 simulation step
    df["thermal_stress"] = df["loading_pct"] / 100.0

    # Drop first row — RoCoV is NaN; NEVER fill forward (leakage risk)
    df = df.dropna(subset=["RoCoV"]).reset_index(drop=True)

    assert df["RoCoV"].isna().sum() == 0, "RoCoV must have no NaN after first-row drop"
    logger.info("Physics features computed. Shape: %s", df.shape)
    return df


# ── UCI dataset loading & preparation ────────────────────────────────────────

def load_uci_dataset() -> pd.DataFrame:
    """
    Load and clean the UCI Electrical Grid Stability dataset.

    Expected columns: tau1, tau2, tau3, tau4, p1, p2, p3, p4,
                      g1, g2, g3, g4, stab, stabf

    Returns:
        Cleaned DataFrame with binary label column 'label'.

    Raises:
        FileNotFoundError: if data/grid_stability.csv does not exist.
    """
    if not UCI_PATH.exists():
        raise FileNotFoundError(
            f"UCI dataset not found at {UCI_PATH}.\n"
            "Download from: https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data\n"
            "Save as: data/grid_stability.csv"
        )

    df = pd.read_csv(UCI_PATH)
    logger.info("UCI dataset loaded. Shape: %s", df.shape)

    # Remove duplicates automatically
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("Removed %d duplicate rows from UCI dataset", n_before - len(df))

    # Map string label → binary int
    if "stabf" in df.columns:
        df["label"] = (df["stabf"] == "unstable").astype(int)
        df = df.drop(columns=["stabf"])
    elif "label" not in df.columns:
        # Fallback: use numeric stab column (negative = unstable)
        df["label"] = (df["stab"] < 0).astype(int)

    if "stab" in df.columns:
        df = df.drop(columns=["stab"])

    # Check for remaining missing values
    missing = df.isna().sum().sum()
    if missing > 0:
        logger.warning("UCI dataset has %d missing values — dropping rows", missing)
        df = df.dropna().reset_index(drop=True)

    logger.info("UCI dataset cleaned. Shape: %s | Label distribution:\n%s",
                df.shape, df["label"].value_counts().to_dict())
    return df


# ── Train/test split + scaling ────────────────────────────────────────────────

def prepare_train_test(
    df: pd.DataFrame,
    label_col: str = "label",
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/test and apply StandardScaler fitted ONLY on training data.

    Leakage prevention assertions:
    - Split happens BEFORE scaling
    - Scaler is fit on X_train only
    - X_test is only transformed, never fitted

    Args:
        df: cleaned DataFrame with features + label column.
        label_col: name of target column.
        fit_scaler: if True, fit and save scaler; if False, load saved scaler.

    Returns:
        (X_train, X_test, y_train, y_test) — all scaled.
    """
    feature_cols = [c for c in df.columns if c not in [label_col, "fault_type"]]
    X = df[feature_cols]
    y = df[label_col]

    # CRITICAL: split BEFORE scaling — prevents leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(X_train)  # Fit ONLY on training data
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Scaler fit on training set and saved to %s", SCALER_PATH)
    else:
        assert SCALER_PATH.exists(), f"Scaler not found at {SCALER_PATH}. Run train.py first."
        scaler = joblib.load(SCALER_PATH)
        logger.info("Loaded scaler from %s", SCALER_PATH)

    # Leakage assertions
    assert hasattr(scaler, "mean_"), "Scaler must be fit before transform"

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    logger.info(
        "Train/test split complete. Train: %d | Test: %d",
        len(X_train_scaled), len(X_test_scaled),
    )
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_feature_names_uci() -> list:
    """Return expected UCI feature column names (excluding label)."""
    return ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Feature Engineering Standalone Run ===")

    # Test UCI pipeline
    uci_df = load_uci_dataset()
    X_train, X_test, y_train, y_test = prepare_train_test(uci_df)

    print(f"\nUCI feature shapes — train: {X_train.shape} | test: {X_test.shape}")
    print(f"NaN in train: {X_train.isna().sum().sum()} | NaN in test: {X_test.isna().sum().sum()}")
    print(f"Feature columns: {X_train.columns.tolist()}")
