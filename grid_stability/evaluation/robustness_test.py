"""
evaluation/robustness_test.py

Injects Gaussian noise at multiple sigma levels and measures accuracy degradation.
Flags if accuracy drops > 15% at sigma = 0.10.
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.feature_engineering import load_uci_dataset, prepare_train_test
from utils.thresholds import ROBUSTNESS_MAX_DROP, ROBUSTNESS_SIGMAS

logger = logging.getLogger(__name__)

_MODEL_DIR = _ROOT / "models" / "saved"
_EVAL_DIR = _ROOT / "evaluation" / "outputs"
_EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_robustness_test() -> pd.DataFrame:
    """
    Inject Gaussian noise at increasing sigma levels, re-run predictions,
    record accuracy drop vs clean baseline.

    Returns:
        DataFrame with columns [sigma, rf_accuracy, xgb_accuracy,
                                 rf_drop, xgb_drop].
    """
    rf = joblib.load(_MODEL_DIR / "random_forest.pkl")
    xgb = joblib.load(_MODEL_DIR / "xgboost.pkl")

    df = load_uci_dataset()
    _, X_test, _, y_test = prepare_train_test(df, fit_scaler=False)

    baseline_rf = rf.score(X_test, y_test)
    baseline_xgb = xgb.score(X_test, y_test)
    logger.info("Baseline — RF: %.4f | XGB: %.4f", baseline_rf, baseline_xgb)

    records = []
    rng = np.random.default_rng(42)

    for sigma in ROBUSTNESS_SIGMAS:
        noise = rng.normal(0, sigma, X_test.shape)
        X_noisy = X_test + noise

        rf_acc = rf.score(X_noisy, y_test)
        xgb_acc = xgb.score(X_noisy, y_test)
        rf_drop = baseline_rf - rf_acc
        xgb_drop = baseline_xgb - xgb_acc

        records.append({
            "sigma": sigma,
            "rf_accuracy": round(rf_acc, 4),
            "xgb_accuracy": round(xgb_acc, 4),
            "rf_drop": round(rf_drop, 4),
            "xgb_drop": round(xgb_drop, 4),
        })

        if sigma == 0.10:
            if rf_drop > ROBUSTNESS_MAX_DROP:
                logger.warning(
                    "RF accuracy dropped %.2f%% at sigma=0.10 (threshold=%.0f%%)",
                    rf_drop * 100, ROBUSTNESS_MAX_DROP * 100,
                )
            if xgb_drop > ROBUSTNESS_MAX_DROP:
                logger.warning(
                    "XGB accuracy dropped %.2f%% at sigma=0.10 (threshold=%.0f%%)",
                    xgb_drop * 100, ROBUSTNESS_MAX_DROP * 100,
                )

    results_df = pd.DataFrame(records)
    return results_df, baseline_rf, baseline_xgb


def plot_robustness(results_df: pd.DataFrame) -> None:
    """Save accuracy degradation plot for both models."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(results_df["sigma"], results_df["rf_accuracy"], "o-",
            color="steelblue", label="Random Forest")
    ax.plot(results_df["sigma"], results_df["xgb_accuracy"], "s-",
            color="tomato", label="XGBoost")
    ax.axhline(results_df["rf_accuracy"].iloc[0], color="steelblue",
               linestyle="--", alpha=0.4, label="RF baseline")
    ax.axhline(results_df["xgb_accuracy"].iloc[0], color="tomato",
               linestyle="--", alpha=0.4, label="XGB baseline")
    ax.set_xlabel("Gaussian Noise σ")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Test — Accuracy vs Noise Level")
    ax.legend()
    ax.grid(alpha=0.3)
    out = _EVAL_DIR / "robustness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Robustness plot saved to %s", out)
    print(f"Robustness plot saved to {out}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_ROOT / "logs" / "project.log"),
        ],
    )
    results_df, baseline_rf, baseline_xgb = run_robustness_test()
    print(f"\nBaseline — RF: {baseline_rf:.4f} | XGB: {baseline_xgb:.4f}")
    print(results_df.to_string(index=False))
    plot_robustness(results_df)
