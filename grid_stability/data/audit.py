"""
data/audit.py — Data Quality Audit

Validates UCI dataset and simulation data for:
- Shape, dtypes, missing values
- Class imbalance (flags >80/20)
- Duplicate rows
- High-correlation features (>0.95)
- Feature discriminative power
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.feature_engineering import load_uci_dataset

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _ROOT / "data" / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_uci_dataset(df: pd.DataFrame) -> dict:
    """
    Run quality checks on UCI dataset.

    Args:
        df: loaded UCI DataFrame.

    Returns:
        Dict of audit results.
    """
    results = {}
    print(f"\n{'═'*60}")
    print("UCI DATASET AUDIT")
    print(f"{'═'*60}")
    print(f"Shape:  {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")

    # Missing values
    missing = df.isna().sum()
    results["missing"] = missing.to_dict()
    print(f"\nMissing values per column:\n{missing[missing > 0] if missing.any() else 'None'}")

    # Class distribution
    label_dist = df["label"].value_counts(normalize=True)
    majority = label_dist.max()
    results["class_imbalance_flag"] = majority > 0.80
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    if results["class_imbalance_flag"]:
        logger.warning("CLASS IMBALANCE: majority class = %.1f%%", majority * 100)
        print(f"⚠ IMBALANCED: majority class = {majority*100:.1f}%")

    # Duplicates
    n_dup = df.duplicated().sum()
    results["duplicates"] = int(n_dup)
    print(f"\nDuplicate rows: {n_dup}")

    # Correlation matrix — flag pairs >0.95
    feature_cols = [c for c in df.columns if c != "label"]
    corr = df[feature_cols].corr().abs()
    high_corr_pairs = [
        (feature_cols[i], feature_cols[j], corr.iloc[i, j])
        for i in range(len(feature_cols))
        for j in range(i + 1, len(feature_cols))
        if corr.iloc[i, j] > 0.95
    ]
    results["high_correlation_pairs"] = high_corr_pairs
    if high_corr_pairs:
        print(f"\nHigh-correlation pairs (>0.95):")
        for f1, f2, c in high_corr_pairs:
            print(f"  {f1} ↔ {f2}: {c:.3f}")
    else:
        print("\nNo high-correlation pairs (>0.95) found.")

    return results


def audit_simulation_data(sim_df: pd.DataFrame) -> None:
    """
    Validate simulation output for quality.

    Args:
        sim_df: DataFrame from grid_simulator.run_simulations().
    """
    print(f"\n{'═'*60}")
    print("SIMULATION DATA AUDIT")
    print(f"{'═'*60}")
    print(f"Shape: {sim_df.shape}")

    nan_count = sim_df.isna().sum().sum()
    inf_count = np.isinf(sim_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    print(f"\nFault type distribution:\n{sim_df['fault_type'].value_counts()}")
    print(f"\nLabel distribution:\n{sim_df['label'].value_counts()}")

    # Check ratio
    ratio = sim_df["label"].value_counts(normalize=True)
    majority = ratio.max()
    if majority > 0.70:
        logger.warning("Simulation data ratio imbalanced: majority=%.1f%%", majority * 100)
        print(f"⚠ Ratio warning: majority class = {majority*100:.1f}%")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Plot per-feature histograms split by class label.
    Flags features with >80% class overlap as low discriminative power.

    Args:
        df: UCI DataFrame with 'label' column.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for label, color in [(0, "steelblue"), (1, "tomato")]:
            subset = df[df["label"] == label][col]
            ax.hist(subset, bins=30, alpha=0.5, color=color,
                    label="Stable" if label == 0 else "Unstable", density=True)

        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Class", fontsize=13)
    plt.tight_layout()
    out = _OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature distribution plot saved to %s", out)
    print(f"\nFeature distribution plot saved to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_uci_dataset()
    results = audit_uci_dataset(df)
    plot_feature_distributions(df)
    print("\nAudit complete.")
