"""
Module 3b — models/evaluate.py

Full evaluation suite for Random Forest and XGBoost.
Generates metrics, plots, latency benchmark, and SHAP consistency check.
"""
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
)
from sklearn.model_selection import learning_curve
import shap

from features.feature_engineering import load_uci_dataset, prepare_train_test
from utils.thresholds import N_ESTIMATORS, RANDOM_STATE

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_MODEL_DIR = _ROOT / "models" / "saved"
_EVAL_DIR = _ROOT / "evaluation" / "outputs"
_EVAL_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH = _MODEL_DIR / "random_forest.pkl"
XGB_PATH = _MODEL_DIR / "xgboost.pkl"


def _load_models_and_data():
    """Load saved models and prepare test data."""
    assert RF_PATH.exists(), f"Model not found: {RF_PATH}. Run models/train.py first."
    assert XGB_PATH.exists(), f"Model not found: {XGB_PATH}. Run models/train.py first."

    rf = joblib.load(RF_PATH)
    xgb = joblib.load(XGB_PATH)

    df = load_uci_dataset()
    X_train, X_test, y_train, y_test = prepare_train_test(df, fit_scaler=False)
    return rf, xgb, X_train, X_test, y_train, y_test


def _compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    """Compute classification metrics for a single model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_w": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_w": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_w": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_m": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_m": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_m": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }


def print_metrics_table(rf_metrics: dict, xgb_metrics: dict) -> None:
    """Print side-by-side formatted metrics comparison table."""
    header = f"{'Metric':<25} {'Random Forest':>15} {'XGBoost':>15}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for key in rf_metrics:
        if key == "model":
            continue
        rf_val = rf_metrics[key]
        xgb_val = xgb_metrics[key]
        print(f"{key:<25} {rf_val:>15.4f} {xgb_val:>15.4f}")
    print(sep)


def plot_confusion_matrices(rf, xgb, X_test, y_test) -> None:
    """Save normalised side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, name in zip(axes, [rf, xgb], ["Random Forest", "XGBoost"]):
        cm = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=["Stable", "Unstable"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\n(Normalised)")
    plt.tight_layout()
    out = _EVAL_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_roc_curves(rf, xgb, X_test, y_test) -> None:
    """Save AUC-ROC curves for both models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Baseline (AUC = 0.50)")
    for model, name, color in [(rf, "Random Forest", "steelblue"), (xgb, "XGBoost", "tomato")]:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC = {auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AUC-ROC Curves")
    ax.legend()
    out = _EVAL_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_learning_curves(rf, xgb, X_train, y_train) -> None:
    """Save learning curves (train vs val score vs training size)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model, name in zip(axes, [rf, xgb], ["Random Forest", "XGBoost"]):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        ax.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                        train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="steelblue")
        ax.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                        val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="tomato")
        ax.plot(train_sizes, train_scores.mean(1), "o-", color="steelblue", label="Train")
        ax.plot(train_sizes, val_scores.mean(1), "o-", color="tomato", label="Validation")
        ax.set_title(f"Learning Curve — {name}")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.legend()
    plt.tight_layout()
    out = _EVAL_DIR / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def plot_feature_importance(rf, feature_names: list) -> None:
    """Save top-10 RF feature importances bar chart."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        [importances[i] for i in reversed(indices)],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Top 10 Feature Importances")
    plt.tight_layout()
    out = _EVAL_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def latency_benchmark(rf, xgb, X_test: pd.DataFrame, n_runs: int = 1000) -> None:
    """Time 1000 single predictions per model and report percentiles."""
    print("\n── Latency Benchmark (ms) ──────────────────────────────")
    for model, name in [(rf, "Random Forest"), (xgb, "XGBoost")]:
        times = []
        for i in range(n_runs):
            row = X_test.iloc[[i % len(X_test)]]
            t0 = time.perf_counter()
            model.predict(row)
            times.append((time.perf_counter() - t0) * 1000)
        times_arr = np.array(times)
        print(
            f"{name:<16} mean={times_arr.mean():.3f}ms  "
            f"median={np.median(times_arr):.3f}ms  "
            f"p95={np.percentile(times_arr, 95):.3f}ms  "
            f"p99={np.percentile(times_arr, 99):.3f}ms"
        )


def shap_consistency_check(rf, X_test: pd.DataFrame) -> None:
    """
    Compare top-5 SHAP features vs top-5 RF feature importances.
    Logs WARNING if Spearman rank correlation < 0.7.
    """
    logger.info("Running SHAP consistency check...")
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test.iloc[:200])
        # shap_values may be a list [class0, class1]; take class-1 mean abs
        if isinstance(shap_values, list):
            shap_mean = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_mean = np.abs(shap_values).mean(axis=0)

        shap_top5 = np.argsort(shap_mean)[::-1][:5].tolist()
        rf_top5 = np.argsort(rf.feature_importances_)[::-1][:5].tolist()

        # Spearman correlation of the ranking positions
        feature_names = X_test.columns.tolist()
        all_features = list(set(shap_top5 + rf_top5))
        shap_ranks = {f: shap_top5.index(f) if f in shap_top5 else 5 for f in all_features}
        rf_ranks = {f: rf_top5.index(f) if f in rf_top5 else 5 for f in all_features}
        from scipy.stats import spearmanr
        corr, _ = spearmanr(list(shap_ranks.values()), list(rf_ranks.values()))

        print(f"\n── SHAP Consistency Check ─────────────────────────────")
        print(f"Top-5 SHAP features: {[feature_names[i] for i in shap_top5]}")
        print(f"Top-5 RF importances: {[feature_names[i] for i in rf_top5]}")
        print(f"Spearman correlation: {corr:.3f}")
        if corr < 0.7:
            logger.warning("SHAP ↔ RF importance correlation %.3f < 0.7", corr)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SHAP consistency check failed: %s", exc)


def run_full_evaluation() -> None:
    """Run the complete evaluation pipeline."""
    rf, xgb, X_train, X_test, y_train, y_test = _load_models_and_data()
    feature_names = X_test.columns.tolist()

    logger.info("Computing metrics...")
    rf_metrics = _compute_metrics(rf, X_test, y_test, "Random Forest")
    xgb_metrics = _compute_metrics(xgb, X_test, y_test, "XGBoost")
    print_metrics_table(rf_metrics, xgb_metrics)

    logger.info("Generating plots...")
    plot_confusion_matrices(rf, xgb, X_test, y_test)
    plot_roc_curves(rf, xgb, X_test, y_test)
    plot_learning_curves(rf, xgb, X_train, y_train)
    plot_feature_importance(rf, feature_names)

    latency_benchmark(rf, xgb, X_test)
    shap_consistency_check(rf, X_test)

    logger.info("All evaluation outputs saved to %s", _EVAL_DIR)
    print(f"\nAll plots saved to: {_EVAL_DIR}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(__file__).parent.parent / "logs" / "project.log"),
        ],
    )
    run_full_evaluation()
