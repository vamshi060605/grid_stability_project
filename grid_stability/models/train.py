"""
Module 3a — models/train.py

Trains Random Forest and XGBoost on UCI dataset.
All hyperparameters loaded from config.yaml.
"""
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from features.feature_engineering import load_uci_dataset, prepare_train_test
from utils.thresholds import N_ESTIMATORS, RANDOM_STATE

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_MODEL_DIR = _ROOT / "models" / "saved"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH = _MODEL_DIR / "random_forest.pkl"
XGB_PATH = _MODEL_DIR / "xgboost.pkl"


def train_models() -> tuple:
    """
    Train Random Forest and XGBoost classifiers on UCI grid stability data.

    Returns:
        Tuple of (rf_model, xgb_model, X_test, y_test).
    """
    logger.info("Loading UCI dataset...")
    df = load_uci_dataset()

    logger.info("Preparing train/test split with scaling...")
    X_train, X_test, y_train, y_test = prepare_train_test(df, fit_scaler=True)

    # ── Random Forest ─────────────────────────────────────────────────────────
    logger.info("Training Random Forest (n_estimators=%d)...", N_ESTIMATORS)
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_PATH)
    logger.info("Random Forest saved to %s", RF_PATH)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, XGB_PATH)
    logger.info("XGBoost saved to %s", XGB_PATH)
    
    import json
    training_means = X_train.mean().to_dict()
    means_path = _MODEL_DIR / "training_means.json"
    with open(means_path, "w") as f:
        json.dump(training_means, f, indent=2)
    logger.info(f"Training means saved to {means_path}")

    rf_train_acc = rf.score(X_train, y_train)
    xgb_train_acc = xgb.score(X_train, y_train)
    logger.info("Training complete — RF train acc: %.4f | XGB train acc: %.4f",
                rf_train_acc, xgb_train_acc)

    return rf, xgb, X_test, y_test


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(__file__).parent.parent / "logs" / "project.log"),
        ],
    )
    rf, xgb, X_test, y_test = train_models()
    print(f"\nTraining complete.")
    print(f"RF test accuracy:  {rf.score(X_test, y_test):.4f}")
    print(f"XGB test accuracy: {xgb.score(X_test, y_test):.4f}")
    print(f"Models saved to: {_MODEL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Operator-in-the-loop feedback mechanism.
# Each confirmed/rejected prediction is weighted as 10 training samples and
# added to a rolling feedback buffer. The model is partially retrained on
# UCI data + feedback buffer on every operator action. This simulates
# production ML systems where operator expertise continuously refines model
# behaviour without full retraining pipelines.
# ─────────────────────────────────────────────────────────────────────────────

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from utils.thresholds import FEEDBACK_WEIGHT, FEEDBACK_BUFFER_MAX, FEEDBACK_BUFFER_TRIM


def _load_feedback_log(feedback_log_path: Path) -> dict:
    """Load feedback log from JSON, creating it if missing."""
    if not feedback_log_path.exists():
        empty = {
            "samples": [],
            "total_confirmations": 0,
            "total_false_alarms": 0,
            "last_retrain_accuracy": None,
        }
        feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_log_path.write_text(json.dumps(empty, indent=2), encoding="utf-8")
        return empty
    return json.loads(feedback_log_path.read_text(encoding="utf-8"))


def _save_feedback_log(feedback_log_path: Path, log: dict) -> None:
    """Save feedback log to JSON."""
    feedback_log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def update_model_with_feedback(
    model_path: Path,
    X_instance: pd.DataFrame,
    confirmed_label: int,
    scaler_path: Path,
    feedback_log_path: Path,
) -> dict:
    """
    Update Random Forest model using operator feedback.

    Retrains on original UCI data + weighted feedback buffer so the model
    incorporates operator corrections without drifting from the base
    distribution.

    Args:
        model_path: Path to saved random_forest.pkl.
        X_instance: Single-row DataFrame (unscaled, UCI feature space).
        confirmed_label: 1 = operator confirms unstable, 0 = false alarm.
        scaler_path: Path to saved scaler.pkl.
        feedback_log_path: Path to data/feedback_log.json.

    Returns:
        Dict with old_accuracy, new_accuracy, delta, samples_added,
        total_feedback_count.
    """
    # ── Load existing model & scaler ──────────────────────────────────────────
    rf_old = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # ── Compute old accuracy on UCI test set ──────────────────────────────────
    from features.feature_engineering import load_uci_dataset, prepare_train_test
    df_uci = load_uci_dataset()
    X_train_uci, X_test_uci, y_train_uci, y_test_uci = prepare_train_test(
        df_uci, fit_scaler=False,
    )
    old_accuracy = accuracy_score(y_test_uci, rf_old.predict(X_test_uci))

    # ── Scale the feedback instance ───────────────────────────────────────────
    feature_names = X_train_uci.columns.tolist()
    X_scaled = pd.DataFrame(
        scaler.transform(X_instance[feature_names]),
        columns=feature_names,
    )

    # ── Update feedback log ───────────────────────────────────────────────────
    log = _load_feedback_log(feedback_log_path)
    verdict = "CONFIRMED" if confirmed_label == 1 else "REJECTED"

    sample_entry = {
        "features": X_scaled.iloc[0].to_dict(),
        "label": confirmed_label,
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weight": FEEDBACK_WEIGHT,
    }
    log["samples"].append(sample_entry)

    if confirmed_label == 1:
        log["total_confirmations"] += 1
    else:
        log["total_false_alarms"] += 1

    # Trim buffer if too large — drop oldest entries
    if len(log["samples"]) > FEEDBACK_BUFFER_MAX:
        log["samples"] = log["samples"][FEEDBACK_BUFFER_TRIM:]

    # ── Build combined training set: UCI + weighted feedback buffer ────────────
    feedback_rows = []
    feedback_labels = []
    for s in log["samples"]:
        row_vals = [s["features"].get(f, 0.0) for f in feature_names]
        for _ in range(s.get("weight", FEEDBACK_WEIGHT)):
            feedback_rows.append(row_vals)
            feedback_labels.append(s["label"])

    if feedback_rows:
        X_feedback = pd.DataFrame(feedback_rows, columns=feature_names)
        y_feedback = pd.Series(feedback_labels, name="label")
        X_combined = pd.concat([X_train_uci, X_feedback], ignore_index=True)
        y_combined = pd.concat([y_train_uci, y_feedback], ignore_index=True)
    else:
        X_combined = X_train_uci
        y_combined = y_train_uci

    # ── Retrain RF on combined data ───────────────────────────────────────────
    from utils.thresholds import N_ESTIMATORS, RANDOM_STATE as RS
    rf_new = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RS,
        n_jobs=-1,
    )
    rf_new.fit(X_combined, y_combined)

    new_accuracy = accuracy_score(y_test_uci, rf_new.predict(X_test_uci))

    # ── Save updated model and log ────────────────────────────────────────────
    joblib.dump(rf_new, model_path)
    log["last_retrain_accuracy"] = round(new_accuracy, 6)
    _save_feedback_log(feedback_log_path, log)

    delta = round(new_accuracy - old_accuracy, 6)
    total_count = log["total_confirmations"] + log["total_false_alarms"]

    logger.info(
        "Feedback update complete — verdict=%s old_acc=%.4f new_acc=%.4f delta=%+.4f buffer=%d",
        verdict, old_accuracy, new_accuracy, delta, len(log["samples"]),
    )

    return {
        "old_accuracy": round(old_accuracy, 6),
        "new_accuracy": round(new_accuracy, 6),
        "delta": delta,
        "samples_added": 1,
        "total_feedback_count": total_count,
    }

