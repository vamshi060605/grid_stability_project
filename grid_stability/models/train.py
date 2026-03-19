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
