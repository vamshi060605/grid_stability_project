"""
main.py — Project entry point.

Runs the full pipeline in order:
  1. Data audit
  2. Model training
  3. Full evaluation
  4. Recommendation engine self-test
  5. Report generation

Usage:
    python main.py              # Full pipeline
    python main.py --train      # Train only
    python main.py --evaluate   # Evaluate only
    python main.py --dashboard  # Launch dashboard
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_LOG_DIR = _ROOT / "logs"
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


def run_train():
    logger.info("=== Step 1: Training models ===")
    from models.train import train_models
    train_models()


def run_evaluate():
    logger.info("=== Step 2: Running evaluation ===")
    from models.evaluate import run_full_evaluation
    run_full_evaluation()


def run_recommendation_test():
    logger.info("=== Step 3: Testing recommendation engine ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, "xai/recommendation_engine.py"],
        capture_output=False,
    )
    return result.returncode == 0


def run_report():
    logger.info("=== Step 4: Generating PDF report ===")
    from reports.generate_summary import generate_pdf
    generate_pdf()


def launch_dashboard():
    logger.info("Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "dashboard/app.py"])


def main():
    parser = argparse.ArgumentParser(description="Grid Stability Project Runner")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
        return

    if args.train:
        run_train()
        return

    if args.evaluate:
        run_evaluate()
        return

    if args.report:
        run_report()
        return

    # Full pipeline
    logger.info("Running full pipeline...")
    run_train()
    run_evaluate()
    run_recommendation_test()
    run_report()
    logger.info("Pipeline complete. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
