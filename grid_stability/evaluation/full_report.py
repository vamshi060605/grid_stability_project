"""
evaluation/full_report.py — Full evaluation report for both models.
Wraps evaluate.py functionality into a standalone report runner.
"""
import logging
from pathlib import Path
import sys

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use('Agg')

from models.evaluate import run_full_evaluation

logger = logging.getLogger(__name__)

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
