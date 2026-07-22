"""Vercel entrypoint for the Moneyball FastAPI application."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from moneyball_predictions.api import app

__all__ = ["app"]
