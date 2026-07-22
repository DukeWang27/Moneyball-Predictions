"""Load precomputed artifacts used by serverless read endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .schemas import BacktestResponse

PACKAGE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
LIVE_MODEL_BUNDLE = ARTIFACT_DIR / "live_model_bundle.joblib"


def backtest_snapshot_path(season: int) -> Path:
    return ARTIFACT_DIR / f"backtest_{season}.json"


def load_live_model_bundle() -> dict[str, Any] | None:
    if not LIVE_MODEL_BUNDLE.exists():
        return None
    try:
        payload = joblib.load(LIVE_MODEL_BUNDLE)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_backtest_snapshot(season: int) -> BacktestResponse | None:
    path = backtest_snapshot_path(season)
    if not path.exists():
        return None
    try:
        return BacktestResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None
