"""Pregame prediction-horizon policy."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum


class PredictionHorizon(StrEnum):
    T24H = "T24H"
    T1H = "T1H"
    MANUAL = "MANUAL"


def horizon_due(
    *,
    now: datetime,
    game_start: datetime,
    horizon: PredictionHorizon,
    tolerance: timedelta = timedelta(minutes=5),
) -> bool:
    if now.tzinfo is None or game_start.tzinfo is None:
        raise ValueError("now and game_start must be timezone-aware")
    target_delta = {
        PredictionHorizon.T24H: timedelta(hours=24),
        PredictionHorizon.T1H: timedelta(hours=1),
        PredictionHorizon.MANUAL: game_start - now,
    }[horizon]
    if horizon is PredictionHorizon.MANUAL:
        return now < game_start
    target = game_start - target_delta
    return target <= now < target + tolerance


def validate_horizon_features(
    *,
    horizon: PredictionHorizon,
    lineups_confirmed: bool,
) -> None:
    if horizon is PredictionHorizon.T1H and not lineups_confirmed:
        raise ValueError("T1H prediction requires both confirmed lineups")
    if horizon is PredictionHorizon.T24H and lineups_confirmed:
        raise ValueError(
            "T24H records must not use eventual confirmed-lineup information; use projections"
        )
