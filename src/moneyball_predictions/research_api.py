"""FastAPI routes for deterministic sabermetric and execution research."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .execution import build_passive_buy_plan, quote_buy_by_dollars, slippage_adjusted_metrics
from .horizons import PredictionHorizon, validate_horizon_features
from .sabermetrics import (
    BatterEventProjection,
    BattingLine,
    PitchingLine,
    ReliefAppearance,
    base_runs_smyth_v1,
    calculate_fip,
    calculate_xfip,
    leverage_adjusted_bullpen_fatigue,
    project_lineup,
    pythagenpat_win_pct,
)
from .storage import database_summary, insert_prediction

router = APIRouter(prefix="/api/v1/research", tags=["research"])


class BatterProjectionInput(BaseModel):
    player_id: int
    batting_slot: int = Field(ge=1, le=9)
    expected_pa: float = Field(gt=0, le=8)
    event_rates: dict[str, float]


class LineupProjectionRequest(BaseModel):
    batters: list[BatterProjectionInput]
    woba_weights: dict[str, float]
    run_values: dict[str, float]


class BattingLineInput(BaseModel):
    at_bats: float = Field(ge=0)
    hits: float = Field(ge=0)
    doubles: float = Field(ge=0)
    triples: float = Field(ge=0)
    home_runs: float = Field(ge=0)
    walks: float = Field(ge=0)
    intentional_walks: float = Field(ge=0)
    hit_by_pitch: float = Field(ge=0)


class BaseRunsRequest(BaseModel):
    offense: BattingLineInput
    defense_allowed: BattingLineInput
    games: float = Field(gt=0)


class PitchingLineInput(BaseModel):
    outs_recorded: int = Field(gt=0)
    strikeouts: int = Field(ge=0)
    walks: int = Field(ge=0)
    hit_batters: int = Field(ge=0)
    home_runs: int = Field(ge=0)
    fly_balls: int = Field(default=0, ge=0)


class FipRequest(BaseModel):
    line: PitchingLineInput
    fip_constant: float
    league_hr_per_fly_ball: float = Field(default=0.10, ge=0, le=1)


class ReliefAppearanceInput(BaseModel):
    reliever_id: int
    ended_at: datetime
    pitches: int = Field(ge=0)
    mean_leverage_index: float = Field(ge=0, le=10)


class FatigueRequest(BaseModel):
    as_of: datetime
    appearances: list[ReliefAppearanceInput]


class VwapRequest(BaseModel):
    order_book: dict[str, Any]
    desired_dollars: float = Field(gt=0)
    model_probability: float = Field(ge=0, le=1)
    maximum_price: float | None = Field(default=None, gt=0, lt=1)
    fee_dollars: float = Field(default=0, ge=0)


class PassivePlanRequest(BaseModel):
    token_id: str = Field(min_length=1)
    fair_probability: float = Field(gt=0, lt=1)
    desired_dollars: float = Field(gt=0)
    best_bid: float = Field(gt=0, lt=1)
    best_ask: float = Field(gt=0, lt=1)
    tick_size: float = Field(gt=0, lt=1)
    required_edge: float = Field(default=0.05, ge=0, lt=1)
    adverse_selection_buffer: float = Field(default=0.01, ge=0, lt=1)


class ImmutablePredictionRequest(BaseModel):
    game_id: int
    horizon: Literal["T24H", "T1H", "MANUAL"]
    as_of: datetime
    model_version: str
    feature_schema_version: str = "v0.13.0"
    code_commit_sha: str = "unknown"
    model_artifact_sha256: str = "unavailable"
    calibration_artifact_sha256: str | None = None
    features: dict[str, Any]
    source_snapshot: dict[str, Any]
    raw_home_probability: float = Field(ge=0, le=1)
    calibrated_home_probability: float = Field(ge=0, le=1)
    home_team: str
    away_team: str
    probable_home_pitcher_id: int | None = None
    probable_away_pitcher_id: int | None = None
    lineups_confirmed: bool = False


def _batting_line(payload: BattingLineInput) -> BattingLine:
    return BattingLine(**payload.model_dump())


@router.post("/lineup")
def lineup_projection(payload: LineupProjectionRequest) -> dict[str, float]:
    try:
        projection = project_lineup(
            [BatterEventProjection(**item.model_dump()) for item in payload.batters],
            woba_weights=payload.woba_weights,
            run_values=payload.run_values,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "xwoba": projection.xwoba,
        "linear_weight_runs_per_pa": projection.linear_weight_runs_per_pa,
        "expected_plate_appearances": projection.expected_plate_appearances,
    }


@router.post("/base-runs")
def base_runs(payload: BaseRunsRequest) -> dict[str, float]:
    offense = base_runs_smyth_v1(_batting_line(payload.offense))
    allowed = base_runs_smyth_v1(_batting_line(payload.defense_allowed))
    return {
        "base_runs_scored": offense,
        "base_runs_allowed": allowed,
        "pythagenpat_win_pct": pythagenpat_win_pct(offense, allowed, payload.games),
    }


@router.post("/fip")
def fip(payload: FipRequest) -> dict[str, float]:
    line = PitchingLine(**payload.line.model_dump())
    return {
        "fip": calculate_fip(line, fip_constant=payload.fip_constant),
        "xfip": calculate_xfip(
            line,
            league_hr_per_fly_ball=payload.league_hr_per_fly_ball,
            fip_constant=payload.fip_constant,
        ),
    }


@router.post("/bullpen-fatigue")
def bullpen_fatigue(payload: FatigueRequest) -> dict[str, dict[int, float]]:
    try:
        result = leverage_adjusted_bullpen_fatigue(
            [ReliefAppearance(**item.model_dump()) for item in payload.appearances],
            as_of=payload.as_of,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"fatigue_by_reliever": result}


@router.post("/execution/vwap")
def execution_vwap(payload: VwapRequest) -> dict[str, Any]:
    try:
        quote = quote_buy_by_dollars(
            payload.order_book,
            payload.desired_dollars,
            maximum_price=payload.maximum_price,
        )
        metrics = slippage_adjusted_metrics(
            model_probability=payload.model_probability,
            quote=quote,
            fee_dollars=payload.fee_dollars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "requested_dollars": float(quote.requested_dollars),
        "filled_dollars": float(quote.filled_dollars),
        "acquired_shares": float(quote.acquired_shares),
        "unfilled_dollars": float(quote.unfilled_dollars),
        "worst_price": float(quote.worst_price) if quote.worst_price is not None else None,
        "levels_consumed": quote.levels_consumed,
        **metrics,
    }


@router.post("/execution/passive-plan")
def passive_plan(payload: PassivePlanRequest) -> dict[str, Any]:
    try:
        plan = build_passive_buy_plan(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if plan is None:
        return {"place_order": False, "reason": "No valid post-only price preserves the edge"}
    return {
        "place_order": True,
        "dry_run": True,
        "token_id": plan.token_id,
        "side": plan.side,
        "limit_price": float(plan.limit_price),
        "size_shares": float(plan.size_shares),
        "notional_dollars": float(plan.notional_dollars),
        "post_only": plan.post_only,
        "order_type": plan.order_type,
    }


@router.post("/predictions/immutable")
def immutable_prediction(payload: ImmutablePredictionRequest) -> dict[str, Any]:
    try:
        horizon = PredictionHorizon(payload.horizon)
        validate_horizon_features(
            horizon=horizon,
            lineups_confirmed=payload.lineups_confirmed,
        )
        result = insert_prediction(**payload.model_dump())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "prediction_id": result.prediction_id,
        "feature_hash_sha256": result.feature_hash_sha256,
        "created": result.created,
    }


@router.get("/database")
def research_database() -> dict[str, int | str]:
    return database_summary()
