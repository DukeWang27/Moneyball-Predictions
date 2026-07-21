"""FastAPI application for Moneyball Predictions."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .backtest import BacktestError, build_mlb_backtest
from .edge_performance import EdgePerformanceError, build_edge_performance
from .live import (
    LivePredictionError,
    build_live_mlb_predictions,
    build_single_scoreboard_game,
)
from .model import log5_probability, pythagorean_expectation
from .odds import devig_two_way_decimal, expected_value
from .player_props import (
    PlayerPropError,
    build_strikeout_prop_board,
    build_strikeout_prop_settlement,
)
from .research_api import router as research_router
from .schemas import (
    BacktestResponse,
    EdgePerformanceResponse,
    LiveMlbResponse,
    PredictionRequest,
    PredictionResponse,
    PropBoardResponse,
    PropSettlementResponse,
    ScoreboardGame,
    SideAnalysis,
)

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"

app = FastAPI(
    title="Moneyball Predictions API",
    version="0.12.1",
    description="MLB moneylines, pitcher props, paper trading, execution research, and leakage-safe model comparison.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(research_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/polymarket/mlb", response_model=LiveMlbResponse)
async def live_mlb_markets(
    bankroll: float = Query(default=100.0, gt=0, le=1_000_000),
    kelly_multiplier: float = Query(default=0.25, gt=0, le=1.0),
    season: int | None = Query(default=None, ge=2000, le=2100),
    days: int = Query(default=2, ge=1, le=7),
) -> LiveMlbResponse:
    try:
        return await build_live_mlb_predictions(
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
            season=season,
            days=days,
        )
    except LivePredictionError as exc:
        raise HTTPException(status_code=502, detail=f"Live data refresh failed: {exc}") from exc


@app.get("/api/v1/backtest/mlb", response_model=BacktestResponse)
async def mlb_backtest(
    season: int = Query(default=2026, ge=2000, le=2100),
    through: date | None = Query(default=None),
    min_games: int = Query(default=10, ge=1, le=40),
) -> BacktestResponse:
    try:
        return await build_mlb_backtest(
            season=season,
            through=through,
            min_games=min_games,
        )
    except BacktestError as exc:
        raise HTTPException(status_code=502, detail=f"Historical backtest failed: {exc}") from exc


@app.get("/api/v1/props/strikeouts", response_model=PropBoardResponse)
async def mlb_strikeout_props(
    stake_dollars: float = Query(default=50.0, gt=0, le=100_000),
    season: int | None = Query(default=None, ge=2000, le=2100),
    days: int = Query(default=3, ge=1, le=7),
) -> PropBoardResponse:
    try:
        return await build_strikeout_prop_board(
            stake_dollars=stake_dollars,
            season=season,
            days=days,
        )
    except PlayerPropError as exc:
        raise HTTPException(status_code=502, detail=f"Player-prop refresh failed: {exc}") from exc


@app.get(
    "/api/v1/props/strikeouts/settlement",
    response_model=PropSettlementResponse,
)
async def mlb_strikeout_prop_settlement(
    game_pk: int = Query(gt=0),
    player_id: int = Query(gt=0),
    threshold: int = Query(ge=1, le=30),
    side: Literal["YES", "NO"] = Query(),
) -> PropSettlementResponse:
    try:
        return await build_strikeout_prop_settlement(
            game_pk=game_pk,
            player_id=player_id,
            threshold=threshold,
            side=side,
        )
    except PlayerPropError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Player-prop settlement failed: {exc}",
        ) from exc


@app.get("/api/v1/edge-performance/mlb", response_model=EdgePerformanceResponse)
async def mlb_edge_performance(
    entry_horizon_minutes: int = Query(default=60, ge=15, le=1440),
) -> EdgePerformanceResponse:
    try:
        return await build_edge_performance(
            entry_horizon_minutes=entry_horizon_minutes,
        )
    except EdgePerformanceError as exc:
        raise HTTPException(status_code=502, detail=f"Edge performance refresh failed: {exc}") from exc


@app.get("/api/v1/mlb/game/{game_pk}", response_model=ScoreboardGame)
async def mlb_game_status(game_pk: int) -> ScoreboardGame:
    try:
        return await build_single_scoreboard_game(game_pk)
    except LivePredictionError as exc:
        raise HTTPException(status_code=502, detail=f"MLB game refresh failed: {exc}") from exc


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    team_a_strength = pythagorean_expectation(
        payload.team_a.runs_scored,
        payload.team_a.runs_allowed,
        payload.team_a.games_played,
    )
    team_b_strength = pythagorean_expectation(
        payload.team_b.runs_scored,
        payload.team_b.runs_allowed,
        payload.team_b.games_played,
    )

    probability_a = log5_probability(team_a_strength, team_b_strength)
    probability_b = 1.0 - probability_a
    market_a, market_b = devig_two_way_decimal(
        payload.decimal_odds_a,
        payload.decimal_odds_b,
    )

    side_a = SideAnalysis(
        team=payload.team_a.name,
        model_probability=probability_a,
        market_probability=market_a,
        edge=probability_a - market_a,
        expected_value=expected_value(probability_a, payload.decimal_odds_a, payload.stake),
    )
    side_b = SideAnalysis(
        team=payload.team_b.name,
        model_probability=probability_b,
        market_probability=market_b,
        edge=probability_b - market_b,
        expected_value=expected_value(probability_b, payload.decimal_odds_b, payload.stake),
    )

    recommended_side: str | None = None
    if side_a.expected_value > 0 or side_b.expected_value > 0:
        recommended_side = max((side_a, side_b), key=lambda side: side.expected_value).team

    return PredictionResponse(
        team_a_strength=team_a_strength,
        team_b_strength=team_b_strength,
        side_a=side_a,
        side_b=side_b,
        recommended_side=recommended_side,
    )


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
