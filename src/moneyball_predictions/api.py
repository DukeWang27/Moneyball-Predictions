"""FastAPI application for Moneyball Predictions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal
import base64
import hmac
import os

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .backtest import BacktestError, build_mlb_backtest
from .edge_performance import EdgePerformanceError, build_edge_performance
from .live import (
    LivePredictionError,
    build_live_mlb_predictions,
    build_single_scoreboard_game,
)
from .lineups import parse_game_lineups
from .mlb import MLB_STATS_BASE_URL
from .model import log5_probability, pythagorean_expectation
from .odds import devig_two_way_decimal, expected_value
from .player_props import (
    PlayerPropError,
    build_strikeout_prop_board,
    build_strikeout_prop_settlement,
)
from .research_api import router as research_router
from .portfolio_api import router as portfolio_router
from .dashboard_api import router as dashboard_router
from .cron_api import router as cron_router
from .db import PLAYER_PROP_ACCOUNT, database_health, ensure_database_initialized, session_scope
from .portfolio import account_metrics
from .runtime import running_on_vercel
from .serverless_artifacts import load_backtest_snapshot
from .repositories import (
    insert_lineup_snapshot as insert_postgres_lineup_snapshot,
    insert_prop_family_decisions,
)
from .schemas import (
    BacktestResponse,
    EdgePerformanceResponse,
    GameLineupsResponse,
    LiveMlbResponse,
    PredictionRequest,
    PredictionResponse,
    PropBoardResponse,
    PropSettlementResponse,
    ScoreboardGame,
    TeamLineupResponse,
    SideAnalysis,
)

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"

app = FastAPI(
    title="Moneyball Predictions API",
    version="0.13.0",
    description="MLB moneylines, pitcher props, paper trading, execution research, and leakage-safe model comparison.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(research_router)
app.include_router(portfolio_router)
app.include_router(dashboard_router)
app.include_router(cron_router)


@app.middleware("http")
async def optional_private_dashboard(request: Request, call_next):
    """Use browser-native Basic Auth when DASHBOARD_PASSWORD is configured."""
    password = os.environ.get("DASHBOARD_PASSWORD", "")
    if not password or request.url.path == "/health" or request.url.path.startswith("/api/internal/cron/"):
        return await call_next(request)
    username = os.environ.get("DASHBOARD_USERNAME", "moneyball")
    header = request.headers.get("authorization", "")
    supplied_user = supplied_password = ""
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
            supplied_user, supplied_password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            pass
    if not (
        hmac.compare_digest(supplied_user, username)
        and hmac.compare_digest(supplied_password, password)
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"},
            headers={"WWW-Authenticate": 'Basic realm="Moneyball Predictions"'},
        )
    response = await call_next(request)
    existing_vary = response.headers.get("Vary", "")
    vary_values = {value.strip() for value in existing_vary.split(",") if value.strip()}
    vary_values.add("Authorization")
    response.headers["Vary"] = ", ".join(sorted(vary_values))
    return response


@app.get("/health")
def health() -> dict:
    try:
        ensure_database_initialized()
        database = database_health()
    except Exception as exc:
        database = {"status": "error", "detail": str(exc)}
    return {"status": "ok", "version": "0.13.0", "database": database}


@app.get("/api/v1/polymarket/mlb", response_model=LiveMlbResponse)
async def live_mlb_markets(
    response: Response,
    bankroll: float = Query(default=100.0, gt=0, le=1_000_000),
    kelly_multiplier: float = Query(default=0.25, gt=0, le=1.0),
    season: int | None = Query(default=None, ge=2000, le=2100),
    days: int = Query(default=2, ge=1, le=7),
) -> LiveMlbResponse:
    response.headers["Cache-Control"] = "public, max-age=5"
    response.headers["Vercel-CDN-Cache-Control"] = "public, s-maxage=15, stale-while-revalidate=30"
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
    response: Response,
    season: int = Query(default=2026, ge=2000, le=2100),
    through: date | None = Query(default=None),
    min_games: int = Query(default=10, ge=1, le=40),
) -> BacktestResponse:
    response.headers["Vercel-CDN-Cache-Control"] = "public, s-maxage=3600, stale-while-revalidate=86400"
    # Prefer the precomputed default snapshot whenever it exists. This keeps the
    # endpoint fast even when Vercel system environment variables are not
    # exposed to the function.
    if through is None and min_games == 10:
        snapshot = load_backtest_snapshot(season)
        if snapshot is not None:
            return snapshot
    if running_on_vercel():
        if through is not None or min_games != 10:
            raise HTTPException(
                status_code=422,
                detail="Custom Model Lab runs are local-only; deploy a precomputed default snapshot.",
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "Model Lab snapshot is missing. Run "
                "python scripts/export_serverless_artifacts.py --season "
                f"{season}, then commit the generated artifacts."
            ),
        )
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
    response: Response,
    season: int | None = Query(default=None, ge=2000, le=2100),
    days: int = Query(default=3, ge=1, le=7),
) -> PropBoardResponse:
    ensure_database_initialized()
    with session_scope() as session:
        bankroll = account_metrics(session, PLAYER_PROP_ACCOUNT).available_cash
    response.headers["Cache-Control"] = "private, max-age=5"
    try:
        board = await build_strikeout_prop_board(
            stake_dollars=max(0.01, bankroll * 0.01),
            available_bankroll=bankroll,
            shrinkage=0.30,
            season=season,
            days=days,
        )
        with session_scope() as session:
            insert_prop_family_decisions(session, board)
        return board
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
    response: Response,
    entry_horizon_minutes: int = Query(default=60, ge=15, le=1440),
) -> EdgePerformanceResponse:
    response.headers["Vercel-CDN-Cache-Control"] = "public, s-maxage=300, stale-while-revalidate=600"
    try:
        return await build_edge_performance(
            entry_horizon_minutes=entry_horizon_minutes,
        )
    except EdgePerformanceError as exc:
        raise HTTPException(status_code=502, detail=f"Edge performance refresh failed: {exc}") from exc


@app.get(
    "/api/v1/mlb/game/{game_pk}/lineups",
    response_model=GameLineupsResponse,
)
async def mlb_game_lineups(game_pk: int) -> GameLineupsResponse:
    """Fetch and archive the current official starting-lineup state for one game."""
    timeout = httpx.Timeout(20.0, connect=8.0)
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Moneyball-Predictions/0.13.0 lineup-status"},
        ) as client:
            response = await client.get(f"{MLB_STATS_BASE_URL}/game/{game_pk}/boxscore")
            response.raise_for_status()
            lineups = parse_game_lineups(
                game_pk=game_pk,
                payload=response.json(),
                captured_at=datetime.now(UTC),
            )
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"MLB lineup refresh failed: {exc}") from exc

    ensure_database_initialized()
    with session_scope() as session:
        insert_postgres_lineup_snapshot(session, lineups.away)
        insert_postgres_lineup_snapshot(session, lineups.home)

    def team_payload(lineup) -> TeamLineupResponse:
        return TeamLineupResponse(
            game_pk=lineup.game_pk,
            side=lineup.side,
            team_id=lineup.team_id,
            status=lineup.status.value,
            captured_at=lineup.captured_at,
            source_hash=lineup.source_hash,
            players=[
                {
                    "player_id": player.player_id,
                    "full_name": player.full_name,
                    "batting_slot": player.batting_slot,
                    "position": player.position,
                }
                for player in lineup.players
            ],
        )

    return GameLineupsResponse(
        game_pk=game_pk,
        status=lineups.status.value,
        both_confirmed=lineups.both_confirmed,
        away=team_payload(lineups.away),
        home=team_payload(lineups.home),
    )


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
