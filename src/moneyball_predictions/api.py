"""FastAPI application for the first Moneyball Predictions MVP."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import log5_probability, pythagorean_expectation
from .odds import devig_two_way_decimal, expected_value
from .schemas import PredictionRequest, PredictionResponse, SideAnalysis

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"

app = FastAPI(
    title="Moneyball Predictions API",
    version="0.1.0",
    description="MLB sabermetric probability and market-edge calculator.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
