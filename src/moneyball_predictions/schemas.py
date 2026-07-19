"""Pydantic request and response models for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TeamSeasonStats(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    runs_scored: float = Field(ge=0)
    runs_allowed: float = Field(ge=0)
    games_played: int = Field(gt=0)

    @model_validator(mode="after")
    def ensure_nonzero_run_environment(self) -> "TeamSeasonStats":
        if self.runs_scored + self.runs_allowed <= 0:
            raise ValueError("runs_scored and runs_allowed cannot both be zero")
        return self


class PredictionRequest(BaseModel):
    team_a: TeamSeasonStats
    team_b: TeamSeasonStats
    decimal_odds_a: float = Field(gt=1.0)
    decimal_odds_b: float = Field(gt=1.0)
    stake: float = Field(default=10.0, gt=0, le=100_000)


class SideAnalysis(BaseModel):
    team: str
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float


class PredictionResponse(BaseModel):
    team_a_strength: float
    team_b_strength: float
    side_a: SideAnalysis
    side_b: SideAnalysis
    recommended_side: str | None
    methodology: str = "Dynamic Pythagorean expectation + Log5; no pitcher adjustment yet"
