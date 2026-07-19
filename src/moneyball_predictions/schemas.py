"""Pydantic request and response models for the prediction API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

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


class LiveMarketSide(BaseModel):
    team: str
    model_probability: float
    market_buy_price: float
    edge: float
    expected_value: float
    expected_roi: float
    full_kelly_fraction: float
    applied_kelly_fraction: float
    kelly_multiplier: float
    kelly_stake: float
    stake: float
    shares: float
    price_source: str
    top_ask_size: float | None = None
    kelly_capped_by_depth: bool = False


class LiveGamePrediction(BaseModel):
    event_id: str
    market_id: str
    game_pk: int
    title: str
    start_time: str
    polymarket_url: str
    away_team: str
    home_team: str
    away_probable_pitcher: str | None = None
    home_probable_pitcher: str | None = None
    venue: str | None = None
    team_a_strength: float
    team_b_strength: float
    side_a: LiveMarketSide
    side_b: LiveMarketSide
    recommended_side: str | None
    signal: Literal["BET", "LEAN", "PASS"]
    recommendation_reason: str
    liquidity: float | None = None
    volume: float | None = None
    methodology: str = (
        "Prior-regressed run differential + Log5 + home field; pitcher adjustment pending"
    )


class RecommendedBet(BaseModel):
    game_pk: int
    title: str
    team: str
    opponent: str
    start_time: str
    market_buy_price: float
    edge: float
    expected_value: float
    expected_roi: float
    full_kelly_fraction: float
    applied_kelly_fraction: float
    kelly_multiplier: float
    kelly_stake: float
    stake: float
    shares: float
    kelly_capped_by_depth: bool = False
    polymarket_url: str


class ScoreboardGame(BaseModel):
    game_pk: int
    game_date: str
    official_date: str | None
    away_team: str
    home_team: str
    away_score: int | None
    home_score: int | None
    abstract_state: str
    detailed_state: str
    current_inning: int | None = None
    inning_state: str | None = None
    inning_ordinal: str | None = None
    away_probable_pitcher: str | None = None
    home_probable_pitcher: str | None = None
    venue: str | None = None
    winner: str | None = None
    polymarket_url: str | None = None
    market_status: str
    live_bet_message: str


class MatchingDiagnostics(BaseModel):
    discovery_warnings: int = 0
    unsupported_events: int = 0
    no_moneyline: int = 0
    unmatched_teams: int = 0
    missing_tokens: int = 0
    no_schedule_match: int = 0
    outside_date_window: int = 0
    no_executable_ask: int = 0
    examples: list[str] = Field(default_factory=list)


class LiveMlbResponse(BaseModel):
    generated_at: datetime
    season: int
    bankroll: float
    kelly_multiplier: float
    date_window_start: str
    date_window_end: str
    recommended_bets: list[RecommendedBet]
    games: list[LiveGamePrediction]
    scoreboard: list[ScoreboardGame]
    diagnostics: MatchingDiagnostics
    skipped_events: list[str] = Field(default_factory=list)


class BacktestCalibrationBucket(BaseModel):
    label: str
    predictions: int
    mean_probability: float | None = None
    actual_rate: float | None = None


class BacktestModelMetrics(BaseModel):
    key: str
    label: str
    prediction_count: int
    accuracy: float | None
    brier_score: float | None
    log_loss: float | None
    calibration: list[BacktestCalibrationBucket]


class BacktestLeakageAudit(BaseModel):
    chronological_order: bool = True
    prediction_before_result_update: bool = True
    current_game_excluded: bool = True
    prior_season_only_for_priors: bool = True
    calibrator_uses_past_only: bool = True
    passed: bool = True
    note: str = (
        "Each prediction is created before the current final score is added. Priors use the "
        "previous season, and calibration uses only earlier settled predictions."
    )


class BacktestGame(BaseModel):
    game_pk: int
    game_date: str
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    away_probability: float
    home_probability: float
    predicted_winner: str
    predicted_probability: float
    winner: str
    correct: bool
    brier: float
    log_loss: float
    baseline_away_probability: float | None = None
    baseline_home_probability: float | None = None
    baseline_predicted_winner: str | None = None
    baseline_predicted_probability: float | None = None
    baseline_correct: bool | None = None
    raw_home_probability: float | None = None
    calibrated: bool = False


class BacktestResponse(BaseModel):
    generated_at: datetime
    season: int
    prior_season: int | None = None
    start_date: date
    end_date: date
    min_games: int
    completed_games: int
    prediction_count: int
    accuracy: float | None
    brier_score: float | None
    log_loss: float | None
    home_baseline_accuracy: float | None
    calibration: list[BacktestCalibrationBucket]
    recent_predictions: list[BacktestGame]
    baseline: BacktestModelMetrics
    enhanced: BacktestModelMetrics
    accuracy_delta: float | None = None
    brier_delta: float | None = None
    log_loss_delta: float | None = None
    calibration_training_games: int = 0
    leakage_audit: BacktestLeakageAudit = Field(default_factory=BacktestLeakageAudit)
    methodology: str = (
        "Leakage-safe walk-forward comparison: old Pythagorean + Log5 versus a prior-"
        "regressed, home-adjusted, past-only calibrated model"
    )
