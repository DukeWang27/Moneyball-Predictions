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
    kelly_capped_by_risk: bool = False
    sabermetric_support: Literal["CONFIRMED", "MIXED", "CONTRARIAN", "UNKNOWN"] = "UNKNOWN"
    sabermetric_support_count: int = 0
    sabermetric_support_total: int = 0
    sabermetric_reasons: list[str] = Field(default_factory=list)
    value_grade: Literal["A", "B", "C", "LEAN", "PASS"] = "PASS"


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
    away_lineup_confirmed: bool = False
    home_lineup_confirmed: bool = False
    away_lineup_names: list[str] = Field(default_factory=list)
    home_lineup_names: list[str] = Field(default_factory=list)
    team_a_strength: float
    team_b_strength: float
    side_a: LiveMarketSide
    side_b: LiveMarketSide
    recommended_side: str | None
    signal: Literal["BET", "LEAN", "PASS"]
    recommendation_reason: str
    liquidity: float | None = None
    volume: float | None = None
    model_version: str = "v0.9.1"
    methodology: str = (
        "v0.9.1 model lab using regressed team strength, alternative starter shrinkage, starter-workload interaction, prior-season park interaction, and validation-only calibration selection"
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
    kelly_capped_by_risk: bool = False
    sabermetric_support: Literal["CONFIRMED", "MIXED", "CONTRARIAN", "UNKNOWN"] = "UNKNOWN"
    sabermetric_support_count: int = 0
    sabermetric_support_total: int = 0
    value_grade: Literal["A", "B", "C"] = "C"
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
    model_version: str = "v0.9.1"
    model_training_rows: int = 0
    model_validation_rows: int = 0
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




class BacktestFeatureDiagnostic(BaseModel):
    name: str
    mean: float
    std: float
    minimum: float
    p05: float
    median: float
    p95: float
    maximum: float
    unique_values: int
    near_constant: bool
    coefficient: float | None = None


class BacktestCalibrationComparison(BaseModel):
    method: str
    selected: bool
    validation_brier: float | None = None
    target_accuracy: float | None = None
    target_brier: float | None = None
    target_log_loss: float | None = None

class BacktestLeakageAudit(BaseModel):
    chronological_order: bool = True
    prediction_before_result_update: bool = True
    current_game_excluded: bool = True
    prior_season_only_for_priors: bool = True
    calibrator_uses_past_only: bool = True
    coefficients_trained_before_target_season: bool = True
    rolling_features_shifted: bool = True
    passed: bool = True
    note: str = (
        "Each prediction is created before the current final score is added. Priors use the "
        "previous season, component starter and reliever features use prior games only, and v0.9 "
        "coefficients are trained before the target season."
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
    enhanced_v05_away_probability: float | None = None
    enhanced_v05_home_probability: float | None = None
    enhanced_v05_predicted_winner: str | None = None
    enhanced_v05_predicted_probability: float | None = None
    enhanced_v05_correct: bool | None = None


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
    optimized: BacktestModelMetrics
    ablations: list[BacktestModelMetrics] = Field(default_factory=list)
    accuracy_delta: float | None = None
    brier_delta: float | None = None
    log_loss_delta: float | None = None
    calibration_training_games: int = 0
    optimized_training_games: int = 0
    optimized_validation_games: int = 0
    optimized_training_starter_coverage: float = 0.0
    optimized_validation_starter_coverage: float = 0.0
    target_starter_coverage: float = 0.0
    optimized_training_component_coverage: float = 0.0
    optimized_validation_component_coverage: float = 0.0
    target_component_coverage: float = 0.0
    optimized_training_bullpen_coverage: float = 0.0
    optimized_validation_bullpen_coverage: float = 0.0
    target_bullpen_coverage: float = 0.0
    optimized_shrinkage: float = 0.0
    optimized_calibration_method: str = "identity"
    optimized_coefficients: dict[str, float] = Field(default_factory=dict)
    optimized_variant: str = ""
    feature_diagnostics: list[BacktestFeatureDiagnostic] = Field(default_factory=list)
    calibration_comparison: list[BacktestCalibrationComparison] = Field(default_factory=list)
    training_seasons: list[int] = Field(default_factory=list)
    calibration_seasons: list[int] = Field(default_factory=list)
    leakage_audit: BacktestLeakageAudit = Field(default_factory=BacktestLeakageAudit)
    methodology: str = (
        "Leakage-safe comparison of v0.4, v0.5, v0.7 proxy, and v0.9.1 challenger stages "
        "trained before the target season using alternative starter shrinkage, starter-workload "
        "interaction, prior-season park factors, and validation-only calibration selection"
    )


class EdgePerformanceMetrics(BaseModel):
    strategy: Literal["All economic edges", "Sabermetric confirmed"]
    min_edge: float
    eligible_signals: int
    settled_bets: int
    wins: int
    losses: int
    win_rate: float | None = None
    average_model_probability: float | None = None
    average_entry_price: float | None = None
    average_edge: float | None = None
    average_expected_roi: float | None = None
    brier_score: float | None = None
    flat_stake_profit: float | None = None
    flat_stake_roi: float | None = None
    average_clv: float | None = None
    positive_clv_rate: float | None = None
    max_drawdown: float | None = None


class EdgePerformanceBet(BaseModel):
    game_pk: int
    game_start: str
    captured_at: str
    actual_entry_horizon_minutes: int
    team: str
    opponent: str
    model_probability: float
    entry_price: float
    edge: float
    expected_roi: float
    sabermetric_support: str = "UNKNOWN"
    sabermetric_support_count: int = 0
    sabermetric_support_total: int = 0
    winner: str | None = None
    won: bool | None = None
    profit_per_dollar: float | None = None
    closing_no_vig_probability: float | None = None
    clv: float | None = None
    model_version: str = "unknown"


class EdgePerformanceResponse(BaseModel):
    generated_at: datetime
    entry_horizon_minutes: int
    snapshots_read: int
    selected_games: int
    settled_games: int
    open_games: int
    first_capture: datetime | None = None
    last_capture: datetime | None = None
    metrics: list[EdgePerformanceMetrics] = Field(default_factory=list)
    recent_candidates: list[EdgePerformanceBet] = Field(default_factory=list)
    note: str
