"""Leakage-safe walk-forward validation and model comparison for MLB games."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from .mlb import MlbDataError, MlbGameState, fetch_mlb_regular_season_schedule
from .model import (
    add_home_field_advantage,
    clip_probability,
    log5_probability,
    logistic,
    logit,
    pythagorean_expectation,
    regressed_pythagorean_expectation,
)
from .schemas import (
    BacktestCalibrationBucket,
    BacktestGame,
    BacktestLeakageAudit,
    BacktestModelMetrics,
    BacktestResponse,
)

EASTERN = ZoneInfo("America/New_York")
DEFAULT_HOME_WIN_RATE = 0.54
DEFAULT_LEAGUE_RUNS_PER_GAME = 4.50
PRIOR_PSEUDO_GAMES = 25.0
PLATT_MIN_SAMPLES = 150
PLATT_REFIT_INTERVAL = 50


class BacktestError(RuntimeError):
    """Raised when historical validation data cannot be built."""


@dataclass
class TeamRunningTotals:
    runs_scored: float = 0.0
    runs_allowed: float = 0.0
    games_played: int = 0

    def add_game(self, runs_scored: int, runs_allowed: int) -> None:
        self.runs_scored += runs_scored
        self.runs_allowed += runs_allowed
        self.games_played += 1

    @property
    def runs_scored_per_game(self) -> float | None:
        return self.runs_scored / self.games_played if self.games_played else None

    @property
    def runs_allowed_per_game(self) -> float | None:
        return self.runs_allowed / self.games_played if self.games_played else None


@dataclass(frozen=True)
class SeasonPriors:
    team_totals: dict[str, TeamRunningTotals]
    league_runs_per_team_game: float
    home_win_rate: float


@dataclass
class ProbabilityRow:
    probability: float
    outcome: float


@dataclass
class PlattCalibrator:
    """A small past-only logistic calibration layer for one raw probability feature."""

    intercept: float = 0.0
    slope: float = 1.0
    fitted: bool = False

    def fit(self, rows: list[ProbabilityRow], regularization: float = 5.0) -> None:
        if len(rows) < PLATT_MIN_SAMPLES:
            self.intercept = 0.0
            self.slope = 1.0
            self.fitted = False
            return

        a = self.intercept if self.fitted else 0.0
        b = self.slope if self.fitted else 1.0
        for _ in range(30):
            grad_a = regularization * a
            grad_b = regularization * (b - 1.0)
            h_aa = regularization
            h_ab = 0.0
            h_bb = regularization

            for row in rows:
                x = logit(clip_probability(row.probability, 1e-5, 1 - 1e-5))
                predicted = logistic(a + b * x)
                residual = predicted - row.outcome
                weight = max(predicted * (1.0 - predicted), 1e-8)
                grad_a += residual
                grad_b += residual * x
                h_aa += weight
                h_ab += weight * x
                h_bb += weight * x * x

            determinant = h_aa * h_bb - h_ab * h_ab
            if abs(determinant) < 1e-12:
                break
            delta_a = (grad_a * h_bb - grad_b * h_ab) / determinant
            delta_b = (h_aa * grad_b - h_ab * grad_a) / determinant
            a -= delta_a
            b -= delta_b
            a = min(max(a, -2.0), 2.0)
            b = min(max(b, 0.10), 2.50)
            if max(abs(delta_a), abs(delta_b)) < 1e-7:
                break

        self.intercept = a
        self.slope = b
        self.fitted = True

    def transform(self, probability: float) -> float:
        if not self.fitted:
            return probability
        return clip_probability(logistic(self.intercept + self.slope * logit(probability)))


def _game_sort_key(game: MlbGameState) -> tuple[str, int]:
    return game.game_date, game.game_pk


def _valid_final_games(games: list[MlbGameState]) -> list[MlbGameState]:
    return [
        game
        for game in sorted(games, key=_game_sort_key)
        if game.is_final
        and game.away_score is not None
        and game.home_score is not None
        and game.away_score != game.home_score
    ]


def _season_priors(games: list[MlbGameState]) -> SeasonPriors:
    totals: dict[str, TeamRunningTotals] = {}
    total_runs = 0.0
    team_games = 0
    home_wins = 0
    finals = _valid_final_games(games)
    for game in finals:
        away = totals.setdefault(game.away_team, TeamRunningTotals())
        home = totals.setdefault(game.home_team, TeamRunningTotals())
        away.add_game(game.away_score or 0, game.home_score or 0)
        home.add_game(game.home_score or 0, game.away_score or 0)
        total_runs += (game.away_score or 0) + (game.home_score or 0)
        team_games += 2
        if (game.home_score or 0) > (game.away_score or 0):
            home_wins += 1

    league_runs = total_runs / team_games if team_games else DEFAULT_LEAGUE_RUNS_PER_GAME
    home_rate = home_wins / len(finals) if finals else DEFAULT_HOME_WIN_RATE
    home_rate = min(max(home_rate, 0.50), 0.58)
    return SeasonPriors(
        team_totals=totals,
        league_runs_per_team_game=league_runs,
        home_win_rate=home_rate,
    )


def _running_league_rate(totals: dict[str, TeamRunningTotals], fallback: float) -> float:
    total_runs = sum(item.runs_scored for item in totals.values())
    team_games = sum(item.games_played for item in totals.values())
    return total_runs / team_games if team_games else fallback


def _prior_rates(
    team: str,
    priors: SeasonPriors,
) -> tuple[float, float]:
    prior = priors.team_totals.get(team)
    if prior and prior.games_played:
        return (
            prior.runs_scored / prior.games_played,
            prior.runs_allowed / prior.games_played,
        )
    return priors.league_runs_per_team_game, priors.league_runs_per_team_game


def _baseline_home_probability(
    away: TeamRunningTotals,
    home: TeamRunningTotals,
) -> float:
    away_strength = pythagorean_expectation(
        away.runs_scored,
        away.runs_allowed,
        away.games_played,
    )
    home_strength = pythagorean_expectation(
        home.runs_scored,
        home.runs_allowed,
        home.games_played,
    )
    return 1.0 - log5_probability(away_strength, home_strength)


def _enhanced_raw_home_probability(
    *,
    away_team: str,
    home_team: str,
    away: TeamRunningTotals,
    home: TeamRunningTotals,
    priors: SeasonPriors,
    running_league_rate: float,
) -> float:
    away_prior_rs, away_prior_ra = _prior_rates(away_team, priors)
    home_prior_rs, home_prior_ra = _prior_rates(home_team, priors)
    league_rate = 0.75 * priors.league_runs_per_team_game + 0.25 * running_league_rate

    away_strength = regressed_pythagorean_expectation(
        runs_scored=away.runs_scored,
        runs_allowed=away.runs_allowed,
        games_played=away.games_played,
        prior_runs_scored_per_game=away_prior_rs,
        prior_runs_allowed_per_game=away_prior_ra,
        league_runs_per_team_game=league_rate,
        pseudo_games=PRIOR_PSEUDO_GAMES,
    )
    home_strength = regressed_pythagorean_expectation(
        runs_scored=home.runs_scored,
        runs_allowed=home.runs_allowed,
        games_played=home.games_played,
        prior_runs_scored_per_game=home_prior_rs,
        prior_runs_allowed_per_game=home_prior_ra,
        league_runs_per_team_game=league_rate,
        pseudo_games=PRIOR_PSEUDO_GAMES,
    )
    neutral_home_probability = log5_probability(home_strength, away_strength)
    return clip_probability(
        add_home_field_advantage(neutral_home_probability, priors.home_win_rate),
        0.10,
        0.90,
    )


def _training_rows_from_prior_season(
    games: list[MlbGameState],
    *,
    min_games: int,
) -> list[ProbabilityRow]:
    """Build calibration rows using only information available before each prior game."""
    totals: dict[str, TeamRunningTotals] = {}
    empty_priors = SeasonPriors({}, DEFAULT_LEAGUE_RUNS_PER_GAME, DEFAULT_HOME_WIN_RATE)
    rows: list[ProbabilityRow] = []

    for game in _valid_final_games(games):
        away = totals.setdefault(game.away_team, TeamRunningTotals())
        home = totals.setdefault(game.home_team, TeamRunningTotals())
        if away.games_played >= min_games and home.games_played >= min_games:
            raw_home = _enhanced_raw_home_probability(
                away_team=game.away_team,
                home_team=game.home_team,
                away=away,
                home=home,
                priors=empty_priors,
                running_league_rate=_running_league_rate(
                    totals, DEFAULT_LEAGUE_RUNS_PER_GAME
                ),
            )
            rows.append(
                ProbabilityRow(
                    probability=raw_home,
                    outcome=1.0 if (game.home_score or 0) > (game.away_score or 0) else 0.0,
                )
            )
        away.add_game(game.away_score or 0, game.home_score or 0)
        home.add_game(game.home_score or 0, game.away_score or 0)
    return rows


def _calibration_buckets(
    predictions: list[BacktestGame],
    *,
    use_baseline: bool = False,
) -> list[BacktestCalibrationBucket]:
    bounds = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    buckets: list[BacktestCalibrationBucket] = []
    for lower, upper in bounds:
        selected: list[tuple[float, bool]] = []
        for prediction in predictions:
            probability = (
                prediction.baseline_predicted_probability
                if use_baseline
                else prediction.predicted_probability
            )
            correct = (
                bool(prediction.baseline_correct)
                if use_baseline
                else prediction.correct
            )
            if probability is not None and lower <= probability < upper:
                selected.append((probability, correct))
        label_upper = 1.0 if upper > 1.0 else upper
        buckets.append(
            BacktestCalibrationBucket(
                label=f"{lower:.0%}-{label_upper:.0%}",
                predictions=len(selected),
                mean_probability=(
                    sum(probability for probability, _ in selected) / len(selected)
                    if selected
                    else None
                ),
                actual_rate=(
                    sum(1 for _, correct in selected if correct) / len(selected)
                    if selected
                    else None
                ),
            )
        )
    return buckets


def _metrics(
    predictions: list[BacktestGame],
    *,
    key: str,
    label: str,
    baseline: bool = False,
) -> BacktestModelMetrics:
    count = len(predictions)
    if not count:
        return BacktestModelMetrics(
            key=key,
            label=label,
            prediction_count=0,
            accuracy=None,
            brier_score=None,
            log_loss=None,
            calibration=_calibration_buckets([], use_baseline=baseline),
        )

    if baseline:
        correct_values = [bool(row.baseline_correct) for row in predictions]
        brier_values = []
        log_values = []
        for row in predictions:
            home_outcome = 1.0 if row.winner == row.home_team else 0.0
            home_probability = float(row.baseline_home_probability or 0.5)
            clipped = clip_probability(home_probability)
            brier_values.append((home_probability - home_outcome) ** 2)
            log_values.append(
                -(
                    home_outcome * math.log(clipped)
                    + (1.0 - home_outcome) * math.log(1.0 - clipped)
                )
            )
    else:
        correct_values = [row.correct for row in predictions]
        brier_values = [row.brier for row in predictions]
        log_values = [row.log_loss for row in predictions]

    return BacktestModelMetrics(
        key=key,
        label=label,
        prediction_count=count,
        accuracy=sum(correct_values) / count,
        brier_score=sum(brier_values) / count,
        log_loss=sum(log_values) / count,
        calibration=_calibration_buckets(predictions, use_baseline=baseline),
    )


def evaluate_walk_forward_games(
    games: list[MlbGameState],
    *,
    season: int,
    start_date: date,
    end_date: date,
    min_games: int = 10,
    prior_games: list[MlbGameState] | None = None,
) -> BacktestResponse:
    """Compare old and enhanced models using only data known before each game."""
    priors = _season_priors(prior_games or [])
    calibration_rows = _training_rows_from_prior_season(
        prior_games or [],
        min_games=max(5, min_games),
    )
    calibrator = PlattCalibrator()
    calibrator.fit(calibration_rows)
    rows_at_last_fit = len(calibration_rows)

    totals: dict[str, TeamRunningTotals] = {}
    predictions: list[BacktestGame] = []
    completed_games = 0

    for game in _valid_final_games(games):
        completed_games += 1
        away = totals.setdefault(game.away_team, TeamRunningTotals())
        home = totals.setdefault(game.home_team, TeamRunningTotals())

        if away.games_played >= min_games and home.games_played >= min_games:
            baseline_home = _baseline_home_probability(away, home)
            baseline_away = 1.0 - baseline_home
            raw_home = _enhanced_raw_home_probability(
                away_team=game.away_team,
                home_team=game.home_team,
                away=away,
                home=home,
                priors=priors,
                running_league_rate=_running_league_rate(
                    totals, priors.league_runs_per_team_game
                ),
            )
            enhanced_home = calibrator.transform(raw_home)
            enhanced_away = 1.0 - enhanced_home

            winner = (
                game.away_team
                if (game.away_score or 0) > (game.home_score or 0)
                else game.home_team
            )
            home_outcome = 1.0 if winner == game.home_team else 0.0

            predicted_winner = (
                game.home_team if enhanced_home >= enhanced_away else game.away_team
            )
            predicted_probability = max(enhanced_home, enhanced_away)
            baseline_winner = (
                game.home_team if baseline_home >= baseline_away else game.away_team
            )
            baseline_probability = max(baseline_home, baseline_away)
            clipped_home = clip_probability(enhanced_home)
            brier = (enhanced_home - home_outcome) ** 2
            log_loss_value = -(
                home_outcome * math.log(clipped_home)
                + (1.0 - home_outcome) * math.log(1.0 - clipped_home)
            )

            predictions.append(
                BacktestGame(
                    game_pk=game.game_pk,
                    game_date=game.game_date,
                    away_team=game.away_team,
                    home_team=game.home_team,
                    away_score=game.away_score or 0,
                    home_score=game.home_score or 0,
                    away_probability=enhanced_away,
                    home_probability=enhanced_home,
                    predicted_winner=predicted_winner,
                    predicted_probability=predicted_probability,
                    winner=winner,
                    correct=predicted_winner == winner,
                    brier=brier,
                    log_loss=log_loss_value,
                    baseline_away_probability=baseline_away,
                    baseline_home_probability=baseline_home,
                    baseline_predicted_winner=baseline_winner,
                    baseline_predicted_probability=baseline_probability,
                    baseline_correct=baseline_winner == winner,
                    raw_home_probability=raw_home,
                    calibrated=calibrator.fitted,
                )
            )

            # The current result is appended only after its prediction is frozen.
            calibration_rows.append(
                ProbabilityRow(probability=raw_home, outcome=home_outcome)
            )
            if len(calibration_rows) - rows_at_last_fit >= PLATT_REFIT_INTERVAL:
                calibrator.fit(calibration_rows)
                rows_at_last_fit = len(calibration_rows)

        # Current scores become available to future games only here.
        away.add_game(game.away_score or 0, game.home_score or 0)
        home.add_game(game.home_score or 0, game.away_score or 0)

    baseline_metrics = _metrics(
        predictions,
        key="baseline",
        label="Old v0.4",
        baseline=True,
    )
    enhanced_metrics = _metrics(
        predictions,
        key="enhanced",
        label="Enhanced v0.5",
    )
    count = len(predictions)
    home_baseline = (
        sum(1 for row in predictions if row.winner == row.home_team) / count
        if count
        else None
    )

    accuracy_delta = None
    brier_delta = None
    log_loss_delta = None
    if baseline_metrics.accuracy is not None and enhanced_metrics.accuracy is not None:
        accuracy_delta = enhanced_metrics.accuracy - baseline_metrics.accuracy
    if baseline_metrics.brier_score is not None and enhanced_metrics.brier_score is not None:
        brier_delta = baseline_metrics.brier_score - enhanced_metrics.brier_score
    if baseline_metrics.log_loss is not None and enhanced_metrics.log_loss is not None:
        log_loss_delta = baseline_metrics.log_loss - enhanced_metrics.log_loss

    return BacktestResponse(
        generated_at=datetime.now(UTC),
        season=season,
        prior_season=season - 1 if prior_games else None,
        start_date=start_date,
        end_date=end_date,
        min_games=min_games,
        completed_games=completed_games,
        prediction_count=count,
        accuracy=enhanced_metrics.accuracy,
        brier_score=enhanced_metrics.brier_score,
        log_loss=enhanced_metrics.log_loss,
        home_baseline_accuracy=home_baseline,
        calibration=enhanced_metrics.calibration,
        recent_predictions=list(reversed(predictions[-20:])),
        baseline=baseline_metrics,
        enhanced=enhanced_metrics,
        accuracy_delta=accuracy_delta,
        brier_delta=brier_delta,
        log_loss_delta=log_loss_delta,
        calibration_training_games=len(calibration_rows),
        leakage_audit=BacktestLeakageAudit(),
    )


async def build_mlb_backtest(
    *,
    season: int,
    through: date | None = None,
    min_games: int = 10,
) -> BacktestResponse:
    """Fetch target and prior seasons, then run a strict chronological backtest."""
    today = datetime.now(EASTERN).date()
    season_start = date(season, 3, 1)
    default_end = date(season, 11, 15)
    if season >= today.year:
        default_end = today - timedelta(days=1)

    end_date = through or default_end
    prior_season = season - 1
    prior_start = date(prior_season, 3, 1)
    prior_end = date(prior_season, 11, 15)

    if end_date < season_start:
        return evaluate_walk_forward_games(
            [],
            season=season,
            start_date=season_start,
            end_date=end_date,
            min_games=min_games,
            prior_games=[],
        )

    timeout = httpx.Timeout(60.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.5.0 backtest"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            prior_games = await fetch_mlb_regular_season_schedule(
                client,
                season=prior_season,
                start_date=prior_start,
                end_date=prior_end,
            )
            games = await fetch_mlb_regular_season_schedule(
                client,
                season=season,
                start_date=season_start,
                end_date=end_date,
            )
    except (httpx.HTTPError, MlbDataError) as exc:
        raise BacktestError(str(exc)) from exc

    return evaluate_walk_forward_games(
        games,
        season=season,
        start_date=season_start,
        end_date=end_date,
        min_games=min_games,
        prior_games=prior_games,
    )
