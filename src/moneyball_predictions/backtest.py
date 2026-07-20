"""Leakage-safe walk-forward validation and model comparison for MLB games."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np

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
from .optimized import (
    FEATURE_NAMES,
    ModelVariant,
    build_target_rows,
    prepare_optimized_model,
    prepare_optimized_model_multifold,
)
from .schemas import (
    BacktestCalibrationBucket,
    BacktestCalibrationComparison,
    BacktestFeatureDiagnostic,
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


def _row_probability_and_correct(
    prediction: BacktestGame,
    mode: str,
) -> tuple[float | None, bool | None, float | None]:
    if mode == "baseline":
        return (
            prediction.baseline_predicted_probability,
            prediction.baseline_correct,
            prediction.baseline_home_probability,
        )
    if mode == "v05":
        return (
            prediction.enhanced_v05_predicted_probability,
            prediction.enhanced_v05_correct,
            prediction.enhanced_v05_home_probability,
        )
    return prediction.predicted_probability, prediction.correct, prediction.home_probability


def _calibration_buckets(
    predictions: list[BacktestGame],
    *,
    mode: str = "optimized",
) -> list[BacktestCalibrationBucket]:
    bounds = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    buckets: list[BacktestCalibrationBucket] = []
    for lower, upper in bounds:
        selected: list[tuple[float, bool]] = []
        for prediction in predictions:
            probability, correct, _ = _row_probability_and_correct(prediction, mode)
            if probability is not None and correct is not None and lower <= probability < upper:
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
    combined: list[tuple[float, bool]] = []
    for prediction in predictions:
        probability, correct, _ = _row_probability_and_correct(prediction, mode)
        if probability is not None and correct is not None and 0.60 <= probability < 0.80:
            combined.append((probability, correct))
    buckets.append(
        BacktestCalibrationBucket(
            label="60%-80% combined",
            predictions=len(combined),
            mean_probability=(
                sum(probability for probability, _ in combined) / len(combined)
                if combined
                else None
            ),
            actual_rate=(
                sum(1 for _, correct in combined if correct) / len(combined)
                if combined
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
    mode: str,
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
            calibration=_calibration_buckets([], mode=mode),
        )

    correct_values: list[bool] = []
    brier_values: list[float] = []
    log_values: list[float] = []
    for row in predictions:
        _, correct, home_probability = _row_probability_and_correct(row, mode)
        if correct is None or home_probability is None:
            continue
        home_outcome = 1.0 if row.winner == row.home_team else 0.0
        clipped = clip_probability(home_probability)
        correct_values.append(correct)
        brier_values.append((home_probability - home_outcome) ** 2)
        log_values.append(
            -(
                home_outcome * math.log(clipped)
                + (1.0 - home_outcome) * math.log(1.0 - clipped)
            )
        )

    actual_count = len(correct_values)
    return BacktestModelMetrics(
        key=key,
        label=label,
        prediction_count=actual_count,
        accuracy=sum(correct_values) / actual_count if actual_count else None,
        brier_score=sum(brier_values) / actual_count if actual_count else None,
        log_loss=sum(log_values) / actual_count if actual_count else None,
        calibration=_calibration_buckets(predictions, mode=mode),
    )


def _variant_metrics(
    predictions: list[BacktestGame],
    probability_map: dict[int, float],
    *,
    key: str,
    label: str,
) -> BacktestModelMetrics:
    selected = [row for row in predictions if row.game_pk in probability_map]
    if not selected:
        return BacktestModelMetrics(
            key=key,
            label=label,
            prediction_count=0,
            accuracy=None,
            brier_score=None,
            log_loss=None,
            calibration=[],
        )
    correct = 0
    briers: list[float] = []
    logs: list[float] = []
    buckets: list[BacktestCalibrationBucket] = []
    pairs: list[tuple[float, bool, float]] = []
    for row in selected:
        home_probability = probability_map[row.game_pk]
        winner = row.home_team if home_probability >= 0.5 else row.away_team
        is_correct = winner == row.winner
        correct += int(is_correct)
        outcome = 1.0 if row.winner == row.home_team else 0.0
        clipped = clip_probability(home_probability)
        briers.append((home_probability - outcome) ** 2)
        logs.append(-(outcome * math.log(clipped) + (1.0 - outcome) * math.log(1.0 - clipped)))
        confidence = max(home_probability, 1.0 - home_probability)
        pairs.append((confidence, is_correct, home_probability))
    for lower, upper in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]:
        values = [(confidence, is_correct) for confidence, is_correct, _ in pairs if lower <= confidence < upper]
        buckets.append(
            BacktestCalibrationBucket(
                label=f"{lower:.0%}-{min(upper, 1.0):.0%}",
                predictions=len(values),
                mean_probability=sum(v for v, _ in values) / len(values) if values else None,
                actual_rate=sum(1 for _, ok in values if ok) / len(values) if values else None,
            )
        )
    count = len(selected)
    return BacktestModelMetrics(
        key=key,
        label=label,
        prediction_count=count,
        accuracy=correct / count,
        brier_score=sum(briers) / count,
        log_loss=sum(logs) / count,
        calibration=buckets,
    )


def _feature_diagnostics(
    rows: list,
    variant: ModelVariant | None,
) -> list[BacktestFeatureDiagnostic]:
    if not rows:
        return []
    coefficients = variant.model.coefficient_map() if variant else {}
    diagnostics: list[BacktestFeatureDiagnostic] = []
    for index, name in enumerate(FEATURE_NAMES):
        values = np.asarray([row.features[index] for row in rows], dtype=float)
        if not len(values):
            continue
        std = float(values.std())
        diagnostics.append(
            BacktestFeatureDiagnostic(
                name=name,
                mean=float(values.mean()),
                std=std,
                minimum=float(values.min()),
                p05=float(np.quantile(values, 0.05)),
                median=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
                maximum=float(values.max()),
                unique_values=int(len(np.unique(np.round(values, 8)))),
                near_constant=std < 1e-3 or len(np.unique(np.round(values, 8))) < 10,
                coefficient=coefficients.get(name),
            )
        )
    return diagnostics


def _calibration_comparison(
    rows: list,
    variant: ModelVariant | None,
) -> list[BacktestCalibrationComparison]:
    if not rows or variant is None:
        return []
    raw = [variant.model.predict(row.features) for row in rows]
    comparisons: list[BacktestCalibrationComparison] = []
    for method, calibrator in sorted(variant.calibration_candidates.items()):
        probabilities = [calibrator.transform(value) for value in raw]
        count = len(rows)
        if not count:
            continue
        correct = 0
        brier = 0.0
        log_loss_value = 0.0
        for probability, row in zip(probabilities, rows, strict=True):
            predicted_home = probability >= 0.5
            actual_home = row.home_win >= 0.5
            correct += int(predicted_home == actual_home)
            brier += (probability - row.home_win) ** 2
            clipped = clip_probability(probability)
            log_loss_value += -(
                row.home_win * math.log(clipped)
                + (1.0 - row.home_win) * math.log(1.0 - clipped)
            )
        comparisons.append(
            BacktestCalibrationComparison(
                method=method,
                selected=(
                    calibrator.method == variant.calibrator.method
                    and abs(calibrator.shrinkage - variant.calibrator.shrinkage) < 1e-9
                ),
                validation_brier=variant.calibration_scores.get(method),
                target_accuracy=correct / count,
                target_brier=brier / count,
                target_log_loss=log_loss_value / count,
            )
        )
    return comparisons


def evaluate_walk_forward_games(
    games: list[MlbGameState],
    *,
    season: int,
    start_date: date,
    end_date: date,
    min_games: int = 10,
    prior_games: list[MlbGameState] | None = None,
    training_games_by_season: dict[int, list[MlbGameState]] | None = None,
) -> BacktestResponse:
    """Compare v0.4, v0.5, v0.7, and v0.9 using only pregame information."""
    optimized_probability: dict[int, float] = {}
    variant_probability_maps: dict[str, dict[int, float]] = {}
    optimized_artifact = None
    selected_optimized_variant = None
    optimized_label = "Component-aware v0.9"
    target_feature_rows = []
    training_seasons: list[int] = []
    calibration_seasons: list[int] = []

    required = (season - 4, season - 3, season - 2, season - 1)
    fallback_required = (season - 3, season - 2, season - 1)
    if training_games_by_season and all(year in training_games_by_season for year in required):
        optimized_artifact, target_prior, target_elo = prepare_optimized_model_multifold(
            earlier_seed_games=training_games_by_season[season - 4],
            earlier_training_games=training_games_by_season[season - 3],
            earlier_validation_games=training_games_by_season[season - 2],
            seed_games=training_games_by_season[season - 3],
            training_games=training_games_by_season[season - 2],
            validation_games=training_games_by_season[season - 1],
            min_games=min_games,
            fold_years=(season - 2, season - 1),
        )
    elif training_games_by_season and all(
        year in training_games_by_season for year in fallback_required
    ):
        optimized_artifact, target_prior, target_elo = prepare_optimized_model(
            seed_games=training_games_by_season[season - 3],
            training_games=training_games_by_season[season - 2],
            validation_games=training_games_by_season[season - 1],
            min_games=min_games,
        )

    if optimized_artifact is not None:
        target_feature_rows, _ = build_target_rows(
            games=games,
            prior_summary=target_prior,
            prior_elo=target_elo,
            min_games=min_games,
        )
        training_seasons = [season - 2]
        calibration_seasons = list(optimized_artifact.selection_folds) or [season - 1]
        for name, variant in optimized_artifact.variants.items():
            variant_probability_maps[name] = {
                row.game.game_pk: variant.predict(row.features) for row in target_feature_rows
            }
        target_component_coverage = (
            sum(1 for row in target_feature_rows if row.component_starter_covered)
            / len(target_feature_rows)
            if target_feature_rows
            else 0.0
        )
        component_ready = (
            optimized_artifact.training_component_coverage >= 0.70
            and optimized_artifact.validation_component_coverage >= 0.70
            and target_component_coverage >= 0.70
        )
        selected_optimized_variant = (
            optimized_artifact.champion
            if component_ready
            else optimized_artifact.variants["v07"]
        )
        selected_variant_name = next(
            (name for name, item in optimized_artifact.variants.items() if item is selected_optimized_variant),
            "v07",
        )
        optimized_probability = variant_probability_maps.get(selected_variant_name, {})
        optimized_label = (
            f"v0.9.1 {selected_optimized_variant.label}"
            if component_ready
            else "v0.7 proxy (pitching sync required)"
        )

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
            v05_home = calibrator.transform(raw_home)
            v05_away = 1.0 - v05_home
            optimized_home = optimized_probability.get(game.game_pk, v05_home)
            optimized_away = 1.0 - optimized_home

            winner = (
                game.away_team
                if (game.away_score or 0) > (game.home_score or 0)
                else game.home_team
            )
            home_outcome = 1.0 if winner == game.home_team else 0.0

            predicted_winner = game.home_team if optimized_home >= 0.5 else game.away_team
            predicted_probability = max(optimized_home, optimized_away)
            v05_winner = game.home_team if v05_home >= 0.5 else game.away_team
            v05_probability = max(v05_home, v05_away)
            baseline_winner = game.home_team if baseline_home >= 0.5 else game.away_team
            baseline_probability = max(baseline_home, baseline_away)
            clipped_home = clip_probability(optimized_home)
            brier = (optimized_home - home_outcome) ** 2
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
                    away_probability=optimized_away,
                    home_probability=optimized_home,
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
                    enhanced_v05_away_probability=v05_away,
                    enhanced_v05_home_probability=v05_home,
                    enhanced_v05_predicted_winner=v05_winner,
                    enhanced_v05_predicted_probability=v05_probability,
                    enhanced_v05_correct=v05_winner == winner,
                )
            )

            # v0.5 online calibration receives the current outcome only after prediction.
            calibration_rows.append(ProbabilityRow(probability=raw_home, outcome=home_outcome))
            if len(calibration_rows) - rows_at_last_fit >= PLATT_REFIT_INTERVAL:
                calibrator.fit(calibration_rows)
                rows_at_last_fit = len(calibration_rows)

        # Every model sees the current score only after its probability is frozen.
        away.add_game(game.away_score or 0, game.home_score or 0)
        home.add_game(game.home_score or 0, game.away_score or 0)

    baseline_metrics = _metrics(
        predictions,
        key="baseline",
        label="Old v0.4",
        mode="baseline",
    )
    enhanced_metrics = _metrics(
        predictions,
        key="enhanced",
        label="Enhanced v0.5",
        mode="v05",
    )
    optimized_metrics = _metrics(
        predictions,
        key="optimized",
        label=optimized_label,
        mode="optimized",
    )

    ablations: list[BacktestModelMetrics] = []
    if optimized_artifact:
        labels = {
            "v07": "v0.7 proxy benchmark",
            "ComponentStarter": "+ aggregate starter components",
            "StarterWorkload": "+ starter quality × expected innings",
            "Park": "+ prior-season park interaction",
            "StarterSplit": "+ separate K/BB/HR starter rates",
            "ParkNeutralSplit": "+ park-neutral team strength",
            "BullpenQuality": "+ bullpen quality (experimental)",
            "Full": "+ fatigue (experimental)",
        }
        for name in (
            "v07",
            "ComponentStarter",
            "StarterWorkload",
            "Park",
            "StarterSplit",
            "ParkNeutralSplit",
            "BullpenQuality",
            "Full",
        ):
            ablations.append(
                _variant_metrics(
                    predictions,
                    variant_probability_maps.get(name, {}),
                    key=name.lower(),
                    label=labels[name],
                )
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
    if enhanced_metrics.accuracy is not None and optimized_metrics.accuracy is not None:
        accuracy_delta = optimized_metrics.accuracy - enhanced_metrics.accuracy
    if enhanced_metrics.brier_score is not None and optimized_metrics.brier_score is not None:
        brier_delta = enhanced_metrics.brier_score - optimized_metrics.brier_score
    if enhanced_metrics.log_loss is not None and optimized_metrics.log_loss is not None:
        log_loss_delta = enhanced_metrics.log_loss - optimized_metrics.log_loss

    return BacktestResponse(
        generated_at=datetime.now(UTC),
        season=season,
        prior_season=season - 1 if prior_games else None,
        start_date=start_date,
        end_date=end_date,
        min_games=min_games,
        completed_games=completed_games,
        prediction_count=count,
        accuracy=optimized_metrics.accuracy,
        brier_score=optimized_metrics.brier_score,
        log_loss=optimized_metrics.log_loss,
        home_baseline_accuracy=home_baseline,
        calibration=optimized_metrics.calibration,
        recent_predictions=list(reversed(predictions[-20:])),
        baseline=baseline_metrics,
        enhanced=enhanced_metrics,
        optimized=optimized_metrics,
        ablations=ablations,
        accuracy_delta=accuracy_delta,
        brier_delta=brier_delta,
        log_loss_delta=log_loss_delta,
        calibration_training_games=len(calibration_rows),
        optimized_training_games=optimized_artifact.training_rows if optimized_artifact else 0,
        optimized_validation_games=optimized_artifact.validation_rows if optimized_artifact else 0,
        optimized_training_starter_coverage=(
            optimized_artifact.training_starter_coverage if optimized_artifact else 0.0
        ),
        optimized_validation_starter_coverage=(
            optimized_artifact.validation_starter_coverage if optimized_artifact else 0.0
        ),
        target_starter_coverage=(
            sum(1 for row in target_feature_rows if row.starter_covered) / len(target_feature_rows)
            if target_feature_rows
            else 0.0
        ),
        optimized_training_component_coverage=(
            optimized_artifact.training_component_coverage if optimized_artifact else 0.0
        ),
        optimized_validation_component_coverage=(
            optimized_artifact.validation_component_coverage if optimized_artifact else 0.0
        ),
        target_component_coverage=(
            sum(1 for row in target_feature_rows if row.component_starter_covered) / len(target_feature_rows)
            if target_feature_rows
            else 0.0
        ),
        optimized_training_bullpen_coverage=(
            optimized_artifact.training_bullpen_coverage if optimized_artifact else 0.0
        ),
        optimized_validation_bullpen_coverage=(
            optimized_artifact.validation_bullpen_coverage if optimized_artifact else 0.0
        ),
        target_bullpen_coverage=(
            sum(1 for row in target_feature_rows if row.bullpen_covered) / len(target_feature_rows)
            if target_feature_rows
            else 0.0
        ),
        optimized_shrinkage=(
            selected_optimized_variant.shrinkage if selected_optimized_variant else 0.0
        ),
        optimized_calibration_method=(
            selected_optimized_variant.calibration_method
            if selected_optimized_variant
            else "identity"
        ),
        optimized_coefficients=(
            selected_optimized_variant.model.coefficient_map()
            if selected_optimized_variant
            else {}
        ),
        optimized_variant=(selected_optimized_variant.label if selected_optimized_variant else ""),
        feature_diagnostics=_feature_diagnostics(
            target_feature_rows, selected_optimized_variant
        ),
        calibration_comparison=_calibration_comparison(
            target_feature_rows, selected_optimized_variant
        ),
        training_seasons=training_seasons,
        calibration_seasons=calibration_seasons,
        leakage_audit=BacktestLeakageAudit(),
    )


async def build_mlb_backtest(
    *,
    season: int,
    through: date | None = None,
    min_games: int = 10,
) -> BacktestResponse:
    """Fetch three pre-target seasons and run an untouched target-season test."""
    today = datetime.now(EASTERN).date()
    season_start = date(season, 3, 1)
    default_end = date(season, 11, 15)
    if season >= today.year:
        default_end = today - timedelta(days=1)

    end_date = through or default_end
    if end_date < season_start:
        return evaluate_walk_forward_games(
            [],
            season=season,
            start_date=season_start,
            end_date=end_date,
            min_games=min_games,
            prior_games=[],
        )

    years = [season - 4, season - 3, season - 2, season - 1]
    timeout = httpx.Timeout(90.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.9.1 backtest"}

    async def fetch_year(client: httpx.AsyncClient, year: int) -> tuple[int, list[MlbGameState]]:
        return (
            year,
            await fetch_mlb_regular_season_schedule(
                client,
                season=year,
                start_date=date(year, 3, 1),
                end_date=date(year, 11, 15),
            ),
        )

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            historical_pairs, games = await asyncio.gather(
                asyncio.gather(*(fetch_year(client, year) for year in years)),
                fetch_mlb_regular_season_schedule(
                    client,
                    season=season,
                    start_date=season_start,
                    end_date=end_date,
                ),
            )
    except (httpx.HTTPError, MlbDataError) as exc:
        raise BacktestError(str(exc)) from exc

    training_games = dict(historical_pairs)
    return evaluate_walk_forward_games(
        games,
        season=season,
        start_date=season_start,
        end_date=end_date,
        min_games=min_games,
        prior_games=training_games.get(season - 1, []),
        training_games_by_season=training_games,
    )

