"""Leakage-safe v0.6 feature engineering and regularized logistic model.

The optimized model uses only game-level information that existed before first pitch:
prior-regressed team strength, Elo, rolling run differential, rolling win rate, and rest.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable

import numpy as np

from .mlb import MlbGameState
from .model import (
    add_home_field_advantage,
    clip_probability,
    log5_probability,
    logit,
    regressed_pythagorean_expectation,
)

DEFAULT_ELO = 1500.0
ELO_K = 18.0
ELO_CARRY = 0.75
DEFAULT_HOME_WIN_RATE = 0.54
DEFAULT_LEAGUE_RUNS = 4.50
PRIOR_PSEUDO_GAMES = 25.0
FEATURE_NAMES = (
    "Base",
    "Elo",
    "Form10",
    "Form30",
    "Wins10",
    "Rest",
)
FEATURE_VARIANTS: dict[str, tuple[int, ...]] = {
    "Base": (0,),
    "Elo": (0, 1),
    "Form": (0, 1, 2, 3, 4),
    "Full": (0, 1, 2, 3, 4, 5),
}


def _parse_game_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _final_games(games: Iterable[MlbGameState]) -> list[MlbGameState]:
    return [
        game
        for game in sorted(games, key=lambda item: (_parse_game_time(item.game_date), item.game_pk))
        if game.is_final
        and game.away_score is not None
        and game.home_score is not None
        and game.away_score != game.home_score
    ]


@dataclass
class TeamSeasonState:
    runs_scored: float = 0.0
    runs_allowed: float = 0.0
    games_played: int = 0
    recent_run_diffs: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    recent_wins: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_game_time: datetime | None = None

    def update(self, *, scored: int, allowed: int, game_time: datetime) -> None:
        self.runs_scored += scored
        self.runs_allowed += allowed
        self.games_played += 1
        self.recent_run_diffs.append(float(scored - allowed))
        self.recent_wins.append(1.0 if scored > allowed else 0.0)
        self.last_game_time = game_time

    def average_run_diff(self, window: int) -> float:
        values = list(self.recent_run_diffs)[-window:]
        return sum(values) / len(values) if values else 0.0

    def win_rate(self, window: int) -> float:
        values = list(self.recent_wins)[-window:]
        return sum(values) / len(values) if values else 0.5


@dataclass(frozen=True)
class TeamPrior:
    runs_scored_per_game: float
    runs_allowed_per_game: float
    win_rate: float


@dataclass(frozen=True)
class SeasonSummary:
    teams: dict[str, TeamPrior]
    league_runs_per_team_game: float
    home_win_rate: float


@dataclass(frozen=True)
class FeatureRow:
    game: MlbGameState
    features: tuple[float, ...]
    home_win: float


@dataclass
class RegularizedLogisticModel:
    feature_indices: tuple[int, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    l2: float = 8.0

    @classmethod
    def fit(
        cls,
        rows: list[FeatureRow],
        *,
        feature_indices: tuple[int, ...],
        l2: float = 8.0,
    ) -> "RegularizedLogisticModel":
        if len(rows) < 20:
            return cls(
                feature_indices=feature_indices,
                means=np.zeros(len(feature_indices)),
                scales=np.ones(len(feature_indices)),
                coefficients=np.zeros(len(feature_indices) + 1),
                l2=l2,
            )

        raw_x = np.asarray(
            [[row.features[index] for index in feature_indices] for row in rows],
            dtype=float,
        )
        y = np.asarray([row.home_win for row in rows], dtype=float)
        means = raw_x.mean(axis=0)
        scales = raw_x.std(axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        standardized = (raw_x - means) / scales
        x = np.column_stack((np.ones(len(rows)), standardized))
        coefficients = np.zeros(x.shape[1], dtype=float)

        penalty = np.eye(x.shape[1], dtype=float) * l2
        penalty[0, 0] = 0.0

        for _ in range(60):
            linear = np.clip(x @ coefficients, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-linear))
            weight = np.clip(probability * (1.0 - probability), 1e-7, None)
            gradient = x.T @ (probability - y) + penalty @ coefficients
            hessian = x.T @ (x * weight[:, None]) + penalty
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            coefficients -= step
            if float(np.max(np.abs(step))) < 1e-7:
                break

        return cls(
            feature_indices=feature_indices,
            means=means,
            scales=scales,
            coefficients=coefficients,
            l2=l2,
        )

    def predict(self, features: tuple[float, ...]) -> float:
        raw = np.asarray([features[index] for index in self.feature_indices], dtype=float)
        standardized = (raw - self.means) / self.scales
        x = np.concatenate(([1.0], standardized))
        linear = float(np.clip(x @ self.coefficients, -30.0, 30.0))
        return clip_probability(1.0 / (1.0 + math.exp(-linear)), 0.05, 0.95)

    def coefficient_map(self) -> dict[str, float]:
        result = {"Intercept": float(self.coefficients[0])}
        for position, feature_index in enumerate(self.feature_indices, start=1):
            result[FEATURE_NAMES[feature_index]] = float(self.coefficients[position])
        return result


@dataclass(frozen=True)
class ModelVariant:
    key: str
    label: str
    model: RegularizedLogisticModel
    shrinkage: float
    validation_brier: float | None

    def predict(self, features: tuple[float, ...]) -> float:
        probability = self.model.predict(features)
        adjusted = (1.0 - self.shrinkage) * probability + self.shrinkage * 0.5
        return clip_probability(adjusted, 0.05, 0.95)


@dataclass(frozen=True)
class OptimizedModelArtifact:
    variants: dict[str, ModelVariant]
    training_rows: int
    validation_rows: int

    @property
    def full(self) -> ModelVariant:
        return self.variants["Full"]


class SeasonFeatureEngine:
    """Chronological feature state for one season."""

    def __init__(
        self,
        *,
        prior_summary: SeasonSummary | None,
        prior_elo: dict[str, float] | None,
    ) -> None:
        self.prior_summary = prior_summary or SeasonSummary(
            teams={},
            league_runs_per_team_game=DEFAULT_LEAGUE_RUNS,
            home_win_rate=DEFAULT_HOME_WIN_RATE,
        )
        self.states: dict[str, TeamSeasonState] = {}
        self.elo: dict[str, float] = {
            team: DEFAULT_ELO + ELO_CARRY * (rating - DEFAULT_ELO)
            for team, rating in (prior_elo or {}).items()
        }
        self.home_wins = 0
        self.completed_games = 0

    def _state(self, team: str) -> TeamSeasonState:
        return self.states.setdefault(team, TeamSeasonState())

    def _prior(self, team: str) -> TeamPrior:
        return self.prior_summary.teams.get(
            team,
            TeamPrior(
                runs_scored_per_game=self.prior_summary.league_runs_per_team_game,
                runs_allowed_per_game=self.prior_summary.league_runs_per_team_game,
                win_rate=0.5,
            ),
        )

    def _running_league_rate(self) -> float:
        runs = sum(state.runs_scored for state in self.states.values())
        team_games = sum(state.games_played for state in self.states.values())
        if not team_games:
            return self.prior_summary.league_runs_per_team_game
        return runs / team_games

    def _base_home_probability(self, away_team: str, home_team: str) -> float:
        away = self._state(away_team)
        home = self._state(home_team)
        away_prior = self._prior(away_team)
        home_prior = self._prior(home_team)
        league_rate = (
            0.75 * self.prior_summary.league_runs_per_team_game
            + 0.25 * self._running_league_rate()
        )
        away_strength = regressed_pythagorean_expectation(
            runs_scored=away.runs_scored,
            runs_allowed=away.runs_allowed,
            games_played=away.games_played,
            prior_runs_scored_per_game=away_prior.runs_scored_per_game,
            prior_runs_allowed_per_game=away_prior.runs_allowed_per_game,
            league_runs_per_team_game=league_rate,
            pseudo_games=PRIOR_PSEUDO_GAMES,
        )
        home_strength = regressed_pythagorean_expectation(
            runs_scored=home.runs_scored,
            runs_allowed=home.runs_allowed,
            games_played=home.games_played,
            prior_runs_scored_per_game=home_prior.runs_scored_per_game,
            prior_runs_allowed_per_game=home_prior.runs_allowed_per_game,
            league_runs_per_team_game=league_rate,
            pseudo_games=PRIOR_PSEUDO_GAMES,
        )
        neutral_home = log5_probability(home_strength, away_strength)
        return clip_probability(
            add_home_field_advantage(neutral_home, self.prior_summary.home_win_rate),
            0.10,
            0.90,
        )

    @staticmethod
    def _rest_days(state: TeamSeasonState, game_time: datetime) -> float:
        if state.last_game_time is None:
            return 3.0
        elapsed = (game_time - state.last_game_time).total_seconds() / 86400.0
        return min(max(elapsed, 0.0), 5.0)

    def features_for_matchup(
        self,
        *,
        away_team: str,
        home_team: str,
        game_time: datetime,
    ) -> tuple[float, ...]:
        away = self._state(away_team)
        home = self._state(home_team)
        base_home = self._base_home_probability(away_team, home_team)
        elo_diff = (
            self.elo.get(home_team, DEFAULT_ELO) - self.elo.get(away_team, DEFAULT_ELO)
        ) / 100.0
        form10 = home.average_run_diff(10) - away.average_run_diff(10)
        form30 = home.average_run_diff(30) - away.average_run_diff(30)
        wins10 = home.win_rate(10) - away.win_rate(10)
        rest = self._rest_days(home, game_time) - self._rest_days(away, game_time)
        rest = min(max(rest, -3.0), 3.0)
        return (logit(base_home), elo_diff, form10, form30, wins10, rest)

    def eligible(self, away_team: str, home_team: str, min_games: int) -> bool:
        return (
            self._state(away_team).games_played >= min_games
            and self._state(home_team).games_played >= min_games
        )

    def update_game(self, game: MlbGameState) -> None:
        if game.away_score is None or game.home_score is None:
            return
        game_time = _parse_game_time(game.game_date)
        away = self._state(game.away_team)
        home = self._state(game.home_team)

        away_rating = self.elo.get(game.away_team, DEFAULT_ELO)
        home_rating = self.elo.get(game.home_team, DEFAULT_ELO)
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))
        actual_home = 1.0 if game.home_score > game.away_score else 0.0
        margin = abs(game.home_score - game.away_score)
        margin_multiplier = min(max(math.log1p(margin), 1.0), 2.5)
        change = ELO_K * margin_multiplier * (actual_home - expected_home)
        self.elo[game.home_team] = home_rating + change
        self.elo[game.away_team] = away_rating - change

        away.update(scored=game.away_score, allowed=game.home_score, game_time=game_time)
        home.update(scored=game.home_score, allowed=game.away_score, game_time=game_time)
        self.completed_games += 1
        if actual_home == 1.0:
            self.home_wins += 1

    def summary(self) -> SeasonSummary:
        teams: dict[str, TeamPrior] = {}
        total_runs = 0.0
        team_games = 0
        for team, state in self.states.items():
            if not state.games_played:
                continue
            teams[team] = TeamPrior(
                runs_scored_per_game=state.runs_scored / state.games_played,
                runs_allowed_per_game=state.runs_allowed / state.games_played,
                win_rate=sum(state.recent_wins) / len(state.recent_wins)
                if state.recent_wins
                else 0.5,
            )
            total_runs += state.runs_scored
            team_games += state.games_played
        league_runs = total_runs / team_games if team_games else DEFAULT_LEAGUE_RUNS
        home_rate = self.home_wins / self.completed_games if self.completed_games else DEFAULT_HOME_WIN_RATE
        home_rate = min(max(home_rate, 0.50), 0.58)
        return SeasonSummary(
            teams=teams,
            league_runs_per_team_game=league_runs,
            home_win_rate=home_rate,
        )


def build_season_rows(
    games: list[MlbGameState],
    *,
    prior_summary: SeasonSummary | None,
    prior_elo: dict[str, float] | None,
    min_games: int,
) -> tuple[list[FeatureRow], SeasonFeatureEngine]:
    engine = SeasonFeatureEngine(prior_summary=prior_summary, prior_elo=prior_elo)
    rows: list[FeatureRow] = []
    for game in _final_games(games):
        game_time = _parse_game_time(game.game_date)
        if engine.eligible(game.away_team, game.home_team, min_games):
            rows.append(
                FeatureRow(
                    game=game,
                    features=engine.features_for_matchup(
                        away_team=game.away_team,
                        home_team=game.home_team,
                        game_time=game_time,
                    ),
                    home_win=1.0 if (game.home_score or 0) > (game.away_score or 0) else 0.0,
                )
            )
        engine.update_game(game)
    return rows, engine


def _brier(probabilities: list[float], rows: list[FeatureRow]) -> float | None:
    if not rows:
        return None
    return sum((probability - row.home_win) ** 2 for probability, row in zip(probabilities, rows, strict=True)) / len(rows)


def _choose_shrinkage(
    model: RegularizedLogisticModel,
    validation_rows: list[FeatureRow],
) -> tuple[float, float | None]:
    if not validation_rows:
        return 0.15, None
    raw = [model.predict(row.features) for row in validation_rows]
    candidates = (0.0, 0.10, 0.20, 0.30, 0.40)
    scored: list[tuple[float, float]] = []
    for shrinkage in candidates:
        probabilities = [(1.0 - shrinkage) * value + shrinkage * 0.5 for value in raw]
        score = _brier(probabilities, validation_rows)
        if score is not None:
            scored.append((score, shrinkage))
    if not scored:
        return 0.15, None
    best_score, best_shrinkage = min(scored)
    return best_shrinkage, best_score


def train_optimized_artifact(
    training_rows: list[FeatureRow],
    validation_rows: list[FeatureRow],
) -> OptimizedModelArtifact:
    variants: dict[str, ModelVariant] = {}
    for key, indices in FEATURE_VARIANTS.items():
        tuning_model = RegularizedLogisticModel.fit(
            training_rows,
            feature_indices=indices,
        )
        shrinkage, validation_brier = _choose_shrinkage(tuning_model, validation_rows)
        final_model = RegularizedLogisticModel.fit(
            [*training_rows, *validation_rows],
            feature_indices=indices,
        )
        variants[key] = ModelVariant(
            key=key.lower(),
            label=key,
            model=final_model,
            shrinkage=shrinkage,
            validation_brier=validation_brier,
        )
    return OptimizedModelArtifact(
        variants=variants,
        training_rows=len(training_rows),
        validation_rows=len(validation_rows),
    )


def prepare_optimized_model(
    *,
    seed_games: list[MlbGameState],
    training_games: list[MlbGameState],
    validation_games: list[MlbGameState],
    min_games: int,
) -> tuple[OptimizedModelArtifact, SeasonSummary, dict[str, float]]:
    _, seed_engine = build_season_rows(
        seed_games,
        prior_summary=None,
        prior_elo=None,
        min_games=min_games,
    )
    training_rows, training_engine = build_season_rows(
        training_games,
        prior_summary=seed_engine.summary(),
        prior_elo=seed_engine.elo,
        min_games=min_games,
    )
    validation_rows, validation_engine = build_season_rows(
        validation_games,
        prior_summary=training_engine.summary(),
        prior_elo=training_engine.elo,
        min_games=min_games,
    )
    artifact = train_optimized_artifact(training_rows, validation_rows)
    return artifact, validation_engine.summary(), validation_engine.elo


def build_target_rows(
    *,
    games: list[MlbGameState],
    prior_summary: SeasonSummary,
    prior_elo: dict[str, float],
    min_games: int,
) -> tuple[list[FeatureRow], SeasonFeatureEngine]:
    return build_season_rows(
        games,
        prior_summary=prior_summary,
        prior_elo=prior_elo,
        min_games=min_games,
    )
