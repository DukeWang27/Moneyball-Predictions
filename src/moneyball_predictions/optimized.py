"""Leakage-safe v0.9 pregame model.

v0.9 keeps the proven v0.7 proxy as a benchmark, then adds component pitching inputs
when official per-game box scores are available in ``data/mlb_boxscores``:

* starter K/BB/HBP/HR component quality with prior-season regression;
* expected starter innings;
* individual-reliever component quality;
* a three-day pitch-count fatigue index; and
* multi-fold calibration selection, park-neutral team strength, and component-rate pitching features.

Every feature is created before the current game is added to state. If the local
box-score cache is absent, component features safely regress to league average and the
v0.7 proxy remains visible in the ablation table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .mlb import MlbBattingLine, MlbGameState, MlbPitchingLine
from .sabermetrics import BattingLine, base_runs_smyth_v1, pythagenpat_win_pct
from .model import (
    add_home_field_advantage,
    clip_probability,
    log5_probability,
    logit,
    regressed_pythagorean_expectation,
)

DEFAULT_HOME_WIN_RATE = 0.54
DEFAULT_LEAGUE_RUNS = 4.50
DEFAULT_FIRST5_RUNS = 2.75
DEFAULT_LATE_RUNS = 1.75
PRIOR_PSEUDO_GAMES = 25.0
STARTER_PSEUDO_STARTS = 6.0
BULLPEN_PSEUDO_GAMES = 15.0
COMPONENT_PSEUDO_BF = 90.0
LOOSE_COMPONENT_PSEUDO_BF = 45.0
WORKLOAD_PSEUDO_STARTS = 5.0
PARK_PSEUDO_GAMES = 40.0

FEATURE_NAMES = (
    "Base",
    "ProxyStarter",
    "ProxyBullpen",
    "StarterComponent",
    "ExpectedInnings",
    "BullpenComponent",
    "BullpenFatigue",
    "StarterComponentLoose",
    "StarterWorkload",
    "ParkInteraction",
    "ParkNeutralBase",
    "StarterKRate",
    "StarterBBRate",
    "StarterHRRate",
    "BaseRuns",
)
FEATURE_VARIANTS: dict[str, tuple[int, ...]] = {
    "v07": (0, 1, 2),
    "ComponentStarter": (0, 3),
    "StarterWorkload": (0, 3, 8),
    "Park": (0, 3, 8, 9),
    "StarterSplit": (0, 11, 12, 13),
    "ParkNeutralSplit": (10, 11, 12, 13, 8, 9),
    "BaseRunsStarter": (14, 11, 12, 13, 8, 9),
    "BullpenQuality": (10, 11, 12, 13, 8, 9, 5),
    "Full": (10, 11, 12, 13, 8, 9, 5, 6),
}
LIVE_CHAMPION_CANDIDATES = (
    "ComponentStarter",
    "StarterWorkload",
    "Park",
    "StarterSplit",
    "ParkNeutralSplit",
    "BaseRunsStarter",
)


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


def _pitcher_key(pitcher_id: int | None, pitcher_name: str | None) -> str | None:
    if pitcher_id is not None:
        return f"id:{pitcher_id}"
    if pitcher_name:
        normalized = " ".join(pitcher_name.casefold().split())
        return f"name:{normalized}"
    return None


def _line_key(line: MlbPitchingLine) -> str:
    return f"id:{line.pitcher_id}"


@dataclass
class TeamSeasonState:
    runs_scored: float = 0.0
    runs_allowed: float = 0.0
    games_played: int = 0

    def update(self, *, scored: float, allowed: float) -> None:
        self.runs_scored += scored
        self.runs_allowed += allowed
        self.games_played += 1


@dataclass
class BaseRunsComponentState:
    at_bats: float = 0.0
    hits: float = 0.0
    doubles: float = 0.0
    triples: float = 0.0
    home_runs: float = 0.0
    walks: float = 0.0
    intentional_walks: float = 0.0
    hit_by_pitch: float = 0.0
    games: int = 0

    def update(self, line: MlbBattingLine) -> None:
        self.at_bats += line.at_bats
        self.hits += line.hits
        self.doubles += line.doubles
        self.triples += line.triples
        self.home_runs += line.home_runs
        self.walks += line.walks
        self.intentional_walks += line.intentional_walks
        self.hit_by_pitch += line.hit_by_pitch
        self.games += 1

    def estimated_runs(self) -> float:
        return base_runs_smyth_v1(
            BattingLine(
                at_bats=self.at_bats,
                hits=self.hits,
                doubles=self.doubles,
                triples=self.triples,
                home_runs=self.home_runs,
                walks=self.walks,
                intentional_walks=self.intentional_walks,
                hit_by_pitch=self.hit_by_pitch,
            )
        )


@dataclass
class PitcherStartState:
    first5_runs_allowed: float = 0.0
    starts: int = 0

    def update(self, runs_allowed: float) -> None:
        self.first5_runs_allowed += runs_allowed
        self.starts += 1


@dataclass
class BullpenState:
    late_runs_allowed: float = 0.0
    games: int = 0

    def update(self, runs_allowed: float) -> None:
        self.late_runs_allowed += runs_allowed
        self.games += 1


@dataclass
class PitcherComponentState:
    batters_faced: float = 0.0
    strikeouts: float = 0.0
    walks: float = 0.0
    hit_batters: float = 0.0
    home_runs: float = 0.0
    outs_recorded: float = 0.0
    pitches_thrown: float = 0.0
    starts: int = 0
    appearances: int = 0
    recent_usage: list[tuple[datetime, int]] = field(default_factory=list)

    def update(self, line: MlbPitchingLine, game_time: datetime) -> None:
        self.batters_faced += line.batters_faced
        self.strikeouts += line.strikeouts
        self.walks += line.walks
        self.hit_batters += line.hit_batters
        self.home_runs += line.home_runs
        self.outs_recorded += line.outs_recorded
        self.pitches_thrown += line.pitches_thrown
        self.starts += int(line.is_starter)
        self.appearances += 1
        self.recent_usage.append((game_time, line.pitches_thrown))
        cutoff = game_time.timestamp() - 5 * 86400
        self.recent_usage = [item for item in self.recent_usage if item[0].timestamp() >= cutoff]


@dataclass(frozen=True)
class TeamPrior:
    runs_scored_per_game: float
    runs_allowed_per_game: float


@dataclass(frozen=True)
class PitcherComponentPrior:
    batters_faced: float
    strikeouts: float
    walks: float
    hit_batters: float
    home_runs: float
    outs_recorded: float
    starts: int
    appearances: int


@dataclass(frozen=True)
class SeasonSummary:
    teams: dict[str, TeamPrior]
    pitcher_first5_rates: dict[str, float]
    bullpen_late_rates: dict[str, float]
    league_runs_per_team_game: float
    league_first5_runs_per_team_game: float
    league_late_runs_per_team_game: float
    home_win_rate: float
    base_runs_teams: dict[str, TeamPrior] = field(default_factory=dict)
    league_base_runs_per_team_game: float = DEFAULT_LEAGUE_RUNS
    pitcher_components: dict[str, PitcherComponentPrior] = field(default_factory=dict)
    team_relievers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    park_run_factors: dict[str, float] = field(default_factory=dict)
    league_component_score: float = 0.0
    league_k_rate: float = 0.22
    league_bb_hbp_rate: float = 0.09
    league_hr_rate: float = 0.03
    league_starter_innings: float = 5.2


@dataclass(frozen=True)
class FeatureRow:
    game: MlbGameState
    features: tuple[float, ...]
    home_win: float
    starter_covered: bool = False
    component_starter_covered: bool = False
    bullpen_covered: bool = False


@dataclass
class RegularizedLogisticModel:
    feature_indices: tuple[int, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    l2: float = 10.0

    @classmethod
    def fit(
        cls,
        rows: list[FeatureRow],
        *,
        feature_indices: tuple[int, ...],
        l2: float = 10.0,
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
            [[row.features[index] for index in feature_indices] for row in rows], dtype=float
        )
        y = np.asarray([row.home_win for row in rows], dtype=float)
        means = raw_x.mean(axis=0)
        scales = np.where(raw_x.std(axis=0) < 1e-8, 1.0, raw_x.std(axis=0))
        x = np.column_stack((np.ones(len(rows)), (raw_x - means) / scales))
        coefficients = np.zeros(x.shape[1], dtype=float)
        penalty = np.eye(x.shape[1], dtype=float) * l2
        penalty[0, 0] = 0.0
        for _ in range(80):
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
        return cls(feature_indices, means, scales, coefficients, l2)

    def predict(self, features: tuple[float, ...]) -> float:
        raw = np.asarray([features[index] for index in self.feature_indices], dtype=float)
        x = np.concatenate(([1.0], (raw - self.means) / self.scales))
        linear = float(np.clip(x @ self.coefficients, -30.0, 30.0))
        return clip_probability(1.0 / (1.0 + math.exp(-linear)), 0.08, 0.92)

    def coefficient_map(self) -> dict[str, float]:
        result = {"Intercept": float(self.coefficients[0])}
        for position, feature_index in enumerate(self.feature_indices, start=1):
            result[FEATURE_NAMES[feature_index]] = float(self.coefficients[position])
        return result


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: str = "identity"
    intercept: float = 0.0
    slope: float = 1.0
    shrinkage: float = 0.0
    x_thresholds: tuple[float, ...] = ()
    y_thresholds: tuple[float, ...] = ()

    def transform(self, probability: float) -> float:
        probability = clip_probability(probability)
        if self.method == "platt":
            value = self.intercept + self.slope * logit(probability)
            calibrated = 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))
            return clip_probability(calibrated, 0.08, 0.92)
        if self.method == "isotonic" and self.x_thresholds and self.y_thresholds:
            calibrated = float(
                np.interp(
                    probability,
                    np.asarray(self.x_thresholds, dtype=float),
                    np.asarray(self.y_thresholds, dtype=float),
                )
            )
            return clip_probability(calibrated, 0.08, 0.92)
        if self.method == "shrink":
            return clip_probability(
                (1.0 - self.shrinkage) * probability + self.shrinkage * 0.5,
                0.08,
                0.92,
            )
        return probability


@dataclass(frozen=True)
class ModelVariant:
    key: str
    label: str
    model: RegularizedLogisticModel
    calibrator: ProbabilityCalibrator
    validation_brier: float | None
    calibration_candidates: dict[str, ProbabilityCalibrator] = field(default_factory=dict)
    calibration_scores: dict[str, float] = field(default_factory=dict)

    @property
    def shrinkage(self) -> float:
        return self.calibrator.shrinkage

    @property
    def calibration_method(self) -> str:
        return self.calibrator.method

    def predict(self, features: tuple[float, ...]) -> float:
        return self.calibrator.transform(self.model.predict(features))


@dataclass(frozen=True)
class OptimizedModelArtifact:
    variants: dict[str, ModelVariant]
    training_rows: int
    validation_rows: int
    training_starter_coverage: float
    validation_starter_coverage: float
    training_component_coverage: float = 0.0
    validation_component_coverage: float = 0.0
    training_bullpen_coverage: float = 0.0
    validation_bullpen_coverage: float = 0.0
    champion_key: str | None = None
    selection_scores: dict[str, float] = field(default_factory=dict)
    selection_folds: tuple[int, ...] = ()

    @property
    def full(self) -> ModelVariant:
        return self.variants["Full"]

    @property
    def champion(self) -> ModelVariant:
        if self.champion_key and self.champion_key in self.variants:
            return self.variants[self.champion_key]
        candidates = [
            self.variants[key]
            for key in LIVE_CHAMPION_CANDIDATES
            if key in self.variants and self.variants[key].validation_brier is not None
        ]
        if not candidates:
            return self.variants["ComponentStarter"]
        best_score = min(float(item.validation_brier or 1.0) for item in candidates)
        # Prefer the simpler challenger when validation scores are effectively tied.
        return next(
            item for item in candidates if float(item.validation_brier or 1.0) <= best_score + 0.0002
        )


class SeasonFeatureEngine:
    """Chronological v0.9 state for one season."""

    def __init__(
        self,
        *,
        prior_summary: SeasonSummary | None,
        prior_elo: dict[str, float] | None = None,
    ) -> None:
        del prior_elo
        self.prior_summary = prior_summary or SeasonSummary(
            teams={},
            pitcher_first5_rates={},
            bullpen_late_rates={},
            league_runs_per_team_game=DEFAULT_LEAGUE_RUNS,
            league_first5_runs_per_team_game=DEFAULT_FIRST5_RUNS,
            league_late_runs_per_team_game=DEFAULT_LATE_RUNS,
            home_win_rate=DEFAULT_HOME_WIN_RATE,
        )
        self.states: dict[str, TeamSeasonState] = {}
        self.park_adjusted_states: dict[str, TeamSeasonState] = {}
        self.pitchers: dict[str, PitcherStartState] = {}
        self.base_runs_offense: dict[str, BaseRunsComponentState] = {}
        self.base_runs_allowed: dict[str, BaseRunsComponentState] = {}
        self.bullpens: dict[str, BullpenState] = {}
        self.components: dict[str, PitcherComponentState] = {}
        self.team_relievers: dict[str, set[str]] = {
            team: set(keys) for team, keys in self.prior_summary.team_relievers.items()
        }
        self.venue_runs: dict[str, float] = {}
        self.venue_games: dict[str, int] = {}
        self.home_wins = 0
        self.completed_games = 0

    @property
    def elo(self) -> dict[str, float]:
        return {}

    def _state(self, team: str) -> TeamSeasonState:
        return self.states.setdefault(team, TeamSeasonState())

    def _park_state(self, team: str) -> TeamSeasonState:
        return self.park_adjusted_states.setdefault(team, TeamSeasonState())

    def _base_runs_offense_state(self, team: str) -> BaseRunsComponentState:
        return self.base_runs_offense.setdefault(team, BaseRunsComponentState())

    def _base_runs_allowed_state(self, team: str) -> BaseRunsComponentState:
        return self.base_runs_allowed.setdefault(team, BaseRunsComponentState())

    def _bullpen(self, team: str) -> BullpenState:
        return self.bullpens.setdefault(team, BullpenState())

    def _component(self, key: str) -> PitcherComponentState:
        return self.components.setdefault(key, PitcherComponentState())

    def _prior(self, team: str) -> TeamPrior:
        return self.prior_summary.teams.get(
            team,
            TeamPrior(
                self.prior_summary.league_runs_per_team_game,
                self.prior_summary.league_runs_per_team_game,
            ),
        )

    def _running_league_rate(self) -> float:
        runs = sum(state.runs_scored for state in self.states.values())
        team_games = sum(state.games_played for state in self.states.values())
        return runs / team_games if team_games else self.prior_summary.league_runs_per_team_game

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
        return clip_probability(
            add_home_field_advantage(
                log5_probability(home_strength, away_strength),
                self.prior_summary.home_win_rate,
            ),
            0.10,
            0.90,
        )

    def _park_neutral_home_probability(self, away_team: str, home_team: str) -> float:
        away = self._park_state(away_team)
        home = self._park_state(home_team)
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
        return clip_probability(
            add_home_field_advantage(
                log5_probability(home_strength, away_strength),
                self.prior_summary.home_win_rate,
            ),
            0.10,
            0.90,
        )

    def _base_runs_home_probability(self, away_team: str, home_team: str) -> float:
        league = self.prior_summary.league_base_runs_per_team_game

        def team_strength(team: str) -> float:
            offense = self._base_runs_offense_state(team)
            allowed = self._base_runs_allowed_state(team)
            prior = self.prior_summary.base_runs_teams.get(
                team, TeamPrior(league, league)
            )
            scored_rate = (
                offense.estimated_runs()
                + prior.runs_scored_per_game * PRIOR_PSEUDO_GAMES
            ) / (offense.games + PRIOR_PSEUDO_GAMES)
            allowed_rate = (
                allowed.estimated_runs()
                + prior.runs_allowed_per_game * PRIOR_PSEUDO_GAMES
            ) / (allowed.games + PRIOR_PSEUDO_GAMES)
            return pythagenpat_win_pct(scored_rate, allowed_rate, 1.0)

        away_strength = team_strength(away_team)
        home_strength = team_strength(home_team)
        return clip_probability(
            add_home_field_advantage(
                log5_probability(home_strength, away_strength),
                self.prior_summary.home_win_rate,
            ),
            0.10,
            0.90,
        )

    def _starter_proxy_rate(self, pitcher_key: str | None) -> float:
        league = self.prior_summary.league_first5_runs_per_team_game
        if pitcher_key is None:
            return league
        current = self.pitchers.get(pitcher_key, PitcherStartState())
        prior = self.prior_summary.pitcher_first5_rates.get(pitcher_key, league)
        return (current.first5_runs_allowed + prior * STARTER_PSEUDO_STARTS) / (
            current.starts + STARTER_PSEUDO_STARTS
        )

    def _bullpen_proxy_rate(self, team: str) -> float:
        league = self.prior_summary.league_late_runs_per_team_game
        current = self._bullpen(team)
        prior = self.prior_summary.bullpen_late_rates.get(team, league)
        return (current.late_runs_allowed + prior * BULLPEN_PSEUDO_GAMES) / (
            current.games + BULLPEN_PSEUDO_GAMES
        )

    def _component_totals(self, key: str | None) -> tuple[float, float, float, float, float, float, float, int]:
        if key is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        current = self.components.get(key, PitcherComponentState())
        prior = self.prior_summary.pitcher_components.get(key)
        return (
            current.batters_faced + (prior.batters_faced if prior else 0.0),
            current.strikeouts + (prior.strikeouts if prior else 0.0),
            current.walks + (prior.walks if prior else 0.0),
            current.hit_batters + (prior.hit_batters if prior else 0.0),
            current.home_runs + (prior.home_runs if prior else 0.0),
            current.outs_recorded + (prior.outs_recorded if prior else 0.0),
            current.pitches_thrown,
            current.starts + (prior.starts if prior else 0),
        )

    def _component_score(
        self,
        key: str | None,
        *,
        pseudo_bf: float = COMPONENT_PSEUDO_BF,
    ) -> float:
        bf, strikeouts, walks, hit_batters, home_runs, _, _, _ = self._component_totals(key)
        league = self.prior_summary.league_component_score
        if bf <= 0:
            return league
        observed = (13.0 * home_runs + 3.0 * (walks + hit_batters) - 2.0 * strikeouts) / bf
        reliability = bf / (bf + pseudo_bf)
        return reliability * observed + (1.0 - reliability) * league

    def _component_rates(
        self,
        key: str | None,
        *,
        pseudo_bf: float = COMPONENT_PSEUDO_BF,
    ) -> tuple[float, float, float]:
        bf, strikeouts, walks, hit_batters, home_runs, _, _, _ = self._component_totals(key)
        if bf <= 0:
            return (
                self.prior_summary.league_k_rate,
                self.prior_summary.league_bb_hbp_rate,
                self.prior_summary.league_hr_rate,
            )
        reliability = bf / (bf + pseudo_bf)
        k_rate = reliability * (strikeouts / bf) + (1.0 - reliability) * self.prior_summary.league_k_rate
        bb_rate = reliability * ((walks + hit_batters) / bf) + (1.0 - reliability) * self.prior_summary.league_bb_hbp_rate
        hr_rate = reliability * (home_runs / bf) + (1.0 - reliability) * self.prior_summary.league_hr_rate
        return k_rate, bb_rate, hr_rate

    def _expected_starter_innings(self, key: str | None) -> float:
        _, _, _, _, _, outs, _, starts = self._component_totals(key)
        prior_innings = self.prior_summary.league_starter_innings
        innings = (outs / 3.0 + prior_innings * WORKLOAD_PSEUDO_STARTS) / (
            starts + WORKLOAD_PSEUDO_STARTS
        )
        return min(max(innings, 3.0), 7.0)

    def _offense_rate(self, team: str) -> float:
        state = self._state(team)
        prior = self._prior(team)
        return (
            state.runs_scored + prior.runs_scored_per_game * PRIOR_PSEUDO_GAMES
        ) / (state.games_played + PRIOR_PSEUDO_GAMES)

    def _park_factor(self, venue: str | None) -> float:
        if not venue:
            return 1.0
        return min(max(self.prior_summary.park_run_factors.get(venue, 1.0), 0.80), 1.25)

    def _fatigue(self, key: str, game_time: datetime) -> float:
        state = self.components.get(key)
        if not state:
            return 0.0
        score = 0.0
        for appearance_time, pitches in state.recent_usage:
            days = (game_time - appearance_time).total_seconds() / 86400.0
            if 0.0 < days <= 1.5:
                score += pitches
            elif days <= 2.5:
                score += 0.6 * pitches
            elif days <= 3.5:
                score += 0.3 * pitches
        return score

    def _bullpen_profile(
        self,
        team: str,
        game_time: datetime,
        starter_key: str | None,
    ) -> tuple[float, float, bool]:
        keys = set(self.team_relievers.get(team, set()))
        keys.discard(starter_key)
        candidates: list[tuple[float, float, float]] = []
        for key in keys:
            bf, _, _, _, _, outs, _, _ = self._component_totals(key)
            if bf <= 0 and outs <= 0:
                continue
            fatigue = self._fatigue(key, game_time)
            quality = self._component_score(key)
            availability = math.exp(-fatigue / 55.0)
            usage = max(outs, 3.0) ** 0.5
            candidates.append((quality, availability, usage))
        if not candidates:
            return self.prior_summary.league_component_score, 0.0, False
        candidates.sort(key=lambda item: (item[0] + (1.0 - item[1]) * 0.20, -item[2]))
        selected = candidates[:5]
        weights = [availability * usage for _, availability, usage in selected]
        denominator = sum(weights)
        quality = (
            sum(item[0] * weight for item, weight in zip(selected, weights, strict=True))
            / denominator
            if denominator > 0
            else self.prior_summary.league_component_score
        )
        fatigue = sum((1.0 - item[1]) for item in selected) / len(selected)
        return quality, fatigue, True

    def features_for_matchup(
        self,
        *,
        away_team: str,
        home_team: str,
        game_time: datetime,
        away_pitcher_id: int | None = None,
        home_pitcher_id: int | None = None,
        away_pitcher_name: str | None = None,
        home_pitcher_name: str | None = None,
        venue: str | None = None,
    ) -> tuple[float, ...]:
        base_home = self._base_home_probability(away_team, home_team)
        park_neutral_home = self._park_neutral_home_probability(away_team, home_team)
        away_key = _pitcher_key(away_pitcher_id, away_pitcher_name)
        home_key = _pitcher_key(home_pitcher_id, home_pitcher_name)
        proxy_starter = self._starter_proxy_rate(away_key) - self._starter_proxy_rate(home_key)
        proxy_bullpen = self._bullpen_proxy_rate(away_team) - self._bullpen_proxy_rate(home_team)
        starter_component = self._component_score(away_key) - self._component_score(home_key)
        starter_component_loose = self._component_score(
            away_key, pseudo_bf=LOOSE_COMPONENT_PSEUDO_BF
        ) - self._component_score(home_key, pseudo_bf=LOOSE_COMPONENT_PSEUDO_BF)
        away_expected_innings = self._expected_starter_innings(away_key)
        home_expected_innings = self._expected_starter_innings(home_key)
        expected_innings = home_expected_innings - away_expected_innings
        starter_workload = (
            self._component_score(away_key, pseudo_bf=LOOSE_COMPONENT_PSEUDO_BF)
            * away_expected_innings
            - self._component_score(home_key, pseudo_bf=LOOSE_COMPONENT_PSEUDO_BF)
            * home_expected_innings
        ) / max(self.prior_summary.league_starter_innings, 1.0)
        park_interaction = (self._park_factor(venue) - 1.0) * (
            self._offense_rate(home_team) - self._offense_rate(away_team)
        )
        away_bullpen, away_fatigue, _ = self._bullpen_profile(away_team, game_time, away_key)
        home_bullpen, home_fatigue, _ = self._bullpen_profile(home_team, game_time, home_key)
        bullpen_component = away_bullpen - home_bullpen
        bullpen_fatigue = away_fatigue - home_fatigue
        away_k, away_bb, away_hr = self._component_rates(away_key)
        home_k, home_bb, home_hr = self._component_rates(home_key)
        starter_k_rate = home_k - away_k
        starter_bb_rate = away_bb - home_bb
        starter_hr_rate = away_hr - home_hr
        base_runs_home = self._base_runs_home_probability(away_team, home_team)
        return (
            logit(base_home),
            proxy_starter,
            proxy_bullpen,
            starter_component,
            expected_innings,
            bullpen_component,
            bullpen_fatigue,
            starter_component_loose,
            starter_workload,
            park_interaction,
            logit(park_neutral_home),
            starter_k_rate,
            starter_bb_rate,
            starter_hr_rate,
            logit(base_runs_home),
        )

    def starter_covered(self, game: MlbGameState) -> bool:
        return (
            _pitcher_key(game.away_probable_pitcher_id, game.away_probable_pitcher) is not None
            and _pitcher_key(game.home_probable_pitcher_id, game.home_probable_pitcher) is not None
            and game.away_first5_runs is not None
            and game.home_first5_runs is not None
        )

    def component_starter_covered(self, game: MlbGameState) -> bool:
        return game.away_starter_line is not None and game.home_starter_line is not None

    def bullpen_covered(self, game: MlbGameState) -> bool:
        return len(game.away_pitching) > 1 and len(game.home_pitching) > 1

    def eligible(self, away_team: str, home_team: str, min_games: int) -> bool:
        return (
            self._state(away_team).games_played >= min_games
            and self._state(home_team).games_played >= min_games
        )

    def _update_pitching_side(
        self,
        team: str,
        lines: tuple[MlbPitchingLine, ...],
        game_time: datetime,
    ) -> None:
        for line in lines:
            key = _line_key(line)
            self._component(key).update(line, game_time)
            if not line.is_starter:
                self.team_relievers.setdefault(team, set()).add(key)

    def update_game(self, game: MlbGameState) -> None:
        if game.away_score is None or game.home_score is None:
            return
        game_time = _parse_game_time(game.game_date)
        self._state(game.away_team).update(scored=game.away_score, allowed=game.home_score)
        self._state(game.home_team).update(scored=game.home_score, allowed=game.away_score)
        park_factor = self._park_factor(game.venue)
        self._park_state(game.away_team).update(
            scored=game.away_score / park_factor,
            allowed=game.home_score / park_factor,
        )
        self._park_state(game.home_team).update(
            scored=game.home_score / park_factor,
            allowed=game.away_score / park_factor,
        )

        if game.away_batting is not None:
            self._base_runs_offense_state(game.away_team).update(game.away_batting)
            self._base_runs_allowed_state(game.home_team).update(game.away_batting)
        if game.home_batting is not None:
            self._base_runs_offense_state(game.home_team).update(game.home_batting)
            self._base_runs_allowed_state(game.away_team).update(game.home_batting)

        if game.away_first5_runs is not None and game.home_first5_runs is not None:
            away_pitcher = _pitcher_key(game.away_probable_pitcher_id, game.away_probable_pitcher)
            home_pitcher = _pitcher_key(game.home_probable_pitcher_id, game.home_probable_pitcher)
            if away_pitcher:
                self.pitchers.setdefault(away_pitcher, PitcherStartState()).update(
                    float(game.home_first5_runs)
                )
            if home_pitcher:
                self.pitchers.setdefault(home_pitcher, PitcherStartState()).update(
                    float(game.away_first5_runs)
                )
            self._bullpen(game.away_team).update(
                max(float(game.home_score - game.home_first5_runs), 0.0)
            )
            self._bullpen(game.home_team).update(
                max(float(game.away_score - game.away_first5_runs), 0.0)
            )

        self._update_pitching_side(game.away_team, game.away_pitching, game_time)
        self._update_pitching_side(game.home_team, game.home_pitching, game_time)
        if game.venue:
            self.venue_runs[game.venue] = self.venue_runs.get(game.venue, 0.0) + (
                game.away_score + game.home_score
            )
            self.venue_games[game.venue] = self.venue_games.get(game.venue, 0) + 1
        self.completed_games += 1
        self.home_wins += int(game.home_score > game.away_score)

    def summary(self) -> SeasonSummary:
        teams: dict[str, TeamPrior] = {}
        total_runs = 0.0
        team_games = 0
        for team, state in self.states.items():
            if state.games_played:
                teams[team] = TeamPrior(
                    state.runs_scored / state.games_played,
                    state.runs_allowed / state.games_played,
                )
                total_runs += state.runs_scored
                team_games += state.games_played
        base_runs_teams: dict[str, TeamPrior] = {}
        total_base_runs = 0.0
        base_runs_games = 0
        for team in set(self.base_runs_offense) | set(self.base_runs_allowed):
            offense = self.base_runs_offense.get(team, BaseRunsComponentState())
            allowed = self.base_runs_allowed.get(team, BaseRunsComponentState())
            games = min(offense.games, allowed.games)
            if games <= 0:
                continue
            scored_rate = offense.estimated_runs() / games
            allowed_rate = allowed.estimated_runs() / games
            base_runs_teams[team] = TeamPrior(scored_rate, allowed_rate)
            total_base_runs += offense.estimated_runs()
            base_runs_games += games

        pitcher_rates = {
            key: state.first5_runs_allowed / state.starts
            for key, state in self.pitchers.items()
            if state.starts
        }
        bullpen_rates = {
            team: state.late_runs_allowed / state.games
            for team, state in self.bullpens.items()
            if state.games
        }
        component_priors = {
            key: PitcherComponentPrior(
                state.batters_faced,
                state.strikeouts,
                state.walks,
                state.hit_batters,
                state.home_runs,
                state.outs_recorded,
                state.starts,
                state.appearances,
            )
            for key, state in self.components.items()
            if state.appearances
        }
        total_bf = sum(state.batters_faced for state in self.components.values())
        total_k = sum(state.strikeouts for state in self.components.values())
        total_bb_hbp = sum(state.walks + state.hit_batters for state in self.components.values())
        total_hr = sum(state.home_runs for state in self.components.values())
        numerator = sum(
            13.0 * state.home_runs
            + 3.0 * (state.walks + state.hit_batters)
            - 2.0 * state.strikeouts
            for state in self.components.values()
        )
        starter_outs = sum(state.outs_recorded for state in self.components.values() if state.starts)
        starts = sum(state.starts for state in self.components.values())
        league_total_runs_per_game = (
            2.0 * total_runs / team_games if team_games else 2.0 * DEFAULT_LEAGUE_RUNS
        )
        park_factors = {
            venue: (
                runs + league_total_runs_per_game * PARK_PSEUDO_GAMES
            )
            / (
                (self.venue_games.get(venue, 0) + PARK_PSEUDO_GAMES)
                * league_total_runs_per_game
            )
            for venue, runs in self.venue_runs.items()
            if self.venue_games.get(venue, 0) > 0 and league_total_runs_per_game > 0
        }
        total_first5 = sum(state.first5_runs_allowed for state in self.pitchers.values())
        pitcher_starts = sum(state.starts for state in self.pitchers.values())
        total_late = sum(state.late_runs_allowed for state in self.bullpens.values())
        bullpen_games = sum(state.games for state in self.bullpens.values())
        home_rate = self.home_wins / self.completed_games if self.completed_games else DEFAULT_HOME_WIN_RATE
        return SeasonSummary(
            teams=teams,
            pitcher_first5_rates=pitcher_rates,
            bullpen_late_rates=bullpen_rates,
            league_runs_per_team_game=total_runs / team_games if team_games else DEFAULT_LEAGUE_RUNS,
            league_first5_runs_per_team_game=(
                total_first5 / pitcher_starts if pitcher_starts else DEFAULT_FIRST5_RUNS
            ),
            league_late_runs_per_team_game=(
                total_late / bullpen_games if bullpen_games else DEFAULT_LATE_RUNS
            ),
            home_win_rate=min(max(home_rate, 0.50), 0.58),
            base_runs_teams=base_runs_teams,
            league_base_runs_per_team_game=(
                total_base_runs / base_runs_games
                if base_runs_games else DEFAULT_LEAGUE_RUNS
            ),
            pitcher_components=component_priors,
            team_relievers={team: tuple(sorted(keys)) for team, keys in self.team_relievers.items()},
            park_run_factors=park_factors,
            league_component_score=numerator / total_bf if total_bf else 0.0,
            league_k_rate=total_k / total_bf if total_bf else 0.22,
            league_bb_hbp_rate=total_bb_hbp / total_bf if total_bf else 0.09,
            league_hr_rate=total_hr / total_bf if total_bf else 0.03,
            league_starter_innings=(starter_outs / 3.0 / starts if starts else 5.2),
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
                        away_pitcher_id=game.away_probable_pitcher_id,
                        home_pitcher_id=game.home_probable_pitcher_id,
                        away_pitcher_name=game.away_probable_pitcher,
                        home_pitcher_name=game.home_probable_pitcher,
                        venue=game.venue,
                    ),
                    home_win=1.0 if game.home_score > game.away_score else 0.0,
                    starter_covered=engine.starter_covered(game),
                    component_starter_covered=engine.component_starter_covered(game),
                    bullpen_covered=engine.bullpen_covered(game),
                )
            )
        engine.update_game(game)
    return rows, engine


def _brier(probabilities: list[float], rows: list[FeatureRow]) -> float | None:
    if not rows:
        return None
    return sum(
        (probability - row.home_win) ** 2
        for probability, row in zip(probabilities, rows, strict=True)
    ) / len(rows)


def _fit_platt(raw: list[float], rows: list[FeatureRow]) -> ProbabilityCalibrator:
    if len(rows) < 50:
        return ProbabilityCalibrator()
    x = np.asarray([logit(value) for value in raw], dtype=float)
    y = np.asarray([row.home_win for row in rows], dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.asarray([0.0, 1.0], dtype=float)
    penalty = np.diag([0.1, 1.0])
    for _ in range(60):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (probability - y) + penalty @ (coefficients - np.asarray([0.0, 1.0]))
        hessian = design.T @ (design * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return ProbabilityCalibrator(
        method="platt",
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
    )


def _fit_isotonic(raw: list[float], rows: list[FeatureRow]) -> ProbabilityCalibrator:
    # Isotonic is flexible and can overfit small samples, so require roughly one full MLB season.
    if len(rows) < 1000:
        return ProbabilityCalibrator()
    y = np.asarray([row.home_win for row in rows], dtype=float)
    model = IsotonicRegression(
        y_min=0.08,
        y_max=0.92,
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(np.asarray(raw, dtype=float), y)
    return ProbabilityCalibrator(
        method="isotonic",
        x_thresholds=tuple(float(value) for value in model.X_thresholds_),
        y_thresholds=tuple(float(value) for value in model.y_thresholds_),
    )


def _calibration_candidates(
    raw: list[float],
    validation_rows: list[FeatureRow],
) -> dict[str, ProbabilityCalibrator]:
    candidates = {
        "identity": ProbabilityCalibrator(),
        "shrink10": ProbabilityCalibrator(method="shrink", shrinkage=0.10),
        "shrink20": ProbabilityCalibrator(method="shrink", shrinkage=0.20),
        "shrink30": ProbabilityCalibrator(method="shrink", shrinkage=0.30),
        "shrink40": ProbabilityCalibrator(method="shrink", shrinkage=0.40),
        "platt": _fit_platt(raw, validation_rows),
    }
    isotonic = _fit_isotonic(raw, validation_rows)
    if isotonic.method == "isotonic":
        candidates["isotonic"] = isotonic
    return candidates


def _choose_calibrator(
    model: RegularizedLogisticModel,
    validation_rows: list[FeatureRow],
) -> tuple[ProbabilityCalibrator, float | None, dict[str, ProbabilityCalibrator], dict[str, float]]:
    if not validation_rows:
        fallback = ProbabilityCalibrator(method="shrink", shrinkage=0.20)
        return fallback, None, {"shrink20": fallback}, {}
    raw = [model.predict(row.features) for row in validation_rows]
    candidates = _calibration_candidates(raw, validation_rows)
    scores: dict[str, float] = {}
    for name, calibrator in candidates.items():
        score = _brier([calibrator.transform(value) for value in raw], validation_rows)
        if score is not None:
            scores[name] = score
    if not scores:
        fallback = ProbabilityCalibrator(method="shrink", shrinkage=0.20)
        return fallback, None, {"shrink20": fallback}, {}
    best_name = min(scores, key=scores.get)
    return candidates[best_name], scores[best_name], candidates, scores


def _coverage(rows: list[FeatureRow], attribute: str) -> float:
    return sum(1 for row in rows if getattr(row, attribute)) / len(rows) if rows else 0.0


def train_optimized_artifact(
    training_rows: list[FeatureRow],
    validation_rows: list[FeatureRow],
) -> OptimizedModelArtifact:
    variants: dict[str, ModelVariant] = {}
    for key, indices in FEATURE_VARIANTS.items():
        model = RegularizedLogisticModel.fit(training_rows, feature_indices=indices, l2=12.0)
        calibrator, validation_brier, candidates, scores = _choose_calibrator(
            model, validation_rows
        )
        variants[key] = ModelVariant(
            key=key.casefold(),
            label=key,
            model=model,
            calibrator=calibrator,
            validation_brier=validation_brier,
            calibration_candidates=candidates,
            calibration_scores=scores,
        )
    return OptimizedModelArtifact(
        variants=variants,
        training_rows=len(training_rows),
        validation_rows=len(validation_rows),
        training_starter_coverage=_coverage(training_rows, "starter_covered"),
        validation_starter_coverage=_coverage(validation_rows, "starter_covered"),
        training_component_coverage=_coverage(training_rows, "component_starter_covered"),
        validation_component_coverage=_coverage(validation_rows, "component_starter_covered"),
        training_bullpen_coverage=_coverage(training_rows, "bullpen_covered"),
        validation_bullpen_coverage=_coverage(validation_rows, "bullpen_covered"),
    )


def _multifold_calibration_choice(
    artifacts: list[OptimizedModelArtifact],
    variant_key: str,
) -> tuple[str, float] | None:
    variants = [artifact.variants.get(variant_key) for artifact in artifacts]
    if any(variant is None for variant in variants):
        return None
    resolved = [variant for variant in variants if variant is not None]
    common_methods = set(resolved[0].calibration_scores)
    for variant in resolved[1:]:
        common_methods &= set(variant.calibration_scores)
    if not common_methods:
        return None
    means = {
        method: float(np.mean([variant.calibration_scores[method] for variant in resolved]))
        for method in common_methods
    }
    simple = {name: score for name, score in means.items() if name != "isotonic"}
    if not simple:
        best_name = min(means, key=means.get)
        return best_name, means[best_name]
    best_simple = min(simple, key=simple.get)
    chosen = best_simple
    if "isotonic" in means:
        isotonic_fold_scores = [variant.calibration_scores["isotonic"] for variant in resolved]
        simple_fold_scores = [variant.calibration_scores[best_simple] for variant in resolved]
        average_gain = means[best_simple] - means["isotonic"]
        worst_fold_damage = max(iso - simple_score for iso, simple_score in zip(
            isotonic_fold_scores, simple_fold_scores, strict=True
        ))
        # Flexible calibration is promoted only when it wins robustly across prior folds.
        if average_gain >= 0.0007 and worst_fold_damage <= 0.0003:
            chosen = "isotonic"
    return chosen, means[chosen]


def apply_multifold_selection(
    final_artifact: OptimizedModelArtifact,
    earlier_artifacts: list[OptimizedModelArtifact],
    *,
    fold_years: tuple[int, ...],
) -> OptimizedModelArtifact:
    artifacts = [*earlier_artifacts, final_artifact]
    updated: dict[str, ModelVariant] = {}
    selection_scores: dict[str, float] = {}
    for key, variant in final_artifact.variants.items():
        choice = _multifold_calibration_choice(artifacts, key)
        if choice is None:
            updated[key] = variant
            if variant.validation_brier is not None:
                selection_scores[key] = float(variant.validation_brier)
            continue
        method, mean_brier = choice
        calibrator = variant.calibration_candidates.get(method, variant.calibrator)
        updated[key] = replace(
            variant,
            calibrator=calibrator,
            validation_brier=mean_brier,
        )
        selection_scores[key] = mean_brier

    eligible = [key for key in LIVE_CHAMPION_CANDIDATES if key in selection_scores]
    champion_key = "ComponentStarter"
    if eligible:
        best = min(selection_scores[key] for key in eligible)
        # Complexity guard: keep the earliest/simplest candidate within two ten-thousandths.
        champion_key = next(key for key in eligible if selection_scores[key] <= best + 0.0002)
    return replace(
        final_artifact,
        variants=updated,
        champion_key=champion_key,
        selection_scores=selection_scores,
        selection_folds=fold_years,
    )


def prepare_optimized_model_multifold(
    *,
    earlier_seed_games: list[MlbGameState],
    earlier_training_games: list[MlbGameState],
    earlier_validation_games: list[MlbGameState],
    seed_games: list[MlbGameState],
    training_games: list[MlbGameState],
    validation_games: list[MlbGameState],
    min_games: int,
    fold_years: tuple[int, ...],
) -> tuple[OptimizedModelArtifact, SeasonSummary, dict[str, float]]:
    earlier_artifact, _, _ = prepare_optimized_model(
        seed_games=earlier_seed_games,
        training_games=earlier_training_games,
        validation_games=earlier_validation_games,
        min_games=min_games,
    )
    final_artifact, prior_summary, prior_elo = prepare_optimized_model(
        seed_games=seed_games,
        training_games=training_games,
        validation_games=validation_games,
        min_games=min_games,
    )
    selected = apply_multifold_selection(
        final_artifact, [earlier_artifact], fold_years=fold_years
    )
    return selected, prior_summary, prior_elo


def prepare_optimized_model(
    *,
    seed_games: list[MlbGameState],
    training_games: list[MlbGameState],
    validation_games: list[MlbGameState],
    min_games: int,
) -> tuple[OptimizedModelArtifact, SeasonSummary, dict[str, float]]:
    _, seed_engine = build_season_rows(
        seed_games, prior_summary=None, prior_elo=None, min_games=min_games
    )
    training_rows, training_engine = build_season_rows(
        training_games,
        prior_summary=seed_engine.summary(),
        prior_elo=None,
        min_games=min_games,
    )
    validation_rows, validation_engine = build_season_rows(
        validation_games,
        prior_summary=training_engine.summary(),
        prior_elo=None,
        min_games=min_games,
    )
    return (
        train_optimized_artifact(training_rows, validation_rows),
        validation_engine.summary(),
        {},
    )


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
