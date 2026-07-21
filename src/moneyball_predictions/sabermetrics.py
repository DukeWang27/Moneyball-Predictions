"""Point-in-time sabermetric feature primitives.

The functions in this module are deliberately pure and deterministic.  They do not
fetch data or mutate model state.  That makes them safe to reuse in live prediction,
walk-forward backtests, and reproducibility tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import exp, log
from typing import Mapping, Sequence

LINEUP_EVENTS = ("ubb", "hbp", "single", "double", "triple", "home_run")


@dataclass(frozen=True)
class BatterEventProjection:
    """Pregame event probabilities for one hitter against one pitcher hand."""

    player_id: int
    batting_slot: int
    expected_pa: float
    event_rates: Mapping[str, float]

    def validate(self) -> None:
        if not 1 <= self.batting_slot <= 9:
            raise ValueError("batting_slot must be between 1 and 9")
        if self.expected_pa <= 0:
            raise ValueError("expected_pa must be positive")
        total = 0.0
        for event in LINEUP_EVENTS:
            value = float(self.event_rates.get(event, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid {event} probability: {value}")
            total += value
        if total > 1.0 + 1e-9:
            raise ValueError("Lineup event probabilities exceed 1.0")


@dataclass(frozen=True)
class LineupProjection:
    xwoba: float
    linear_weight_runs_per_pa: float
    expected_plate_appearances: float


@dataclass(frozen=True)
class BattingLine:
    at_bats: float
    hits: float
    doubles: float
    triples: float
    home_runs: float
    walks: float
    intentional_walks: float
    hit_by_pitch: float

    @property
    def singles(self) -> float:
        return max(0.0, self.hits - self.doubles - self.triples - self.home_runs)

    @property
    def total_bases(self) -> float:
        return (
            self.singles
            + 2.0 * self.doubles
            + 3.0 * self.triples
            + 4.0 * self.home_runs
        )


@dataclass(frozen=True)
class PitchingLine:
    outs_recorded: int
    strikeouts: int
    walks: int
    hit_batters: int
    home_runs: int
    fly_balls: int = 0

    @property
    def innings(self) -> float:
        return self.outs_recorded / 3.0


@dataclass(frozen=True)
class ReliefAppearance:
    reliever_id: int
    ended_at: datetime
    pitches: int
    mean_leverage_index: float


def shrink_rate(
    successes: float,
    trials: float,
    *,
    prior_rate: float,
    prior_strength: float,
) -> float:
    """Return a beta-binomial style posterior mean for an event rate."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid successes/trials")
    if not 0.0 <= prior_rate <= 1.0:
        raise ValueError("prior_rate must be in [0, 1]")
    if prior_strength < 0:
        raise ValueError("prior_strength cannot be negative")
    denominator = trials + prior_strength
    if denominator <= 0:
        return prior_rate
    return (successes + prior_strength * prior_rate) / denominator


def project_lineup(
    batters: Sequence[BatterEventProjection],
    *,
    woba_weights: Mapping[str, float],
    run_values: Mapping[str, float],
) -> LineupProjection:
    """Combine nine handedness-specific batter projections using expected PA weights."""
    if len(batters) != 9:
        raise ValueError("A confirmed MLB lineup must contain exactly 9 batters")
    slots = {batter.batting_slot for batter in batters}
    if slots != set(range(1, 10)):
        raise ValueError("Lineup must contain batting slots 1 through 9 exactly once")

    total_pa = 0.0
    total_woba = 0.0
    total_runs = 0.0
    for batter in batters:
        batter.validate()
        total_pa += batter.expected_pa
        for event in LINEUP_EVENTS:
            probability = float(batter.event_rates.get(event, 0.0))
            total_woba += batter.expected_pa * probability * float(woba_weights[event])
            total_runs += batter.expected_pa * probability * float(run_values[event])

    if total_pa <= 0:
        raise ValueError("Lineup expected PA must be positive")
    return LineupProjection(
        xwoba=total_woba / total_pa,
        linear_weight_runs_per_pa=total_runs / total_pa,
        expected_plate_appearances=total_pa,
    )


def base_runs_smyth_v1(line: BattingLine) -> float:
    """Return a versioned Smyth-style Base Runs estimate.

    Versioning matters because several Base Runs coefficient sets exist.  This exact
    formula should not be changed silently after predictions have been stored.
    """
    if min(
        line.at_bats,
        line.hits,
        line.doubles,
        line.triples,
        line.home_runs,
        line.walks,
        line.intentional_walks,
        line.hit_by_pitch,
    ) < 0:
        raise ValueError("Batting totals cannot be negative")

    unintentional_walks = max(0.0, line.walks - line.intentional_walks)
    a = line.hits + unintentional_walks + line.hit_by_pitch - line.home_runs
    b = (
        1.4 * line.total_bases
        - 0.6 * line.hits
        - 3.0 * line.home_runs
        + 0.1 * (unintentional_walks + line.hit_by_pitch)
    ) * 1.02
    c = max(0.0, line.at_bats - line.hits)
    d = line.home_runs
    advancement = max(0.0, b)
    if advancement + c <= 0:
        return max(0.0, d)
    return max(0.0, a * advancement / (advancement + c) + d)


def pythagenpat_win_pct(
    runs_scored: float,
    runs_allowed: float,
    games: float,
    *,
    gamma: float = 0.287,
) -> float:
    """Return Pythagenpat expected win percentage with a dynamic run exponent."""
    if games <= 0:
        raise ValueError("games must be positive")
    if runs_scored < 0 or runs_allowed < 0:
        raise ValueError("runs cannot be negative")
    epsilon = 1e-12
    scored = max(runs_scored, epsilon)
    allowed = max(runs_allowed, epsilon)
    runs_per_game = (scored + allowed) / games
    exponent = max(runs_per_game, epsilon) ** gamma
    scored_term = scored**exponent
    allowed_term = allowed**exponent
    return scored_term / (scored_term + allowed_term)


def calculate_fip_constant(
    *,
    league_era: float,
    league_home_runs: float,
    league_walks: float,
    league_hit_batters: float,
    league_strikeouts: float,
    league_outs_recorded: float,
) -> float:
    if league_outs_recorded <= 0:
        raise ValueError("league_outs_recorded must be positive")
    innings = league_outs_recorded / 3.0
    component = (
        13.0 * league_home_runs
        + 3.0 * (league_walks + league_hit_batters)
        - 2.0 * league_strikeouts
    ) / innings
    return league_era - component


def calculate_fip(line: PitchingLine, *, fip_constant: float) -> float:
    if line.outs_recorded <= 0:
        raise ValueError("Pitcher must have recorded at least one out")
    numerator = (
        13.0 * line.home_runs
        + 3.0 * (line.walks + line.hit_batters)
        - 2.0 * line.strikeouts
    )
    return numerator / line.innings + fip_constant


def calculate_xfip(
    line: PitchingLine,
    *,
    league_hr_per_fly_ball: float,
    fip_constant: float,
) -> float:
    if line.outs_recorded <= 0:
        raise ValueError("Pitcher must have recorded at least one out")
    if not 0.0 <= league_hr_per_fly_ball <= 1.0:
        raise ValueError("league_hr_per_fly_ball must be in [0, 1]")
    expected_home_runs = line.fly_balls * league_hr_per_fly_ball
    numerator = (
        13.0 * expected_home_runs
        + 3.0 * (line.walks + line.hit_batters)
        - 2.0 * line.strikeouts
    )
    return numerator / line.innings + fip_constant


def expected_pitching_runs_allowed(
    *,
    starter_runs_per_nine: float,
    expected_starter_innings: float,
    bullpen_runs_per_nine: float,
) -> float:
    """Blend starter and bullpen quality by expected innings rather than independently."""
    if starter_runs_per_nine < 0 or bullpen_runs_per_nine < 0:
        raise ValueError("Run rates cannot be negative")
    starter_innings = min(max(expected_starter_innings, 0.0), 9.0)
    bullpen_innings = 9.0 - starter_innings
    return (
        starter_runs_per_nine * starter_innings / 9.0
        + bullpen_runs_per_nine * bullpen_innings / 9.0
    )


def leverage_adjusted_bullpen_fatigue(
    appearances: Sequence[ReliefAppearance],
    *,
    as_of: datetime,
    lookback_hours: float = 72.0,
    half_life_hours: float = 36.0,
    leverage_alpha: float = 0.40,
    leverage_cap: float = 3.0,
) -> dict[int, float]:
    """Return per-reliever fatigue from workload, recency, and game-state leverage."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if lookback_hours <= 0 or half_life_hours <= 0:
        raise ValueError("lookback_hours and half_life_hours must be positive")
    decay_rate = log(2.0) / half_life_hours
    fatigue: dict[int, float] = defaultdict(float)
    appearance_dates: dict[int, set[date]] = defaultdict(set)

    as_of_utc = as_of.astimezone(UTC)
    for appearance in appearances:
        if appearance.ended_at.tzinfo is None:
            raise ValueError("appearance timestamps must be timezone-aware")
        if appearance.pitches < 0:
            raise ValueError("pitches cannot be negative")
        age_hours = (
            as_of_utc - appearance.ended_at.astimezone(UTC)
        ).total_seconds() / 3600.0
        if age_hours < 0 or age_hours > lookback_hours:
            continue
        recency = exp(-decay_rate * age_hours)
        leverage = min(max(appearance.mean_leverage_index, 0.0), leverage_cap)
        leverage_multiplier = 1.0 + leverage_alpha * max(0.0, leverage - 1.0)
        fatigue[appearance.reliever_id] += appearance.pitches * recency * leverage_multiplier
        appearance_dates[appearance.reliever_id].add(appearance.ended_at.date())

    for reliever_id, dates in appearance_dates.items():
        if len(dates) >= 2:
            fatigue[reliever_id] += 8.0
        if len(dates) >= 3:
            fatigue[reliever_id] += 12.0
    return dict(fatigue)


def availability_from_fatigue(
    fatigue_score: float,
    *,
    midpoint: float = 45.0,
    scale: float = 10.0,
) -> float:
    """Map a fatigue score to an estimated availability weight in [0, 1]."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    value = 1.0 / (1.0 + exp((fatigue_score - midpoint) / scale))
    return min(max(value, 0.0), 1.0)
