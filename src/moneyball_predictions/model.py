"""Core sabermetric probability functions used by live and historical models."""

from __future__ import annotations

import math


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")


def clip_probability(value: float, lower: float = 1e-6, upper: float = 1 - 1e-6) -> float:
    """Keep a probability away from exactly zero and one."""
    if lower <= 0 or upper >= 1 or lower >= upper:
        raise ValueError("probability bounds must satisfy 0 < lower < upper < 1")
    return min(max(value, lower), upper)


def logit(value: float) -> float:
    """Convert a probability into log-odds."""
    value = clip_probability(value)
    return math.log(value / (1.0 - value))


def logistic(value: float) -> float:
    """Convert log-odds into a probability without numeric overflow."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def dynamic_pythagorean_exponent(
    runs_scored: float,
    runs_allowed: float,
    games_played: int,
) -> float:
    """Return the dynamic Pythagorean exponent from per-game run environment."""
    if runs_scored < 0 or runs_allowed < 0:
        raise ValueError("runs_scored and runs_allowed cannot be negative")
    if games_played <= 0:
        raise ValueError("games_played must be positive")

    runs_per_game_environment = (runs_scored + runs_allowed) / games_played
    if runs_per_game_environment == 0:
        raise ValueError("at least one run must have been scored or allowed")
    return runs_per_game_environment**0.287


def pythagorean_expectation(
    runs_scored: float,
    runs_allowed: float,
    games_played: int,
) -> float:
    """Estimate team strength using a dynamic-exponent Pythagorean expectation."""
    exponent = dynamic_pythagorean_exponent(runs_scored, runs_allowed, games_played)
    scored_component = runs_scored**exponent
    allowed_component = runs_allowed**exponent
    denominator = scored_component + allowed_component

    if denominator == 0:
        raise ValueError("cannot compute expectation when both run totals are zero")
    return scored_component / denominator


def regressed_pythagorean_expectation(
    *,
    runs_scored: float,
    runs_allowed: float,
    games_played: int,
    prior_runs_scored_per_game: float,
    prior_runs_allowed_per_game: float,
    league_runs_per_team_game: float,
    pseudo_games: float = 25.0,
    prior_team_weight: float = 0.70,
) -> float:
    """Estimate team strength after shrinking current rates toward known priors.

    The current season is combined with a fixed number of prior-season pseudo-games.
    Prior team rates are themselves partially shrunk toward the prior league run
    environment. All arguments are expected to be information known before the game.
    """
    if games_played < 0:
        raise ValueError("games_played cannot be negative")
    if runs_scored < 0 or runs_allowed < 0:
        raise ValueError("run totals cannot be negative")
    if pseudo_games <= 0:
        raise ValueError("pseudo_games must be positive")
    if not 0 <= prior_team_weight <= 1:
        raise ValueError("prior_team_weight must be between 0 and 1")
    if min(
        prior_runs_scored_per_game,
        prior_runs_allowed_per_game,
        league_runs_per_team_game,
    ) <= 0:
        raise ValueError("prior run rates must be positive")

    prior_rs = (
        prior_team_weight * prior_runs_scored_per_game
        + (1.0 - prior_team_weight) * league_runs_per_team_game
    )
    prior_ra = (
        prior_team_weight * prior_runs_allowed_per_game
        + (1.0 - prior_team_weight) * league_runs_per_team_game
    )
    adjusted_games = games_played + pseudo_games
    adjusted_runs_scored = runs_scored + prior_rs * pseudo_games
    adjusted_runs_allowed = runs_allowed + prior_ra * pseudo_games
    return pythagorean_expectation(
        adjusted_runs_scored,
        adjusted_runs_allowed,
        max(1, round(adjusted_games)),
    )


def log5_probability(team_a_strength: float, team_b_strength: float) -> float:
    """Estimate the probability that team A beats team B using the Log5 method."""
    _validate_probability(team_a_strength, "team_a_strength")
    _validate_probability(team_b_strength, "team_b_strength")

    numerator = team_a_strength - (team_a_strength * team_b_strength)
    denominator = (
        team_a_strength + team_b_strength - (2 * team_a_strength * team_b_strength)
    )
    if denominator == 0:
        raise ValueError("Log5 is undefined for these two strengths")
    return numerator / denominator


def add_home_field_advantage(
    neutral_home_probability: float,
    equal_team_home_win_rate: float,
) -> float:
    """Apply a home-field intercept on the log-odds scale.

    When the teams are equal and neutral_home_probability is 0.5, the result equals
    equal_team_home_win_rate. This is preferable to simply adding percentage points.
    """
    _validate_probability(neutral_home_probability, "neutral_home_probability")
    _validate_probability(equal_team_home_win_rate, "equal_team_home_win_rate")
    if equal_team_home_win_rate in {0.0, 1.0}:
        raise ValueError("equal_team_home_win_rate must be strictly between 0 and 1")
    return logistic(logit(neutral_home_probability) + logit(equal_team_home_win_rate))
