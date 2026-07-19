"""Core, deterministic sabermetric formulas used by the MVP."""

from __future__ import annotations


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")


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
