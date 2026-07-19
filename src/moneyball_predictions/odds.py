"""Odds conversion, de-vigging, expected value, and bankroll-sizing helpers."""

from __future__ import annotations


def decimal_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def devig_two_way_decimal(decimal_odds_a: float, decimal_odds_b: float) -> tuple[float, float]:
    """Normalize a two-way sportsbook market so the two probabilities sum to one."""
    raw_a = decimal_to_implied_probability(decimal_odds_a)
    raw_b = decimal_to_implied_probability(decimal_odds_b)
    overround = raw_a + raw_b
    return raw_a / overround, raw_b / overround


def expected_value(model_probability: float, decimal_odds: float, stake: float) -> float:
    """Return expected profit, in dollars, for a conventional decimal-odds wager."""
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1 inclusive")
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1.0")
    if stake < 0:
        raise ValueError("stake cannot be negative")

    profit_if_win = stake * (decimal_odds - 1.0)
    loss_if_lose = stake
    return (model_probability * profit_if_win) - ((1.0 - model_probability) * loss_if_lose)


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    """Return full Kelly fraction, floored at zero when the wager has no positive edge."""
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1 inclusive")
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1.0")

    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    return max(0.0, ((model_probability * b) - q) / b)
