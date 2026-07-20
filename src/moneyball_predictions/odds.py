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


def prediction_market_expected_value(
    model_probability: float,
    buy_price: float,
    stake: float,
) -> float:
    """Expected dollar profit when buying outcome shares at an executable ask price."""
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1 inclusive")
    if not 0.0 < buy_price < 1.0:
        raise ValueError("buy_price must be between 0 and 1 exclusive")
    if stake < 0:
        raise ValueError("stake cannot be negative")

    shares = stake / buy_price
    profit_if_win = shares - stake
    return (model_probability * profit_if_win) - ((1.0 - model_probability) * stake)


def prediction_market_kelly_fraction(model_probability: float, buy_price: float) -> float:
    """Return full Kelly fraction for a binary $1 prediction-market share.

    A share bought for ``buy_price`` pays $1 on a win and $0 on a loss. The
    equivalent decimal odds are ``1 / buy_price``. The simplified full-Kelly
    fraction is ``(model_probability - buy_price) / (1 - buy_price)``.
    """
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1 inclusive")
    if not 0.0 < buy_price < 1.0:
        raise ValueError("buy_price must be between 0 and 1 exclusive")

    return max(0.0, (model_probability - buy_price) / (1.0 - buy_price))


def prediction_market_kelly_stake(
    model_probability: float,
    buy_price: float,
    bankroll: float,
    multiplier: float = 0.25,
    top_ask_size: float | None = None,
    max_bankroll_fraction: float = 1.0,
) -> tuple[float, float, float, bool]:
    """Return a liquidity-aware fractional-Kelly stake.

    Returns ``(stake, full_kelly_fraction, applied_fraction, capped_by_depth)``.
    ``multiplier`` may be 0.25 for quarter Kelly, 0.5 for half Kelly, or 1.0 for
    full Kelly. When top-of-book size is known, the stake is capped so the
    displayed best ask remains executable for the whole paper order.
    """
    if bankroll < 0:
        raise ValueError("bankroll cannot be negative")
    if not 0.0 < multiplier <= 1.0:
        raise ValueError("multiplier must be greater than 0 and at most 1")
    if top_ask_size is not None and top_ask_size < 0:
        raise ValueError("top_ask_size cannot be negative")
    if not 0.0 < max_bankroll_fraction <= 1.0:
        raise ValueError("max_bankroll_fraction must be greater than 0 and at most 1")

    full_fraction = prediction_market_kelly_fraction(model_probability, buy_price)
    applied_fraction = min(full_fraction * multiplier, max_bankroll_fraction)
    raw_stake = bankroll * applied_fraction
    capped_by_depth = False

    if top_ask_size is not None:
        top_ask_notional = top_ask_size * buy_price
        if top_ask_notional < raw_stake:
            raw_stake = top_ask_notional
            capped_by_depth = True

    stake = min(bankroll, max(0.0, raw_stake))
    return round(stake, 2), full_fraction, applied_fraction, capped_by_depth
