"""Decision layer for nested pitcher strikeout contracts.

Every threshold for one pitcher/game is evaluated from one Poisson distribution.
Only one contract can become BEST_BET; the rest remain visible alternatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from scipy.stats import poisson

from .execution import quote_buy_by_dollars

Side = Literal["YES", "NO"]
Decision = Literal["BEST_BET", "ALTERNATIVE", "LEAN", "PASS", "WARNING"]


@dataclass(frozen=True)
class MarketContract:
    market_id: str
    threshold: int
    side: Side
    token_id: str
    asks: tuple[tuple[float, float], ...]
    polymarket_url: str


@dataclass(frozen=True)
class ContractEvaluation:
    market_id: str
    threshold: int
    side: Side
    token_id: str
    raw_probability: float
    conservative_probability: float
    executable_price: float | None
    edge: float | None
    expected_roi: float | None
    full_kelly_fraction: float
    selected_kelly_fraction: float
    proposed_stake: float
    expected_log_growth: float | None
    fully_fillable: bool
    levels_consumed: int
    decision: Decision
    polymarket_url: str


@dataclass(frozen=True)
class FamilyDecision:
    evaluations: tuple[ContractEvaluation, ...]
    best_market_id: str | None
    best_side: Side | None
    best_threshold: int | None
    warning_code: str | None
    warning_message: str | None


def probability_at_least(expected_strikeouts: float, threshold: int) -> float:
    if expected_strikeouts < 0:
        raise ValueError("expected_strikeouts cannot be negative")
    if threshold < 1:
        raise ValueError("threshold must be positive")
    return float(poisson.sf(threshold - 1, mu=expected_strikeouts))


def shrink_probability(probability: float, shrinkage: float = 0.30) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    return 0.5 + (1.0 - shrinkage) * (probability - 0.5)


def expected_roi(probability: float, price: float) -> float:
    if not 0 < price < 1:
        raise ValueError("price must be between zero and one")
    return probability / price - 1.0


def full_kelly_fraction(probability: float, price: float) -> float:
    if not 0 < price < 1:
        raise ValueError("price must be between zero and one")
    return max(0.0, (probability - price) / (1.0 - price))


def expected_log_growth(
    *, probability: float, price: float, bankroll_fraction: float
) -> float:
    if bankroll_fraction <= 0:
        return 0.0
    if bankroll_fraction >= 1:
        return float("-inf")
    win_return_multiple = (1.0 / price) - 1.0
    wealth_if_win = 1.0 + bankroll_fraction * win_return_multiple
    wealth_if_loss = 1.0 - bankroll_fraction
    if wealth_if_win <= 0 or wealth_if_loss <= 0:
        return float("-inf")
    return probability * math.log(wealth_if_win) + (1.0 - probability) * math.log(
        wealth_if_loss
    )


def validate_probability_ladder(probabilities: dict[int, float], tolerance: float = 1e-9) -> bool:
    previous = 1.0
    for threshold in sorted(probabilities):
        probability = probabilities[threshold]
        if probability > previous + tolerance:
            return False
        previous = probability
    return True


def market_ladder_warning(
    yes_prices: dict[int, float],
    no_prices: dict[int, float],
    *,
    tolerance: float = 0.03,
) -> tuple[str | None, str | None]:
    thresholds = sorted(set(yes_prices) | set(no_prices))
    for lower, higher in zip(thresholds, thresholds[1:], strict=False):
        if lower in yes_prices and higher in yes_prices:
            if yes_prices[higher] > yes_prices[lower] + tolerance:
                return (
                    "INVERTED_YES_PRICES",
                    f"{higher}+ YES is priced above {lower}+ YES beyond the tolerance.",
                )
        if lower in no_prices and higher in no_prices:
            if no_prices[higher] + tolerance < no_prices[lower]:
                return (
                    "INVERTED_NO_PRICES",
                    f"{higher}+ NO is priced below {lower}+ NO beyond the tolerance.",
                )
    return None, None


def _raw_probability(expected_strikeouts: float, threshold: int, side: Side) -> float:
    yes_probability = probability_at_least(expected_strikeouts, threshold)
    return yes_probability if side == "YES" else 1.0 - yes_probability


def _quote_contract(
    contract: MarketContract,
    *,
    probability: float,
    bankroll: float,
    kelly_multiplier: float,
    max_bankroll_fraction: float,
) -> ContractEvaluation:
    max_stake = max(0.0, bankroll * max_bankroll_fraction)
    if max_stake < 0.01 or not contract.asks:
        return ContractEvaluation(
            market_id=contract.market_id,
            threshold=contract.threshold,
            side=contract.side,
            token_id=contract.token_id,
            raw_probability=probability,
            conservative_probability=probability,
            executable_price=None,
            edge=None,
            expected_roi=None,
            full_kelly_fraction=0.0,
            selected_kelly_fraction=0.0,
            proposed_stake=0.0,
            expected_log_growth=None,
            fully_fillable=False,
            levels_consumed=0,
            decision="PASS",
            polymarket_url=contract.polymarket_url,
        )

    # First pass uses the portfolio risk cap.  The second pass re-quotes at the
    # actual Kelly stake so slippage and fillability match the proposed bet.
    preliminary = quote_buy_by_dollars(
        {"asks": [{"price": price, "size": size} for price, size in contract.asks]},
        max_stake,
    )
    if preliminary.acquired_shares <= 0:
        price = None
    else:
        price = float(preliminary.filled_dollars / preliminary.acquired_shares)
    if price is None or not 0 < price < 1:
        return ContractEvaluation(
            market_id=contract.market_id,
            threshold=contract.threshold,
            side=contract.side,
            token_id=contract.token_id,
            raw_probability=probability,
            conservative_probability=probability,
            executable_price=None,
            edge=None,
            expected_roi=None,
            full_kelly_fraction=0.0,
            selected_kelly_fraction=0.0,
            proposed_stake=0.0,
            expected_log_growth=None,
            fully_fillable=False,
            levels_consumed=0,
            decision="PASS",
            polymarket_url=contract.polymarket_url,
        )

    full_kelly = full_kelly_fraction(probability, price)
    selected_fraction = min(kelly_multiplier * full_kelly, max_bankroll_fraction)
    proposed_stake = min(bankroll * selected_fraction, max_stake)
    if proposed_stake < 0.01:
        proposed_stake = 0.0
        final_quote = preliminary
    else:
        final_quote = quote_buy_by_dollars(
            {"asks": [{"price": p, "size": size} for p, size in contract.asks]},
            proposed_stake,
        )
    if final_quote.acquired_shares <= 0:
        final_price = None
    else:
        final_price = float(final_quote.filled_dollars / final_quote.acquired_shares)
    if final_price is None or not 0 < final_price < 1:
        edge = roi = log_growth = None
        full_kelly = selected_fraction = proposed_stake = 0.0
    else:
        edge = probability - final_price
        roi = expected_roi(probability, final_price)
        full_kelly = full_kelly_fraction(probability, final_price)
        selected_fraction = min(kelly_multiplier * full_kelly, max_bankroll_fraction)
        proposed_stake = min(bankroll * selected_fraction, max_stake)
        log_growth = expected_log_growth(
            probability=probability,
            price=final_price,
            bankroll_fraction=(proposed_stake / bankroll if bankroll > 0 else 0.0),
        )
    return ContractEvaluation(
        market_id=contract.market_id,
        threshold=contract.threshold,
        side=contract.side,
        token_id=contract.token_id,
        raw_probability=probability,
        conservative_probability=probability,
        executable_price=final_price,
        edge=edge,
        expected_roi=roi,
        full_kelly_fraction=full_kelly,
        selected_kelly_fraction=selected_fraction,
        proposed_stake=proposed_stake,
        expected_log_growth=log_growth,
        fully_fillable=bool(final_quote.fully_fillable),
        levels_consumed=final_quote.levels_consumed,
        decision="PASS",
        polymarket_url=contract.polymarket_url,
    )


def choose_best_contract(
    contracts: list[MarketContract],
    *,
    expected_strikeouts: float,
    bankroll: float,
    shrinkage: float = 0.30,
    kelly_multiplier: float = 0.25,
    max_bankroll_fraction: float = 0.01,
    minimum_edge: float = 0.05,
    minimum_expected_roi: float = 0.05,
    require_confirmed_lineup: bool = True,
    lineup_confirmed: bool = True,
) -> FamilyDecision:
    if bankroll < 0:
        raise ValueError("bankroll cannot be negative")
    yes_probabilities = {
        threshold: probability_at_least(expected_strikeouts, threshold)
        for threshold in {contract.threshold for contract in contracts}
    }
    if not validate_probability_ladder(yes_probabilities):
        warning = "INVERTED_PROBABILITIES"
        message = "Poisson probabilities increased at a higher strikeout threshold."
    else:
        warning = message = None

    evaluations: list[ContractEvaluation] = []
    for contract in contracts:
        raw = _raw_probability(expected_strikeouts, contract.threshold, contract.side)
        conservative = shrink_probability(raw, shrinkage)
        evaluation = _quote_contract(
            contract,
            probability=conservative,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
            max_bankroll_fraction=max_bankroll_fraction,
        )
        evaluations.append(
            replace(
                evaluation,
                raw_probability=raw,
                conservative_probability=conservative,
            )
        )

    yes_prices = {
        evaluation.threshold: evaluation.executable_price
        for evaluation in evaluations
        if evaluation.side == "YES" and evaluation.executable_price is not None
    }
    no_prices = {
        evaluation.threshold: evaluation.executable_price
        for evaluation in evaluations
        if evaluation.side == "NO" and evaluation.executable_price is not None
    }
    market_warning, market_message = market_ladder_warning(yes_prices, no_prices)
    if warning is None and market_warning:
        warning, message = market_warning, market_message
    if warning is None and require_confirmed_lineup and not lineup_confirmed:
        warning = "LINEUP_NOT_CONFIRMED"
        message = "The opponent starting lineup is not confirmed; betting is disabled."

    if warning:
        evaluations = [replace(item, decision="WARNING") for item in evaluations]
        return FamilyDecision(tuple(evaluations), None, None, None, warning, message)

    eligible = [
        item
        for item in evaluations
        if item.fully_fillable
        and item.executable_price is not None
        and item.edge is not None
        and item.expected_roi is not None
        and item.expected_log_growth is not None
        and item.proposed_stake >= 0.01
        and item.edge >= minimum_edge
        and item.expected_roi >= minimum_expected_roi
        and item.expected_log_growth > 0
    ]
    best = max(
        eligible,
        key=lambda item: (
            item.expected_log_growth or float("-inf"),
            item.expected_roi or float("-inf"),
            item.edge or float("-inf"),
            -item.threshold,
        ),
        default=None,
    )
    marked: list[ContractEvaluation] = []
    for item in evaluations:
        if best is not None and item.market_id == best.market_id and item.side == best.side:
            decision: Decision = "BEST_BET"
        elif item.edge is not None and item.expected_roi is not None and item.edge > 0:
            decision = "ALTERNATIVE" if item.fully_fillable else "LEAN"
        else:
            decision = "PASS"
        marked.append(replace(item, decision=decision))
    return FamilyDecision(
        evaluations=tuple(marked),
        best_market_id=best.market_id if best else None,
        best_side=best.side if best else None,
        best_threshold=best.threshold if best else None,
        warning_code=None,
        warning_message=None,
    )
