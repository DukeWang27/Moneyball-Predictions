import pytest

from moneyball_predictions.prop_decision import (
    MarketContract,
    choose_best_contract,
    probability_at_least,
    validate_probability_ladder,
)


def contract(threshold: int, side: str, price: float) -> MarketContract:
    return MarketContract(
        market_id=f"m-{threshold}",
        threshold=threshold,
        side=side,
        token_id=f"t-{threshold}-{side}",
        asks=((price, 100.0),),
        polymarket_url="https://example.com",
    )


def test_poisson_probability_ladder_is_monotonic() -> None:
    ladder = {threshold: probability_at_least(3.8, threshold) for threshold in (2, 3, 4, 5)}
    assert validate_probability_ladder(ladder)
    assert ladder[2] >= ladder[3] >= ladder[4] >= ladder[5]


def test_selector_outputs_exactly_one_best_bet() -> None:
    contracts = [
        contract(2, "YES", 0.82), contract(2, "NO", 0.25),
        contract(3, "YES", 0.55), contract(3, "NO", 0.49),
        contract(4, "YES", 0.34), contract(4, "NO", 0.69),
    ]
    decision = choose_best_contract(
        contracts,
        expected_strikeouts=3.8,
        bankroll=100.0,
        shrinkage=0.30,
        lineup_confirmed=True,
    )
    best = [item for item in decision.evaluations if item.decision == "BEST_BET"]
    assert len(best) == 1
    assert decision.best_market_id == best[0].market_id
    assert best[0].proposed_stake <= 1.0


def test_inverted_market_ladder_disables_betting() -> None:
    contracts = [
        contract(2, "YES", 0.50), contract(2, "NO", 0.45),
        contract(3, "YES", 0.70), contract(3, "NO", 0.55),
    ]
    decision = choose_best_contract(
        contracts,
        expected_strikeouts=3.0,
        bankroll=100.0,
        lineup_confirmed=True,
    )
    assert decision.warning_code == "INVERTED_YES_PRICES"
    assert decision.best_market_id is None
    assert all(item.decision == "WARNING" for item in decision.evaluations)


def test_unconfirmed_lineup_disables_betting() -> None:
    decision = choose_best_contract(
        [contract(3, "YES", 0.30), contract(3, "NO", 0.75)],
        expected_strikeouts=4.0,
        bankroll=100.0,
        lineup_confirmed=False,
    )
    assert decision.warning_code == "LINEUP_NOT_CONFIRMED"
    assert decision.best_market_id is None
