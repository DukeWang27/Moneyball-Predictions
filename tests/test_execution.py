from decimal import Decimal

import pytest

from moneyball_predictions.execution import (
    build_passive_buy_plan,
    choose_passive_buy_price,
    quote_buy_by_dollars,
    slippage_adjusted_metrics,
)


def test_vwap_walks_multiple_ask_levels() -> None:
    quote = quote_buy_by_dollars(
        {
            "asks": [
                {"price": "0.55", "size": "100"},
                {"price": "0.50", "size": "20"},
                {"price": "0.52", "size": "30"},
            ]
        },
        "20",
    )
    assert quote.fully_fillable is True
    assert quote.levels_consumed == 2
    assert quote.vwap is not None
    assert Decimal("0.50") < quote.vwap < Decimal("0.52")
    metrics = slippage_adjusted_metrics(model_probability=0.60, quote=quote)
    assert metrics["edge"] > 0


def test_vwap_reports_unfilled_notional() -> None:
    quote = quote_buy_by_dollars(
        {"asks": [{"price": "0.50", "size": "10"}]},
        "20",
    )
    assert quote.fully_fillable is False
    assert quote.unfilled_dollars == Decimal("15.0")


def test_passive_price_never_crosses_ask() -> None:
    price = choose_passive_buy_price(
        fair_probability="0.62",
        best_bid="0.53",
        best_ask="0.56",
        tick_size="0.01",
        required_edge="0.05",
        adverse_selection_buffer="0.01",
    )
    assert price == Decimal("0.54")
    assert price < Decimal("0.56")


def test_passive_plan_rejects_when_edge_cannot_be_preserved() -> None:
    plan = build_passive_buy_plan(
        token_id="abc",
        fair_probability=0.55,
        desired_dollars=10,
        best_bid=0.53,
        best_ask=0.54,
        tick_size=0.01,
        required_edge=0.05,
        adverse_selection_buffer=0.01,
    )
    assert plan is None


def test_bad_spread_is_rejected() -> None:
    with pytest.raises(ValueError):
        choose_passive_buy_price(
            fair_probability=0.60,
            best_bid=0.55,
            best_ask=0.55,
            tick_size=0.01,
            required_edge=0.05,
            adverse_selection_buffer=0.01,
        )
