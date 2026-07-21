#!/usr/bin/env python3
"""Run deterministic v0.12 sabermetric and execution examples."""

from decimal import Decimal

from moneyball_predictions.execution import build_passive_buy_plan, quote_buy_by_dollars
from moneyball_predictions.sabermetrics import BattingLine, base_runs_smyth_v1, pythagenpat_win_pct
from moneyball_predictions.storage import database_summary


def main() -> None:
    offense = BattingLine(1000, 280, 55, 8, 45, 105, 5, 10)
    defense = BattingLine(1000, 230, 38, 4, 24, 75, 3, 6)
    scored = base_runs_smyth_v1(offense)
    allowed = base_runs_smyth_v1(defense)
    print(f"Base Runs scored: {scored:.2f}")
    print(f"Base Runs allowed: {allowed:.2f}")
    print(f"Pythagenpat strength: {pythagenpat_win_pct(scored, allowed, 100):.3f}")

    quote = quote_buy_by_dollars(
        {
            "asks": [
                {"price": "0.50", "size": "20"},
                {"price": "0.52", "size": "60"},
                {"price": "0.55", "size": "100"},
            ]
        },
        Decimal("50"),
    )
    print(f"$50 executable VWAP: {quote.vwap}")
    print(f"Fully fillable: {quote.fully_fillable}")

    plan = build_passive_buy_plan(
        token_id="example",
        fair_probability="0.62",
        desired_dollars="10",
        best_bid="0.53",
        best_ask="0.56",
        tick_size="0.01",
        required_edge="0.05",
        adverse_selection_buffer="0.01",
    )
    print(f"Dry-run passive plan: {plan}")
    print(f"Research database: {database_summary()}")


if __name__ == "__main__":
    main()
