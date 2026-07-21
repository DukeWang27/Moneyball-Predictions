"""Deterministic Polymarket execution research helpers.

This module never submits a real order.  It prices a desired notional against the full
ask book and builds a post-only order plan that a separately gated venue adapter can
later submit.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    size_shares: Decimal


@dataclass(frozen=True)
class ExecutionQuote:
    requested_dollars: Decimal
    filled_dollars: Decimal
    acquired_shares: Decimal
    vwap: Decimal | None
    worst_price: Decimal | None
    unfilled_dollars: Decimal
    fully_fillable: bool
    levels_consumed: int


@dataclass(frozen=True)
class PassiveOrderPlan:
    token_id: str
    side: str
    limit_price: Decimal
    size_shares: Decimal
    notional_dollars: Decimal
    fair_probability: Decimal
    required_edge: Decimal
    adverse_selection_buffer: Decimal
    best_bid: Decimal
    best_ask: Decimal
    tick_size: Decimal
    post_only: bool = True
    order_type: str = "GTC"
    dry_run: bool = True


def as_decimal(value: Any, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"Invalid {field}: {value!r}")
    return parsed


def parse_book_levels(raw_levels: Sequence[Any]) -> list[BookLevel]:
    levels: list[BookLevel] = []
    for raw in raw_levels:
        if isinstance(raw, Mapping):
            raw_price = raw.get("price")
            raw_size = raw.get("size")
        else:
            raw_price = getattr(raw, "price", None)
            raw_size = getattr(raw, "size", None)
        price = as_decimal(raw_price, "book price")
        size = as_decimal(raw_size, "book size")
        if Decimal("0") < price < Decimal("1") and size > 0:
            levels.append(BookLevel(price=price, size_shares=size))
    return levels


def quote_buy_by_dollars(
    order_book: Mapping[str, Any],
    desired_dollars: Decimal | float | str,
    *,
    maximum_price: Decimal | float | str | None = None,
) -> ExecutionQuote:
    """Walk asks from cheapest to most expensive for a desired dollar notional."""
    requested = as_decimal(desired_dollars, "desired_dollars")
    if requested <= 0:
        raise ValueError("desired_dollars must be positive")
    max_price = as_decimal(maximum_price, "maximum_price") if maximum_price is not None else None

    raw_asks = order_book.get("asks", [])
    if not isinstance(raw_asks, Sequence) or isinstance(raw_asks, (str, bytes)):
        raise ValueError("order_book asks must be a sequence")
    asks = sorted(parse_book_levels(raw_asks), key=lambda level: level.price)

    remaining = requested
    spent = Decimal("0")
    shares = Decimal("0")
    worst_price: Decimal | None = None
    levels_consumed = 0

    for level in asks:
        if remaining <= 0:
            break
        if max_price is not None and level.price > max_price:
            break
        level_capacity_dollars = level.price * level.size_shares
        dollars_at_level = min(remaining, level_capacity_dollars)
        shares_at_level = dollars_at_level / level.price
        spent += dollars_at_level
        shares += shares_at_level
        remaining -= dollars_at_level
        worst_price = level.price
        levels_consumed += 1

    vwap = spent / shares if shares > 0 else None
    tolerance = Decimal("0.000001")
    return ExecutionQuote(
        requested_dollars=requested,
        filled_dollars=spent,
        acquired_shares=shares,
        vwap=vwap,
        worst_price=worst_price,
        unfilled_dollars=max(Decimal("0"), remaining),
        fully_fillable=remaining <= tolerance,
        levels_consumed=levels_consumed,
    )


def slippage_adjusted_metrics(
    *,
    model_probability: float,
    quote: ExecutionQuote,
    fee_dollars: Decimal | float | str = Decimal("0"),
) -> dict[str, float | bool | None]:
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be in [0, 1]")
    if quote.acquired_shares <= 0:
        return {
            "vwap": None,
            "all_in_cost_per_share": None,
            "edge": None,
            "expected_profit": None,
            "expected_roi": None,
            "fully_fillable": False,
        }
    fee = as_decimal(fee_dollars, "fee_dollars")
    if fee < 0:
        raise ValueError("fee_dollars cannot be negative")
    total_cost = quote.filled_dollars + fee
    all_in_cost_per_share = total_cost / quote.acquired_shares
    expected_payout = Decimal(str(model_probability)) * quote.acquired_shares
    expected_profit = expected_payout - total_cost
    expected_roi = expected_profit / total_cost if total_cost > 0 else Decimal("0")
    return {
        "vwap": float(quote.vwap) if quote.vwap is not None else None,
        "all_in_cost_per_share": float(all_in_cost_per_share),
        "edge": model_probability - float(all_in_cost_per_share),
        "expected_profit": float(expected_profit),
        "expected_roi": float(expected_roi),
        "fully_fillable": quote.fully_fillable,
    }


def floor_to_tick(price: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    return (price / tick_size).to_integral_value(rounding=ROUND_FLOOR) * tick_size


def choose_passive_buy_price(
    *,
    fair_probability: Decimal | float | str,
    best_bid: Decimal | float | str,
    best_ask: Decimal | float | str,
    tick_size: Decimal | float | str,
    required_edge: Decimal | float | str,
    adverse_selection_buffer: Decimal | float | str,
) -> Decimal | None:
    fair = as_decimal(fair_probability, "fair_probability")
    bid = as_decimal(best_bid, "best_bid")
    ask = as_decimal(best_ask, "best_ask")
    tick = as_decimal(tick_size, "tick_size")
    edge = as_decimal(required_edge, "required_edge")
    buffer = as_decimal(adverse_selection_buffer, "adverse_selection_buffer")
    if not Decimal("0") < bid < ask < Decimal("1"):
        raise ValueError("Expected 0 < best_bid < best_ask < 1")
    if not Decimal("0") < fair < Decimal("1"):
        raise ValueError("fair_probability must be between 0 and 1")
    if edge < 0 or buffer < 0:
        raise ValueError("edge and buffer cannot be negative")

    fair_cap = fair - edge - buffer
    highest_post_only = ask - tick
    cap = min(fair_cap, highest_post_only)
    # Do not submit a low-probability queue order behind the current best bid.
    # It preserves theoretical edge but is not a competitive maker quote.
    if cap < bid or cap <= 0:
        return None
    candidate = min(cap, bid + tick)
    candidate = floor_to_tick(candidate, tick)
    if candidate <= 0 or candidate >= ask:
        return None
    return candidate


def build_passive_buy_plan(
    *,
    token_id: str,
    fair_probability: Decimal | float | str,
    desired_dollars: Decimal | float | str,
    best_bid: Decimal | float | str,
    best_ask: Decimal | float | str,
    tick_size: Decimal | float | str,
    required_edge: Decimal | float | str = "0.05",
    adverse_selection_buffer: Decimal | float | str = "0.01",
) -> PassiveOrderPlan | None:
    price = choose_passive_buy_price(
        fair_probability=fair_probability,
        best_bid=best_bid,
        best_ask=best_ask,
        tick_size=tick_size,
        required_edge=required_edge,
        adverse_selection_buffer=adverse_selection_buffer,
    )
    if price is None:
        return None
    notional = as_decimal(desired_dollars, "desired_dollars")
    if notional <= 0:
        raise ValueError("desired_dollars must be positive")
    size = notional / price
    return PassiveOrderPlan(
        token_id=token_id,
        side="BUY",
        limit_price=price,
        size_shares=size,
        notional_dollars=notional,
        fair_probability=as_decimal(fair_probability, "fair_probability"),
        required_edge=as_decimal(required_edge, "required_edge"),
        adverse_selection_buffer=as_decimal(
            adverse_selection_buffer, "adverse_selection_buffer"
        ),
        best_bid=as_decimal(best_bid, "best_bid"),
        best_ask=as_decimal(best_ask, "best_ask"),
        tick_size=as_decimal(tick_size, "tick_size"),
    )
