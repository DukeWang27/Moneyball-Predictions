"""Polymarket sports discovery and executable order-book pricing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .teams import canonical_team_name, extract_team_pair

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"


class PolymarketDataError(RuntimeError):
    """Raised when Polymarket cannot provide usable MLB market data."""


@dataclass(frozen=True)
class PolymarketTeam:
    id: str
    name: str
    abbreviation: str | None = None
    alias: str | None = None


@dataclass(frozen=True)
class OrderBookTop:
    token_id: str
    best_bid: float | None
    best_ask: float | None
    ask_size: float | None
    last_trade_price: float | None
    bids: tuple[tuple[float, float], ...] = ()
    asks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class MarketDiagnostic:
    category: str
    title: str
    detail: str


@dataclass(frozen=True)
class MlbMoneylineMarket:
    event_id: str
    event_title: str
    event_slug: str
    market_id: str
    start_time: str | None
    team_a: str
    team_b: str
    token_a: str
    token_b: str
    reference_price_a: float | None
    reference_price_b: float | None
    liquidity: float | None
    volume: float | None
    accepting_orders: bool

    @property
    def polymarket_url(self) -> str:
        return f"https://polymarket.com/event/{self.event_slug}"


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _parse_identifier_values(value: Any) -> list[str]:
    values = _parse_json_list(value)
    if values:
        return values
    if value is None:
        return []
    return re.findall(r"\d+", str(value))


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= parsed <= 1_000_000_000:
        return None
    return parsed


_UNSUPPORTED_EVENT_TERMS = (
    "world series champion",
    "league champion",
    "division champion",
    "mvp",
    "cy young",
    "hank aaron",
    "rookie of the year",
    "manager of the year",
    "come back player",
    "comeback player",
    "platinum glove",
    "home runs leader",
    "doubles leader",
    "triples leader",
    "stolen bases leader",
    "runs leader",
    "rbis leader",
    "batting average leader",
    "era leader",
    "strikeouts leader",
    "next manager",
    "win totals",
    "make postseason",
    "100+ games",
    "longest winning streak",
    "longest win streak",
    "perfect game",
    "no-hitter",
    "no hitters",
    "scorigami",
    "cover athlete",
    "intentional walks",
    "most home runs",
    "highest abs",
    "player to hit",
    "player props",
    "first 5 innings",
    "first five innings",
    "first 5 inning",
    "first five inning",
)


def _is_standard_game_event(event: dict[str, Any]) -> bool:
    title = str(event.get("title") or "").casefold()
    slug = str(event.get("slug") or "").casefold()
    ticker = str(event.get("ticker") or "").casefold()
    if any(term in title for term in _UNSUPPORTED_EVENT_TERMS):
        return False
    if extract_team_pair(str(event.get("title") or "")):
        return True
    return bool(
        re.fullmatch(r"mlb-[a-z0-9-]+-[a-z0-9-]+-\d{4}-\d{2}-\d{2}", slug)
        or re.fullmatch(r"mlb-[a-z0-9-]+-[a-z0-9-]+-\d{4}-\d{2}-\d{2}", ticker)
    )


def _market_score(market: dict[str, Any]) -> int:
    if market.get("closed") or not market.get("active", True):
        return -10_000

    market_type = str(market.get("sportsMarketType") or market.get("marketType") or "").casefold()
    question = str(market.get("question") or "").casefold()
    group_title = str(market.get("groupItemTitle") or "").casefold()
    outcomes = [outcome.casefold() for outcome in _parse_json_list(market.get("outcomes"))]

    score = 0
    if market_type in {"moneyline", "money_line", "game winner", "winner"}:
        score += 100
    elif "moneyline" in market_type:
        score += 90
    if group_title in {"moneyline", "winner", "game winner", "match winner"}:
        score += 40
    if len(outcomes) == 2 and set(outcomes) != {"over", "under"}:
        score += 10

    excluded = (
        "spread",
        "run line",
        "total",
        "over/under",
        "first inning",
        "first 5",
        "first five",
        "player prop",
        "score",
    )
    if any(term in market_type or term in question or term in group_title for term in excluded):
        score -= 100
    return score


def _resolve_market_teams(
    market: dict[str, Any],
    event_title: str,
    polymarket_teams: dict[str, PolymarketTeam],
) -> tuple[str, str] | None:
    outcomes = _parse_json_list(market.get("outcomes"))
    short_outcomes = _parse_json_list(market.get("shortOutcomes"))

    if len(outcomes) == 2:
        canonical_outcomes = [canonical_team_name(outcome) for outcome in outcomes]
        if all(canonical_outcomes) and canonical_outcomes[0] != canonical_outcomes[1]:
            return canonical_outcomes[0], canonical_outcomes[1]  # type: ignore[return-value]

    if len(short_outcomes) == 2:
        canonical_short = [canonical_team_name(outcome) for outcome in short_outcomes]
        if all(canonical_short) and canonical_short[0] != canonical_short[1]:
            return canonical_short[0], canonical_short[1]  # type: ignore[return-value]

    team_a = polymarket_teams.get(str(market.get("teamAID") or ""))
    team_b = polymarket_teams.get(str(market.get("teamBID") or ""))
    if team_a and team_b:
        canonical_a = (
            canonical_team_name(team_a.name)
            or canonical_team_name(team_a.abbreviation)
            or canonical_team_name(team_a.alias)
        )
        canonical_b = (
            canonical_team_name(team_b.name)
            or canonical_team_name(team_b.abbreviation)
            or canonical_team_name(team_b.alias)
        )
        if canonical_a and canonical_b and canonical_a != canonical_b:
            return canonical_a, canonical_b

    return extract_team_pair(event_title)


def _looks_like_mlb_event(event: dict[str, Any]) -> bool:
    slug = str(event.get("slug") or "").casefold()
    ticker = str(event.get("ticker") or "").casefold()
    if slug.startswith("mlb-") or ticker.startswith("mlb-"):
        return True

    tags = event.get("tags") if isinstance(event.get("tags"), list) else []
    if any(str(tag.get("slug") or "").casefold() == "mlb" for tag in tags if isinstance(tag, dict)):
        return True

    raw_markets = event.get("markets") if isinstance(event.get("markets"), list) else []
    return any(
        str(market.get("sportsMarketType") or "").casefold() == "moneyline"
        and (market.get("teamAID") or len(_parse_json_list(market.get("outcomes"))) == 2)
        for market in raw_markets
        if isinstance(market, dict)
    )


async def _fetch_sports_metadata(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    response = await client.get(f"{GAMMA_BASE_URL}/sports")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise PolymarketDataError("Polymarket sports metadata had an unexpected shape")
    return payload


async def _fetch_tag_id_by_slug(client: httpx.AsyncClient, slug: str) -> str | None:
    response = await client.get(f"{GAMMA_BASE_URL}/tags/slug/{slug}")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    tag_id = str(payload.get("id") or "").strip()
    return tag_id or None


async def _fetch_polymarket_teams(client: httpx.AsyncClient) -> dict[str, PolymarketTeam]:
    response = await client.get(
        f"{GAMMA_BASE_URL}/teams",
        params={"league": "mlb", "limit": 100},
    )
    response.raise_for_status()
    payload = response.json()
    teams: dict[str, PolymarketTeam] = {}
    if not isinstance(payload, list):
        return teams
    for raw in payload:
        if str(raw.get("league") or "").casefold() not in {"mlb", "baseball"}:
            continue
        team_id = str(raw.get("id") or "")
        name = str(raw.get("name") or "").strip()
        if not team_id or not name:
            continue
        teams[team_id] = PolymarketTeam(
            id=team_id,
            name=name,
            abbreviation=(str(raw.get("abbreviation")) if raw.get("abbreviation") else None),
            alias=(str(raw.get("alias")) if raw.get("alias") else None),
        )
    return teams


async def _fetch_events_for_filter(
    client: httpx.AsyncClient,
    filter_name: str,
    filter_value: str,
) -> list[dict[str, Any]]:
    response = await client.get(
        f"{GAMMA_BASE_URL}/events",
        params={
            filter_name: filter_value,
            "active": True,
            "closed": False,
            "limit": 100,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def _request_diagnostic(label: str, exc: httpx.HTTPStatusError) -> MarketDiagnostic:
    status = exc.response.status_code
    return MarketDiagnostic(
        category="discovery_warning",
        title="Polymarket discovery",
        detail=f"Candidate {label} returned HTTP {status}; fallback attempted",
    )


async def _discover_mlb_events(
    client: httpx.AsyncClient,
) -> tuple[dict[str, dict[str, Any]], list[MarketDiagnostic]]:
    diagnostics: list[MarketDiagnostic] = []
    events_by_id: dict[str, dict[str, Any]] = {}
    attempted: set[tuple[str, str]] = set()
    successful_request = False

    async def try_candidate(filter_name: str, filter_value: str) -> bool:
        nonlocal successful_request
        candidate = (filter_name, filter_value)
        if candidate in attempted:
            return False
        attempted.add(candidate)
        label = f"{filter_name}={filter_value}"
        try:
            events = await _fetch_events_for_filter(client, filter_name, filter_value)
        except httpx.HTTPStatusError as exc:
            diagnostics.append(_request_diagnostic(label, exc))
            return False
        successful_request = True
        for event in events:
            if not isinstance(event, dict) or not _looks_like_mlb_event(event):
                continue
            event_id = str(event.get("id") or "")
            if event_id:
                events_by_id[event_id] = event
        return bool(events_by_id)

    if await try_candidate("tag_slug", "mlb"):
        return events_by_id, diagnostics

    try:
        tag_id = await _fetch_tag_id_by_slug(client, "mlb")
    except httpx.HTTPStatusError as exc:
        diagnostics.append(_request_diagnostic("tag lookup slug=mlb", exc))
        tag_id = None
    if tag_id and await try_candidate("tag_id", tag_id):
        return events_by_id, diagnostics

    try:
        sports = await _fetch_sports_metadata(client)
    except (httpx.HTTPStatusError, PolymarketDataError) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            diagnostics.append(_request_diagnostic("sports metadata", exc))
        else:
            diagnostics.append(
                MarketDiagnostic("discovery_warning", "Polymarket discovery", str(exc))
            )
        sports = []

    mlb_sport = next(
        (
            sport
            for sport in sports
            if str(sport.get("sport") or "").casefold() in {"mlb", "baseball"}
        ),
        None,
    )
    if mlb_sport:
        for candidate_tag_id in _parse_identifier_values(mlb_sport.get("tags")):
            if await try_candidate("tag_id", candidate_tag_id):
                return events_by_id, diagnostics

    if not successful_request:
        details = "; ".join(item.detail for item in diagnostics[-3:])
        raise PolymarketDataError(
            f"Polymarket MLB event discovery failed: {details or 'no request succeeded'}"
        )

    return events_by_id, diagnostics


async def fetch_mlb_moneyline_markets(
    client: httpx.AsyncClient,
) -> tuple[list[MlbMoneylineMarket], list[MarketDiagnostic]]:
    """Discover standard MLB game moneylines and classify unsupported events cleanly."""
    events_by_id, diagnostics = await _discover_mlb_events(client)
    polymarket_teams = await _fetch_polymarket_teams(client)
    markets: list[MlbMoneylineMarket] = []

    for event in events_by_id.values():
        event_title = str(event.get("title") or "Unknown event")
        if not _is_standard_game_event(event):
            diagnostics.append(
                MarketDiagnostic(
                    category="unsupported_event",
                    title=event_title,
                    detail="Not a standard full-game MLB matchup",
                )
            )
            continue

        raw_markets = event.get("markets") or []
        if not isinstance(raw_markets, list):
            diagnostics.append(
                MarketDiagnostic("no_market_list", event_title, "Event had no market list")
            )
            continue
        candidates = sorted(
            (market for market in raw_markets if isinstance(market, dict)),
            key=_market_score,
            reverse=True,
        )
        if not candidates or _market_score(candidates[0]) <= 0:
            diagnostics.append(
                MarketDiagnostic(
                    "no_moneyline",
                    event_title,
                    "No standard full-game moneyline market",
                )
            )
            continue

        market = candidates[0]
        teams = _resolve_market_teams(market, event_title, polymarket_teams)
        tokens = _parse_json_list(market.get("clobTokenIds"))
        reference_prices = [
            _safe_float(value) for value in _parse_json_list(market.get("outcomePrices"))
        ]
        if not teams:
            diagnostics.append(
                MarketDiagnostic(
                    "unmatched_teams",
                    event_title,
                    "Could not map both teams to canonical MLB names",
                )
            )
            continue
        if len(tokens) != 2:
            diagnostics.append(
                MarketDiagnostic(
                    "missing_tokens",
                    event_title,
                    "Expected two CLOB outcome tokens",
                )
            )
            continue

        markets.append(
            MlbMoneylineMarket(
                event_id=str(event.get("id") or ""),
                event_title=event_title,
                event_slug=str(event.get("slug") or ""),
                market_id=str(market.get("id") or ""),
                start_time=(
                    str(
                        market.get("gameStartTime")
                        or market.get("eventStartTime")
                        or event.get("startDate")
                        or ""
                    )
                    or None
                ),
                team_a=teams[0],
                team_b=teams[1],
                token_a=tokens[0],
                token_b=tokens[1],
                reference_price_a=(reference_prices[0] if len(reference_prices) == 2 else None),
                reference_price_b=(reference_prices[1] if len(reference_prices) == 2 else None),
                liquidity=_safe_float(market.get("liquidityNum") or market.get("liquidity")),
                volume=_safe_float(market.get("volumeNum") or market.get("volume")),
                accepting_orders=market.get("acceptingOrders") is not False,
            )
        )

    markets.sort(key=lambda market: market.start_time or "")
    return markets, diagnostics


async def fetch_order_book_tops(
    client: httpx.AsyncClient,
    token_ids: list[str],
) -> dict[str, OrderBookTop]:
    """Fetch multiple books in one public CLOB request and return top-of-book prices."""
    if not token_ids:
        return {}
    response = await client.post(
        f"{CLOB_BASE_URL}/books",
        json=[{"token_id": token_id} for token_id in token_ids],
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise PolymarketDataError("Polymarket CLOB books response had an unexpected shape")

    result: dict[str, OrderBookTop] = {}
    for book in payload:
        token_id = str(book.get("asset_id") or "")
        if not token_id:
            continue
        bids = book.get("bids") if isinstance(book.get("bids"), list) else []
        asks = book.get("asks") if isinstance(book.get("asks"), list) else []
        best_bid_order = max(
            bids,
            key=lambda order: _safe_float(order.get("price")) or -1,
            default=None,
        )
        best_ask_order = min(
            asks,
            key=lambda order: _safe_float(order.get("price")) or 2,
            default=None,
        )
        parsed_bids = tuple(
            (price, size)
            for order in bids
            if (price := _safe_float(order.get("price"))) is not None
            and (size := _safe_float(order.get("size"))) is not None
            and 0 < price < 1
            and size > 0
        )
        parsed_asks = tuple(
            (price, size)
            for order in asks
            if (price := _safe_float(order.get("price"))) is not None
            and (size := _safe_float(order.get("size"))) is not None
            and 0 < price < 1
            and size > 0
        )
        result[token_id] = OrderBookTop(
            token_id=token_id,
            best_bid=_safe_float(best_bid_order.get("price")) if best_bid_order else None,
            best_ask=_safe_float(best_ask_order.get("price")) if best_ask_order else None,
            ask_size=_safe_float(best_ask_order.get("size")) if best_ask_order else None,
            last_trade_price=_safe_float(book.get("last_trade_price")),
            bids=tuple(sorted(parsed_bids, key=lambda item: item[0], reverse=True)),
            asks=tuple(sorted(parsed_asks, key=lambda item: item[0])),
        )
    return result
