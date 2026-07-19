import httpx
import pytest

from moneyball_predictions.polymarket import (
    fetch_mlb_moneyline_markets,
    fetch_order_book_tops,
)


def _mlb_event() -> dict:
    return {
        "id": "event-1",
        "ticker": "mlb-nyy-lad-2026-07-20",
        "title": "Yankees vs. Dodgers",
        "slug": "mlb-nyy-lad-2026-07-20",
        "tags": [{"slug": "mlb"}],
        "markets": [
            {
                "id": "total",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "sportsMarketType": "total",
                "outcomes": '["Over", "Under"]',
            },
            {
                "id": "moneyline",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "sportsMarketType": "moneyline",
                "outcomes": '["NYY", "LAD"]',
                "clobTokenIds": '["token-a", "token-b"]',
                "outcomePrices": '["0.52", "0.48"]',
                "gameStartTime": "2026-07-20T23:05:00Z",
            },
        ],
    }


@pytest.mark.asyncio
async def test_discovers_moneyline_with_tag_slug_and_maps_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events":
            assert request.url.params.get("tag_slug") == "mlb"
            assert "order" not in request.url.params
            assert "series_id" not in request.url.params
            return httpx.Response(200, json=[_mlb_event()])
        if request.url.path == "/teams":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        markets, skipped = await fetch_mlb_moneyline_markets(client)

    assert skipped == []
    assert len(markets) == 1
    assert markets[0].team_a == "New York Yankees"
    assert markets[0].team_b == "Los Angeles Dodgers"
    assert markets[0].token_a == "token-a"


@pytest.mark.asyncio
async def test_422_on_slug_falls_back_to_canonical_tag_id() -> None:
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        if request.url.path == "/events" and request.url.params.get("tag_slug") == "mlb":
            return httpx.Response(422, json={"detail": "bad filter"})
        if request.url.path == "/tags/slug/mlb":
            return httpx.Response(200, json={"id": 100381, "slug": "mlb"})
        if request.url.path == "/events" and request.url.params.get("tag_id") == "100381":
            return httpx.Response(200, json=[_mlb_event()])
        if request.url.path == "/teams":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        markets, skipped = await fetch_mlb_moneyline_markets(client)

    assert len(markets) == 1
    assert any("HTTP 422" in item.detail for item in skipped)
    assert not any("series_id" in url for url in requests_seen)


@pytest.mark.asyncio
async def test_bad_sports_tag_is_skipped_and_next_tag_can_work() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events" and request.url.params.get("tag_slug") == "mlb":
            return httpx.Response(422, json={"detail": "bad filter"})
        if request.url.path == "/tags/slug/mlb":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/sports":
            return httpx.Response(200, json=[{"sport": "mlb", "tags": "1,100381"}])
        if request.url.path == "/events" and request.url.params.get("tag_id") == "1":
            return httpx.Response(422, json={"detail": "invalid tag"})
        if request.url.path == "/events" and request.url.params.get("tag_id") == "100381":
            return httpx.Response(200, json=[_mlb_event()])
        if request.url.path == "/teams":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        markets, skipped = await fetch_mlb_moneyline_markets(client)

    assert len(markets) == 1
    assert sum("HTTP 422" in item.detail for item in skipped) == 2


@pytest.mark.asyncio
async def test_empty_successful_discovery_returns_no_markets_instead_of_crashing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events":
            return httpx.Response(200, json=[])
        if request.url.path == "/tags/slug/mlb":
            return httpx.Response(200, json={"id": 100381, "slug": "mlb"})
        if request.url.path == "/sports":
            return httpx.Response(200, json=[])
        if request.url.path == "/teams":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        markets, skipped = await fetch_mlb_moneyline_markets(client)

    assert markets == []
    assert skipped == []


@pytest.mark.asyncio
async def test_reads_best_ask_from_batch_books() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/books"
        return httpx.Response(
            200,
            json=[
                {
                    "asset_id": "token-a",
                    "bids": [{"price": "0.49", "size": "50"}],
                    "asks": [
                        {"price": "0.53", "size": "20"},
                        {"price": "0.51", "size": "10"},
                    ],
                    "last_trade_price": "0.50",
                }
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        books = await fetch_order_book_tops(client, ["token-a"])

    assert books["token-a"].best_ask == pytest.approx(0.51)
    assert books["token-a"].ask_size == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_futures_are_filtered_as_unsupported_not_team_mismatch() -> None:
    future = {
        "id": "future-1",
        "title": "MLB World Series Champion 2026",
        "slug": "mlb-world-series-champion-2026",
        "tags": [{"slug": "mlb"}],
        "markets": [
            {
                "id": "future-market",
                "active": True,
                "closed": False,
                "sportsMarketType": "winner",
                "outcomes": '["Dodgers", "Yankees"]',
                "clobTokenIds": '["a", "b"]',
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events":
            return httpx.Response(200, json=[future])
        if request.url.path == "/teams":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        markets, diagnostics = await fetch_mlb_moneyline_markets(client)

    assert markets == []
    assert diagnostics[0].category == "unsupported_event"
