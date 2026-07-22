import pytest

from moneyball_predictions.player_props import (
    build_pitcher_projection,
    parse_strikeout_market,
    project_strikeout_probability,
    settle_strikeout_selection,
)


def test_parse_yes_no_strikeout_market() -> None:
    market = parse_strikeout_market(
        {
            "id": "market-1",
            "conditionId": "0xabc",
            "question": "Will Gerrit Cole record 7+ strikeouts?",
            "slug": "gerrit-cole-7-strikeouts",
            "active": True,
            "closed": False,
            "sportsMarketType": "pitcher_strikeouts",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["yes-token", "no-token"]',
            "eventStartTime": "2026-07-21T23:05:00Z",
        }
    )
    assert market is not None
    assert market.player_name == "Gerrit Cole"
    assert market.threshold == 7
    assert market.yes_token_id == "yes-token"
    assert market.no_token_id == "no-token"


def test_poisson_strikeout_probability_is_bounded() -> None:
    matchup_k, expected_bf, expected_k, probability = project_strikeout_probability(
        threshold=7,
        pitcher_k_rate=0.30,
        lineup_k_rate=0.24,
        league_k_rate=0.225,
        projected_innings=6.0,
        opponent_reach_rate=0.31,
    )
    assert 0 < matchup_k < 1
    assert expected_bf > 20
    assert expected_k > 0
    assert 0 < probability < 1


def test_pitcher_projection_shrinks_small_samples() -> None:
    projection = build_pitcher_projection(
        {
            "battersFaced": 20,
            "strikeOuts": 12,
            "gamesStarted": 1,
            "inningsPitched": "5.0",
        }
    )
    assert projection.k_rate < 0.60
    assert projection.k_rate > 0.225
    assert projection.projected_innings == pytest.approx(5.1666666667)


def test_strikeout_prop_settlement_yes_and_no() -> None:
    assert settle_strikeout_selection(strikeouts=7, threshold=7, side="YES") is True
    assert settle_strikeout_selection(strikeouts=6, threshold=7, side="YES") is False
    assert settle_strikeout_selection(strikeouts=6, threshold=7, side="NO") is True
    assert settle_strikeout_selection(strikeouts=7, threshold=7, side="NO") is False


@pytest.mark.asyncio
async def test_market_discovery_scans_every_valid_strikeout_type() -> None:
    from moneyball_predictions.player_props import fetch_active_strikeout_markets

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self) -> None:
            self.scanned: list[str] = []

        async def get(self, url: str, params=None):
            if url.endswith("/sports/market-types"):
                return Response(["pitcher_strikeouts", "pitcher_strikeouts_alt"])
            market_type = (params or {}).get("sports_market_types")
            self.scanned.append(market_type)
            suffix = "main" if market_type == "pitcher_strikeouts" else "alt"
            return Response(
                [
                    {
                        "id": f"market-{suffix}",
                        "question": f"Will Example Pitcher record {4 if suffix == 'main' else 5}+ strikeouts?",
                        "slug": f"example-{suffix}",
                        "sportsMarketType": market_type,
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": f'["yes-{suffix}", "no-{suffix}"]',
                    }
                ]
            )

    client = Client()
    markets = await fetch_active_strikeout_markets(client)

    assert {market.market_id for market in markets} == {"market-main", "market-alt"}
    assert client.scanned == ["pitcher_strikeouts", "pitcher_strikeouts_alt"]
