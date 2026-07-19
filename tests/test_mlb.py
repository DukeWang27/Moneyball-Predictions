import httpx
import pytest

from moneyball_predictions.mlb import fetch_team_season_stats


@pytest.mark.asyncio
async def test_parses_mlb_standings_run_totals() -> None:
    records = []
    team_records = []
    names = [
        "New York Yankees", "Boston Red Sox", "Tampa Bay Rays", "Toronto Blue Jays",
        "Baltimore Orioles", "Cleveland Guardians", "Detroit Tigers", "Minnesota Twins",
        "Kansas City Royals", "Chicago White Sox", "Houston Astros", "Seattle Mariners",
        "Texas Rangers", "Los Angeles Angels", "Athletics", "Atlanta Braves",
        "New York Mets", "Philadelphia Phillies", "Washington Nationals", "Miami Marlins",
    ]
    for index, name in enumerate(names, start=1):
        team_records.append({
            "team": {"id": index, "name": name},
            "gamesPlayed": 100,
            "runsScored": 500 + index,
            "runsAllowed": 450 + index,
        })
    records.append({"teamRecords": team_records})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"records": records})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        stats = await fetch_team_season_stats(client, 2026)

    assert stats["New York Yankees"].games_played == 100
    assert stats["Athletics"].runs_scored == 515


@pytest.mark.asyncio
async def test_parses_live_schedule_score_and_inning() -> None:
    from datetime import date

    from moneyball_predictions.mlb import fetch_mlb_schedule

    payload = {
        "dates": [
            {
                "date": "2026-07-19",
                "games": [
                    {
                        "gamePk": 123,
                        "gameDate": "2026-07-19T23:20:00Z",
                        "officialDate": "2026-07-19",
                        "status": {
                            "abstractGameState": "Live",
                            "detailedState": "In Progress",
                            "codedGameState": "I",
                        },
                        "teams": {
                            "away": {
                                "team": {"name": "Los Angeles Dodgers"},
                                "score": 3,
                                "probablePitcher": {"fullName": "Pitcher A"},
                            },
                            "home": {
                                "team": {"name": "New York Yankees"},
                                "score": 2,
                                "probablePitcher": {"fullName": "Pitcher B"},
                            },
                        },
                        "linescore": {
                            "currentInning": 6,
                            "inningState": "Top",
                            "currentInningOrdinal": "6th",
                        },
                        "venue": {"name": "Yankee Stadium"},
                    }
                ],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/schedule"
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        games = await fetch_mlb_schedule(client, date(2026, 7, 19), date(2026, 7, 19))

    assert games[0].is_live
    assert games[0].away_score == 3
    assert games[0].inning_ordinal == "6th"
