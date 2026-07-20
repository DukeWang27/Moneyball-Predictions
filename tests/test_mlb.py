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
                                "probablePitcher": {"id": 11, "fullName": "Pitcher A"},
                            },
                            "home": {
                                "team": {"name": "New York Yankees"},
                                "score": 2,
                                "probablePitcher": {"id": 22, "fullName": "Pitcher B"},
                            },
                        },
                        "linescore": {
                            "currentInning": 6,
                            "inningState": "Top",
                            "currentInningOrdinal": "6th",
                            "innings": [
                                {"away": {"runs": 1}, "home": {"runs": 0}},
                                {"away": {"runs": 0}, "home": {"runs": 1}},
                                {"away": {"runs": 2}, "home": {"runs": 0}},
                                {"away": {"runs": 0}, "home": {"runs": 1}},
                                {"away": {"runs": 0}, "home": {"runs": 0}},
                                {"away": {"runs": 0}, "home": {"runs": 0}},
                            ],
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
    assert games[0].away_probable_pitcher_id == 11
    assert games[0].home_probable_pitcher_id == 22
    assert games[0].away_first5_runs == 3
    assert games[0].home_first5_runs == 2


def test_attaches_component_pitching_lines_from_boxscore() -> None:
    from moneyball_predictions.mlb import MlbGameState, attach_boxscore_payload

    game = MlbGameState(
        game_pk=777,
        game_date="2026-07-18T23:00:00Z",
        official_date="2026-07-18",
        away_team="Boston Red Sox",
        home_team="New York Yankees",
        away_score=3,
        home_score=4,
        abstract_state="Final",
        detailed_state="Final",
        coded_state="F",
        current_inning=9,
        inning_state="Bottom",
        inning_ordinal="9th",
        away_probable_pitcher="Away Starter",
        home_probable_pitcher="Home Starter",
        venue="Yankee Stadium",
    )
    payload = {
        "teams": {
            "away": {
                "pitchers": [101, 102],
                "players": {
                    "ID101": {
                        "person": {"fullName": "Away Starter"},
                        "stats": {
                            "pitching": {
                                "inningsPitched": "5.2",
                                "battersFaced": 23,
                                "strikeOuts": 7,
                                "baseOnBalls": 2,
                                "hitBatsmen": 1,
                                "homeRuns": 1,
                                "pitchesThrown": 94,
                                "earnedRuns": 3,
                            }
                        },
                    },
                    "ID102": {
                        "person": {"fullName": "Away Reliever"},
                        "stats": {
                            "pitching": {
                                "inningsPitched": "2.1",
                                "battersFaced": 9,
                                "strikeOuts": 2,
                                "baseOnBalls": 0,
                                "hitBatsmen": 0,
                                "homeRuns": 0,
                                "pitchesThrown": 28,
                                "earnedRuns": 1,
                            }
                        },
                    },
                },
            },
            "home": {
                "pitchers": [201],
                "players": {
                    "ID201": {
                        "person": {"fullName": "Home Starter"},
                        "stats": {
                            "pitching": {
                                "inningsPitched": "9.0",
                                "battersFaced": 33,
                                "strikeOuts": 10,
                                "baseOnBalls": 1,
                                "hitBatsmen": 0,
                                "homeRuns": 1,
                                "pitchesThrown": 109,
                                "earnedRuns": 3,
                            }
                        },
                    }
                },
            },
        }
    }

    enriched = attach_boxscore_payload(game, payload)
    assert enriched.away_starter_line is not None
    assert enriched.away_starter_line.outs_recorded == 17
    assert enriched.away_starter_line.strikeouts == 7
    assert len(enriched.away_pitching) == 2
    assert enriched.away_pitching[1].is_starter is False
    assert enriched.home_starter_line is not None
    assert enriched.home_starter_line.pitches_thrown == 109


def test_boxscore_parser_captures_confirmed_lineups() -> None:
    from moneyball_predictions.mlb import MlbGameState, attach_boxscore_payload

    game = MlbGameState(
        game_pk=888,
        game_date="2026-07-20T23:00:00Z",
        official_date="2026-07-20",
        away_team="Boston Red Sox",
        home_team="New York Yankees",
        away_score=None,
        home_score=None,
        abstract_state="Preview",
        detailed_state="Scheduled",
        coded_state="S",
        current_inning=None,
        inning_state=None,
        inning_ordinal=None,
        away_probable_pitcher="Away Starter",
        home_probable_pitcher="Home Starter",
        venue="Yankee Stadium",
    )
    players = {}
    away_batters = []
    home_batters = []
    for index in range(1, 10):
        away_id = 1000 + index
        home_id = 2000 + index
        away_batters.append(away_id)
        home_batters.append(home_id)
        players[f"ID{away_id}"] = {
            "person": {"fullName": f"Away {index}"},
            "battingOrder": f"{index}00",
            "stats": {"pitching": {}},
        }
        players[f"ID{home_id}"] = {
            "person": {"fullName": f"Home {index}"},
            "battingOrder": f"{index}00",
            "stats": {"pitching": {}},
        }
    payload = {
        "teams": {
            "away": {"batters": away_batters, "pitchers": [], "players": players},
            "home": {"batters": home_batters, "pitchers": [], "players": players},
        }
    }
    enriched = attach_boxscore_payload(game, payload)
    assert len(enriched.away_lineup_ids) == 9
    assert len(enriched.home_lineup_names) == 9
    assert enriched.away_lineup_names[0] == "Away 1"
