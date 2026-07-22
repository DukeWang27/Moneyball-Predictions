from datetime import UTC, datetime
from pathlib import Path

from moneyball_predictions.lineups import LineupStatus, parse_game_lineups
from moneyball_predictions.storage import (
    database_summary,
    insert_lineup_snapshot,
    latest_lineup_snapshots,
)


def _team_payload(prefix: int, *, count: int = 9, include_sub: bool = False) -> dict:
    batters: list[int] = []
    players: dict[str, dict] = {}
    for slot in range(1, count + 1):
        player_id = prefix + slot
        batters.append(player_id)
        players[f"ID{player_id}"] = {
            "person": {"fullName": f"Starter {player_id}"},
            "position": {"abbreviation": "CF" if slot == 1 else "1B"},
            "battingOrder": f"{slot}00",
        }
    if include_sub:
        substitute_id = prefix + 99
        batters.insert(0, substitute_id)
        players[f"ID{substitute_id}"] = {
            "person": {"fullName": "Pinch Hitter"},
            "position": {"abbreviation": "PH"},
            "battingOrder": "101",
        }
    return {"team": {"id": prefix}, "batters": batters, "players": players}


def test_strict_parser_excludes_substitutions_and_orders_slots() -> None:
    payload = {
        "teams": {
            "away": _team_payload(1000, include_sub=True),
            "home": _team_payload(2000),
        }
    }
    lineups = parse_game_lineups(
        game_pk=42,
        payload=payload,
        captured_at=datetime(2026, 7, 21, 22, tzinfo=UTC),
    )
    assert lineups.both_confirmed
    assert lineups.status is LineupStatus.CONFIRMED
    assert len(lineups.away.players) == 9
    assert lineups.away.players[0].player_id == 1001
    assert [player.batting_slot for player in lineups.away.players] == list(range(1, 10))
    assert all(player.full_name != "Pinch Hitter" for player in lineups.away.players)


def test_parser_reports_partial_and_pending_states() -> None:
    partial = parse_game_lineups(
        game_pk=43,
        payload={"teams": {"away": _team_payload(1000, count=5), "home": {}}},
        captured_at=datetime(2026, 7, 21, 22, tzinfo=UTC),
    )
    assert partial.status is LineupStatus.PARTIAL
    assert partial.away.status is LineupStatus.PARTIAL
    assert partial.home.status is LineupStatus.PENDING


def test_lineup_snapshots_are_immutable_and_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "research.sqlite3"
    first = parse_game_lineups(
        game_pk=44,
        payload={"teams": {"away": _team_payload(1000), "home": _team_payload(2000)}},
        captured_at=datetime(2026, 7, 21, 22, tzinfo=UTC),
    )
    assert insert_lineup_snapshot(first.away, path=path) is True
    assert insert_lineup_snapshot(first.away, path=path) is False

    changed_payload = {"teams": {"away": _team_payload(1000), "home": _team_payload(2000)}}
    changed_payload["teams"]["away"]["players"]["ID1001"]["position"] = {
        "abbreviation": "DH"
    }
    changed = parse_game_lineups(
        game_pk=44,
        payload=changed_payload,
        captured_at=datetime(2026, 7, 21, 22, 5, tzinfo=UTC),
    )
    assert insert_lineup_snapshot(changed.away, path=path) is True
    assert database_summary(path)["lineup_snapshots"] == 2
    latest = latest_lineup_snapshots(44, path=path)
    assert latest["away"]["positions"][0] == "DH"
