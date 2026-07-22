"""Deterministic MLB starting-lineup parsing and status tracking.

The MLB boxscore feed may contain later substitutions alongside the announced
starting order.  This module defines one canonical rule for all product areas:
only battingOrder values 100, 200, ..., 900 count as the original nine starters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal


class LineupStatus(StrEnum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class LineupPlayer:
    player_id: int
    full_name: str
    batting_slot: int
    position: str | None = None


@dataclass(frozen=True)
class TeamLineup:
    game_pk: int
    side: Literal["away", "home"]
    team_id: int | None
    status: LineupStatus
    players: tuple[LineupPlayer, ...]
    captured_at: datetime
    source_hash: str

    @property
    def confirmed(self) -> bool:
        return self.status is LineupStatus.CONFIRMED and len(self.players) == 9


@dataclass(frozen=True)
class GameLineups:
    game_pk: int
    away: TeamLineup
    home: TeamLineup

    @property
    def both_confirmed(self) -> bool:
        return self.away.confirmed and self.home.confirmed

    @property
    def status(self) -> LineupStatus:
        if self.both_confirmed:
            return LineupStatus.CONFIRMED
        if self.away.players or self.home.players:
            return LineupStatus.PARTIAL
        return LineupStatus.PENDING


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_team_lineup(
    *,
    game_pk: int,
    side: Literal["away", "home"],
    payload: object,
    captured_at: datetime,
) -> TeamLineup:
    """Parse only the announced starting nine from one MLB boxscore team payload."""
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    if not isinstance(payload, dict):
        payload = {}

    team_payload = payload.get("team") or {}
    team_id = _safe_int(team_payload.get("id")) if isinstance(team_payload, dict) else None
    batter_ids = payload.get("batters")
    players_payload = payload.get("players")
    if not isinstance(batter_ids, list):
        batter_ids = []
    if not isinstance(players_payload, dict):
        players_payload = {}

    lineup_by_slot: dict[int, LineupPlayer] = {}
    for raw_player_id in batter_ids:
        player_id = _safe_int(raw_player_id)
        if player_id is None:
            continue
        player_payload = (
            players_payload.get(f"ID{player_id}")
            or players_payload.get(str(player_id))
            or {}
        )
        if not isinstance(player_payload, dict):
            continue

        batting_order = _safe_int(player_payload.get("battingOrder"))
        # 101, 201, etc. are substitutions and are intentionally excluded.
        if batting_order is None or batting_order % 100 != 0:
            continue
        batting_slot = batting_order // 100
        if batting_slot not in range(1, 10):
            continue

        person = player_payload.get("person") or {}
        position = player_payload.get("position") or {}
        if not isinstance(person, dict):
            person = {}
        if not isinstance(position, dict):
            position = {}
        full_name = str(
            person.get("fullName")
            or person.get("fullFMLName")
            or player_id
        ).strip()
        position_text = str(position.get("abbreviation") or "").strip() or None
        lineup_by_slot[batting_slot] = LineupPlayer(
            player_id=player_id,
            full_name=full_name,
            batting_slot=batting_slot,
            position=position_text,
        )

    players = tuple(lineup_by_slot[slot] for slot in sorted(lineup_by_slot))
    if len(players) == 9 and {player.batting_slot for player in players} == set(range(1, 10)):
        status = LineupStatus.CONFIRMED
    elif players:
        status = LineupStatus.PARTIAL
    else:
        status = LineupStatus.PENDING

    hash_payload = {
        "game_pk": game_pk,
        "side": side,
        "team_id": team_id,
        "status": status.value,
        "players": [
            {
                "player_id": player.player_id,
                "batting_slot": player.batting_slot,
                "position": player.position,
            }
            for player in players
        ],
    }
    return TeamLineup(
        game_pk=game_pk,
        side=side,
        team_id=team_id,
        status=status,
        players=players,
        captured_at=captured_at.astimezone(UTC),
        source_hash=_canonical_hash(hash_payload),
    )


def parse_game_lineups(
    *,
    game_pk: int,
    payload: object,
    captured_at: datetime | None = None,
) -> GameLineups:
    """Parse both clubs from an MLB boxscore response."""
    timestamp = captured_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    teams: dict[str, object] = {}
    if isinstance(payload, dict) and isinstance(payload.get("teams"), dict):
        teams = payload["teams"]
    return GameLineups(
        game_pk=game_pk,
        away=parse_team_lineup(
            game_pk=game_pk,
            side="away",
            payload=teams.get("away"),
            captured_at=timestamp,
        ),
        home=parse_team_lineup(
            game_pk=game_pk,
            side="home",
            payload=teams.get("home"),
            captured_at=timestamp,
        ),
    )
