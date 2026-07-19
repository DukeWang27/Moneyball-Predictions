"""Read MLB standings, schedules, scores, and game status from Stats API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from .teams import canonical_team_name

MLB_STATS_BASE_URL = "https://statsapi.mlb.com/api/v1"


class MlbDataError(RuntimeError):
    """Raised when the MLB Stats API does not return usable data."""


@dataclass(frozen=True)
class MlbTeamStats:
    team_id: int
    name: str
    runs_scored: float
    runs_allowed: float
    games_played: int


@dataclass(frozen=True)
class MlbGameState:
    game_pk: int
    game_date: str
    official_date: str | None
    away_team: str
    home_team: str
    away_score: int | None
    home_score: int | None
    abstract_state: str
    detailed_state: str
    coded_state: str | None
    current_inning: int | None
    inning_state: str | None
    inning_ordinal: str | None
    away_probable_pitcher: str | None
    home_probable_pitcher: str | None
    venue: str | None

    @property
    def is_live(self) -> bool:
        return self.abstract_state.casefold() == "live"

    @property
    def is_final(self) -> bool:
        return self.abstract_state.casefold() == "final"

    @property
    def is_pregame(self) -> bool:
        return not self.is_live and not self.is_final

    @property
    def winner(self) -> str | None:
        if not self.is_final or self.away_score is None or self.home_score is None:
            return None
        if self.away_score > self.home_score:
            return self.away_team
        if self.home_score > self.away_score:
            return self.home_team
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_schedule_payload(payload: object) -> list[MlbGameState]:
    if not isinstance(payload, dict):
        raise MlbDataError("MLB schedule response had an unexpected shape")

    games: list[MlbGameState] = []
    for date_group in payload.get("dates", []):
        if not isinstance(date_group, dict):
            continue
        official_group_date = _optional_text(date_group.get("date"))
        for raw_game in date_group.get("games", []):
            if not isinstance(raw_game, dict):
                continue
            teams = raw_game.get("teams") or {}
            away = teams.get("away") or {}
            home = teams.get("home") or {}
            away_team_payload = away.get("team") or {}
            home_team_payload = home.get("team") or {}
            away_raw = _optional_text(away_team_payload.get("name"))
            home_raw = _optional_text(home_team_payload.get("name"))
            away_name = canonical_team_name(away_raw) if away_raw else None
            home_name = canonical_team_name(home_raw) if home_raw else None
            game_pk = _safe_int(raw_game.get("gamePk"))
            game_date = _optional_text(raw_game.get("gameDate"))
            if not away_name or not home_name or game_pk is None or not game_date:
                continue

            status = raw_game.get("status") or {}
            linescore = raw_game.get("linescore") or {}
            venue = raw_game.get("venue") or {}
            away_pitcher = away.get("probablePitcher") or {}
            home_pitcher = home.get("probablePitcher") or {}

            games.append(
                MlbGameState(
                    game_pk=game_pk,
                    game_date=game_date,
                    official_date=_optional_text(raw_game.get("officialDate"))
                    or official_group_date,
                    away_team=away_name,
                    home_team=home_name,
                    away_score=_safe_int(away.get("score")),
                    home_score=_safe_int(home.get("score")),
                    abstract_state=str(status.get("abstractGameState") or "Preview"),
                    detailed_state=str(status.get("detailedState") or "Scheduled"),
                    coded_state=_optional_text(status.get("codedGameState")),
                    current_inning=_safe_int(linescore.get("currentInning")),
                    inning_state=_optional_text(linescore.get("inningState")),
                    inning_ordinal=_optional_text(linescore.get("currentInningOrdinal")),
                    away_probable_pitcher=_optional_text(away_pitcher.get("fullName")),
                    home_probable_pitcher=_optional_text(home_pitcher.get("fullName")),
                    venue=_optional_text(venue.get("name")),
                )
            )
    return games


async def fetch_team_season_stats(
    client: httpx.AsyncClient,
    season: int,
) -> dict[str, MlbTeamStats]:
    """Fetch AL and NL standings, keyed by canonical MLB team name."""
    response = await client.get(
        f"{MLB_STATS_BASE_URL}/standings",
        params={
            "leagueId": "103,104",
            "season": season,
            "standingsTypes": "regularSeason",
        },
    )
    response.raise_for_status()
    payload = response.json()

    teams: dict[str, MlbTeamStats] = {}
    for division_record in payload.get("records", []):
        for record in division_record.get("teamRecords", []):
            team_payload = record.get("team") or {}
            raw_name = str(team_payload.get("name") or "").strip()
            canonical = canonical_team_name(raw_name) or raw_name

            try:
                team_id = int(team_payload["id"])
                games_played = int(record["gamesPlayed"])
                runs_scored = float(record["runsScored"])
                runs_allowed = float(record["runsAllowed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MlbDataError("MLB standings were missing run-total fields") from exc

            if not canonical or games_played <= 0:
                continue
            teams[canonical] = MlbTeamStats(
                team_id=team_id,
                name=canonical,
                runs_scored=runs_scored,
                runs_allowed=runs_allowed,
                games_played=games_played,
            )

    if len(teams) < 20:
        raise MlbDataError(
            f"MLB standings returned only {len(teams)} usable teams; expected most of the league"
        )
    return teams


async def fetch_mlb_schedule(
    client: httpx.AsyncClient,
    start_date: date,
    end_date: date,
) -> list[MlbGameState]:
    """Fetch official MLB games, scores, innings, statuses, and probable pitchers."""
    response = await client.get(
        f"{MLB_STATS_BASE_URL}/schedule",
        params={
            "sportId": 1,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "hydrate": "linescore,team,probablePitcher",
        },
    )
    response.raise_for_status()
    return _parse_schedule_payload(response.json())


async def fetch_mlb_game(
    client: httpx.AsyncClient,
    game_pk: int,
) -> MlbGameState | None:
    """Fetch one official MLB game by gamePk for paper-bet settlement."""
    response = await client.get(
        f"{MLB_STATS_BASE_URL}/schedule",
        params={
            "sportId": 1,
            "gamePk": game_pk,
            "hydrate": "linescore,team,probablePitcher",
        },
    )
    response.raise_for_status()
    games = _parse_schedule_payload(response.json())
    return games[0] if games else None
