"""Read MLB standings, schedules, scores, and game status from Stats API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import json
import os
from pathlib import Path

import httpx

from .lineups import LineupStatus, parse_game_lineups
from .teams import canonical_team_name

MLB_STATS_BASE_URL = "https://statsapi.mlb.com/api/v1"


class MlbDataError(RuntimeError):
    """Raised when the MLB Stats API does not return usable data."""


@dataclass(frozen=True)
class MlbPitchingLine:
    """One pitcher's game line, parsed from the official MLB box score."""

    pitcher_id: int
    name: str | None
    team: str
    is_starter: bool
    outs_recorded: int
    batters_faced: int
    strikeouts: int
    walks: int
    hit_batters: int
    home_runs: int
    pitches_thrown: int
    earned_runs: int

    @property
    def innings_pitched(self) -> float:
        return self.outs_recorded / 3.0


def _innings_to_outs(value: object) -> int:
    text = str(value or "0").strip()
    if not text:
        return 0
    try:
        whole_text, _, fraction_text = text.partition(".")
        whole = int(whole_text or 0)
        fraction = int(fraction_text[:1] or 0)
        if fraction not in {0, 1, 2}:
            return max(int(round(float(text) * 3)), 0)
        return max(whole * 3 + fraction, 0)
    except (TypeError, ValueError):
        return 0




@dataclass(frozen=True)
class MlbBattingLine:
    """One team's batting totals from the official game box score."""

    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    intentional_walks: int = 0
    hit_by_pitch: int = 0

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
    game_type: str | None = None
    away_team_id: int | None = None
    home_team_id: int | None = None
    away_probable_pitcher_id: int | None = None
    home_probable_pitcher_id: int | None = None
    away_first5_runs: int | None = None
    home_first5_runs: int | None = None
    away_pitching: tuple[MlbPitchingLine, ...] = ()
    home_pitching: tuple[MlbPitchingLine, ...] = ()
    away_lineup_ids: tuple[int, ...] = ()
    home_lineup_ids: tuple[int, ...] = ()
    away_lineup_names: tuple[str, ...] = ()
    home_lineup_names: tuple[str, ...] = ()
    away_lineup_slots: tuple[int, ...] = ()
    home_lineup_slots: tuple[int, ...] = ()
    away_lineup_positions: tuple[str | None, ...] = ()
    home_lineup_positions: tuple[str | None, ...] = ()
    away_lineup_status: str = LineupStatus.PENDING.value
    home_lineup_status: str = LineupStatus.PENDING.value
    away_lineup_hash: str | None = None
    home_lineup_hash: str | None = None
    away_batting: MlbBattingLine | None = None
    home_batting: MlbBattingLine | None = None

    @property
    def away_starter_line(self) -> MlbPitchingLine | None:
        return next((line for line in self.away_pitching if line.is_starter), None)

    @property
    def home_starter_line(self) -> MlbPitchingLine | None:
        return next((line for line in self.home_pitching if line.is_starter), None)

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


def _first_five_runs(linescore: object, side: str) -> int | None:
    if not isinstance(linescore, dict):
        return None
    innings = linescore.get("innings")
    if not isinstance(innings, list) or len(innings) < 5:
        return None
    total = 0
    for inning in innings[:5]:
        if not isinstance(inning, dict):
            return None
        side_payload = inning.get(side)
        if not isinstance(side_payload, dict):
            return None
        runs = _safe_int(side_payload.get("runs"))
        if runs is None:
            runs = 0
        total += runs
    return total


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
                    game_type=_optional_text(raw_game.get("gameType")),
                    away_team_id=_safe_int(away_team_payload.get("id")),
                    home_team_id=_safe_int(home_team_payload.get("id")),
                    away_probable_pitcher_id=_safe_int(away_pitcher.get("id")),
                    home_probable_pitcher_id=_safe_int(home_pitcher.get("id")),
                    away_first5_runs=_first_five_runs(linescore, "away"),
                    home_first5_runs=_first_five_runs(linescore, "home"),
                )
            )
    return games


def _parse_boxscore_side(payload: object, *, team: str) -> tuple[MlbPitchingLine, ...]:
    if not isinstance(payload, dict):
        return ()
    pitcher_ids = payload.get("pitchers")
    players = payload.get("players")
    if not isinstance(pitcher_ids, list) or not isinstance(players, dict):
        return ()

    lines: list[MlbPitchingLine] = []
    for position, raw_id in enumerate(pitcher_ids):
        pitcher_id = _safe_int(raw_id)
        if pitcher_id is None:
            continue
        player = players.get(f"ID{pitcher_id}") or players.get(str(pitcher_id)) or {}
        if not isinstance(player, dict):
            continue
        person = player.get("person") or {}
        stats = player.get("stats") or {}
        pitching = stats.get("pitching") or {}
        if not isinstance(pitching, dict):
            continue
        lines.append(
            MlbPitchingLine(
                pitcher_id=pitcher_id,
                name=_optional_text(person.get("fullName")),
                team=team,
                is_starter=position == 0,
                outs_recorded=_innings_to_outs(pitching.get("inningsPitched")),
                batters_faced=_safe_int(pitching.get("battersFaced")) or 0,
                strikeouts=_safe_int(pitching.get("strikeOuts")) or 0,
                walks=_safe_int(pitching.get("baseOnBalls")) or 0,
                hit_batters=_safe_int(pitching.get("hitBatsmen")) or 0,
                home_runs=_safe_int(pitching.get("homeRuns")) or 0,
                pitches_thrown=_safe_int(pitching.get("pitchesThrown")) or 0,
                earned_runs=_safe_int(pitching.get("earnedRuns")) or 0,
            )
        )
    return tuple(lines)


def _parse_team_batting(payload: object) -> MlbBattingLine | None:
    if not isinstance(payload, dict):
        return None
    team_stats = payload.get("teamStats") or {}
    batting = team_stats.get("batting") if isinstance(team_stats, dict) else None
    if not isinstance(batting, dict):
        return None
    return MlbBattingLine(
        at_bats=_safe_int(batting.get("atBats")) or 0,
        hits=_safe_int(batting.get("hits")) or 0,
        doubles=_safe_int(batting.get("doubles")) or 0,
        triples=_safe_int(batting.get("triples")) or 0,
        home_runs=_safe_int(batting.get("homeRuns")) or 0,
        walks=_safe_int(batting.get("baseOnBalls")) or 0,
        intentional_walks=_safe_int(batting.get("intentionalWalks")) or 0,
        hit_by_pitch=_safe_int(batting.get("hitByPitch")) or 0,
    )


def attach_boxscore_payload(game: MlbGameState, payload: object) -> MlbGameState:
    """Return a game enriched with official per-pitcher box-score lines."""
    if not isinstance(payload, dict):
        return game
    teams = payload.get("teams") or {}
    if not isinstance(teams, dict):
        return game
    lineups = parse_game_lineups(game_pk=game.game_pk, payload=payload)
    return replace(
        game,
        away_pitching=_parse_boxscore_side(teams.get("away"), team=game.away_team),
        home_pitching=_parse_boxscore_side(teams.get("home"), team=game.home_team),
        away_lineup_ids=tuple(player.player_id for player in lineups.away.players),
        home_lineup_ids=tuple(player.player_id for player in lineups.home.players),
        away_lineup_names=tuple(player.full_name for player in lineups.away.players),
        home_lineup_names=tuple(player.full_name for player in lineups.home.players),
        away_lineup_slots=tuple(player.batting_slot for player in lineups.away.players),
        home_lineup_slots=tuple(player.batting_slot for player in lineups.home.players),
        away_lineup_positions=tuple(player.position for player in lineups.away.players),
        home_lineup_positions=tuple(player.position for player in lineups.home.players),
        away_lineup_status=lineups.away.status.value,
        home_lineup_status=lineups.home.status.value,
        away_lineup_hash=lineups.away.source_hash,
        home_lineup_hash=lineups.home.source_hash,
        away_batting=_parse_team_batting(teams.get("away")),
        home_batting=_parse_team_batting(teams.get("home")),
    )


def pitching_cache_dir() -> Path:
    return Path(os.environ.get("MONEYBALL_PITCHING_CACHE", "data/mlb_boxscores"))


def cached_boxscore_path(game_pk: int, *, cache_dir: Path | None = None) -> Path:
    root = cache_dir or pitching_cache_dir()
    return root / f"{game_pk}.json"


def enrich_games_from_boxscore_cache(
    games: list[MlbGameState],
    *,
    cache_dir: Path | None = None,
) -> list[MlbGameState]:
    enriched: list[MlbGameState] = []
    for game in games:
        path = cached_boxscore_path(game.game_pk, cache_dir=cache_dir)
        if not path.exists():
            enriched.append(game)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            enriched.append(game)
            continue
        enriched.append(attach_boxscore_payload(game, payload))
    return enriched


async def fetch_mlb_boxscore(
    client: httpx.AsyncClient,
    game_pk: int,
) -> object:
    response = await client.get(f"{MLB_STATS_BASE_URL}/game/{game_pk}/boxscore")
    response.raise_for_status()
    return response.json()


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


async def fetch_mlb_regular_season_schedule(
    client: httpx.AsyncClient,
    *,
    season: int,
    start_date: date,
    end_date: date,
) -> list[MlbGameState]:
    """Fetch regular-season MLB games for walk-forward validation."""
    response = await client.get(
        f"{MLB_STATS_BASE_URL}/schedule",
        params={
            "sportId": 1,
            "season": season,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "gameTypes": "R",
            "hydrate": "team,probablePitcher,linescore",
        },
    )
    response.raise_for_status()
    games = _parse_schedule_payload(response.json())
    regular = [game for game in games if game.game_type in {None, "R"}]
    return enrich_games_from_boxscore_cache(regular)
