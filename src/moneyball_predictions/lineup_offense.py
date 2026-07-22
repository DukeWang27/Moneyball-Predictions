"""Confirmed-lineup offense projection and conservative moneyline repricing.

This is a point-in-time T1H overlay.  It compares each announced nine-man lineup
with that club's own season offense against the same opposing pitcher hand.  The
delta avoids crediting an elite club twice merely because its normal hitters are
already embedded in the team-strength baseline.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx

from .mlb import MLB_STATS_BASE_URL, MlbGameState
from .sabermetrics import BatterEventProjection, LineupProjection, project_lineup

# Stable 2026 research constants.  Version these instead of changing them silently.
WOBA_WEIGHTS = {
    "ubb": 0.690,
    "hbp": 0.720,
    "single": 0.880,
    "double": 1.247,
    "triple": 1.578,
    "home_run": 2.031,
}
LINEAR_RUN_VALUES = {
    "ubb": 0.33,
    "hbp": 0.34,
    "single": 0.47,
    "double": 0.78,
    "triple": 1.09,
    "home_run": 1.40,
}
LEAGUE_EVENT_RATES = {
    "ubb": 0.079,
    "hbp": 0.011,
    "single": 0.141,
    "double": 0.045,
    "triple": 0.004,
    "home_run": 0.032,
}
LINEUP_PRIOR_PA = 180.0
TEAM_PRIOR_PA = 450.0
SLOT_WEIGHTS = (1.08, 1.06, 1.04, 1.03, 1.01, 0.99, 0.97, 0.94, 0.91)
CACHE_TTL = timedelta(minutes=30)


@dataclass(frozen=True)
class HittingProjection:
    xwoba: float
    linear_runs_per_pa: float
    plate_appearances: float
    data_available: bool
    split_available: bool


@dataclass(frozen=True)
class ConfirmedLineupAdjustment:
    home_lineup_xwoba: float
    away_lineup_xwoba: float
    home_team_baseline_xwoba: float
    away_team_baseline_xwoba: float
    home_lineup_delta: float
    away_lineup_delta: float
    lineup_advantage: float
    logit_adjustment: float
    home_probability_before: float
    home_probability_after: float
    home_pitcher_hand: str
    away_pitcher_hand: str
    hitters_with_stats: int
    used_handedness_splits: bool

    @property
    def probability_change(self) -> float:
        return self.home_probability_after - self.home_probability_before


@dataclass(frozen=True)
class _CacheValue:
    expires_at: datetime
    value: dict[str, Any]
    split_available: bool


_STAT_CACHE: dict[tuple[str, int, int, str | None], _CacheValue] = {}
_HAND_CACHE: dict[int, tuple[datetime, str]] = {}
_ADJUSTMENT_CACHE: dict[tuple[object, ...], tuple[datetime, ConfirmedLineupAdjustment]] = {}
_CACHE_LOCK = asyncio.Lock()


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _extract_stat(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    stats = payload.get("stats")
    if not isinstance(stats, list):
        return {}
    for block in stats:
        if not isinstance(block, dict):
            continue
        splits = block.get("splits")
        if not isinstance(splits, list):
            continue
        for split in splits:
            if not isinstance(split, dict):
                continue
            stat = split.get("stat")
            if isinstance(stat, dict):
                return stat
    return {}


def _event_counts(stat: dict[str, Any]) -> tuple[float, dict[str, float]]:
    at_bats = _safe_float(stat.get("atBats"))
    hits = _safe_float(stat.get("hits"))
    doubles = _safe_float(stat.get("doubles"))
    triples = _safe_float(stat.get("triples"))
    home_runs = _safe_float(stat.get("homeRuns"))
    walks = _safe_float(stat.get("baseOnBalls"))
    intentional_walks = _safe_float(stat.get("intentionalWalks"))
    hit_by_pitch = _safe_float(stat.get("hitByPitch"))
    sacrifice_flies = _safe_float(stat.get("sacFlies"))
    plate_appearances = _safe_float(stat.get("plateAppearances"))
    if plate_appearances <= 0:
        plate_appearances = at_bats + walks + hit_by_pitch + sacrifice_flies
    singles = max(0.0, hits - doubles - triples - home_runs)
    return plate_appearances, {
        "ubb": max(0.0, walks - intentional_walks),
        "hbp": max(0.0, hit_by_pitch),
        "single": singles,
        "double": max(0.0, doubles),
        "triple": max(0.0, triples),
        "home_run": max(0.0, home_runs),
    }


def _event_rates(
    stat: dict[str, Any],
    *,
    prior_pa: float,
) -> tuple[float, dict[str, float], bool]:
    plate_appearances, counts = _event_counts(stat)
    denominator = plate_appearances + prior_pa
    rates = {
        event: (counts[event] + prior_pa * LEAGUE_EVENT_RATES[event]) / denominator
        for event in LEAGUE_EVENT_RATES
    }
    return plate_appearances, rates, plate_appearances > 0


def _projection_from_stat(stat: dict[str, Any], *, prior_pa: float) -> HittingProjection:
    pa, rates, available = _event_rates(stat, prior_pa=prior_pa)
    xwoba = sum(rates[event] * WOBA_WEIGHTS[event] for event in WOBA_WEIGHTS)
    runs = sum(rates[event] * LINEAR_RUN_VALUES[event] for event in LINEAR_RUN_VALUES)
    return HittingProjection(
        xwoba=xwoba,
        linear_runs_per_pa=runs,
        plate_appearances=pa,
        data_available=available,
        split_available=False,
    )


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def apply_lineup_logit_adjustment(
    home_probability: float,
    lineup_advantage: float,
    *,
    coefficient: float | None = None,
    maximum_logit_adjustment: float | None = None,
) -> tuple[float, float]:
    """Apply a bounded lineup-vs-team-baseline adjustment in log-odds space."""
    if not 0.0 <= home_probability <= 1.0:
        raise ValueError("home_probability must be in [0, 1]")
    resolved_coefficient = coefficient
    if resolved_coefficient is None:
        resolved_coefficient = float(os.environ.get("MONEYBALL_LINEUP_LOGIT_COEFFICIENT", "4.0"))
    resolved_cap = maximum_logit_adjustment
    if resolved_cap is None:
        resolved_cap = float(os.environ.get("MONEYBALL_LINEUP_LOGIT_CAP", "0.20"))
    adjustment = max(
        -abs(resolved_cap),
        min(abs(resolved_cap), resolved_coefficient * lineup_advantage),
    )
    return _logistic(_logit(home_probability) + adjustment), adjustment


async def _fetch_stat(
    client: httpx.AsyncClient,
    *,
    entity: Literal["people", "teams"],
    entity_id: int,
    season: int,
    sit_code: str | None,
) -> tuple[dict[str, Any], bool]:
    key = (entity, entity_id, season, sit_code)
    now = datetime.now(UTC)
    async with _CACHE_LOCK:
        cached = _STAT_CACHE.get(key)
        if cached and cached.expires_at > now:
            return cached.value, cached.split_available

    params: dict[str, str | int] = {
        "stats": "season",
        "group": "hitting",
        "season": season,
    }
    if sit_code:
        params["sitCodes"] = sit_code
    response = await client.get(f"{MLB_STATS_BASE_URL}/{entity}/{entity_id}/stats", params=params)
    response.raise_for_status()
    stat = _extract_stat(response.json())
    split_available = bool(stat) and bool(sit_code)

    if not stat and sit_code:
        # Some players or endpoints have no split record.  Fall back to the same
        # point-in-time season totals rather than inventing a split.
        stat, _ = await _fetch_stat(
            client,
            entity=entity,
            entity_id=entity_id,
            season=season,
            sit_code=None,
        )
        split_available = False

    async with _CACHE_LOCK:
        _STAT_CACHE[key] = _CacheValue(
            expires_at=now + CACHE_TTL,
            value=stat,
            split_available=split_available,
        )
    return stat, split_available


async def _pitcher_hand(client: httpx.AsyncClient, pitcher_id: int) -> str:
    now = datetime.now(UTC)
    async with _CACHE_LOCK:
        cached = _HAND_CACHE.get(pitcher_id)
        if cached and cached[0] > now:
            return cached[1]
    response = await client.get(f"{MLB_STATS_BASE_URL}/people/{pitcher_id}")
    response.raise_for_status()
    payload = response.json()
    people = payload.get("people") if isinstance(payload, dict) else None
    hand = "R"
    if isinstance(people, list) and people and isinstance(people[0], dict):
        pitch_hand = people[0].get("pitchHand") or {}
        if isinstance(pitch_hand, dict):
            candidate = str(pitch_hand.get("code") or "R").upper()
            if candidate in {"L", "R"}:
                hand = candidate
    async with _CACHE_LOCK:
        _HAND_CACHE[pitcher_id] = (now + CACHE_TTL, hand)
    return hand


async def _project_lineup(
    client: httpx.AsyncClient,
    *,
    player_ids: tuple[int, ...],
    opposing_pitcher_hand: str,
    season: int,
) -> tuple[LineupProjection, int, bool]:
    if len(player_ids) != 9:
        raise ValueError("Confirmed lineup must contain exactly nine hitters")
    sit_code = "vl" if opposing_pitcher_hand == "L" else "vr"
    fetched = await asyncio.gather(
        *(
            _fetch_stat(
                client,
                entity="people",
                entity_id=player_id,
                season=season,
                sit_code=sit_code,
            )
            for player_id in player_ids
        ),
        return_exceptions=True,
    )
    batters: list[BatterEventProjection] = []
    hitters_with_stats = 0
    all_split = True
    for slot, (player_id, expected_pa, result) in enumerate(
        zip(player_ids, SLOT_WEIGHTS, fetched, strict=True),
        start=1,
    ):
        if isinstance(result, Exception):
            stat: dict[str, Any] = {}
            split_available = False
        else:
            stat, split_available = result
        _, rates, available = _event_rates(stat, prior_pa=LINEUP_PRIOR_PA)
        hitters_with_stats += int(available)
        all_split = all_split and split_available
        batters.append(
            BatterEventProjection(
                player_id=player_id,
                batting_slot=slot,
                expected_pa=expected_pa,
                event_rates=rates,
            )
        )
    return (
        project_lineup(
            batters,
            woba_weights=WOBA_WEIGHTS,
            run_values=LINEAR_RUN_VALUES,
        ),
        hitters_with_stats,
        all_split,
    )


async def _team_baseline(
    client: httpx.AsyncClient,
    *,
    team_id: int,
    opposing_pitcher_hand: str,
    season: int,
) -> tuple[HittingProjection, bool]:
    sit_code = "vl" if opposing_pitcher_hand == "L" else "vr"
    stat, split_available = await _fetch_stat(
        client,
        entity="teams",
        entity_id=team_id,
        season=season,
        sit_code=sit_code,
    )
    projection = _projection_from_stat(stat, prior_pa=TEAM_PRIOR_PA)
    return (
        HittingProjection(
            xwoba=projection.xwoba,
            linear_runs_per_pa=projection.linear_runs_per_pa,
            plate_appearances=projection.plate_appearances,
            data_available=projection.data_available,
            split_available=split_available,
        ),
        split_available,
    )


async def build_confirmed_lineup_adjustment(
    client: httpx.AsyncClient,
    *,
    game: MlbGameState,
    season: int,
    home_probability: float,
) -> ConfirmedLineupAdjustment | None:
    """Reprice a home win probability after both official lineups are present."""
    if (
        len(game.away_lineup_ids) != 9
        or len(game.home_lineup_ids) != 9
        or game.away_probable_pitcher_id is None
        or game.home_probable_pitcher_id is None
        or game.away_team_id is None
        or game.home_team_id is None
    ):
        return None

    cache_key = (
        season,
        game.game_pk,
        game.away_lineup_hash,
        game.home_lineup_hash,
        game.away_probable_pitcher_id,
        game.home_probable_pitcher_id,
        round(home_probability, 8),
    )
    now = datetime.now(UTC)
    async with _CACHE_LOCK:
        cached = _ADJUSTMENT_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

    away_hand, home_hand = await asyncio.gather(
        _pitcher_hand(client, game.away_probable_pitcher_id),
        _pitcher_hand(client, game.home_probable_pitcher_id),
    )
    # Home hitters face the away starter; away hitters face the home starter.
    home_lineup_result, away_lineup_result, home_team_result, away_team_result = (
        await asyncio.gather(
            _project_lineup(
                client,
                player_ids=game.home_lineup_ids,
                opposing_pitcher_hand=away_hand,
                season=season,
            ),
            _project_lineup(
                client,
                player_ids=game.away_lineup_ids,
                opposing_pitcher_hand=home_hand,
                season=season,
            ),
            _team_baseline(
                client,
                team_id=game.home_team_id,
                opposing_pitcher_hand=away_hand,
                season=season,
            ),
            _team_baseline(
                client,
                team_id=game.away_team_id,
                opposing_pitcher_hand=home_hand,
                season=season,
            ),
        )
    )
    home_lineup, home_coverage, home_split = home_lineup_result
    away_lineup, away_coverage, away_split = away_lineup_result
    home_team, home_team_split = home_team_result
    away_team, away_team_split = away_team_result

    # Require enough real player records and team baselines before the lineups can
    # unlock an actionable bet.  Missing hitters still receive a league prior, but
    # too much fallback data should not masquerade as a confirmed T1H model.
    if (
        home_coverage < 7
        or away_coverage < 7
        or not home_team.data_available
        or not away_team.data_available
    ):
        return None

    home_delta = home_lineup.xwoba - home_team.xwoba
    away_delta = away_lineup.xwoba - away_team.xwoba
    lineup_advantage = home_delta - away_delta
    adjusted_probability, logit_adjustment = apply_lineup_logit_adjustment(
        home_probability,
        lineup_advantage,
    )
    result = ConfirmedLineupAdjustment(
        home_lineup_xwoba=home_lineup.xwoba,
        away_lineup_xwoba=away_lineup.xwoba,
        home_team_baseline_xwoba=home_team.xwoba,
        away_team_baseline_xwoba=away_team.xwoba,
        home_lineup_delta=home_delta,
        away_lineup_delta=away_delta,
        lineup_advantage=lineup_advantage,
        logit_adjustment=logit_adjustment,
        home_probability_before=home_probability,
        home_probability_after=adjusted_probability,
        home_pitcher_hand=home_hand,
        away_pitcher_hand=away_hand,
        hitters_with_stats=home_coverage + away_coverage,
        used_handedness_splits=(
            home_split and away_split and home_team_split and away_team_split
        ),
    )
    async with _CACHE_LOCK:
        _ADJUSTMENT_CACHE[cache_key] = (now + CACHE_TTL, result)
    return result
