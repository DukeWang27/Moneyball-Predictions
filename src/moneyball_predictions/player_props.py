"""Live Polymarket MLB pitcher-strikeout prop research.

The module is read-only.  It discovers active strikeout props, resolves the pitcher to
an official MLB player, builds a transparent Poisson baseline, and prices YES/NO at
the full executable ask-book VWAP for the requested research stake.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal
from zoneinfo import ZoneInfo

import httpx
from rapidfuzz import fuzz, process
from scipy.stats import poisson

from .execution import quote_buy_by_dollars, slippage_adjusted_metrics
from .mlb import (
    MLB_STATS_BASE_URL,
    MlbGameState,
    attach_boxscore_payload,
    fetch_mlb_boxscore,
    fetch_mlb_game,
    fetch_mlb_schedule,
)
from .polymarket import GAMMA_BASE_URL, fetch_order_book_tops
from .schemas import (
    PropBoardResponse,
    PropSettlementResponse,
    PropSideQuote,
    StrikeoutPropPrediction,
)

MODEL_VERSION = "v0.12.1-poisson-k"
EASTERN = ZoneInfo("America/New_York")
LEAGUE_K_RATE = 0.225
LEAGUE_REACH_RATE = 0.315
PITCHER_PRIOR_BF = 120.0
LINEUP_PRIOR_PA = 180.0
STARTER_PRIOR_STARTS = 5.0
STARTER_PRIOR_IP = 5.2


class PlayerPropError(RuntimeError):
    """Raised when the live player-prop board cannot be produced."""


@dataclass(frozen=True)
class ParsedStrikeoutMarket:
    market_id: str
    condition_id: str | None
    question: str
    slug: str
    player_name: str
    threshold: int
    start_time: str | None
    yes_token_id: str
    no_token_id: str
    volume: float | None
    liquidity: float | None

    @property
    def polymarket_url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"


@dataclass(frozen=True)
class PlayerIdentity:
    player_id: int
    full_name: str
    position: str | None = None


@dataclass(frozen=True)
class PitcherProjectionInputs:
    k_rate: float
    projected_innings: float
    batters_faced: int
    starts: int


@dataclass(frozen=True)
class LineupProjectionInputs:
    k_rate: float
    reach_rate: float
    confirmed: bool
    hitters_used: int


def normalize_person_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized.casefold())
    return " ".join(normalized.split())


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


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _threshold_from_line(raw_line: float, *, plus_notation: bool) -> int:
    if plus_notation and raw_line.is_integer():
        return max(1, int(raw_line))
    # "Over 6.5" and line=6.5 both mean seven or more.
    return max(1, math.floor(raw_line) + 1)


_PROP_PATTERNS = (
    re.compile(
        r"will\s+(?P<name>.+?)\s+(?:record|have|get)\s+(?P<line>\d+(?:\.5)?)\+?\s+"
        r"(?:strikeouts?|ks?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>.+?)\s+(?:to record\s+)?(?:over|at least)\s+(?P<line>\d+(?:\.5)?)\s+"
        r"(?:strikeouts?|ks?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>.+?)\s+(?:strikeouts?|ks?)\s+(?:over|o)\s*(?P<line>\d+(?:\.5)?)",
        re.IGNORECASE,
    ),
)


def parse_strikeout_market(market: dict[str, Any]) -> ParsedStrikeoutMarket | None:
    """Parse common Polymarket strikeout-prop question shapes defensively."""
    if market.get("closed") or not market.get("active", True):
        return None

    question = str(market.get("question") or "").strip()
    market_type = str(market.get("sportsMarketType") or "").casefold()
    group_title = str(market.get("groupItemTitle") or "").strip()
    searchable = " ".join((question, market_type, group_title)).casefold()
    if "strikeout" not in searchable and not re.search(r"\bks?\b", searchable):
        return None

    player_name: str | None = None
    threshold: int | None = None
    for pattern in _PROP_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        player_name = match.group("name").strip(" :-")
        raw = float(match.group("line"))
        plus = "+" in match.group(0) and ".5" not in match.group("line")
        threshold = _threshold_from_line(raw, plus_notation=plus)
        break

    if threshold is None:
        raw_line = _safe_float(market.get("line"))
        if raw_line is not None:
            threshold = _threshold_from_line(raw_line, plus_notation=False)

    if not player_name and group_title:
        candidate = re.sub(r"\b(?:strikeouts?|ks?|over|under)\b.*$", "", group_title, flags=re.I)
        candidate = candidate.strip(" :-")
        if candidate:
            player_name = candidate

    if not player_name or threshold is None:
        return None

    outcomes = _parse_json_list(market.get("outcomes"))
    tokens = _parse_json_list(market.get("clobTokenIds"))
    if len(outcomes) != len(tokens) or len(tokens) < 2:
        return None

    token_by_outcome = {
        outcome.strip().casefold(): token
        for outcome, token in zip(outcomes, tokens, strict=True)
    }
    yes_token = token_by_outcome.get("yes") or token_by_outcome.get("over")
    no_token = token_by_outcome.get("no") or token_by_outcome.get("under")
    if not yes_token or not no_token:
        return None

    market_id = str(market.get("id") or "").strip()
    slug = str(market.get("slug") or market_id).strip()
    if not market_id or not slug:
        return None

    return ParsedStrikeoutMarket(
        market_id=market_id,
        condition_id=str(market.get("conditionId") or "").strip() or None,
        question=question,
        slug=slug,
        player_name=player_name,
        threshold=threshold,
        start_time=(
            str(market.get("eventStartTime") or market.get("gameStartTime") or "").strip()
            or None
        ),
        yes_token_id=yes_token,
        no_token_id=no_token,
        volume=_safe_float(market.get("volume")),
        liquidity=_safe_float(market.get("liquidity")),
    )


async def _valid_strikeout_market_types(client: httpx.AsyncClient) -> list[str]:
    try:
        response = await client.get(f"{GAMMA_BASE_URL}/sports/market-types")
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError:
        return []
    if isinstance(payload, dict):
        values = payload.get("marketTypes") or payload.get("market_types") or []
    else:
        values = payload
    if not isinstance(values, list):
        return []
    return [
        str(value)
        for value in values
        if "strikeout" in str(value).casefold() or "pitcher_k" in str(value).casefold()
    ]


async def fetch_active_strikeout_markets(
    client: httpx.AsyncClient,
    *,
    max_pages: int = 6,
    page_size: int = 500,
) -> list[ParsedStrikeoutMarket]:
    """Discover active strikeout props without assuming one permanent type name."""
    parsed: dict[str, ParsedStrikeoutMarket] = {}
    market_types = await _valid_strikeout_market_types(client)
    parameter_sets: list[dict[str, Any]] = [
        {"sports_market_types": market_type}
        for market_type in market_types
    ]
    # Defensive fallback: scan active markets when metadata does not expose a strikeout type.
    parameter_sets.append({})
    for extra_params in parameter_sets:
        for page in range(max_pages):
            response = await client.get(
                f"{GAMMA_BASE_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": page * page_size,
                    **extra_params,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise PlayerPropError("Polymarket markets response had an unexpected shape")
            for raw in payload:
                if not isinstance(raw, dict):
                    continue
                market = parse_strikeout_market(raw)
                if market:
                    parsed[market.market_id] = market
            if len(payload) < page_size:
                break
        if parsed and extra_params:
            break
    return list(parsed.values())


class MlbPlayerDirectory:
    """Point-in-time MLB player-name directory with a persistent local cache."""

    def __init__(self, *, cache_dir: Path = Path("data/player_cache")) -> None:
        self.cache_dir = cache_dir
        self._players: dict[str, PlayerIdentity] = {}

    def _cache_path(self, season: int) -> Path:
        return self.cache_dir / f"mlb_players_{season}.json"

    def _load_payload(self, payload: dict[str, Any]) -> None:
        players: dict[str, PlayerIdentity] = {}
        for person in payload.get("people", []):
            if not isinstance(person, dict) or person.get("id") is None:
                continue
            identity = PlayerIdentity(
                player_id=int(person["id"]),
                full_name=str(person.get("fullName") or person.get("firstLastName") or "").strip(),
                position=(
                    str((person.get("primaryPosition") or {}).get("abbreviation") or "").strip()
                    or None
                ),
            )
            if not identity.full_name:
                continue
            aliases = {
                person.get("fullName"),
                person.get("firstLastName"),
                person.get("boxscoreName"),
                person.get("useName"),
                identity.full_name,
            }
            for alias in aliases:
                if alias:
                    players[normalize_person_name(str(alias))] = identity
        self._players = players

    async def ensure_loaded(self, client: httpx.AsyncClient, season: int) -> None:
        path = self._cache_path(season)
        if path.exists():
            try:
                self._load_payload(json.loads(path.read_text(encoding="utf-8")))
                if self._players:
                    return
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        response = await client.get(
            f"{MLB_STATS_BASE_URL}/sports/1/players",
            params={"season": season},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise PlayerPropError("MLB player directory had an unexpected shape")
        self._load_payload(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def resolve(self, name: str, *, minimum_score: float = 94.0) -> PlayerIdentity:
        normalized = normalize_person_name(name)
        exact = self._players.get(normalized)
        if exact:
            return exact
        result = process.extractOne(normalized, self._players.keys(), scorer=fuzz.WRatio)
        if result is None:
            raise LookupError(f"No MLB player match for {name!r}")
        matched, score, _ = result
        if score < minimum_score:
            raise LookupError(f"Ambiguous MLB player match for {name!r}: {matched!r} ({score:.1f})")
        return self._players[matched]


class MlbStatsProjectionCache:
    """Caches current-season MLB stats during one prop-board refresh."""

    def __init__(self, client: httpx.AsyncClient, season: int) -> None:
        self.client = client
        self.season = season
        self._cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def player_stats(self, player_id: int, group: Literal["pitching", "hitting"]) -> dict[str, Any]:
        key = (player_id, group)
        async with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        response = await self.client.get(
            f"{MLB_STATS_BASE_URL}/people/{player_id}/stats",
            params={"stats": "season", "group": group, "season": self.season},
        )
        response.raise_for_status()
        payload = response.json()
        stat: dict[str, Any] = {}
        try:
            raw = payload["stats"][0]["splits"][0]["stat"]
            if isinstance(raw, dict):
                stat = raw
        except (KeyError, IndexError, TypeError):
            stat = {}
        async with self._lock:
            self._cache[key] = stat
        return stat


def _innings_value(value: Any) -> float:
    text = str(value or "0").strip()
    whole_text, _, fraction_text = text.partition(".")
    try:
        whole = int(whole_text or 0)
        fraction = int(fraction_text[:1] or 0)
    except ValueError:
        return 0.0
    if fraction not in {0, 1, 2}:
        return max(_safe_float(value) or 0.0, 0.0)
    return max(whole + fraction / 3.0, 0.0)


def build_pitcher_projection(stat: dict[str, Any]) -> PitcherProjectionInputs:
    batters_faced = int(_safe_float(stat.get("battersFaced")) or 0)
    strikeouts = int(_safe_float(stat.get("strikeOuts")) or 0)
    starts = int(_safe_float(stat.get("gamesStarted")) or 0)
    innings = _innings_value(stat.get("inningsPitched"))
    k_rate = (strikeouts + PITCHER_PRIOR_BF * LEAGUE_K_RATE) / (
        batters_faced + PITCHER_PRIOR_BF
    )
    projected_innings = (innings + STARTER_PRIOR_STARTS * STARTER_PRIOR_IP) / (
        starts + STARTER_PRIOR_STARTS
    )
    return PitcherProjectionInputs(
        k_rate=min(max(k_rate, 0.05), 0.50),
        projected_innings=min(max(projected_innings, 3.0), 7.5),
        batters_faced=batters_faced,
        starts=starts,
    )


async def build_lineup_projection(
    stats_cache: MlbStatsProjectionCache,
    hitter_ids: Iterable[int],
) -> LineupProjectionInputs:
    ids = tuple(hitter_ids)
    if len(ids) != 9:
        return LineupProjectionInputs(
            k_rate=LEAGUE_K_RATE,
            reach_rate=LEAGUE_REACH_RATE,
            confirmed=False,
            hitters_used=0,
        )
    stats = await asyncio.gather(
        *(stats_cache.player_stats(player_id, "hitting") for player_id in ids),
        return_exceptions=True,
    )
    total_pa = total_k = total_reach = 0.0
    hitters_used = 0
    slot_weights = (1.08, 1.06, 1.04, 1.03, 1.01, 0.99, 0.97, 0.94, 0.91)
    for weight, raw in zip(slot_weights, stats, strict=True):
        if isinstance(raw, Exception) or not raw:
            continue
        pa = _safe_float(raw.get("plateAppearances")) or 0.0
        if pa <= 0:
            continue
        strikeouts = _safe_float(raw.get("strikeOuts")) or 0.0
        hits = _safe_float(raw.get("hits")) or 0.0
        walks = _safe_float(raw.get("baseOnBalls")) or 0.0
        hbp = _safe_float(raw.get("hitByPitch")) or 0.0
        shrunk_k = (strikeouts + LINEUP_PRIOR_PA * LEAGUE_K_RATE) / (pa + LINEUP_PRIOR_PA)
        raw_reach = min(max((hits + walks + hbp) / pa, 0.10), 0.60)
        shrunk_reach = (pa * raw_reach + LINEUP_PRIOR_PA * LEAGUE_REACH_RATE) / (
            pa + LINEUP_PRIOR_PA
        )
        total_pa += weight
        total_k += weight * shrunk_k
        total_reach += weight * shrunk_reach
        hitters_used += 1
    if total_pa <= 0:
        return LineupProjectionInputs(
            k_rate=LEAGUE_K_RATE,
            reach_rate=LEAGUE_REACH_RATE,
            confirmed=False,
            hitters_used=0,
        )
    return LineupProjectionInputs(
        k_rate=min(max(total_k / total_pa, 0.10), 0.40),
        reach_rate=min(max(total_reach / total_pa, 0.20), 0.45),
        confirmed=hitters_used == 9,
        hitters_used=hitters_used,
    )


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def project_strikeout_probability(
    *,
    threshold: int,
    pitcher_k_rate: float,
    lineup_k_rate: float,
    league_k_rate: float,
    projected_innings: float,
    opponent_reach_rate: float,
) -> tuple[float, float, float, float]:
    """Return matchup K%, expected BF, expected K, and P(K >= threshold)."""
    matchup_k_rate = _logistic(
        _logit(pitcher_k_rate) + _logit(lineup_k_rate) - _logit(league_k_rate)
    )
    reach_rate = min(max(opponent_reach_rate, 0.15), 0.50)
    expected_batters_faced = projected_innings * 3.0 / (1.0 - reach_rate)
    expected_strikeouts = expected_batters_faced * matchup_k_rate
    probability = float(poisson.sf(threshold - 1, mu=expected_strikeouts))
    return matchup_k_rate, expected_batters_faced, expected_strikeouts, probability


def _parse_start(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _find_game_for_pitcher(
    market: ParsedStrikeoutMarket,
    identity: PlayerIdentity,
    schedule: list[MlbGameState],
) -> tuple[MlbGameState, Literal["away", "home"]] | None:
    market_time = _parse_start(market.start_time)
    candidates: list[tuple[float, MlbGameState, Literal["away", "home"]]] = []
    target_name = normalize_person_name(identity.full_name)
    for game in schedule:
        sides: tuple[tuple[Literal["away", "home"], int | None, str | None], ...] = (
            ("away", game.away_probable_pitcher_id, game.away_probable_pitcher),
            ("home", game.home_probable_pitcher_id, game.home_probable_pitcher),
        )
        for side, player_id, name in sides:
            if player_id != identity.player_id and normalize_person_name(name or "") != target_name:
                continue
            game_time = _parse_start(game.game_date)
            distance = abs((game_time - market_time).total_seconds()) if game_time and market_time else 0.0
            candidates.append((distance, game, side))
    if not candidates:
        return None
    _, game, side = min(candidates, key=lambda item: item[0])
    return game, side


def _prop_side_quote(
    *,
    side: Literal["YES", "NO"],
    token_id: str,
    probability: float,
    asks: tuple[tuple[float, float], ...],
    stake_dollars: float,
) -> PropSideQuote:
    quote = quote_buy_by_dollars(
        {"asks": [{"price": price, "size": size} for price, size in asks]},
        stake_dollars,
    )
    if quote.acquired_shares <= 0:
        return PropSideQuote(
            side=side,
            token_id=token_id,
            model_probability=probability,
            executable_price=None,
            edge=None,
            expected_roi=None,
            fully_fillable=False,
            levels_consumed=0,
        )
    metrics = slippage_adjusted_metrics(model_probability=probability, quote=quote)
    return PropSideQuote(
        side=side,
        token_id=token_id,
        model_probability=probability,
        executable_price=metrics["all_in_cost_per_share"],
        edge=metrics["edge"],
        expected_roi=metrics["expected_roi"],
        fully_fillable=bool(metrics["fully_fillable"]),
        levels_consumed=quote.levels_consumed,
    )


def _signal_for_quotes(
    yes: PropSideQuote,
    no: PropSideQuote,
) -> tuple[Literal["BET", "LEAN", "PASS"], Literal["YES", "NO"] | None]:
    candidates = [quote for quote in (yes, no) if quote.edge is not None]
    if not candidates:
        return "PASS", None
    selected = max(candidates, key=lambda quote: (quote.edge or -1, quote.expected_roi or -1))
    if (
        selected.fully_fillable
        and (selected.edge or 0.0) >= 0.05
        and (selected.expected_roi or 0.0) >= 0.05
    ):
        return "BET", selected.side
    if (selected.edge or 0.0) > 0:
        return "LEAN", selected.side
    return "PASS", None


async def _enrich_upcoming_lineups(
    client: httpx.AsyncClient,
    schedule: list[MlbGameState],
) -> list[MlbGameState]:
    payloads = await asyncio.gather(
        *(fetch_mlb_boxscore(client, game.game_pk) for game in schedule),
        return_exceptions=True,
    )
    return [
        attach_boxscore_payload(game, payload) if not isinstance(payload, Exception) else game
        for game, payload in zip(schedule, payloads, strict=True)
    ]


async def build_strikeout_prop_board(
    *,
    stake_dollars: float = 50.0,
    season: int | None = None,
    days: int = 3,
) -> PropBoardResponse:
    """Build a read-only, slippage-adjusted pitcher strikeout prop board."""
    now = datetime.now(UTC)
    resolved_season = season or now.year
    start_date = now.astimezone(EASTERN).date()
    end_date = start_date + timedelta(days=max(days, 1) - 1)
    diagnostics: dict[str, int] = {
        "discovered": 0,
        "mapped_players": 0,
        "matched_games": 0,
        "priced_markets": 0,
        "skipped_unmapped": 0,
        "skipped_no_game": 0,
        "skipped_no_book": 0,
        "skipped_started": 0,
    }
    timeout = httpx.Timeout(45.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.12.0 prop-research"}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            markets, schedule = await asyncio.gather(
                fetch_active_strikeout_markets(client),
                fetch_mlb_schedule(client, start_date, end_date),
            )
            diagnostics["discovered"] = len(markets)
            if not markets:
                return PropBoardResponse(
                    generated_at=now,
                    season=resolved_season,
                    stake_dollars=stake_dollars,
                    model_version=MODEL_VERSION,
                    props=[],
                    diagnostics=diagnostics,
                )
            schedule = await _enrich_upcoming_lineups(client, schedule)
            directory = MlbPlayerDirectory()
            await directory.ensure_loaded(client, resolved_season)
            stats_cache = MlbStatsProjectionCache(client, resolved_season)

            mapped: list[tuple[ParsedStrikeoutMarket, PlayerIdentity, MlbGameState, str]] = []
            for market in markets:
                try:
                    identity = directory.resolve(market.player_name)
                except LookupError:
                    diagnostics["skipped_unmapped"] += 1
                    continue
                diagnostics["mapped_players"] += 1
                match = _find_game_for_pitcher(market, identity, schedule)
                if not match:
                    diagnostics["skipped_no_game"] += 1
                    continue
                game, side = match
                game_start = _parse_start(game.game_date)
                if not game.is_pregame or (game_start is not None and game_start <= now):
                    diagnostics["skipped_started"] += 1
                    continue
                diagnostics["matched_games"] += 1
                mapped.append((market, identity, game, side))

            token_ids = [
                token
                for market, _, _, _ in mapped
                for token in (market.yes_token_id, market.no_token_id)
            ]
            books = await fetch_order_book_tops(client, token_ids)
            predictions: list[StrikeoutPropPrediction] = []
            for market, identity, game, side in mapped:
                yes_book = books.get(market.yes_token_id)
                no_book = books.get(market.no_token_id)
                if not yes_book or not no_book:
                    diagnostics["skipped_no_book"] += 1
                    continue
                pitcher_stat = await stats_cache.player_stats(identity.player_id, "pitching")
                pitcher = build_pitcher_projection(pitcher_stat)
                opponent_lineup_ids = game.home_lineup_ids if side == "away" else game.away_lineup_ids
                lineup = await build_lineup_projection(stats_cache, opponent_lineup_ids)
                matchup_k, expected_bf, expected_k, probability_yes = project_strikeout_probability(
                    threshold=market.threshold,
                    pitcher_k_rate=pitcher.k_rate,
                    lineup_k_rate=lineup.k_rate,
                    league_k_rate=LEAGUE_K_RATE,
                    projected_innings=pitcher.projected_innings,
                    opponent_reach_rate=lineup.reach_rate,
                )
                probability_no = 1.0 - probability_yes
                yes_quote = _prop_side_quote(
                    side="YES",
                    token_id=market.yes_token_id,
                    probability=probability_yes,
                    asks=yes_book.asks,
                    stake_dollars=stake_dollars,
                )
                no_quote = _prop_side_quote(
                    side="NO",
                    token_id=market.no_token_id,
                    probability=probability_no,
                    asks=no_book.asks,
                    stake_dollars=stake_dollars,
                )
                signal, recommended_side = _signal_for_quotes(yes_quote, no_quote)
                opponent = game.home_team if side == "away" else game.away_team
                predictions.append(
                    StrikeoutPropPrediction(
                        market_id=market.market_id,
                        question=market.question,
                        player_id=identity.player_id,
                        player_name=identity.full_name,
                        game_pk=game.game_pk,
                        opponent=opponent,
                        start_time=game.game_date,
                        threshold=market.threshold,
                        projected_innings=pitcher.projected_innings,
                        expected_batters_faced=expected_bf,
                        pitcher_k_rate=pitcher.k_rate,
                        opponent_lineup_k_rate=lineup.k_rate,
                        matchup_k_rate=matchup_k,
                        expected_strikeouts=expected_k,
                        probability_yes=probability_yes,
                        probability_no=probability_no,
                        yes=yes_quote,
                        no=no_quote,
                        recommended_side=recommended_side,
                        signal=signal,
                        lineup_status=(
                            "CONFIRMED" if lineup.confirmed else "LEAGUE_FALLBACK"
                        ),
                        hitters_used=lineup.hitters_used,
                        polymarket_url=market.polymarket_url,
                        volume=market.volume,
                        liquidity=market.liquidity,
                        model_version=MODEL_VERSION,
                    )
                )
                diagnostics["priced_markets"] += 1
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        raise PlayerPropError(str(exc)) from exc

    signal_rank = {"BET": 2, "LEAN": 1, "PASS": 0}
    predictions.sort(
        key=lambda prop: (
            signal_rank[prop.signal],
            max(prop.yes.edge or -1.0, prop.no.edge or -1.0),
        ),
        reverse=True,
    )
    return PropBoardResponse(
        generated_at=now,
        season=resolved_season,
        stake_dollars=stake_dollars,
        model_version=MODEL_VERSION,
        props=predictions,
        diagnostics=diagnostics,
    )


def settle_strikeout_selection(
    *,
    strikeouts: int,
    threshold: int,
    side: Literal["YES", "NO"],
) -> bool:
    """Return whether a binary strikeout-prop selection won."""
    if strikeouts < 0:
        raise ValueError("strikeouts cannot be negative")
    if threshold < 1:
        raise ValueError("threshold must be positive")
    over_hit = strikeouts >= threshold
    return over_hit if side == "YES" else not over_hit


async def build_strikeout_prop_settlement(
    *,
    game_pk: int,
    player_id: int,
    threshold: int,
    side: Literal["YES", "NO"],
) -> PropSettlementResponse:
    """Grade one browser paper prop from the official MLB game box score."""
    timeout = httpx.Timeout(20.0, connect=8.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.12.1 prop-settlement"}
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            game = await fetch_mlb_game(client, game_pk)
            if game is None:
                raise PlayerPropError(f"MLB game {game_pk} was not found")

            detail = game.detailed_state.casefold()
            if "cancel" in detail or "postpon" in detail:
                return PropSettlementResponse(
                    game_pk=game_pk,
                    player_id=player_id,
                    threshold=threshold,
                    side=side,
                    game_status=game.detailed_state,
                    strikeouts=None,
                    status="refunded",
                    won=None,
                )

            if not game.is_final:
                return PropSettlementResponse(
                    game_pk=game_pk,
                    player_id=player_id,
                    threshold=threshold,
                    side=side,
                    game_status=game.detailed_state,
                    strikeouts=None,
                    status="open",
                    won=None,
                )

            payload = await fetch_mlb_boxscore(client, game_pk)
            enriched = attach_boxscore_payload(game, payload)
            line = next(
                (
                    item
                    for item in (*enriched.away_pitching, *enriched.home_pitching)
                    if item.pitcher_id == player_id
                ),
                None,
            )
            if line is None:
                return PropSettlementResponse(
                    game_pk=game_pk,
                    player_id=player_id,
                    threshold=threshold,
                    side=side,
                    game_status=game.detailed_state,
                    strikeouts=None,
                    status="refunded",
                    won=None,
                )

            won = settle_strikeout_selection(
                strikeouts=line.strikeouts,
                threshold=threshold,
                side=side,
            )
            return PropSettlementResponse(
                game_pk=game_pk,
                player_id=player_id,
                threshold=threshold,
                side=side,
                game_status=game.detailed_state,
                strikeouts=line.strikeouts,
                status="won" if won else "lost",
                won=won,
            )
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        raise PlayerPropError(str(exc)) from exc
