"""Join MLB scores and season strength with Polymarket MLB moneylines."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from .lineup_offense import ConfirmedLineupAdjustment, build_confirmed_lineup_adjustment
from .lineups import LineupStatus, parse_game_lineups
from .market_archive import append_market_snapshots
from .reproducibility import sha256_json
from .runtime import running_on_vercel
from .serverless_artifacts import load_live_model_bundle
from .storage import insert_lineup_snapshot, insert_prediction
from .mlb import (
    MlbDataError,
    MlbGameState,
    attach_boxscore_payload,
    fetch_mlb_boxscore,
    fetch_mlb_game,
    fetch_mlb_regular_season_schedule,
    fetch_mlb_schedule,
    fetch_team_season_stats,
)
from .model import (
    add_home_field_advantage,
    log5_probability,
    regressed_pythagorean_expectation,
)
from .odds import prediction_market_expected_value, prediction_market_kelly_stake
from .optimized import (
    SeasonFeatureEngine,
    build_target_rows,
)
from .polymarket import (
    MarketDiagnostic,
    MlbMoneylineMarket,
    PolymarketDataError,
    fetch_mlb_moneyline_markets,
    fetch_order_book_tops,
)
from .schemas import (
    LiveGamePrediction,
    LiveMlbResponse,
    LiveMarketSide,
    MatchingDiagnostics,
    RecommendedBet,
    ScoreboardGame,
)
from .tournament import (
    CalibratedPredictor,
    ModelTournamentArtifact,
    prepare_model_tournament_multifold,
)

EASTERN = ZoneInfo("America/New_York")
MIN_BET_EDGE = 0.05
MIN_BET_ROI = 0.05
MODEL_CACHE_MINUTES = 20
MAX_PAPER_STAKE_FRACTION = 0.01
MIN_COMPONENT_COVERAGE = 0.70


@dataclass
class LiveOptimizedContext:
    artifact: ModelTournamentArtifact
    engine: SeasonFeatureEngine
    built_at: datetime


_LIVE_MODEL_CACHE: dict[int, LiveOptimizedContext] = {}


class LivePredictionError(RuntimeError):
    """Raised when a live prediction refresh cannot be completed."""


def _parse_start_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _market_side(
    team: str,
    model_probability: float,
    ask_price: float,
    bankroll: float,
    kelly_multiplier: float,
    ask_size: float | None,
    *,
    sabermetric_support: str = "UNKNOWN",
    sabermetric_support_count: int = 0,
    sabermetric_support_total: int = 0,
    sabermetric_reasons: list[str] | None = None,
) -> LiveMarketSide:
    risk_capped = (
        prediction_market_kelly_stake(
            model_probability=model_probability,
            buy_price=ask_price,
            bankroll=bankroll,
            multiplier=kelly_multiplier,
            top_ask_size=None,
        )[2]
        > MAX_PAPER_STAKE_FRACTION
    )
    kelly_stake, full_kelly, applied_kelly, capped_by_depth = (
        prediction_market_kelly_stake(
            model_probability=model_probability,
            buy_price=ask_price,
            bankroll=bankroll,
            multiplier=kelly_multiplier,
            top_ask_size=ask_size,
            max_bankroll_fraction=MAX_PAPER_STAKE_FRACTION,
        )
    )
    expected_roi = prediction_market_expected_value(model_probability, ask_price, 1.0)
    expected_value = prediction_market_expected_value(
        model_probability, ask_price, kelly_stake
    )
    shares = kelly_stake / ask_price if kelly_stake > 0 else 0.0
    return LiveMarketSide(
        team=team,
        model_probability=model_probability,
        market_buy_price=ask_price,
        edge=model_probability - ask_price,
        expected_value=expected_value,
        expected_roi=expected_roi,
        full_kelly_fraction=full_kelly,
        applied_kelly_fraction=applied_kelly,
        kelly_multiplier=kelly_multiplier,
        kelly_stake=kelly_stake,
        stake=kelly_stake,
        shares=shares,
        price_source="CLOB best ask",
        top_ask_size=ask_size,
        kelly_capped_by_depth=capped_by_depth,
        kelly_capped_by_risk=risk_capped,
        sabermetric_support=sabermetric_support,
        sabermetric_support_count=sabermetric_support_count,
        sabermetric_support_total=sabermetric_support_total,
        sabermetric_reasons=sabermetric_reasons or [],
        value_grade="PASS",
    )



def _sabermetric_support_for_team(
    *,
    team: str,
    game: MlbGameState,
    own_strength: float,
    opponent_strength: float,
    features: tuple[float, ...] | None,
) -> tuple[str, int, int, list[str]]:
    """Summarize whether independent baseball driver groups support a side.

    This is a strategy filter, not a second probability model. The model probability
    already contains the underlying statistics, so support is never added to edge.
    """
    is_home = team == game.home_team
    active: list[tuple[str, bool]] = []

    strength_gap = own_strength - opponent_strength
    if abs(strength_gap) >= 0.005:
        active.append(("Bill James team strength", strength_gap > 0))

    if features is not None:
        starter_home_advantage = features[3]
        starter_advantage = starter_home_advantage if is_home else -starter_home_advantage
        if abs(starter_advantage) >= 0.02:
            active.append(("starter components", starter_advantage > 0))

        park_home_advantage = features[9]
        park_advantage = park_home_advantage if is_home else -park_home_advantage
        if abs(park_advantage) >= 0.005:
            active.append(("park matchup", park_advantage > 0))

    support_count = sum(1 for _, supports in active if supports)
    total = len(active)
    reasons = [
        f"{name} {'supports' if supports else 'opposes'} {team}"
        for name, supports in active
    ]
    if total == 0:
        label = "UNKNOWN"
    elif total >= 2 and support_count == total:
        label = "CONFIRMED"
    elif support_count > 0:
        label = "MIXED"
    else:
        label = "CONTRARIAN"
    return label, support_count, total, reasons


def _value_grade(side: LiveMarketSide) -> str:
    if side.edge >= MIN_BET_EDGE and side.expected_roi >= MIN_BET_ROI:
        if side.sabermetric_support == "CONFIRMED":
            return "A"
        if side.sabermetric_support == "MIXED":
            return "B"
        return "C"
    if side.expected_roi > 0:
        return "LEAN"
    return "PASS"

def _same_matchup(market: MlbMoneylineMarket, game: MlbGameState) -> bool:
    return {market.team_a, market.team_b} == {game.away_team, game.home_team}


def _match_schedule_game(
    market: MlbMoneylineMarket,
    schedule: list[MlbGameState],
) -> MlbGameState | None:
    candidates = [game for game in schedule if _same_matchup(market, game)]
    if not candidates:
        return None

    market_time = _parse_start_time(market.start_time)
    if market_time is None:
        return candidates[0] if len(candidates) == 1 else None

    ranked = sorted(
        candidates,
        key=lambda game: abs(
            (_parse_start_time(game.game_date) or datetime.max.replace(tzinfo=UTC)) - market_time
        ),
    )
    best = ranked[0]
    best_time = _parse_start_time(best.game_date)
    if best_time is None:
        return None
    if abs(best_time - market_time) > timedelta(hours=18):
        return None
    return best


def paper_bet_eligibility(
    *,
    lineup_status: str,
    lineup_model_used: bool,
    starters_confirmed: bool,
    signal: str,
    edge: float,
    expected_roi: float,
) -> tuple[bool, str]:
    """Return the deterministic moneyline paper-bet gate and its reason."""
    if not starters_confirmed:
        return False, "Waiting for both probable starters"
    if lineup_status != LineupStatus.CONFIRMED.value:
        return False, "Waiting for both official starting lineups"
    if not lineup_model_used:
        return False, "Lineups are confirmed, but lineup offense repricing is unavailable"
    if signal != "BET":
        return False, "Current executable price does not qualify"
    if edge < MIN_BET_EDGE:
        return False, "Edge below 5% threshold"
    if expected_roi < MIN_BET_ROI:
        return False, "Expected ROI below 5% threshold"
    return True, "Eligible"


def _signal_for_sides(
    side_a: LiveMarketSide,
    side_b: LiveMarketSide,
) -> tuple[str, str | None, str]:
    side_a.value_grade = _value_grade(side_a)
    side_b.value_grade = _value_grade(side_b)
    best = max((side_a, side_b), key=lambda side: side.expected_roi)
    if best.edge >= MIN_BET_EDGE and best.expected_roi >= MIN_BET_ROI:
        return (
            "BET",
            best.team,
            (
                f"Paper-bet candidate: {best.team} has a {best.edge:.1%} model edge "
                f"and {best.expected_roi:.1%} expected ROI at the current ask."
            ),
        )
    if best.expected_roi > 0:
        return (
            "LEAN",
            best.team,
            (
                f"Lean {best.team}, but pass under the current 5% edge and 5% ROI "
                "paper-bet thresholds."
            ),
        )
    return (
        "PASS",
        None,
        "Pass: neither side has positive expected value at the executable asks.",
    )


def _diagnostic_summary(
    diagnostics: list[MarketDiagnostic],
    runtime_counts: Counter[str],
) -> MatchingDiagnostics:
    source_counts = Counter(item.category for item in diagnostics)
    examples: list[str] = []
    for item in diagnostics:
        if item.category in {"discovery_warning", "unmatched_teams", "no_moneyline"}:
            examples.append(f"{item.title}: {item.detail}")
        if len(examples) >= 8:
            break

    return MatchingDiagnostics(
        discovery_warnings=source_counts["discovery_warning"],
        unsupported_events=source_counts["unsupported_event"],
        no_moneyline=source_counts["no_moneyline"],
        unmatched_teams=source_counts["unmatched_teams"],
        missing_tokens=source_counts["missing_tokens"],
        no_schedule_match=runtime_counts["no_schedule_match"],
        outside_date_window=runtime_counts["outside_date_window"],
        no_executable_ask=runtime_counts["no_executable_ask"],
        examples=examples,
    )


def _scoreboard_row(
    game: MlbGameState,
    market: MlbMoneylineMarket | None,
) -> ScoreboardGame:
    if game.is_live:
        market_status = (
            "Matched Polymarket event; sports orders are cleared at first pitch"
            if market
            else "No matched Polymarket full-game moneyline"
        )
        live_bet_message = (
            "No live bet signal. The current model is pregame-only and does not use the "
            "score, inning, outs, baserunners, or current pitchers."
        )
    elif game.is_final:
        market_status = "Game complete"
        live_bet_message = "Final score available for paper-bet settlement."
    else:
        market_status = (
            "Pregame Polymarket moneyline matched"
            if market and market.accepting_orders
            else "No executable pregame Polymarket moneyline matched"
        )
        live_bet_message = "Use the pregame recommendation card when an executable ask exists."

    return ScoreboardGame(
        game_pk=game.game_pk,
        game_date=game.game_date,
        official_date=game.official_date,
        away_team=game.away_team,
        home_team=game.home_team,
        away_team_id=game.away_team_id,
        home_team_id=game.home_team_id,
        away_score=game.away_score,
        home_score=game.home_score,
        abstract_state=game.abstract_state,
        detailed_state=game.detailed_state,
        current_inning=game.current_inning,
        inning_state=game.inning_state,
        inning_ordinal=game.inning_ordinal,
        away_probable_pitcher=game.away_probable_pitcher,
        home_probable_pitcher=game.home_probable_pitcher,
        venue=game.venue,
        winner=game.winner,
        polymarket_url=market.polymarket_url if market else None,
        market_status=market_status,
        live_bet_message=live_bet_message,
    )



async def _attach_pregame_lineups(
    client: httpx.AsyncClient,
    schedule: list[MlbGameState],
) -> list[MlbGameState]:
    """Fetch, archive, and attach the current official lineup state for each pregame."""
    pregame = [game for game in schedule if game.is_pregame]
    if not pregame:
        return schedule
    payloads = await asyncio.gather(
        *(fetch_mlb_boxscore(client, game.game_pk) for game in pregame),
        return_exceptions=True,
    )
    enriched: dict[int, MlbGameState] = {}
    for game, payload in zip(pregame, payloads, strict=True):
        if isinstance(payload, Exception):
            continue
        try:
            lineups = parse_game_lineups(game_pk=game.game_pk, payload=payload)
            insert_lineup_snapshot(lineups.away)
            insert_lineup_snapshot(lineups.home)
        except (OSError, sqlite3.Error, TypeError, ValueError):
            # Snapshot storage must never make the live board unavailable.
            pass
        enriched[game.game_pk] = attach_boxscore_payload(game, payload)
    return [enriched.get(game.game_pk, game) for game in schedule]


async def _fetch_regular_year(
    client: httpx.AsyncClient,
    year: int,
    end_date: date | None = None,
) -> tuple[int, list[MlbGameState]]:
    return (
        year,
        await fetch_mlb_regular_season_schedule(
            client,
            season=year,
            start_date=date(year, 3, 1),
            end_date=end_date or date(year, 11, 15),
        ),
    )


async def _live_optimized_context(
    client: httpx.AsyncClient,
    *,
    season: int,
    through: date,
    min_games: int = 10,
) -> LiveOptimizedContext:
    cached = _LIVE_MODEL_CACHE.get(season)
    now = datetime.now(UTC)
    if cached and now - cached.built_at < timedelta(minutes=MODEL_CACHE_MINUTES):
        return cached

    if running_on_vercel():
        # Training the multi-season tournament during an HTTP request exceeds
        # Hobby's function budget. Load the frozen pre-deployment artifact and
        # only build the current-season feature state at request time.
        bundle = load_live_model_bundle()
        if bundle is None:
            raise LivePredictionError(
                "Serverless model artifact is missing; run scripts/export_serverless_artifacts.py"
            )
        artifact = bundle.get("artifact")
        prior_summary = bundle.get("prior_summary")
        prior_elo = bundle.get("prior_elo")
        if artifact is None or prior_summary is None or prior_elo is None:
            raise LivePredictionError("Serverless model artifact is incomplete")
        current_pair = await _fetch_regular_year(client, season, through)
    else:
        years = [season - 4, season - 3, season - 2, season - 1]
        historical_pairs, current_pair = await asyncio.gather(
            asyncio.gather(*(_fetch_regular_year(client, year) for year in years)),
            _fetch_regular_year(client, season, through),
        )
        historical = dict(historical_pairs)
        preparation = prepare_model_tournament_multifold(
            earlier_seed_games=historical[season - 4],
            earlier_training_games=historical[season - 3],
            earlier_validation_games=historical[season - 2],
            seed_games=historical[season - 3],
            training_games=historical[season - 2],
            validation_games=historical[season - 1],
            min_games=min_games,
            fold_years=(season - 2, season - 1),
        )
        artifact = preparation.artifact
        prior_summary = preparation.prior_summary
        prior_elo = preparation.prior_elo

    _, engine = build_target_rows(
        games=current_pair[1],
        prior_summary=prior_summary,
        prior_elo=prior_elo,
        min_games=min_games,
    )
    context = LiveOptimizedContext(artifact=artifact, engine=engine, built_at=now)
    _LIVE_MODEL_CACHE[season] = context
    return context


def _archive_immutable_horizon_prediction(
    *,
    now: datetime,
    game: MlbGameState,
    model_version: str,
    features: tuple[float, ...] | None,
    raw_home_probability: float,
    calibrated_home_probability: float,
    artifact: ModelTournamentArtifact | None,
    team_strengths: tuple[float, float],
    lineup_adjustment: ConfirmedLineupAdjustment | None,
) -> None:
    """Best-effort freeze at T-24h or T-1h; never break the dashboard."""
    game_time = _parse_start_time(game.game_date)
    if game_time is None:
        return
    minutes = (game_time - now).total_seconds() / 60.0
    both_lineups = len(game.away_lineup_ids) == 9 and len(game.home_lineup_ids) == 9
    if 3.0 <= minutes <= 180.0 and both_lineups and lineup_adjustment is not None:
        horizon = "T1H"
        frozen_as_of = game_time - timedelta(hours=1)
    elif 1380.0 <= minutes <= 1500.0 and not both_lineups:
        horizon = "T24H"
        frozen_as_of = game_time - timedelta(hours=24)
    else:
        return

    feature_names = (
        "Base", "ProxyStarter", "ProxyBullpen", "StarterComponent",
        "ExpectedInnings", "BullpenComponent", "BullpenFatigue",
        "StarterComponentLoose", "StarterWorkload", "ParkInteraction",
        "ParkNeutralBase", "StarterKRate", "StarterBBRate", "StarterHRRate",
        "BaseRuns",
    )
    feature_payload = (
        dict(zip(feature_names, features, strict=True))
        if features is not None
        else {
            "fallback_team_a_strength": team_strengths[0],
            "fallback_team_b_strength": team_strengths[1],
        }
    )
    if lineup_adjustment is not None:
        feature_payload.update({
            "HomeLineupXwOBA": lineup_adjustment.home_lineup_xwoba,
            "AwayLineupXwOBA": lineup_adjustment.away_lineup_xwoba,
            "HomeTeamBaselineXwOBA": lineup_adjustment.home_team_baseline_xwoba,
            "AwayTeamBaselineXwOBA": lineup_adjustment.away_team_baseline_xwoba,
            "HomeLineupDelta": lineup_adjustment.home_lineup_delta,
            "AwayLineupDelta": lineup_adjustment.away_lineup_delta,
            "LineupAdvantage": lineup_adjustment.lineup_advantage,
            "LineupLogitAdjustment": lineup_adjustment.logit_adjustment,
        })
    artifact_payload = {
        "champion_key": artifact.champion_key if artifact else "fallback",
        "selection_folds": list(artifact.selection_folds) if artifact else [],
        "model_version": model_version,
    }
    source_snapshot = {
        "game_pk": game.game_pk,
        "game_start": game.game_date,
        "away_team": game.away_team,
        "home_team": game.home_team,
        "away_pitcher_id": game.away_probable_pitcher_id,
        "home_pitcher_id": game.home_probable_pitcher_id,
        "away_lineup_ids": list(game.away_lineup_ids),
        "home_lineup_ids": list(game.home_lineup_ids),
        "away_lineup_hash": game.away_lineup_hash,
        "home_lineup_hash": game.home_lineup_hash,
        "lineup_model_used": lineup_adjustment is not None,
    }
    try:
        insert_prediction(
            game_id=game.game_pk,
            horizon=horizon,
            as_of=frozen_as_of,
            model_version=model_version,
            feature_schema_version="v0.13.0",
            code_commit_sha=os.environ.get("MONEYBALL_CODE_COMMIT", "unknown"),
            model_artifact_sha256=sha256_json(artifact_payload),
            calibration_artifact_sha256=None,
            features=feature_payload,
            source_snapshot=source_snapshot,
            raw_home_probability=raw_home_probability,
            calibrated_home_probability=calibrated_home_probability,
            home_team=game.home_team,
            away_team=game.away_team,
            probable_home_pitcher_id=game.home_probable_pitcher_id,
            probable_away_pitcher_id=game.away_probable_pitcher_id,
            lineups_confirmed=both_lineups,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return


async def _bounded_live_optimized_context(
    client: httpx.AsyncClient,
    *,
    season: int,
    through: date,
) -> LiveOptimizedContext | None:
    """Return the optimized context without allowing it to time out the board."""
    timeout_seconds = 20.0 if running_on_vercel() else 180.0
    try:
        return await asyncio.wait_for(
            _live_optimized_context(client, season=season, through=through),
            timeout=timeout_seconds,
        )
    except (TimeoutError, LivePredictionError, httpx.HTTPError, ValueError, TypeError):
        return None


async def build_live_mlb_predictions(
    bankroll: float = 100.0,
    kelly_multiplier: float = 0.25,
    season: int | None = None,
    days: int = 2,
) -> LiveMlbResponse:
    """Fetch MLB scores and price upcoming games with the v0.13.0 lineup-gated tournament or its explicit v0.7 fallback."""
    now = datetime.now(UTC)
    now_eastern = now.astimezone(EASTERN)
    resolved_season = season or now_eastern.year
    start_date = now_eastern.date()
    end_date = start_date + timedelta(days=max(days, 1) - 1)
    timeout = httpx.Timeout(90.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.13.0 lineup-gated-dashboard"}
    runtime_counts: Counter[str] = Counter()
    optimized_context: LiveOptimizedContext | None = None

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            markets_task = fetch_mlb_moneyline_markets(client)
            team_stats_task = fetch_team_season_stats(client, resolved_season)
            prior_stats_task = fetch_team_season_stats(client, resolved_season - 1)
            schedule_task = fetch_mlb_schedule(client, start_date, end_date)
            model_task = _bounded_live_optimized_context(
                client,
                season=resolved_season,
                through=start_date,
            )
            markets_result, team_stats, prior_team_stats, schedule, model_result = (
                await asyncio.gather(
                    markets_task,
                    team_stats_task,
                    prior_stats_task,
                    schedule_task,
                    model_task,
                    return_exceptions=True,
                )
            )

            if isinstance(markets_result, Exception):
                raise markets_result
            if isinstance(team_stats, Exception):
                raise team_stats
            if isinstance(prior_team_stats, Exception):
                raise prior_team_stats
            if isinstance(schedule, Exception):
                raise schedule
            if not isinstance(model_result, Exception):
                optimized_context = model_result
            schedule = await _attach_pregame_lineups(client, schedule)

            markets, market_diagnostics = markets_result
            market_to_game: dict[str, MlbGameState] = {}
            game_to_market: dict[int, MlbMoneylineMarket] = {}
            for market in markets:
                game = _match_schedule_game(market, schedule)
                if not game:
                    market_time = _parse_start_time(market.start_time)
                    if market_time:
                        local_date = market_time.astimezone(EASTERN).date()
                        if local_date < start_date or local_date > end_date:
                            runtime_counts["outside_date_window"] += 1
                        else:
                            runtime_counts["no_schedule_match"] += 1
                    else:
                        runtime_counts["no_schedule_match"] += 1
                    continue
                market_to_game[market.market_id] = game
                game_to_market.setdefault(game.game_pk, market)

            pregame_markets = [
                market
                for market in markets
                if market.market_id in market_to_game
                and market_to_game[market.market_id].is_pregame
                and market.accepting_orders
            ]
            books = await fetch_order_book_tops(
                client,
                [
                    token
                    for market in pregame_markets
                    for token in (market.token_a, market.token_b)
                ],
            )
    except (httpx.HTTPError, MlbDataError, PolymarketDataError) as exc:
        raise LivePredictionError(str(exc)) from exc

    games: list[LiveGamePrediction] = []
    recommended_bets: list[RecommendedBet] = []
    archive_rows: list[dict[str, object]] = []
    lineup_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=8.0),
        follow_redirects=True,
        headers={"User-Agent": "Moneyball-Predictions/0.13.0 lineup-offense"},
    )
    prior_values = list(prior_team_stats.values())
    prior_games_total = sum(item.games_played for item in prior_values)
    prior_league_runs = (
        sum(item.runs_scored for item in prior_values) / prior_games_total
        if prior_values and prior_games_total > 0
        else 4.5
    )

    for market in pregame_markets:
        official_game = market_to_game[market.market_id]
        stats_a = team_stats.get(market.team_a)
        stats_b = team_stats.get(market.team_b)
        if not stats_a or not stats_b:
            runtime_counts["no_schedule_match"] += 1
            continue

        prior_a = prior_team_stats.get(market.team_a)
        prior_b = prior_team_stats.get(market.team_b)
        prior_a_rs = (
            prior_a.runs_scored / prior_a.games_played
            if prior_a and prior_a.games_played
            else prior_league_runs
        )
        prior_a_ra = (
            prior_a.runs_allowed / prior_a.games_played
            if prior_a and prior_a.games_played
            else prior_league_runs
        )
        prior_b_rs = (
            prior_b.runs_scored / prior_b.games_played
            if prior_b and prior_b.games_played
            else prior_league_runs
        )
        prior_b_ra = (
            prior_b.runs_allowed / prior_b.games_played
            if prior_b and prior_b.games_played
            else prior_league_runs
        )
        strength_a = regressed_pythagorean_expectation(
            runs_scored=stats_a.runs_scored,
            runs_allowed=stats_a.runs_allowed,
            games_played=stats_a.games_played,
            prior_runs_scored_per_game=prior_a_rs,
            prior_runs_allowed_per_game=prior_a_ra,
            league_runs_per_team_game=prior_league_runs,
        )
        strength_b = regressed_pythagorean_expectation(
            runs_scored=stats_b.runs_scored,
            runs_allowed=stats_b.runs_allowed,
            games_played=stats_b.games_played,
            prior_runs_scored_per_game=prior_b_rs,
            prior_runs_allowed_per_game=prior_b_ra,
            league_runs_per_team_game=prior_league_runs,
        )

        neutral_a = log5_probability(strength_a, strength_b)
        if market.team_a == official_game.home_team:
            fallback_probability_a = add_home_field_advantage(neutral_a, 0.54)
        else:
            fallback_home_b = add_home_field_advantage(1.0 - neutral_a, 0.54)
            fallback_probability_a = 1.0 - fallback_home_b

        probability_a = fallback_probability_a
        raw_home_probability = (
            fallback_probability_a
            if market.team_a == official_game.home_team
            else 1.0 - fallback_probability_a
        )
        home_probability = raw_home_probability
        model_version = "v0.5 fallback"
        matchup_features: tuple[float, ...] | None = None
        if optimized_context:
            game_time = _parse_start_time(official_game.game_date)
            if game_time is not None:
                matchup_features = optimized_context.engine.features_for_matchup(
                    away_team=official_game.away_team,
                    home_team=official_game.home_team,
                    game_time=game_time,
                    away_pitcher_id=official_game.away_probable_pitcher_id,
                    home_pitcher_id=official_game.home_probable_pitcher_id,
                    away_pitcher_name=official_game.away_probable_pitcher,
                    home_pitcher_name=official_game.home_probable_pitcher,
                    venue=official_game.venue,
                )
                linear_artifact = optimized_context.artifact.linear_artifact
                component_ready = bool(
                    linear_artifact
                    and linear_artifact.training_component_coverage >= MIN_COMPONENT_COVERAGE
                    and linear_artifact.validation_component_coverage >= MIN_COMPONENT_COVERAGE
                )
                if component_ready:
                    candidate = optimized_context.artifact.champion
                    if isinstance(candidate.predictor, CalibratedPredictor):
                        raw_home_probability = candidate.predictor.raw_predictor.predict(
                            matchup_features
                        )
                    else:
                        raw_home_probability = candidate.predict(matchup_features)
                    home_probability = candidate.predict(matchup_features)
                    selected_label = candidate.label
                    selected_calibration = candidate.calibration_method
                else:
                    variant = (
                        linear_artifact.variants["v07"]
                        if linear_artifact is not None
                        else None
                    )
                    fallback_home_probability = (
                        fallback_probability_a
                        if market.team_a == official_game.home_team
                        else 1.0 - fallback_probability_a
                    )
                    home_probability = (
                        variant.predict(matchup_features)
                        if variant is not None
                        else fallback_home_probability
                    )
                    raw_home_probability = home_probability
                    selected_label = "v0.7 proxy"
                    selected_calibration = "pitching sync required"
                probability_a = (
                    home_probability
                    if market.team_a == official_game.home_team
                    else 1.0 - home_probability
                )
                model_version = (
                    f"v0.13.0 {selected_label} ({selected_calibration})"
                    if component_ready
                    else "v0.7 proxy · run pitching sync"
                )
        early_home_probability = home_probability
        lineup_adjustment: ConfirmedLineupAdjustment | None = None
        both_lineups_confirmed = (
            official_game.away_lineup_status == LineupStatus.CONFIRMED.value
            and official_game.home_lineup_status == LineupStatus.CONFIRMED.value
            and len(official_game.away_lineup_ids) == 9
            and len(official_game.home_lineup_ids) == 9
        )
        any_lineup_available = bool(
            official_game.away_lineup_ids or official_game.home_lineup_ids
        )
        lineup_status = (
            LineupStatus.CONFIRMED.value
            if both_lineups_confirmed
            else LineupStatus.PARTIAL.value
            if any_lineup_available
            else LineupStatus.PENDING.value
        )
        if both_lineups_confirmed:
            try:
                lineup_adjustment = await build_confirmed_lineup_adjustment(
                    lineup_client,
                    game=official_game,
                    season=resolved_season,
                    home_probability=home_probability,
                )
            except (httpx.HTTPError, TypeError, ValueError):
                lineup_adjustment = None
            if lineup_adjustment is not None:
                home_probability = lineup_adjustment.home_probability_after
                model_version = f"{model_version} + confirmed-lineup offense"

        probability_a = (
            home_probability
            if market.team_a == official_game.home_team
            else 1.0 - home_probability
        )
        probability_b = 1.0 - probability_a

        book_a = books.get(market.token_a)
        book_b = books.get(market.token_b)
        ask_a = book_a.best_ask if book_a else None
        ask_b = book_b.best_ask if book_b else None
        if ask_a is None or ask_b is None or not 0 < ask_a < 1 or not 0 < ask_b < 1:
            runtime_counts["no_executable_ask"] += 1
            continue

        support_a = _sabermetric_support_for_team(
            team=market.team_a,
            game=official_game,
            own_strength=strength_a,
            opponent_strength=strength_b,
            features=matchup_features,
        )
        support_b = _sabermetric_support_for_team(
            team=market.team_b,
            game=official_game,
            own_strength=strength_b,
            opponent_strength=strength_a,
            features=matchup_features,
        )
        side_a = _market_side(
            market.team_a,
            probability_a,
            ask_a,
            bankroll,
            kelly_multiplier,
            book_a.ask_size if book_a else None,
            sabermetric_support=support_a[0],
            sabermetric_support_count=support_a[1],
            sabermetric_support_total=support_a[2],
            sabermetric_reasons=support_a[3],
        )
        side_b = _market_side(
            market.team_b,
            probability_b,
            ask_b,
            bankroll,
            kelly_multiplier,
            book_b.ask_size if book_b else None,
            sabermetric_support=support_b[0],
            sabermetric_support_count=support_b[1],
            sabermetric_support_total=support_b[2],
            sabermetric_reasons=support_b[3],
        )
        priced_signal, priced_side, priced_reason = _signal_for_sides(side_a, side_b)
        starters_confirmed = (
            official_game.away_probable_pitcher_id is not None
            and official_game.home_probable_pitcher_id is not None
        )
        selected_for_gate = (
            side_a if priced_side == side_a.team else side_b if priced_side == side_b.team else None
        )
        gate_edge = selected_for_gate.edge if selected_for_gate is not None else 0.0
        gate_roi = selected_for_gate.expected_roi if selected_for_gate is not None else 0.0
        gate_allowed, gate_reason = paper_bet_eligibility(
            lineup_status=lineup_status,
            lineup_model_used=lineup_adjustment is not None,
            starters_confirmed=starters_confirmed,
            signal=priced_signal,
            edge=gate_edge,
            expected_roi=gate_roi,
        )
        lineup_bet_eligible = (
            lineup_status == LineupStatus.CONFIRMED.value
            and lineup_adjustment is not None
            and starters_confirmed
        )
        if lineup_status == LineupStatus.PENDING.value:
            signal = "WAIT"
            recommended_side = None
            reason = (
                "Early estimate only: official starting lineups are not available yet. "
                "The app will poll MLB and reprice this game automatically after both lineups arrive."
            )
            lineup_note = "Waiting for both official starting lineups"
        elif lineup_status == LineupStatus.PARTIAL.value:
            signal = "WAIT"
            recommended_side = None
            reason = (
                "Early estimate only: one or both official batting orders are incomplete. "
                "Paper betting remains locked until all 18 starters are available."
            )
            lineup_note = "Partial lineup received; waiting for all 18 starters"
        elif lineup_adjustment is None:
            signal = "WAIT"
            recommended_side = None
            reason = (
                "Both lineups are posted, but the lineup-specific offense projection lacks "
                "enough point-in-time player data. Paper betting remains locked."
            )
            lineup_note = "Lineups confirmed; offense repricing unavailable"
        else:
            signal = priced_signal
            recommended_side = priced_side
            reason = (
                f"{priced_reason} Confirmed-lineup adjustment changed the home probability "
                f"by {lineup_adjustment.probability_change:+.1%}."
            )
            lineup_note = (
                "Both lineups confirmed and probability repriced using the nine announced "
                "hitters against the opposing starter hand"
            )
        if signal == "BET" and not gate_allowed:
            signal = "WAIT"
            recommended_side = None
            reason = f"{gate_reason}. The pregame estimate is visible, but paper betting is locked."

        prediction = LiveGamePrediction(
            event_id=market.event_id,
            market_id=market.market_id,
            game_pk=official_game.game_pk,
            title=f"{official_game.away_team} at {official_game.home_team}",
            start_time=official_game.game_date,
            polymarket_url=market.polymarket_url,
            away_team=official_game.away_team,
            home_team=official_game.home_team,
            away_team_id=official_game.away_team_id,
            home_team_id=official_game.home_team_id,
            away_probable_pitcher=official_game.away_probable_pitcher,
            home_probable_pitcher=official_game.home_probable_pitcher,
            venue=official_game.venue,
            away_lineup_confirmed=len(official_game.away_lineup_ids) == 9,
            home_lineup_confirmed=len(official_game.home_lineup_ids) == 9,
            away_lineup_names=list(official_game.away_lineup_names),
            home_lineup_names=list(official_game.home_lineup_names),
            away_lineup=[
                {
                    "player_id": player_id,
                    "full_name": name,
                    "batting_slot": slot,
                    "position": position,
                }
                for player_id, name, slot, position in zip(
                    official_game.away_lineup_ids,
                    official_game.away_lineup_names,
                    official_game.away_lineup_slots,
                    official_game.away_lineup_positions,
                    strict=True,
                )
            ],
            home_lineup=[
                {
                    "player_id": player_id,
                    "full_name": name,
                    "batting_slot": slot,
                    "position": position,
                }
                for player_id, name, slot, position in zip(
                    official_game.home_lineup_ids,
                    official_game.home_lineup_names,
                    official_game.home_lineup_slots,
                    official_game.home_lineup_positions,
                    strict=True,
                )
            ],
            lineup_status=lineup_status,
            lineup_model_used=lineup_adjustment is not None,
            lineup_bet_eligible=lineup_bet_eligible,
            lineup_note=lineup_note,
            early_home_probability=early_home_probability,
            lineup_adjusted_home_probability=(
                lineup_adjustment.home_probability_after if lineup_adjustment else None
            ),
            home_lineup_xwoba=(lineup_adjustment.home_lineup_xwoba if lineup_adjustment else None),
            away_lineup_xwoba=(lineup_adjustment.away_lineup_xwoba if lineup_adjustment else None),
            home_lineup_delta=(lineup_adjustment.home_lineup_delta if lineup_adjustment else None),
            away_lineup_delta=(lineup_adjustment.away_lineup_delta if lineup_adjustment else None),
            lineup_probability_change=(
                lineup_adjustment.probability_change if lineup_adjustment else None
            ),
            team_a_strength=strength_a,
            team_b_strength=strength_b,
            side_a=side_a,
            side_b=side_b,
            recommended_side=recommended_side,
            signal=signal,
            recommendation_reason=reason,
            liquidity=market.liquidity,
            volume=market.volume,
            model_version=model_version,
        )
        _archive_immutable_horizon_prediction(
            now=now,
            game=official_game,
            model_version=model_version,
            features=matchup_features,
            raw_home_probability=raw_home_probability,
            calibrated_home_probability=(
                probability_a
                if market.team_a == official_game.home_team
                else 1.0 - probability_a
            ),
            artifact=optimized_context.artifact if optimized_context else None,
            team_strengths=(strength_a, strength_b),
            lineup_adjustment=lineup_adjustment,
        )
        games.append(prediction)
        archive_rows.append(
            {
                "game_pk": official_game.game_pk,
                "game_start": official_game.game_date,
                "market_id": market.market_id,
                "event_id": market.event_id,
                "team_a": market.team_a,
                "team_b": market.team_b,
                "token_a": market.token_a,
                "token_b": market.token_b,
                "best_bid_a": book_a.best_bid if book_a else None,
                "best_ask_a": ask_a,
                "ask_size_a": book_a.ask_size if book_a else None,
                "bids_a": ([{"price": p, "size": q} for p, q in book_a.bids] if book_a else []),
                "asks_a": ([{"price": p, "size": q} for p, q in book_a.asks] if book_a else []),
                "best_bid_b": book_b.best_bid if book_b else None,
                "best_ask_b": ask_b,
                "ask_size_b": book_b.ask_size if book_b else None,
                "bids_b": ([{"price": p, "size": q} for p, q in book_b.bids] if book_b else []),
                "asks_b": ([{"price": p, "size": q} for p, q in book_b.asks] if book_b else []),
                "model_probability_a": probability_a,
                "model_probability_b": probability_b,
                "model_version": model_version,
                "team_a_strength": strength_a,
                "team_b_strength": strength_b,
                "sabermetric_support_a": side_a.sabermetric_support,
                "sabermetric_support_count_a": side_a.sabermetric_support_count,
                "sabermetric_support_total_a": side_a.sabermetric_support_total,
                "sabermetric_reasons_a": side_a.sabermetric_reasons,
                "value_grade_a": side_a.value_grade,
                "sabermetric_support_b": side_b.sabermetric_support,
                "sabermetric_support_count_b": side_b.sabermetric_support_count,
                "sabermetric_support_total_b": side_b.sabermetric_support_total,
                "sabermetric_reasons_b": side_b.sabermetric_reasons,
                "value_grade_b": side_b.value_grade,
                "model_features": (
                    dict(zip((
                        "Base", "ProxyStarter", "ProxyBullpen", "StarterComponent",
                        "ExpectedInnings", "BullpenComponent", "BullpenFatigue",
                        "StarterComponentLoose", "StarterWorkload", "ParkInteraction",
                        "ParkNeutralBase", "StarterKRate", "StarterBBRate", "StarterHRRate",
                        "BaseRuns",
                    ), matchup_features, strict=True))
                    if matchup_features is not None else None
                ),
                "prediction_horizon_minutes": max(
                    0, int(((_parse_start_time(official_game.game_date) or now) - now).total_seconds() / 60)
                ),
                "away_probable_pitcher_id": official_game.away_probable_pitcher_id,
                "home_probable_pitcher_id": official_game.home_probable_pitcher_id,
                "away_lineup_confirmed": len(official_game.away_lineup_ids) == 9,
                "home_lineup_confirmed": len(official_game.home_lineup_ids) == 9,
                "away_lineup_ids": list(official_game.away_lineup_ids),
                "home_lineup_ids": list(official_game.home_lineup_ids),
                "away_lineup_hash": official_game.away_lineup_hash,
                "home_lineup_hash": official_game.home_lineup_hash,
                "lineup_status": lineup_status,
                "lineup_model_used": lineup_adjustment is not None,
                "early_home_probability": early_home_probability,
                "lineup_adjusted_home_probability": (
                    lineup_adjustment.home_probability_after if lineup_adjustment else None
                ),
                "lineup_probability_change": (
                    lineup_adjustment.probability_change if lineup_adjustment else None
                ),
                "signal": signal,
                "recommended_side": recommended_side,
            }
        )

        if signal == "BET" and recommended_side:
            selected = side_a if side_a.team == recommended_side else side_b
            opponent = side_b.team if selected is side_a else side_a.team
            recommended_bets.append(
                RecommendedBet(
                    game_pk=official_game.game_pk,
                    title=prediction.title,
                    team=selected.team,
                    opponent=opponent,
                    start_time=prediction.start_time,
                    market_buy_price=selected.market_buy_price,
                    edge=selected.edge,
                    expected_value=selected.expected_value,
                    expected_roi=selected.expected_roi,
                    full_kelly_fraction=selected.full_kelly_fraction,
                    applied_kelly_fraction=selected.applied_kelly_fraction,
                    kelly_multiplier=selected.kelly_multiplier,
                    kelly_stake=selected.kelly_stake,
                    stake=selected.kelly_stake,
                    shares=selected.shares,
                    kelly_capped_by_depth=selected.kelly_capped_by_depth,
                    kelly_capped_by_risk=selected.kelly_capped_by_risk,
                    sabermetric_support=selected.sabermetric_support,
                    sabermetric_support_count=selected.sabermetric_support_count,
                    sabermetric_support_total=selected.sabermetric_support_total,
                    value_grade=selected.value_grade if selected.value_grade in {"A", "B", "C"} else "C",
                    polymarket_url=prediction.polymarket_url,
                )
            )

    await lineup_client.aclose()
    games.sort(key=lambda game: _parse_start_time(game.start_time) or datetime.max.replace(tzinfo=UTC))
    grade_rank = {"A": 3, "B": 2, "C": 1}
    recommended_bets.sort(
        key=lambda bet: (grade_rank.get(bet.value_grade, 0), bet.expected_value, bet.expected_roi),
        reverse=True,
    )
    scoreboard = [
        _scoreboard_row(game, game_to_market.get(game.game_pk))
        for game in sorted(schedule, key=lambda item: item.game_date)
    ]
    diagnostic_summary = _diagnostic_summary(market_diagnostics, runtime_counts)
    append_market_snapshots(archive_rows)

    return LiveMlbResponse(
        generated_at=datetime.now(UTC),
        season=resolved_season,
        model_version=(
            (
                f"v0.13.0 {optimized_context.artifact.champion.label} "
                f"({optimized_context.artifact.champion.calibration_method})"
                if optimized_context.artifact.linear_artifact is not None
                and optimized_context.artifact.linear_artifact.validation_component_coverage
                >= MIN_COMPONENT_COVERAGE
                else "v0.7 proxy · run pitching sync"
            )
            if optimized_context
            else "v0.5 fallback"
        ),
        model_training_rows=(optimized_context.artifact.training_rows if optimized_context else 0),
        model_validation_rows=(optimized_context.artifact.validation_rows if optimized_context else 0),
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        date_window_start=start_date.isoformat(),
        date_window_end=end_date.isoformat(),
        recommended_bets=recommended_bets[:5],
        games=games,
        scoreboard=scoreboard,
        diagnostics=diagnostic_summary,
        skipped_events=diagnostic_summary.examples,
    )


async def build_single_scoreboard_game(game_pk: int) -> ScoreboardGame:
    """Fetch one game for resolving a locally stored paper bet."""
    timeout = httpx.Timeout(15.0, connect=8.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.13.0 paper-settlement"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            game = await fetch_mlb_game(client, game_pk)
    except (httpx.HTTPError, MlbDataError) as exc:
        raise LivePredictionError(str(exc)) from exc
    if game is None:
        raise LivePredictionError(f"MLB game {game_pk} was not found")
    return _scoreboard_row(game, None)
