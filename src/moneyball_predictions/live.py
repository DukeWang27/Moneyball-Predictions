"""Join MLB scores and season strength with Polymarket MLB moneylines."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from .mlb import (
    MlbDataError,
    MlbGameState,
    fetch_mlb_game,
    fetch_mlb_schedule,
    fetch_team_season_stats,
)
from .model import (
    add_home_field_advantage,
    log5_probability,
    regressed_pythagorean_expectation,
)
from .odds import prediction_market_expected_value, prediction_market_kelly_stake
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

EASTERN = ZoneInfo("America/New_York")
MIN_BET_EDGE = 0.05
MIN_BET_ROI = 0.05


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
) -> LiveMarketSide:
    kelly_stake, full_kelly, applied_kelly, capped_by_depth = (
        prediction_market_kelly_stake(
            model_probability=model_probability,
            buy_price=ask_price,
            bankroll=bankroll,
            multiplier=kelly_multiplier,
            top_ask_size=ask_size,
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
    )


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


def _signal_for_sides(
    side_a: LiveMarketSide,
    side_b: LiveMarketSide,
) -> tuple[str, str | None, str]:
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


async def build_live_mlb_predictions(
    bankroll: float = 100.0,
    kelly_multiplier: float = 0.25,
    season: int | None = None,
    days: int = 2,
) -> LiveMlbResponse:
    """Fetch current MLB scores and pregame Polymarket edges for a short date window."""
    now = datetime.now(UTC)
    now_eastern = now.astimezone(EASTERN)
    resolved_season = season or now_eastern.year
    start_date = now_eastern.date()
    end_date = start_date + timedelta(days=max(days, 1) - 1)
    timeout = httpx.Timeout(20.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.5.0 research-dashboard"}
    runtime_counts: Counter[str] = Counter()

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            markets, market_diagnostics = await fetch_mlb_moneyline_markets(client)
            team_stats = await fetch_team_season_stats(client, resolved_season)
            prior_team_stats = await fetch_team_season_stats(client, resolved_season - 1)
            schedule = await fetch_mlb_schedule(client, start_date, end_date)

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
    for market in pregame_markets:
        official_game = market_to_game[market.market_id]
        stats_a = team_stats.get(market.team_a)
        stats_b = team_stats.get(market.team_b)
        if not stats_a or not stats_b:
            runtime_counts["no_schedule_match"] += 1
            continue

        prior_a = prior_team_stats.get(market.team_a)
        prior_b = prior_team_stats.get(market.team_b)
        prior_values = list(prior_team_stats.values())
        prior_league_runs = (
            sum(item.runs_scored for item in prior_values)
            / sum(item.games_played for item in prior_values)
            if prior_values and sum(item.games_played for item in prior_values) > 0
            else 4.5
        )
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
            probability_a = add_home_field_advantage(neutral_a, 0.54)
        else:
            probability_b_home = add_home_field_advantage(1.0 - neutral_a, 0.54)
            probability_a = 1.0 - probability_b_home
        probability_b = 1.0 - probability_a

        book_a = books.get(market.token_a)
        book_b = books.get(market.token_b)
        ask_a = book_a.best_ask if book_a else None
        ask_b = book_b.best_ask if book_b else None
        if ask_a is None or ask_b is None or not 0 < ask_a < 1 or not 0 < ask_b < 1:
            runtime_counts["no_executable_ask"] += 1
            continue

        side_a = _market_side(
            market.team_a,
            probability_a,
            ask_a,
            bankroll,
            kelly_multiplier,
            book_a.ask_size if book_a else None,
        )
        side_b = _market_side(
            market.team_b,
            probability_b,
            ask_b,
            bankroll,
            kelly_multiplier,
            book_b.ask_size if book_b else None,
        )
        signal, recommended_side, reason = _signal_for_sides(side_a, side_b)

        prediction = LiveGamePrediction(
            event_id=market.event_id,
            market_id=market.market_id,
            game_pk=official_game.game_pk,
            title=f"{official_game.away_team} at {official_game.home_team}",
            start_time=official_game.game_date,
            polymarket_url=market.polymarket_url,
            away_team=official_game.away_team,
            home_team=official_game.home_team,
            away_probable_pitcher=official_game.away_probable_pitcher,
            home_probable_pitcher=official_game.home_probable_pitcher,
            venue=official_game.venue,
            team_a_strength=strength_a,
            team_b_strength=strength_b,
            side_a=side_a,
            side_b=side_b,
            recommended_side=recommended_side,
            signal=signal,
            recommendation_reason=reason,
            liquidity=market.liquidity,
            volume=market.volume,
        )
        games.append(prediction)

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
                    polymarket_url=prediction.polymarket_url,
                )
            )

    games.sort(key=lambda game: _parse_start_time(game.start_time) or datetime.max.replace(tzinfo=UTC))
    recommended_bets.sort(
        key=lambda bet: (bet.expected_value, bet.expected_roi), reverse=True
    )
    scoreboard = [
        _scoreboard_row(game, game_to_market.get(game.game_pk))
        for game in sorted(schedule, key=lambda item: item.game_date)
    ]
    diagnostic_summary = _diagnostic_summary(market_diagnostics, runtime_counts)

    return LiveMlbResponse(
        generated_at=datetime.now(UTC),
        season=resolved_season,
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
    headers = {"User-Agent": "Moneyball-Predictions/0.5.0 paper-settlement"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            game = await fetch_mlb_game(client, game_pk)
    except (httpx.HTTPError, MlbDataError) as exc:
        raise LivePredictionError(str(exc)) from exc
    if game is None:
        raise LivePredictionError(f"MLB game {game_pk} was not found")
    return _scoreboard_row(game, None)
