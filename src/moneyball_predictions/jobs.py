"""Bounded, idempotent jobs suitable for HTTP cron invocations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select

from .db import LineupSnapshotRecord, PaperBet, initialize_database, session_scope
from .lineups import parse_game_lineups
from .mlb import fetch_mlb_boxscore, fetch_mlb_game, fetch_mlb_schedule
from .player_props import build_strikeout_prop_settlement
from .portfolio import settle_bet
from .repositories import finish_cron_run, insert_lineup_snapshot, start_cron_run


def _parse_start(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _poll_due(minutes_until_game: float, last_capture: datetime | None, now: datetime) -> bool:
    """Adaptive cadence without requiring a permanently running process."""
    if minutes_until_game > 360 or minutes_until_game < 0:
        return False
    if minutes_until_game > 120:
        interval_seconds = 600
    elif minutes_until_game > 15:
        interval_seconds = 120
    else:
        interval_seconds = 60
    if last_capture is None:
        return True
    if last_capture.tzinfo is None:
        last_capture = last_capture.replace(tzinfo=UTC)
    return (now - last_capture.astimezone(UTC)).total_seconds() >= interval_seconds


async def poll_lineups_once() -> dict[str, int]:
    initialize_database()
    now = datetime.now(UTC)
    counts = {
        "eligible_games": 0,
        "fetched_games": 0,
        "new_snapshots": 0,
        "confirmed_games": 0,
        "skipped_cadence": 0,
        "errors": 0,
    }
    timeout = httpx.Timeout(30.0, connect=8.0)
    with session_scope() as session:
        run = start_cron_run(session, "lineups")
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Moneyball-Predictions/0.13.0 lineup-cron"},
            ) as client:
                schedule = await fetch_mlb_schedule(
                    client,
                    now.date(),
                    now.date() + timedelta(days=1),
                )
                for game in schedule:
                    game_start = _parse_start(game.game_date)
                    if game_start is None or not game.is_pregame:
                        continue
                    minutes = (game_start - now).total_seconds() / 60.0
                    if minutes > 360 or minutes < 0:
                        continue
                    counts["eligible_games"] += 1
                    latest = session.execute(
                        select(LineupSnapshotRecord.captured_at_utc)
                        .where(LineupSnapshotRecord.game_pk == game.game_pk)
                        .order_by(LineupSnapshotRecord.captured_at_utc.desc())
                        .limit(1)
                    ).scalar_one_or_none()
                    if not _poll_due(minutes, latest, now):
                        counts["skipped_cadence"] += 1
                        continue
                    try:
                        payload = await fetch_mlb_boxscore(client, game.game_pk)
                        lineups = parse_game_lineups(
                            game_pk=game.game_pk,
                            payload=payload,
                            captured_at=datetime.now(UTC),
                        )
                        counts["fetched_games"] += 1
                        counts["new_snapshots"] += int(
                            insert_lineup_snapshot(session, lineups.away)
                        )
                        counts["new_snapshots"] += int(
                            insert_lineup_snapshot(session, lineups.home)
                        )
                        if lineups.both_confirmed:
                            counts["confirmed_games"] += 1
                    except (httpx.HTTPError, TypeError, ValueError):
                        counts["errors"] += 1
            finish_cron_run(session, run, counts=counts)
        except Exception as exc:
            finish_cron_run(session, run, counts=counts, error=exc)
            raise
    return counts


async def settle_paper_bets_once() -> dict[str, int]:
    initialize_database()
    counts = {"open": 0, "settled": 0, "refunded": 0, "errors": 0}
    with session_scope() as session:
        run = start_cron_run(session, "settle-paper-bets")
        try:
            open_bets = list(
                session.scalars(select(PaperBet).where(PaperBet.status == "OPEN")).all()
            )
            counts["open"] = len(open_bets)
            timeout = httpx.Timeout(20.0, connect=8.0)
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Moneyball-Predictions/0.13.0 settlement-cron"},
            ) as client:
                for bet in open_bets:
                    try:
                        if bet.market_type == "STRIKEOUT_PROP":
                            result = await build_strikeout_prop_settlement(
                                game_pk=bet.game_pk,
                                player_id=int(bet.player_id or 0),
                                threshold=int(bet.prop_threshold or 0),
                                side=bet.prop_side or "YES",
                            )
                            if result.status == "open":
                                continue
                            status = result.status.upper()
                            settle_bet(
                                session,
                                bet_id=bet.bet_id,
                                status=status,
                                actual_strikeouts=result.strikeouts,
                                actual_result=(
                                    f"{result.strikeouts} strikeouts"
                                    if result.strikeouts is not None
                                    else result.game_status
                                ),
                            )
                        else:
                            game = await fetch_mlb_game(client, bet.game_pk)
                            if game is None or not game.is_final:
                                detail = (game.detailed_state if game else "").casefold()
                                if "cancel" in detail or "postpon" in detail:
                                    settle_bet(
                                        session,
                                        bet_id=bet.bet_id,
                                        status="REFUNDED",
                                        actual_result=game.detailed_state if game else None,
                                    )
                                else:
                                    continue
                            else:
                                won = game.winner == bet.team
                                settle_bet(
                                    session,
                                    bet_id=bet.bet_id,
                                    status="WON" if won else "LOST",
                                    actual_result=game.winner,
                                )
                        if bet.status == "REFUNDED":
                            counts["refunded"] += 1
                        else:
                            counts["settled"] += 1
                    except Exception:
                        counts["errors"] += 1
            finish_cron_run(session, run, counts=counts)
        except Exception as exc:
            finish_cron_run(session, run, counts=counts, error=exc)
            raise
    return counts
