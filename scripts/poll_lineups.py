"""Poll official MLB starting lineups and archive only changed states.

Run beside the FastAPI server.  The poller does not calculate bets; it creates the
point-in-time lineup history that the live endpoint and later backtests can trust.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

import httpx

from moneyball_predictions.lineups import parse_game_lineups
from moneyball_predictions.mlb import fetch_mlb_boxscore, fetch_mlb_schedule
from moneyball_predictions.storage import insert_lineup_snapshot


def _parse_start(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def poll_once(
    *,
    hours_before: float = 6.0,
    stop_minutes_before: float = 2.0,
) -> dict[str, int]:
    now = datetime.now(UTC)
    timeout = httpx.Timeout(30.0, connect=8.0)
    counts = {
        "eligible_games": 0,
        "fetched_games": 0,
        "new_snapshots": 0,
        "confirmed_games": 0,
        "errors": 0,
    }
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Moneyball-Predictions/0.12.2 lineup-poller"},
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
            if minutes > hours_before * 60.0 or minutes < stop_minutes_before:
                continue
            counts["eligible_games"] += 1
            try:
                payload = await fetch_mlb_boxscore(client, game.game_pk)
                lineups = parse_game_lineups(
                    game_pk=game.game_pk,
                    payload=payload,
                    captured_at=datetime.now(UTC),
                )
                counts["fetched_games"] += 1
                counts["new_snapshots"] += int(insert_lineup_snapshot(lineups.away))
                counts["new_snapshots"] += int(insert_lineup_snapshot(lineups.home))
                if lineups.both_confirmed:
                    counts["confirmed_games"] += 1
                print(
                    f"{game.game_pk} {game.away_team} at {game.home_team} · "
                    f"away={lineups.away.status.value}({len(lineups.away.players)}/9) · "
                    f"home={lineups.home.status.value}({len(lineups.home.players)}/9)"
                )
            except (httpx.HTTPError, OSError, TypeError, ValueError) as exc:
                counts["errors"] += 1
                print(f"{game.game_pk}: lineup fetch failed: {exc}")
    return counts


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--hours-before", type=float, default=6.0)
    parser.add_argument("--stop-minutes-before", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        started = datetime.now(UTC)
        try:
            counts = await poll_once(
                hours_before=args.hours_before,
                stop_minutes_before=args.stop_minutes_before,
            )
            print("Lineup poll summary:", counts)
        except (httpx.HTTPError, OSError, TypeError, ValueError) as exc:
            print(f"Lineup poll failed: {exc}")
        if args.once:
            return
        elapsed = (datetime.now(UTC) - started).total_seconds()
        await asyncio.sleep(max(1.0, args.interval - elapsed))


if __name__ == "__main__":
    asyncio.run(main())
