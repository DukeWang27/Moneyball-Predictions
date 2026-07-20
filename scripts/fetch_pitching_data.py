#!/usr/bin/env python3
"""Build the local MLB box-score cache used by the v0.8 component model.

The command is resumable. Existing game JSON files are skipped, so rerunning it only
fetches new or previously failed games.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

import httpx

from moneyball_predictions.mlb import (
    cached_boxscore_path,
    fetch_mlb_boxscore,
    fetch_mlb_regular_season_schedule,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025, 2026],
        help="Seasons to cache (default: 2023 2024 2025 2026)",
    )
    parser.add_argument(
        "--through",
        type=date.fromisoformat,
        default=None,
        help="Optional final date for the newest season, YYYY-MM-DD",
    )
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/mlb_boxscores"))
    return parser


async def _fetch_one(
    client: httpx.AsyncClient,
    game_pk: int,
    *,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    path = cached_boxscore_path(game_pk, cache_dir=cache_dir)
    if path.exists():
        return game_pk, "cached"
    async with semaphore:
        for attempt in range(4):
            try:
                payload = await fetch_mlb_boxscore(client, game_pk)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                temporary.replace(path)
                return game_pk, "downloaded"
            except (httpx.HTTPError, OSError) as exc:
                if attempt == 3:
                    return game_pk, f"failed: {exc}"
                await asyncio.sleep(0.5 * (2**attempt))
    return game_pk, "failed"


async def main() -> None:
    args = _parser().parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(60.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.8.1 pitching-cache"}
    semaphore = asyncio.Semaphore(max(args.concurrency, 1))

    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        all_game_pks: list[int] = []
        newest = max(args.seasons)
        for season in sorted(set(args.seasons)):
            end = args.through if season == newest and args.through else date(season, 11, 15)
            games = await fetch_mlb_regular_season_schedule(
                client,
                season=season,
                start_date=date(season, 3, 1),
                end_date=end,
            )
            finals = [game for game in games if game.is_final]
            all_game_pks.extend(game.game_pk for game in finals)
            print(f"{season}: found {len(finals)} completed regular-season games")

        unique = sorted(set(all_game_pks))
        print(f"Caching {len(unique)} unique box scores in {args.cache_dir}...")
        tasks = [
            _fetch_one(
                client,
                game_pk,
                cache_dir=args.cache_dir,
                semaphore=semaphore,
            )
            for game_pk in unique
        ]
        downloaded = cached = failed = 0
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            _, status = await task
            if status == "downloaded":
                downloaded += 1
            elif status == "cached":
                cached += 1
            else:
                failed += 1
                print(status)
            if index % 250 == 0 or index == len(unique):
                print(
                    f"{index}/{len(unique)} complete · new {downloaded} · cached {cached} · failed {failed}"
                )

    if failed:
        raise SystemExit(f"Finished with {failed} failed downloads. Rerun the command to retry.")
    print("Pitching cache ready.")


if __name__ == "__main__":
    asyncio.run(main())
