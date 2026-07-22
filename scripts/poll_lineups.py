"""Local loop around the same bounded lineup job used by HTTP cron."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from moneyball_predictions.jobs import poll_lineups_once


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        started = datetime.now(UTC)
        try:
            print("Lineup poll summary:", await poll_lineups_once())
        except Exception as exc:
            print(f"Lineup poll failed: {exc}")
        if args.once:
            return
        elapsed = (datetime.now(UTC) - started).total_seconds()
        await asyncio.sleep(max(1.0, args.interval - elapsed))


if __name__ == "__main__":
    asyncio.run(main())
