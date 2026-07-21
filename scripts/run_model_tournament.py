#!/usr/bin/env python3
"""Run the leakage-safe model tournament and save a compact JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from moneyball_predictions.backtest import build_mlb_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--min-games", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_registry/latest.json"),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    result = await build_mlb_backtest(
        season=args.season,
        min_games=args.min_games,
    )
    payload = {
        "generated_at": result.generated_at.isoformat(),
        "target_season": result.season,
        "selection_folds": result.tournament_selection_folds,
        "champion": result.tournament_champion,
        "promoted": result.tournament_promoted,
        "decision": result.tournament_decision,
        "target_metrics": result.optimized.model_dump(),
        "models": [item.model_dump() for item in result.tournament_models],
        "leakage_audit": result.leakage_audit.model_dump(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Champion: {result.tournament_champion}")
    print(result.tournament_decision)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
