#!/usr/bin/env python3
"""Precompute expensive model artifacts before deploying to Vercel Hobby."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import joblib

from moneyball_predictions.backtest import build_mlb_backtest
from moneyball_predictions.mlb import fetch_mlb_regular_season_schedule
from moneyball_predictions.optimized import build_target_rows
from moneyball_predictions.serverless_artifacts import (
    ARTIFACT_DIR,
    LIVE_MODEL_BUNDLE,
    backtest_snapshot_path,
)
from moneyball_predictions.tournament import prepare_model_tournament_multifold

EASTERN = ZoneInfo("America/New_York")


async def _fetch_year(client: httpx.AsyncClient, year: int):
    return (
        year,
        await fetch_mlb_regular_season_schedule(
            client,
            season=year,
            start_date=date(year, 3, 1),
            end_date=date(year, 11, 15),
        ),
    )


async def export_artifacts(season: int) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    years = [season - 4, season - 3, season - 2, season - 1]
    timeout = httpx.Timeout(180.0, connect=15.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.13.0 artifact-export"}

    print(f"Fetching training seasons: {', '.join(map(str, years))}")
    through = datetime.now(EASTERN).date()
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        historical_pairs = await asyncio.gather(*(_fetch_year(client, year) for year in years))
        current_games = await fetch_mlb_regular_season_schedule(
            client,
            season=season,
            start_date=date(season, 3, 1),
            end_date=through,
        )

    historical = dict(historical_pairs)
    print("Training guarded model tournament...")
    preparation = prepare_model_tournament_multifold(
        earlier_seed_games=historical[season - 4],
        earlier_training_games=historical[season - 3],
        earlier_validation_games=historical[season - 2],
        seed_games=historical[season - 3],
        training_games=historical[season - 2],
        validation_games=historical[season - 1],
        min_games=10,
        fold_years=(season - 2, season - 1),
    )
    _, current_engine = build_target_rows(
        games=current_games,
        prior_summary=preparation.prior_summary,
        prior_elo=preparation.prior_elo,
        min_games=10,
    )
    joblib.dump(
        {
            "season": season,
            "generated_at": datetime.now(EASTERN).isoformat(),
            "engine_through": through.isoformat(),
            "artifact": preparation.artifact,
            "prior_summary": preparation.prior_summary,
            "prior_elo": preparation.prior_elo,
            "engine": current_engine,
        },
        LIVE_MODEL_BUNDLE,
        compress=3,
    )
    print(f"Wrote live model bundle: {LIVE_MODEL_BUNDLE}")

    print(f"Building Model Lab snapshot for {season}...")
    snapshot = await build_mlb_backtest(season=season, min_games=10)
    snapshot_path = backtest_snapshot_path(season)
    snapshot_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote Model Lab snapshot: {snapshot_path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        default=datetime.now(EASTERN).year,
        help="Target season for the live artifact and Model Lab snapshot.",
    )
    args = parser.parse_args()
    await export_artifacts(args.season)


if __name__ == "__main__":
    asyncio.run(main())
