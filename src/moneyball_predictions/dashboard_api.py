"""Lightweight dashboard summary and cached section metadata."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Response
from sqlalchemy import func, select

from .db import (
    CronRunRecord,
    LineupSnapshotRecord,
    ensure_database_initialized,
    session_scope,
)
from .portfolio import all_account_metrics

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary")
def dashboard_summary(response: Response) -> dict:
    """Return only fast Postgres-backed state for the initial page render."""
    ensure_database_initialized()
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    with session_scope() as session:
        accounts = all_account_metrics(session)
        confirmed_games = session.scalar(
            select(func.count(func.distinct(LineupSnapshotRecord.game_pk))).where(
                LineupSnapshotRecord.status == "CONFIRMED"
            )
        ) or 0
        latest_cron = session.scalar(
            select(CronRunRecord).order_by(CronRunRecord.started_at.desc()).limit(1)
        )
    return {
        "generated_at": datetime.now(UTC),
        "version": "0.13.0",
        "accounts": accounts,
        "data_freshness": {
            "confirmed_lineup_games_archived": int(confirmed_games),
            "latest_job": (
                {
                    "job_name": latest_cron.job_name,
                    "status": latest_cron.status,
                    "started_at": latest_cron.started_at,
                    "completed_at": latest_cron.completed_at,
                }
                if latest_cron
                else None
            ),
        },
        "sections": {
            "moneylines": "/api/v1/polymarket/mlb",
            "player_props": "/api/v1/props/strikeouts",
            "portfolio": "/api/v1/paper/accounts",
            "edge_research": "/api/v1/edge-performance/mlb",
            "model_lab": "/api/v1/backtest/mlb",
        },
    }
