"""Protected HTTP endpoints for external scheduling on Vercel Hobby."""

from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Header, HTTPException, Response

from .jobs import poll_lineups_once, settle_paper_bets_once

router = APIRouter(prefix="/api/internal/cron", tags=["internal cron"])


def _authorize(authorization: str | None) -> None:
    secret = os.environ.get("CRON_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="CRON_SECRET is not configured")
    supplied = authorization or ""
    expected = f"Bearer {secret}"
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid cron authorization")


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store, max-age=0"


@router.get("/lineups")
async def cron_lineups(
    response: Response,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    _no_store(response)
    return await poll_lineups_once()


@router.get("/settle-paper-bets")
async def cron_settle_paper_bets(
    response: Response,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    _no_store(response)
    return await settle_paper_bets_once()
