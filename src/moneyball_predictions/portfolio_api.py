"""FastAPI routes for the two server-owned paper ledgers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from .db import ensure_database_initialized, session_scope
from .portfolio import (
    BetPlacement,
    DuplicateBetError,
    InsufficientCashError,
    PortfolioError,
    all_account_metrics,
    import_legacy_browser_account,
    list_bets,
    place_bet,
    reset_accounts,
    serialize_bet,
)
from .schemas import LegacyPaperImportRequest, PaperBetPlacementRequest

router = APIRouter(prefix="/api/v1/paper", tags=["paper portfolio"])


def _private_no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store, max-age=0"


@router.get("/accounts")
def paper_accounts(response: Response) -> dict:
    ensure_database_initialized()
    _private_no_store(response)
    with session_scope() as session:
        return all_account_metrics(session)


@router.get("/bets")
def paper_bets(response: Response, account_type: str | None = None) -> dict:
    ensure_database_initialized()
    _private_no_store(response)
    with session_scope() as session:
        return {"bets": list_bets(session, account_type=account_type)}


@router.post("/bets", status_code=201)
def create_paper_bet(payload: PaperBetPlacementRequest, response: Response) -> dict:
    ensure_database_initialized()
    _private_no_store(response)
    try:
        with session_scope() as session:
            bet = place_bet(session, BetPlacement(**payload.model_dump()))
            session.refresh(bet)
            result = serialize_bet(bet)
            metrics = all_account_metrics(session)
    except DuplicateBetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InsufficientCashError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PortfolioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"bet": result, "accounts": metrics}


@router.post("/import-browser")
def import_browser_account(payload: LegacyPaperImportRequest, response: Response) -> dict:
    ensure_database_initialized()
    _private_no_store(response)
    try:
        with session_scope() as session:
            counts = import_legacy_browser_account(session, payload.model_dump())
            metrics = all_account_metrics(session)
    except PortfolioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**counts, "accounts": metrics}


@router.delete("/reset")
def reset_paper_accounts(response: Response) -> dict:
    ensure_database_initialized()
    _private_no_store(response)
    with session_scope() as session:
        reset_accounts(session)
        return all_account_metrics(session)
