"""Postgres-backed paper portfolio services.

The server owns all balances and grading math.  The frontend only submits a bet
request and renders the resulting ledger.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import (
    MONEYLINE_ACCOUNT,
    PLAYER_PROP_ACCOUNT,
    PaperAccount,
    PaperBet,
    initialize_database,
    session_scope,
)

CENT = Decimal("0.01")


class PortfolioError(RuntimeError):
    pass


class InsufficientCashError(PortfolioError):
    pass


class DuplicateBetError(PortfolioError):
    pass


@dataclass(frozen=True)
class BetPlacement:
    account_type: Literal["MONEYLINE", "PLAYER_PROP"]
    market_type: Literal["MONEYLINE", "STRIKEOUT_PROP"]
    game_pk: int
    market_id: str
    selection: str
    model_version: str
    model_probability: float
    entry_price: float
    entry_edge: float
    stake: float
    expected_roi: float | None = None
    token_id: str | None = None
    team: str | None = None
    opponent: str | None = None
    player_id: int | None = None
    player_name: str | None = None
    prop_threshold: int | None = None
    prop_side: Literal["YES", "NO"] | None = None
    placed_at: datetime | None = None
    start_time: datetime | None = None
    polymarket_url: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AccountMetrics:
    account_type: str
    starting_bankroll: float
    current_bankroll: float
    available_cash: float
    open_exposure: float
    realized_pnl: float
    total_return: float
    open_bets: int
    settled_bets: int
    wins: int
    losses: int
    accuracy: float | None
    average_entry_edge: float | None
    brier_score: float | None


def _money(value: Decimal | float | int) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def _as_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _account(session: Session, account_type: str) -> PaperAccount:
    account = session.scalar(
        select(PaperAccount).where(PaperAccount.account_type == account_type)
    )
    if account is None:
        raise PortfolioError(f"Paper account {account_type!r} is not initialized")
    return account


def account_metrics(session: Session, account_type: str) -> AccountMetrics:
    account = _account(session, account_type)
    bets = list(
        session.scalars(
            select(PaperBet)
            .where(PaperBet.account_id == account.account_id)
            .order_by(PaperBet.placed_at.asc())
        ).all()
    )
    open_bets = [bet for bet in bets if bet.status == "OPEN"]
    graded = [bet for bet in bets if bet.status in {"WON", "LOST"}]
    wins = sum(bet.status == "WON" for bet in graded)
    losses = sum(bet.status == "LOST" for bet in graded)
    realized_pnl = sum(
        (bet.realized_pnl or Decimal("0")) for bet in bets if bet.status != "OPEN"
    )
    open_exposure = sum((bet.stake for bet in open_bets), Decimal("0"))
    current_bankroll = account.starting_bankroll + realized_pnl
    available_cash = current_bankroll - open_exposure
    probability_bets = [
        bet
        for bet in graded
        if bet.model_probability is not None and 0 <= bet.model_probability <= 1
    ]
    brier = None
    if probability_bets:
        brier = sum(
            (float(bet.model_probability) - (1.0 if bet.status == "WON" else 0.0)) ** 2
            for bet in probability_bets
        ) / len(probability_bets)
    edge_bets = [bet for bet in bets if bet.entry_edge is not None]
    average_edge = (
        sum(float(bet.entry_edge) for bet in edge_bets) / len(edge_bets)
        if edge_bets
        else None
    )
    settled = wins + losses
    return AccountMetrics(
        account_type=account.account_type,
        starting_bankroll=float(account.starting_bankroll),
        current_bankroll=float(current_bankroll),
        available_cash=float(available_cash),
        open_exposure=float(open_exposure),
        realized_pnl=float(realized_pnl),
        total_return=float(realized_pnl / account.starting_bankroll)
        if account.starting_bankroll
        else 0.0,
        open_bets=len(open_bets),
        settled_bets=settled,
        wins=wins,
        losses=losses,
        accuracy=(wins / settled) if settled else None,
        average_entry_edge=average_edge,
        brier_score=brier,
    )


def all_account_metrics(session: Session) -> dict[str, Any]:
    moneyline = account_metrics(session, MONEYLINE_ACCOUNT)
    props = account_metrics(session, PLAYER_PROP_ACCOUNT)
    return {
        "moneyline": moneyline.__dict__,
        "player_props": props.__dict__,
        "combined": {
            "starting_bankroll": moneyline.starting_bankroll + props.starting_bankroll,
            "current_bankroll": moneyline.current_bankroll + props.current_bankroll,
            "available_cash": moneyline.available_cash + props.available_cash,
            "open_exposure": moneyline.open_exposure + props.open_exposure,
            "realized_pnl": moneyline.realized_pnl + props.realized_pnl,
            "total_return": (
                (moneyline.realized_pnl + props.realized_pnl)
                / (moneyline.starting_bankroll + props.starting_bankroll)
            ),
        },
    }


def list_bets(session: Session, account_type: str | None = None) -> list[dict[str, Any]]:
    statement: Select[tuple[PaperBet]] = select(PaperBet).order_by(PaperBet.placed_at.desc())
    if account_type:
        account = _account(session, account_type)
        statement = statement.where(PaperBet.account_id == account.account_id)
    rows = list(session.scalars(statement).all())
    return [serialize_bet(row) for row in rows]


def serialize_bet(bet: PaperBet) -> dict[str, Any]:
    return {
        "bet_id": bet.bet_id,
        "account_type": bet.account.account_type if bet.account else None,
        "market_type": bet.market_type,
        "game_pk": bet.game_pk,
        "market_id": bet.market_id,
        "token_id": bet.token_id,
        "selection": bet.selection,
        "team": bet.team,
        "opponent": bet.opponent,
        "player_id": bet.player_id,
        "player_name": bet.player_name,
        "prop_threshold": bet.prop_threshold,
        "prop_side": bet.prop_side,
        "model_version": bet.model_version,
        "model_probability": bet.model_probability,
        "entry_price": bet.entry_price,
        "entry_edge": bet.entry_edge,
        "expected_roi": bet.expected_roi,
        "stake": float(bet.stake),
        "shares": float(bet.shares),
        "status": bet.status,
        "actual_result": bet.actual_result,
        "actual_strikeouts": bet.actual_strikeouts,
        "realized_pnl": float(bet.realized_pnl) if bet.realized_pnl is not None else None,
        "placed_at": bet.placed_at,
        "settled_at": bet.settled_at,
        "start_time": bet.start_time,
        "polymarket_url": bet.polymarket_url,
        "metadata": bet.metadata_json,
    }


def _validate_placement(payload: BetPlacement) -> None:
    if payload.account_type == MONEYLINE_ACCOUNT and payload.market_type != "MONEYLINE":
        raise PortfolioError("Moneyline bets must use the moneyline account")
    if payload.account_type == PLAYER_PROP_ACCOUNT and payload.market_type != "STRIKEOUT_PROP":
        raise PortfolioError("Strikeout props must use the player-prop account")
    if not 0.0 <= payload.model_probability <= 1.0:
        raise PortfolioError("model_probability must be between zero and one")
    if not 0.0 < payload.entry_price < 1.0:
        raise PortfolioError("entry_price must be between zero and one")
    if payload.stake < 0.01:
        raise PortfolioError("stake must be at least $0.01")
    if payload.market_type == "STRIKEOUT_PROP":
        if payload.player_id is None or payload.prop_threshold is None or payload.prop_side is None:
            raise PortfolioError("Strikeout prop player, threshold, and side are required")


def place_bet(session: Session, payload: BetPlacement) -> PaperBet:
    _validate_placement(payload)
    account = _account(session, payload.account_type)
    metrics = account_metrics(session, payload.account_type)
    stake = _money(payload.stake)
    if stake > _money(metrics.available_cash):
        raise InsufficientCashError(
            f"Requested ${stake} but only ${metrics.available_cash:.2f} is available"
        )
    existing = session.scalar(
        select(PaperBet).where(
            PaperBet.account_id == account.account_id,
            PaperBet.market_id == payload.market_id,
        )
    )
    if existing is not None:
        raise DuplicateBetError("A paper bet already exists for this market")
    if payload.market_type == "STRIKEOUT_PROP":
        open_same_pitcher = session.scalar(
            select(PaperBet).where(
                PaperBet.account_id == account.account_id,
                PaperBet.market_type == "STRIKEOUT_PROP",
                PaperBet.game_pk == payload.game_pk,
                PaperBet.player_id == payload.player_id,
                PaperBet.status == "OPEN",
            )
        )
        if open_same_pitcher is not None:
            raise DuplicateBetError(
                "Only one open strikeout prop is allowed per pitcher and game"
            )
    placed_at = _as_utc(payload.placed_at)
    bet = PaperBet(
        account_id=account.account_id,
        market_type=payload.market_type,
        game_pk=payload.game_pk,
        market_id=payload.market_id,
        token_id=payload.token_id,
        selection=payload.selection,
        team=payload.team,
        opponent=payload.opponent,
        player_id=payload.player_id,
        player_name=payload.player_name,
        prop_threshold=payload.prop_threshold,
        prop_side=payload.prop_side,
        model_version=payload.model_version,
        model_probability=payload.model_probability,
        entry_price=payload.entry_price,
        entry_edge=payload.entry_edge,
        expected_roi=payload.expected_roi,
        stake=stake,
        shares=(stake / Decimal(str(payload.entry_price))).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        ),
        status="OPEN",
        placed_at=placed_at,
        start_time=_as_utc(payload.start_time) if payload.start_time else None,
        polymarket_url=payload.polymarket_url,
        metadata_json=payload.metadata or {},
    )
    session.add(bet)
    try:
        session.flush()
    except IntegrityError as exc:
        raise DuplicateBetError("A paper bet already exists for this market") from exc
    return bet


def settle_bet(
    session: Session,
    *,
    bet_id: str,
    status: Literal["WON", "LOST", "REFUNDED"],
    actual_result: str | None = None,
    actual_strikeouts: int | None = None,
    settled_at: datetime | None = None,
) -> PaperBet:
    bet = session.get(PaperBet, bet_id)
    if bet is None:
        raise PortfolioError("Paper bet was not found")
    if bet.status != "OPEN":
        return bet
    if status == "WON":
        realized = Decimal(str(bet.shares)) - Decimal(str(bet.stake))
    elif status == "LOST":
        realized = -Decimal(str(bet.stake))
    else:
        realized = Decimal("0")
    bet.status = status
    bet.actual_result = actual_result
    bet.actual_strikeouts = actual_strikeouts
    bet.realized_pnl = _money(realized)
    bet.settled_at = _as_utc(settled_at)
    session.flush()
    return bet


def reset_accounts(session: Session) -> None:
    session.query(PaperBet).delete()
    session.flush()


def _legacy_market_id(bet: dict[str, Any], bet_type: str) -> str:
    supplied = str(bet.get("marketId") or "").strip()
    if supplied:
        return supplied
    identity = "|".join(
        str(value)
        for value in (
            bet_type,
            bet.get("gamePk"),
            bet.get("team"),
            bet.get("playerId"),
            bet.get("threshold"),
            bet.get("side"),
        )
    )
    return "legacy-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def import_legacy_browser_account(
    session: Session,
    payload: dict[str, Any],
) -> dict[str, int]:
    """Import the old mixed localStorage account into the two new ledgers.

    Both new ledgers retain the required $100 starting bankroll.  Bets are split by
    market type and each balance is reconstructed from its own open exposure and
    realized results.
    """
    bets = payload.get("bets") if isinstance(payload, dict) else None
    if not isinstance(bets, list):
        raise PortfolioError("Legacy account must contain a bets array")
    imported = skipped = 0
    for raw in bets:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        is_prop = raw.get("betType") == "strikeout_prop"
        account_type = PLAYER_PROP_ACCOUNT if is_prop else MONEYLINE_ACCOUNT
        market_type = "STRIKEOUT_PROP" if is_prop else "MONEYLINE"
        status = str(raw.get("status") or "open").upper()
        if status not in {"OPEN", "WON", "LOST", "REFUNDED"}:
            status = "OPEN"
        price = float(raw.get("price") or 0)
        stake = float(raw.get("stake") or 0)
        probability = float(raw.get("modelProbability") or 0.5)
        if not (0 < price < 1 and stake >= 0.01 and 0 <= probability <= 1):
            skipped += 1
            continue
        market_id = _legacy_market_id(raw, market_type)
        account = _account(session, account_type)
        exists = session.scalar(
            select(PaperBet).where(
                PaperBet.account_id == account.account_id,
                PaperBet.market_id == market_id,
            )
        )
        if exists:
            skipped += 1
            continue
        placed_at_raw = raw.get("placedAt")
        start_time_raw = raw.get("startTime")
        try:
            placed_at = datetime.fromisoformat(str(placed_at_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            placed_at = datetime.now(UTC)
        try:
            start_time = datetime.fromisoformat(str(start_time_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            start_time = None
        bet = PaperBet(
            account_id=account.account_id,
            market_type=market_type,
            game_pk=int(raw.get("gamePk") or 0),
            market_id=market_id,
            selection=(
                f"{raw.get('playerName')} {raw.get('threshold')}+ K {raw.get('side')}"
                if is_prop
                else str(raw.get("team") or "Unknown")
            ),
            team=None if is_prop else str(raw.get("team") or "Unknown"),
            opponent=str(raw.get("opponent") or "") or None,
            player_id=int(raw.get("playerId")) if raw.get("playerId") is not None else None,
            player_name=str(raw.get("playerName") or "") or None,
            prop_threshold=int(raw.get("threshold")) if raw.get("threshold") is not None else None,
            prop_side=str(raw.get("side") or "") or None,
            model_version=str(raw.get("modelVersion") or "legacy-browser"),
            model_probability=probability,
            entry_price=price,
            entry_edge=float(raw.get("edge") or 0),
            expected_roi=(
                float(raw.get("expectedRoi"))
                if raw.get("expectedRoi") is not None
                else None
            ),
            stake=_money(stake),
            shares=(Decimal(str(stake)) / Decimal(str(price))).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            ),
            status=status,
            actual_strikeouts=(
                int(raw.get("actualStrikeouts"))
                if raw.get("actualStrikeouts") is not None
                else None
            ),
            realized_pnl=(
                _money(float(raw.get("profit") or 0)) if status != "OPEN" else None
            ),
            placed_at=_as_utc(placed_at),
            settled_at=datetime.now(UTC) if status != "OPEN" else None,
            start_time=_as_utc(start_time) if start_time else None,
            polymarket_url=str(raw.get("polymarketUrl") or "") or None,
            metadata_json={"legacy_id": raw.get("id")},
        )
        session.add(bet)
        imported += 1
    session.flush()
    return {"imported": imported, "skipped": skipped}


def initialized_metrics() -> dict[str, Any]:
    initialize_database()
    with session_scope() as session:
        return all_account_metrics(session)
