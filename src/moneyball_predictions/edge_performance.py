"""Evaluate archived pregame market edges against settled MLB results.

This module deliberately separates two concepts:

* economic edge: model probability minus the executable market ask; and
* sabermetric support: whether independent baseball driver groups agree with the side.

Bill James-style team strength helps produce the model probability. It does not replace
price when deciding whether a contract is valuable. The confirmed strategy therefore
requires both a positive economic edge and supporting baseball drivers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any, Iterable

import httpx

from .market_archive import read_market_snapshots
from .mlb import MlbDataError, MlbGameState, fetch_mlb_schedule
from .schemas import EdgePerformanceBet, EdgePerformanceMetrics, EdgePerformanceResponse

EDGE_THRESHOLDS = (0.00, 0.02, 0.03, 0.05, 0.08)
MAX_ENTRY_AGE_MINUTES = 24 * 60


class EdgePerformanceError(RuntimeError):
    """Raised when archived edge performance cannot be evaluated."""


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _horizon_minutes(row: dict[str, Any]) -> float | None:
    captured = _parse_datetime(row.get("captured_at"))
    start = _parse_datetime(row.get("game_start"))
    if captured is None or start is None:
        return None
    return (start - captured).total_seconds() / 60.0


def _select_snapshots(
    rows: Iterable[dict[str, Any]],
    *,
    entry_horizon_minutes: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select one honest entry snapshot and one closing snapshot per game.

    Entry is the latest available snapshot captured at least ``entry_horizon_minutes``
    before first pitch. Closing is the last snapshot before first pitch. Entries older
    than 24 hours are omitted instead of pretending they represent the requested horizon.
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        game_pk = _safe_int(row.get("game_pk"))
        horizon = _horizon_minutes(row)
        if game_pk is None or horizon is None or horizon <= 0:
            continue
        grouped[game_pk].append(row)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for game_rows in grouped.values():
        pregame = sorted(
            game_rows,
            key=lambda item: _parse_datetime(item.get("captured_at")) or datetime.min.replace(tzinfo=UTC),
        )
        entry_candidates = [
            row
            for row in pregame
            if entry_horizon_minutes <= (_horizon_minutes(row) or -1) <= MAX_ENTRY_AGE_MINUTES
        ]
        if not entry_candidates:
            continue
        entry = entry_candidates[-1]
        close = pregame[-1]
        selected.append((entry, close))
    return selected


def _midpoint(row: dict[str, Any], side: str) -> float | None:
    bid = _safe_float(row.get(f"best_bid_{side}"))
    ask = _safe_float(row.get(f"best_ask_{side}"))
    if bid is not None and ask is not None and 0 < bid < 1 and 0 < ask < 1:
        return (bid + ask) / 2.0
    if ask is not None and 0 < ask < 1:
        return ask
    if bid is not None and 0 < bid < 1:
        return bid
    return None


def _closing_no_vig_probability(row: dict[str, Any], side: str) -> float | None:
    mid_a = _midpoint(row, "a")
    mid_b = _midpoint(row, "b")
    if mid_a is None or mid_b is None or mid_a + mid_b <= 0:
        return None
    probability_a = mid_a / (mid_a + mid_b)
    return probability_a if side == "a" else 1.0 - probability_a


def _best_side(row: dict[str, Any]) -> dict[str, Any] | None:
    sides: list[dict[str, Any]] = []
    for side in ("a", "b"):
        team = str(row.get(f"team_{side}") or "").strip()
        probability = _safe_float(row.get(f"model_probability_{side}"))
        price = _safe_float(row.get(f"best_ask_{side}"))
        if not team or probability is None or price is None or not 0 < price < 1:
            continue
        expected_roi = probability / price - 1.0
        sides.append(
            {
                "side": side,
                "team": team,
                "probability": probability,
                "price": price,
                "edge": probability - price,
                "expected_roi": expected_roi,
                "support": str(row.get(f"sabermetric_support_{side}") or "UNKNOWN"),
                "support_count": _safe_int(row.get(f"sabermetric_support_count_{side}")) or 0,
                "support_total": _safe_int(row.get(f"sabermetric_support_total_{side}")) or 0,
            }
        )
    return max(sides, key=lambda item: item["expected_roi"]) if sides else None


def _candidate_rows(
    snapshots: list[tuple[dict[str, Any], dict[str, Any]]],
    games: dict[int, MlbGameState],
) -> list[EdgePerformanceBet]:
    candidates: list[EdgePerformanceBet] = []
    for entry, close in snapshots:
        game_pk = _safe_int(entry.get("game_pk"))
        selected = _best_side(entry)
        if game_pk is None or selected is None:
            continue
        game = games.get(game_pk)
        winner = game.winner if game is not None else None
        won = winner == selected["team"] if winner else None
        profit = None
        if won is True:
            profit = 1.0 / selected["price"] - 1.0
        elif won is False:
            profit = -1.0
        close_probability = _closing_no_vig_probability(close, selected["side"])
        clv = (
            close_probability - selected["price"]
            if close_probability is not None
            else None
        )
        opponent_side = "b" if selected["side"] == "a" else "a"
        candidates.append(
            EdgePerformanceBet(
                game_pk=game_pk,
                game_start=str(entry.get("game_start") or ""),
                captured_at=str(entry.get("captured_at") or ""),
                actual_entry_horizon_minutes=max(int(round(_horizon_minutes(entry) or 0)), 0),
                team=selected["team"],
                opponent=str(entry.get(f"team_{opponent_side}") or ""),
                model_probability=selected["probability"],
                entry_price=selected["price"],
                edge=selected["edge"],
                expected_roi=selected["expected_roi"],
                sabermetric_support=selected["support"],
                sabermetric_support_count=selected["support_count"],
                sabermetric_support_total=selected["support_total"],
                winner=winner,
                won=won,
                profit_per_dollar=profit,
                closing_no_vig_probability=close_probability,
                clv=clv,
                model_version=str(entry.get("model_version") or "unknown"),
            )
        )
    return sorted(candidates, key=lambda item: item.game_start)


def _max_drawdown(profits: list[float]) -> float | None:
    if not profits:
        return None
    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for profit in profits:
        cumulative += profit
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return worst


def _metrics(
    candidates: list[EdgePerformanceBet],
    *,
    threshold: float,
    confirmed_only: bool,
) -> EdgePerformanceMetrics:
    qualified = [
        row
        for row in candidates
        if row.edge >= threshold
        and row.expected_roi >= threshold
        and (not confirmed_only or row.sabermetric_support == "CONFIRMED")
    ]
    settled = [row for row in qualified if row.won is not None and row.profit_per_dollar is not None]
    wins = sum(1 for row in settled if row.won)
    profits = [float(row.profit_per_dollar) for row in settled if row.profit_per_dollar is not None]
    clv_rows = [row.clv for row in settled if row.clv is not None]
    brier_values = [
        (row.model_probability - (1.0 if row.won else 0.0)) ** 2 for row in settled
    ]
    return EdgePerformanceMetrics(
        strategy="Sabermetric confirmed" if confirmed_only else "All economic edges",
        min_edge=threshold,
        eligible_signals=len(qualified),
        settled_bets=len(settled),
        wins=wins,
        losses=len(settled) - wins,
        win_rate=wins / len(settled) if settled else None,
        average_model_probability=(
            sum(row.model_probability for row in settled) / len(settled) if settled else None
        ),
        average_entry_price=(
            sum(row.entry_price for row in settled) / len(settled) if settled else None
        ),
        average_edge=sum(row.edge for row in settled) / len(settled) if settled else None,
        average_expected_roi=(
            sum(row.expected_roi for row in settled) / len(settled) if settled else None
        ),
        brier_score=sum(brier_values) / len(brier_values) if brier_values else None,
        flat_stake_profit=sum(profits) if profits else None,
        flat_stake_roi=sum(profits) / len(profits) if profits else None,
        average_clv=sum(clv_rows) / len(clv_rows) if clv_rows else None,
        positive_clv_rate=(
            sum(1 for value in clv_rows if value > 0) / len(clv_rows) if clv_rows else None
        ),
        max_drawdown=_max_drawdown(profits),
    )


def analyze_archived_snapshots(
    rows: list[dict[str, Any]],
    games: dict[int, MlbGameState],
    *,
    entry_horizon_minutes: int = 60,
) -> EdgePerformanceResponse:
    selected = _select_snapshots(rows, entry_horizon_minutes=entry_horizon_minutes)
    candidates = _candidate_rows(selected, games)
    metrics = [
        _metrics(candidates, threshold=threshold, confirmed_only=confirmed)
        for threshold in EDGE_THRESHOLDS
        for confirmed in (False, True)
    ]
    settled_games = sum(1 for row in candidates if row.won is not None)
    open_games = len(candidates) - settled_games
    capture_times = [
        parsed
        for parsed in (_parse_datetime(row.get("captured_at")) for row in rows)
        if parsed is not None
    ]
    recent = sorted(candidates, key=lambda item: item.game_start, reverse=True)[:20]
    return EdgePerformanceResponse(
        generated_at=datetime.now(UTC),
        entry_horizon_minutes=entry_horizon_minutes,
        snapshots_read=len(rows),
        selected_games=len(candidates),
        settled_games=settled_games,
        open_games=open_games,
        first_capture=min(capture_times) if capture_times else None,
        last_capture=max(capture_times) if capture_times else None,
        metrics=metrics,
        recent_candidates=recent,
        note=(
            "Economic edge is model probability minus the executable ask. Sabermetric "
            "confirmation is a filter requiring independent team-strength and pitching/park "
            "drivers to agree; it is not added to the probability a second time. Results cover "
            "only snapshots this app actually archived before first pitch."
        ),
    )


async def build_edge_performance(
    *,
    entry_horizon_minutes: int = 60,
) -> EdgePerformanceResponse:
    rows = read_market_snapshots()
    if not rows:
        return analyze_archived_snapshots(
            [], {}, entry_horizon_minutes=entry_horizon_minutes
        )

    dates = [
        parsed.date()
        for parsed in (_parse_datetime(row.get("game_start")) for row in rows)
        if parsed is not None
    ]
    if not dates:
        return analyze_archived_snapshots(
            rows, {}, entry_horizon_minutes=entry_horizon_minutes
        )

    start_date = min(dates)
    end_date = max(max(dates), date.today())
    timeout = httpx.Timeout(90.0, connect=10.0)
    headers = {"User-Agent": "Moneyball-Predictions/0.9.1 edge-performance"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            schedule = await fetch_mlb_schedule(client, start_date, end_date)
    except (httpx.HTTPError, MlbDataError) as exc:
        raise EdgePerformanceError(str(exc)) from exc

    games = {game.game_pk: game for game in schedule}
    return analyze_archived_snapshots(
        rows,
        games,
        entry_horizon_minutes=entry_horizon_minutes,
    )
