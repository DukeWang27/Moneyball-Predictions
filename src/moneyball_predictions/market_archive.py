"""Immutable market snapshots backed by Postgres in deployment.

A local JSONL path remains available for offline research and backwards-compatible
fixtures. Set ``MONEYBALL_MARKET_ARCHIVE`` to force JSONL mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .db import MarketSnapshotRecord, ensure_database_initialized, session_scope


def snapshot_root() -> Path:
    return Path(os.environ.get("MONEYBALL_MARKET_ARCHIVE", "data/market_snapshots"))


def _jsonl_mode() -> bool:
    return "MONEYBALL_MARKET_ARCHIVE" in os.environ


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(rows: list[dict[str, Any]], generated: datetime) -> Path | None:
    path = snapshot_root() / f"{generated.date().isoformat()}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                payload = {"captured_at": generated.isoformat(), **row}
                handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
    except OSError:
        return None
    return path


def append_market_snapshots(rows: list[dict[str, Any]]) -> Path | str | None:
    """Persist one immutable capture of each market row.

    Persistence remains best-effort so a database outage does not break the live board.
    """
    if not rows:
        return None
    generated = datetime.now(UTC)
    if _jsonl_mode():
        return _append_jsonl(rows, generated)
    try:
        ensure_database_initialized()
        with session_scope() as session:
            for row in rows:
                market_id = str(row.get("market_id") or "").strip()
                if not market_id:
                    continue
                payload = dict(row)
                session.add(
                    MarketSnapshotRecord(
                        game_pk=int(row["game_pk"]) if row.get("game_pk") is not None else None,
                        market_id=market_id,
                        market_type=str(row.get("market_type") or "MONEYLINE"),
                        captured_at_utc=generated,
                        orderbook_hash=_canonical_hash(payload),
                        payload_json=payload,
                    )
                )
        return "postgres"
    except Exception:
        # Vercel has an ephemeral filesystem, but this fallback is still useful locally.
        return _append_jsonl(rows, generated)


def _read_jsonl() -> list[dict[str, Any]]:
    root = snapshot_root()
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def read_market_snapshots() -> list[dict[str, Any]]:
    """Read archived snapshots in chronological order."""
    if _jsonl_mode():
        return _read_jsonl()
    try:
        ensure_database_initialized()
        with session_scope() as session:
            records = list(
                session.scalars(
                    select(MarketSnapshotRecord).order_by(
                        MarketSnapshotRecord.captured_at_utc.asc()
                    )
                ).all()
            )
        return [
            {"captured_at": record.captured_at_utc.isoformat(), **record.payload_json}
            for record in records
        ]
    except Exception:
        return _read_jsonl()
