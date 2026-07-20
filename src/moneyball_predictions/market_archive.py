"""Append-only local archive for future CLV and execution research."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def snapshot_root() -> Path:
    return Path(os.environ.get("MONEYBALL_MARKET_ARCHIVE", "data/market_snapshots"))


def append_market_snapshots(rows: list[dict[str, Any]]) -> Path | None:
    """Append snapshots to a date-partitioned JSONL file.

    Archiving is intentionally best-effort so a local disk error never breaks the live
    prediction dashboard.
    """
    if not rows:
        return None
    generated = datetime.now(UTC)
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


def read_market_snapshots() -> list[dict[str, Any]]:
    """Read every valid archived JSONL snapshot in chronological file order."""
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
