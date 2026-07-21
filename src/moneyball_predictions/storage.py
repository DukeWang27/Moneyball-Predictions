"""SQLite-backed immutable prediction and execution research registry."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .reproducibility import canonical_json, feature_sha256, sha256_json

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prediction_runs (
    prediction_id TEXT PRIMARY KEY,
    game_id INTEGER NOT NULL,
    horizon TEXT NOT NULL CHECK (horizon IN ('T24H', 'T1H', 'MANUAL')),
    as_of_utc TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    code_commit_sha TEXT NOT NULL,
    model_artifact_sha256 TEXT NOT NULL,
    calibration_artifact_sha256 TEXT,
    feature_hash_sha256 TEXT NOT NULL,
    source_snapshot_sha256 TEXT NOT NULL,
    feature_json TEXT NOT NULL,
    raw_home_probability REAL NOT NULL CHECK (raw_home_probability BETWEEN 0 AND 1),
    calibrated_home_probability REAL NOT NULL CHECK (calibrated_home_probability BETWEEN 0 AND 1),
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    probable_home_pitcher_id INTEGER,
    probable_away_pitcher_id INTEGER,
    lineups_confirmed INTEGER NOT NULL CHECK (lineups_confirmed IN (0, 1)),
    created_at_utc TEXT NOT NULL,
    UNIQUE (game_id, horizon, as_of_utc, model_version, feature_hash_sha256)
);

CREATE INDEX IF NOT EXISTS idx_prediction_game_horizon
ON prediction_runs (game_id, horizon, as_of_utc);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_frozen_horizon
ON prediction_runs (game_id, horizon, model_version)
WHERE horizon IN ('T24H', 'T1H');

CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    game_id INTEGER NOT NULL,
    prediction_id TEXT,
    token_id TEXT NOT NULL,
    captured_at_utc TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    bid_depth_json TEXT NOT NULL,
    ask_depth_json TEXT NOT NULL,
    orderbook_hash TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES prediction_runs(prediction_id)
);

CREATE TABLE IF NOT EXISTS orders (
    local_order_id TEXT PRIMARY KEY,
    venue_order_id TEXT UNIQUE,
    prediction_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    side TEXT NOT NULL,
    token_id TEXT NOT NULL,
    order_type TEXT NOT NULL,
    post_only INTEGER NOT NULL,
    limit_price REAL NOT NULL,
    requested_shares REAL NOT NULL,
    required_edge REAL NOT NULL,
    model_probability REAL NOT NULL,
    status TEXT NOT NULL,
    submitted_at_utc TEXT NOT NULL,
    cancelled_at_utc TEXT,
    FOREIGN KEY (prediction_id) REFERENCES prediction_runs(prediction_id),
    FOREIGN KEY (snapshot_id) REFERENCES market_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    local_order_id TEXT NOT NULL,
    venue_trade_id TEXT UNIQUE,
    fill_price REAL NOT NULL,
    fill_shares REAL NOT NULL,
    fee_dollars REAL NOT NULL DEFAULT 0,
    maker INTEGER NOT NULL CHECK (maker IN (0, 1)),
    filled_at_utc TEXT NOT NULL,
    FOREIGN KEY (local_order_id) REFERENCES orders(local_order_id)
);
"""


def research_db_path() -> Path:
    return Path(os.environ.get("MONEYBALL_RESEARCH_DB", "data/moneyball_research.sqlite3"))


def connect(path: Path | None = None) -> sqlite3.Connection:
    resolved = path or research_db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


@dataclass(frozen=True)
class ImmutablePrediction:
    prediction_id: str
    feature_hash_sha256: str
    created: bool


def insert_prediction(
    *,
    game_id: int,
    horizon: str,
    as_of: datetime,
    model_version: str,
    feature_schema_version: str,
    code_commit_sha: str,
    model_artifact_sha256: str,
    calibration_artifact_sha256: str | None,
    features: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    raw_home_probability: float,
    calibrated_home_probability: float,
    home_team: str,
    away_team: str,
    probable_home_pitcher_id: int | None,
    probable_away_pitcher_id: int | None,
    lineups_confirmed: bool,
    path: Path | None = None,
) -> ImmutablePrediction:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if horizon not in {"T24H", "T1H", "MANUAL"}:
        raise ValueError("Unsupported prediction horizon")
    if not 0.0 <= raw_home_probability <= 1.0:
        raise ValueError("raw_home_probability must be in [0, 1]")
    if not 0.0 <= calibrated_home_probability <= 1.0:
        raise ValueError("calibrated_home_probability must be in [0, 1]")

    feature_hash = feature_sha256(features)
    source_hash = sha256_json(source_snapshot)
    as_of_text = as_of.astimezone(UTC).isoformat()
    prediction_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"moneyball:{game_id}:{horizon}:{as_of_text}:{model_version}:{feature_hash}",
    ))
    created_at = datetime.now(UTC).isoformat()

    with connect(path) as connection:
        before = connection.total_changes
        connection.execute(
            """
            INSERT OR IGNORE INTO prediction_runs (
                prediction_id, game_id, horizon, as_of_utc, model_version,
                feature_schema_version, code_commit_sha, model_artifact_sha256,
                calibration_artifact_sha256, feature_hash_sha256,
                source_snapshot_sha256, feature_json, raw_home_probability,
                calibrated_home_probability, home_team, away_team,
                probable_home_pitcher_id, probable_away_pitcher_id,
                lineups_confirmed, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                game_id,
                horizon,
                as_of_text,
                model_version,
                feature_schema_version,
                code_commit_sha,
                model_artifact_sha256,
                calibration_artifact_sha256,
                feature_hash,
                source_hash,
                canonical_json(features),
                raw_home_probability,
                calibrated_home_probability,
                home_team,
                away_team,
                probable_home_pitcher_id,
                probable_away_pitcher_id,
                int(lineups_confirmed),
                created_at,
            ),
        )
        created = connection.total_changes > before
        if not created and horizon in {"T24H", "T1H"}:
            existing = connection.execute(
                """
                SELECT prediction_id, feature_hash_sha256
                FROM prediction_runs
                WHERE game_id = ? AND horizon = ? AND model_version = ?
                ORDER BY created_at_utc ASC LIMIT 1
                """,
                (game_id, horizon, model_version),
            ).fetchone()
            if existing is not None:
                return ImmutablePrediction(
                    str(existing["prediction_id"]),
                    str(existing["feature_hash_sha256"]),
                    False,
                )
    return ImmutablePrediction(prediction_id, feature_hash, created)


def insert_market_snapshot(
    *,
    game_id: int,
    prediction_id: str | None,
    token_id: str,
    captured_at: datetime,
    bids: list[dict[str, Any]],
    asks: list[dict[str, Any]],
    best_bid: float | None,
    best_ask: float | None,
    path: Path | None = None,
) -> str:
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    payload = {"bids": bids, "asks": asks}
    orderbook_hash = sha256_json(payload)
    snapshot_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"moneyball-book:{game_id}:{token_id}:{captured_at.astimezone(UTC).isoformat()}:{orderbook_hash}",
    ))
    with connect(path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO market_snapshots (
                snapshot_id, game_id, prediction_id, token_id, captured_at_utc,
                best_bid, best_ask, bid_depth_json, ask_depth_json, orderbook_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                game_id,
                prediction_id,
                token_id,
                captured_at.astimezone(UTC).isoformat(),
                best_bid,
                best_ask,
                json.dumps(bids, separators=(",", ":"), sort_keys=True),
                json.dumps(asks, separators=(",", ":"), sort_keys=True),
                orderbook_hash,
            ),
        )
    return snapshot_id


def database_summary(path: Path | None = None) -> dict[str, int | str]:
    resolved = path or research_db_path()
    with connect(resolved) as connection:
        counts = {}
        for table in ("prediction_runs", "market_snapshots", "orders", "fills"):
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
            counts[table] = int(row["count"] if row is not None else 0)
    return {"path": str(resolved), **counts}
