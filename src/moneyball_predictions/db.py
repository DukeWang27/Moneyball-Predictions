"""Serverless-friendly SQLAlchemy storage for Moneyball Predictions.

The production deployment uses a pooled Neon Postgres connection through
``DATABASE_URL``.  Local development defaults to a small SQLite database so the
application can still be run before Neon is configured.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Iterator

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    text,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


MONEYLINE_ACCOUNT = "MONEYLINE"
PLAYER_PROP_ACCOUNT = "PLAYER_PROP"
ACCOUNT_TYPES = {MONEYLINE_ACCOUNT, PLAYER_PROP_ACCOUNT}
DEFAULT_STARTING_BANKROLL = Decimal("100.00")


def utcnow() -> datetime:
    return datetime.now(UTC)


def normalize_database_url(url: str) -> str:
    """Return a SQLAlchemy 2 URL with an explicit psycopg driver."""
    value = url.strip()
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value[len("postgres://") :]
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def database_url() -> str:
    return normalize_database_url(
        os.environ.get(
            "DATABASE_URL",
            "sqlite+pysqlite:///data/moneyball_app.sqlite3",
        )
    )


class Base(DeclarativeBase):
    pass


class PaperAccount(Base):
    __tablename__ = "paper_accounts"
    __table_args__ = (
        CheckConstraint(
            "account_type IN ('MONEYLINE', 'PLAYER_PROP')",
            name="ck_paper_accounts_type",
        ),
    )

    account_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_type: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    starting_bankroll: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=DEFAULT_STARTING_BANKROLL
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    bets: Mapped[list["PaperBet"]] = relationship(
        back_populates="account", cascade="all, delete-orphan"
    )


class PaperBet(Base):
    __tablename__ = "paper_bets"
    __table_args__ = (
        CheckConstraint(
            "market_type IN ('MONEYLINE', 'STRIKEOUT_PROP')",
            name="ck_paper_bets_market_type",
        ),
        CheckConstraint(
            "status IN ('OPEN', 'WON', 'LOST', 'REFUNDED')",
            name="ck_paper_bets_status",
        ),
        CheckConstraint("entry_price > 0 AND entry_price < 1", name="ck_paper_bets_price"),
        CheckConstraint("model_probability >= 0 AND model_probability <= 1", name="ck_paper_bets_probability"),
        CheckConstraint("stake > 0", name="ck_paper_bets_stake"),
        UniqueConstraint("account_id", "market_id", name="uq_paper_bet_account_market"),
        Index("idx_paper_bets_account_status", "account_id", "status"),
        Index("idx_paper_bets_game", "game_pk"),
    )

    bet_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("paper_accounts.account_id", ondelete="CASCADE"), nullable=False
    )
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False)
    market_id: Mapped[str] = mapped_column(String(255), nullable=False)
    token_id: Mapped[str | None] = mapped_column(String(255))
    selection: Mapped[str] = mapped_column(String(255), nullable=False)
    team: Mapped[str | None] = mapped_column(String(160))
    opponent: Mapped[str | None] = mapped_column(String(160))
    player_id: Mapped[int | None] = mapped_column(BigInteger)
    player_name: Mapped[str | None] = mapped_column(String(160))
    prop_threshold: Mapped[int | None] = mapped_column(Integer)
    prop_side: Mapped[str | None] = mapped_column(String(8))
    model_version: Mapped[str] = mapped_column(String(160), nullable=False)
    model_probability: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_edge: Mapped[float] = mapped_column(Float, nullable=False)
    expected_roi: Mapped[float | None] = mapped_column(Float)
    stake: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    shares: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    actual_result: Mapped[str | None] = mapped_column(Text)
    actual_strikeouts: Mapped[int | None] = mapped_column(Integer)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    polymarket_url: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    account: Mapped[PaperAccount] = relationship(back_populates="bets")


class LineupSnapshotRecord(Base):
    __tablename__ = "lineup_snapshots"
    __table_args__ = (
        CheckConstraint("side IN ('away', 'home')", name="ck_lineup_side"),
        CheckConstraint(
            "status IN ('PENDING', 'PARTIAL', 'CONFIRMED')",
            name="ck_lineup_status",
        ),
        UniqueConstraint("game_pk", "side", "source_hash_sha256", name="uq_lineup_hash"),
        Index("idx_lineup_game_time", "game_pk", "captured_at_utc"),
    )

    lineup_snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    team_id: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    captured_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    players_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)


class PredictionRunRecord(Base):
    __tablename__ = "prediction_runs"
    __table_args__ = (
        UniqueConstraint(
            "game_pk",
            "horizon",
            "as_of_utc",
            "model_version",
            "feature_hash_sha256",
            name="uq_prediction_run",
        ),
        Index("idx_prediction_game_horizon", "game_pk", "horizon", "as_of_utc"),
    )

    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False)
    horizon: Mapped[str] = mapped_column(String(16), nullable=False)
    as_of_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(160), nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    features_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    raw_home_probability: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_home_probability: Mapped[float] = mapped_column(Float, nullable=False)
    home_team: Mapped[str] = mapped_column(String(160), nullable=False)
    away_team: Mapped[str] = mapped_column(String(160), nullable=False)
    lineups_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class MarketSnapshotRecord(Base):
    __tablename__ = "market_snapshots"
    __table_args__ = (
        UniqueConstraint("market_id", "captured_at_utc", name="uq_market_snapshot"),
        Index("idx_market_snapshots_time", "captured_at_utc"),
        Index("idx_market_snapshots_game", "game_pk"),
    )

    snapshot_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    game_pk: Mapped[int | None] = mapped_column(BigInteger)
    market_id: Mapped[str] = mapped_column(String(255), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    captured_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    orderbook_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class PropFamilyDecisionRecord(Base):
    __tablename__ = "prop_family_decisions"
    __table_args__ = (
        UniqueConstraint(
            "game_pk", "player_id", "captured_at_utc", name="uq_prop_family_decision"
        ),
        Index("idx_prop_family_game_player", "game_pk", "player_id"),
    )

    decision_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    game_pk: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_name: Mapped[str] = mapped_column(String(160), nullable=False)
    captured_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_strikeouts: Mapped[float] = mapped_column(Float, nullable=False)
    shrinkage: Mapped[float] = mapped_column(Float, nullable=False)
    best_market_id: Mapped[str | None] = mapped_column(String(255))
    best_side: Mapped[str | None] = mapped_column(String(8))
    warning_code: Mapped[str | None] = mapped_column(String(64))
    evaluation_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class CronRunRecord(Base):
    __tablename__ = "cron_runs"
    __table_args__ = (Index("idx_cron_runs_job_started", "job_name", "started_at"),)

    cron_run_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_name: Mapped[str] = mapped_column(String(80), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    counts_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_text: Mapped[str | None] = mapped_column(Text)


@lru_cache(maxsize=4)
def engine_for_url(url: str) -> Engine:
    if url.startswith("sqlite"):
        database = make_url(url).database
        if database and database != ":memory:":
            from pathlib import Path
            Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **options)


def get_engine() -> Engine:
    return engine_for_url(database_url())


def reset_engine_cache() -> None:
    engine_for_url.cache_clear()
    ensure_database_for_url.cache_clear()


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    session = Session(engine or get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def seed_accounts(session: Session) -> None:
    existing = set(session.scalars(select(PaperAccount.account_type)).all())
    for account_type in sorted(ACCOUNT_TYPES - existing):
        session.add(
            PaperAccount(
                account_type=account_type,
                starting_bankroll=DEFAULT_STARTING_BANKROLL,
            )
        )
    session.flush()


def initialize_database(engine: Engine | None = None) -> None:
    resolved = engine or get_engine()
    Base.metadata.create_all(resolved)
    with session_scope(resolved) as session:
        seed_accounts(session)


@lru_cache(maxsize=4)
def ensure_database_for_url(url: str) -> None:
    initialize_database(engine_for_url(url))


def ensure_database_initialized() -> None:
    ensure_database_for_url(database_url())


def database_health(engine: Engine | None = None) -> dict[str, Any]:
    resolved = engine or get_engine()
    with resolved.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "dialect": resolved.dialect.name,
        "accounts": len(ACCOUNT_TYPES),
    }
