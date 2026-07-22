"""SQLAlchemy repositories shared by request handlers and scheduled jobs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import CronRunRecord, LineupSnapshotRecord, PropFamilyDecisionRecord
from .lineups import TeamLineup


def insert_lineup_snapshot(session: Session, lineup: TeamLineup) -> bool:
    existing = session.scalar(
        select(LineupSnapshotRecord.lineup_snapshot_id).where(
            LineupSnapshotRecord.game_pk == lineup.game_pk,
            LineupSnapshotRecord.side == lineup.side,
            LineupSnapshotRecord.source_hash_sha256 == lineup.source_hash,
        )
    )
    if existing:
        return False
    snapshot_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"moneyball-lineup:{lineup.game_pk}:{lineup.side}:{lineup.source_hash}",
        )
    )
    captured_at = lineup.captured_at
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=UTC)
    session.add(
        LineupSnapshotRecord(
            lineup_snapshot_id=snapshot_id,
            game_pk=lineup.game_pk,
            side=lineup.side,
            team_id=lineup.team_id,
            status=lineup.status.value,
            captured_at_utc=captured_at.astimezone(UTC),
            source_hash_sha256=lineup.source_hash,
            players_json=[
                {
                    "player_id": player.player_id,
                    "full_name": player.full_name,
                    "batting_slot": player.batting_slot,
                    "position": player.position,
                }
                for player in lineup.players
            ],
        )
    )
    session.flush()
    return True


def start_cron_run(session: Session, job_name: str) -> CronRunRecord:
    row = CronRunRecord(
        job_name=job_name,
        started_at=datetime.now(UTC),
        status="RUNNING",
        counts_json={},
    )
    session.add(row)
    session.flush()
    return row


def finish_cron_run(
    session: Session,
    row: CronRunRecord,
    *,
    counts: dict[str, Any],
    error: Exception | None = None,
) -> None:
    row.completed_at = datetime.now(UTC)
    row.counts_json = counts
    row.status = "FAILED" if error else "COMPLETED"
    row.error_text = str(error) if error else None
    session.flush()


def insert_prop_family_decisions(session: Session, board) -> int:
    created = 0
    for family in board.families:
        session.add(
            PropFamilyDecisionRecord(
                game_pk=family.game_pk,
                player_id=family.player_id,
                player_name=family.player_name,
                captured_at_utc=board.generated_at,
                expected_strikeouts=family.expected_strikeouts,
                shrinkage=family.shrinkage,
                best_market_id=family.best_market_id,
                best_side=family.best_side,
                warning_code=family.warning_code,
                evaluation_json=family.model_dump(mode="json"),
            )
        )
        created += 1
    session.flush()
    return created
