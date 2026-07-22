"""Postgres foundation with separate paper ledgers.

Revision ID: 20260721_0001
Revises:
Create Date: 2026-07-21
"""

from alembic import op

from moneyball_predictions.db import Base

revision = "20260721_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
