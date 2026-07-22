"""Apply migrations and seed the two $100 paper accounts."""

from __future__ import annotations

from alembic import command
from alembic.config import Config

from moneyball_predictions.db import ensure_database_initialized, database_health


def main() -> None:
    command.upgrade(Config("alembic.ini"), "head")
    ensure_database_initialized()
    print(database_health())
    print("Seeded MONEYLINE $100 and PLAYER_PROP $100 accounts.")


if __name__ == "__main__":
    main()
