"""Import an exported moneyballPaperAccountV1 JSON file into Postgres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moneyball_predictions.db import ensure_database_initialized, session_scope
from moneyball_predictions.portfolio import all_account_metrics, import_legacy_browser_account


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    ensure_database_initialized()
    with session_scope() as session:
        result = import_legacy_browser_account(session, payload)
        print(result)
        print(json.dumps(all_account_metrics(session), indent=2))


if __name__ == "__main__":
    main()
