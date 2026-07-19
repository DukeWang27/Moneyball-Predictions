#!/usr/bin/env python3
"""Fetch small canonical reference datasets into the gitignored data directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

DEFAULT_MLB_BASE_URL = "https://statsapi.mlb.com/api"


def fetch_json(url: str) -> object:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed trusted API base URL
        return json.load(response)


def fetch_mlb_teams(output_dir: Path) -> Path:
    base_url = os.getenv("MLB_STATS_API_BASE_URL", DEFAULT_MLB_BASE_URL).rstrip("/")
    query = urlencode({"sportId": 1, "hydrate": "venue"})
    payload = fetch_json(f"{base_url}/v1/teams?{query}")
    destination = output_dir / "mlb_teams.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mlb-teams"], default="mlb-teams")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "mlb-teams":
        path = fetch_mlb_teams(args.output_dir)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
