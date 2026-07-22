#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  if [[ -n "${LINEUP_PID:-}" ]]; then
    kill "$LINEUP_PID" 2>/dev/null || true
    wait "$LINEUP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1
python scripts/poll_lineups.py --interval 60 &
LINEUP_PID=$!
python -m uvicorn moneyball_predictions.api:app --reload
