#!/usr/bin/env bash
set -Eeuo pipefail

LINEUP_PID=""
UVICORN_PID=""

cleanup() {
  trap - INT TERM EXIT

  echo
  echo "Stopping Moneyball Predictions..."

  for pid in "$UVICORN_PID" "$LINEUP_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  sleep 0.5

  for pid in "$UVICORN_PID" "$LINEUP_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done

  for pid in "$UVICORN_PID" "$LINEUP_PID"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done

  echo "Stopped."
}

trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

export PYTHONUNBUFFERED=1

python scripts/init_postgres.py

python scripts/poll_lineups.py --interval 60 &
LINEUP_PID=$!

# No --reload here. It creates an additional child process that can
# sometimes remain alive after Control+C.
python -m uvicorn moneyball_predictions.api:app \
  --host 127.0.0.1 \
  --port 8000 &
UVICORN_PID=$!

echo
echo "Moneyball Predictions is running:"
echo "  Dashboard: http://127.0.0.1:8000"
echo "  API PID: $UVICORN_PID"
echo "  Lineup poller PID: $LINEUP_PID"
echo
echo "Press Control+C once to stop both processes."

wait "$UVICORN_PID"
