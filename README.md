# Moneyball Predictions v0.12.2

Local FastAPI application for MLB pregame forecasting, Polymarket moneyline and pitcher-prop research, browser paper trading, archived edge/CLV analysis, and deterministic model experimentation.

## v0.12.2 confirmed-lineup process

Every pregame moneyline now has one standardized lineup state:

- `PENDING`: neither official nine-man batting order is available.
- `PARTIAL`: one lineup or an incomplete lineup is available.
- `CONFIRMED`: both teams have exactly nine official starters.

The canonical parser accepts only MLB `battingOrder` values `100, 200, ..., 900`. Later substitutions such as `101` or `201` are excluded from the announced starting nine.

Moneyline paper betting is locked until:

1. both probable starters have official MLB IDs;
2. both official lineups contain nine starters;
3. lineup-specific offense repricing succeeds; and
4. the repriced side clears the 5% edge and 5% expected-ROI gates.

Before confirmation, the dashboard still shows the early model probability and market price, but the signal is `WAIT` and no paper-bet button appears.

## Confirmed-lineup repricing

For each hitter, v0.12.2 requests current-season hitting statistics against the opposing starter's handedness when the split is available. It creates a shrunken event-rate projection for unintentional walks, hit-by-pitches, singles, doubles, triples, and home runs, then calculates a lineup xwOBA with batting-order weights.

To avoid double-counting elite teams, the adjustment is based on:

```text
(home confirmed-lineup xwOBA - home team baseline xwOBA)
-
(away confirmed-lineup xwOBA - away team baseline xwOBA)
```

That difference is applied conservatively in log-odds space and capped. This is a transparent T1H overlay, not yet a historically promoted replacement for the v0.10 champion. Its coefficient must be validated with archived point-in-time lineups before being increased.

Optional environment controls:

```bash
export MONEYBALL_LINEUP_LOGIT_COEFFICIENT=4.0
export MONEYBALL_LINEUP_LOGIT_CAP=0.20
```

## Immutable lineup storage

The research database now includes `lineup_snapshots`. A new row is inserted only when a team's lineup hash changes. `PENDING`, `PARTIAL`, `CONFIRMED`, and later lineup changes remain permanently auditable.

## Install

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
unzip -o ~/Downloads/moneyball-polymarket-v0.12.2.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Expected:

```text
85 passed
```

## Initialize or migrate storage

```bash
python scripts/init_research_db.py
```

The command preserves existing predictions, market snapshots, paper-bet browser data, and other SQLite records while adding the lineup table.

## Start the server and automatic lineup poller together

```bash
bash scripts/start_local.sh
```

This starts:

```text
FastAPI / Uvicorn
60-second official MLB lineup polling
immutable lineup snapshot storage
```

Press `Control + C` once to stop both processes.

Open:

```text
http://127.0.0.1:8000
```

Hard refresh on macOS with `Command + Shift + R`.

The header should say:

```text
MLB · Polymarket board + confirmed-lineup lab · v0.12.2
```

## Poll or inspect lineups manually

One polling pass:

```bash
python scripts/poll_lineups.py --once
```

Inspect one game's current official lineup state:

```bash
curl -s http://127.0.0.1:8000/api/v1/mlb/game/GAME_PK/lineups \
  | python -m json.tool
```

## Important research limitation

Historical final box scores reveal who ultimately started, but they do not by themselves establish when a lineup became publicly available. Forward lineup snapshots created by this release are therefore the trusted source for future T24H-versus-T1H CLV and ROI testing.

Real-money order submission remains disabled. Paper trading and order-book research remain local.
