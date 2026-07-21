# Moneyball Predictions v0.12.1

Local FastAPI application for MLB pregame forecasting, Polymarket moneyline and player-prop research, browser paper trading, archived edge/CLV analysis, and deterministic model experimentation.

## v0.12 additions

- Adds a live, read-only **pitcher strikeout prop board**.
- Discovers active Polymarket strikeout markets without assuming one permanent market-type name.
- Maps market player names to official MLB player IDs with exact matching, a persistent player cache, and conservative fuzzy fallback.
- Matches the pitcher to the official MLB schedule and confirmed probable starter.
- Builds a transparent Poisson strikeout baseline from:
  - regressed pitcher strikeout rate,
  - projected starter innings,
  - confirmed opposing-lineup strikeout rate when available,
  - league fallback when lineups are not yet confirmed.
- Prices both YES and NO by walking the full Polymarket ask book for the chosen dollar stake.
- Requires a fully fillable order plus 5% executable edge and 5% expected ROI for a prop `BET` signal.
- Adds official MLB team logos and player headshots.
- Adds a slate/navy institutional theme with neon `BET`, yellow `LEAN`, gray `PASS`, and monospace numerical tables.
- Adds a JSON-driven Release Notes modal from `static/changelog.json`.
- Preserves v0.11 Base Runs, FIP/xFIP, lineup linear-weights research, leverage fatigue, deterministic hashing, SQLite records, and dry-run maker planning.

## Scope and safety

Real-money submission is still disabled. The prop board, VWAP calculations, and passive-order planner are research tools. The optional CLOB SDK is not needed for paper research.

The Poisson strikeout model is an auditable baseline. It must be forward-tested and compared with negative-binomial or mixture models before it is treated as a mature prop model.

## Install

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
unzip -o ~/Downloads/moneyball-polymarket-v0.12.1.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Expected: `76 passed`.

## Initialize deterministic storage

```bash
python scripts/init_research_db.py
```

## Refresh historical MLB box scores

```bash
python scripts/fetch_pitching_data.py --seasons 2021 2022 2023 2024 2025 2026
```

The command is resumable and skips files already cached.

## Start the dashboard

```bash
python -m uvicorn moneyball_predictions.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Hard-refresh on macOS with `Command + Shift + R`.

The header should say:

```text
MLB · Polymarket board + institutional prop lab · v0.12.1
```

## Strikeout prop API

```bash
curl -s "http://127.0.0.1:8000/api/v1/props/strikeouts?stake_dollars=50&days=3" \
  | python -m json.tool
```

Important returned fields:

```text
projected_innings
expected_batters_faced
pitcher_k_rate
opponent_lineup_k_rate
matchup_k_rate
expected_strikeouts
probability_yes
probability_no
yes.executable_price
no.executable_price
edge
expected_roi
signal
lineup_status
```

When no active mapped strikeout markets exist, the endpoint returns an empty `props` list and diagnostics rather than fabricating markets.

## Research execution endpoints

### Dollar-stake VWAP

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/research/execution/vwap \
  -H 'Content-Type: application/json' \
  --data @examples/orderbook_vwap_request.json | python -m json.tool
```

### Dry-run passive maker plan

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/research/execution/passive-plan \
  -H 'Content-Type: application/json' \
  --data @examples/passive_plan_request.json | python -m json.tool
```

### Immutable prediction test

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/research/predictions/immutable \
  -H 'Content-Type: application/json' \
  --data @examples/immutable_prediction_request.json | python -m json.tool
```

## Optional trading SDK

```bash
python -m pip install -e ".[trading]"
```

Do not add wallet secrets until maker-order simulation, stale-book rejection, cancellations, exposure limits, heartbeat handling, and the global kill switch have been forward-tested.
