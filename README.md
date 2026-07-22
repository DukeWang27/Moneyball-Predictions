# Moneyball Predictions v0.13.0

MLB moneyline and pitcher-strikeout research dashboard built with FastAPI, vanilla JavaScript, Polymarket market data, MLB data, and Postgres-backed paper trading.

## Major changes

- FastAPI entrypoint for Vercel (`index.py`)
- Neon/Postgres persistence with SQLAlchemy and Alembic
- Separate $100 moneyline and $100 player-prop paper ledgers
- Server-side balance, P&L, accuracy, average edge, and Brier calculations
- Legacy browser-bet importer
- Grouped pitcher strikeout families
- One `BEST_BET` contract per pitcher/game
- Poisson probability ladders with monotonicity checks
- YES/NO market ladder validation
- Conservative 30% probability shrinkage until historical prop calibration is available
- Expected ROI, Kelly fraction, and expected-log-growth ranking
- Lazy-loaded tabbed frontend
- Protected lineup and settlement cron endpoints
- Postgres market snapshots and lineup snapshots

## Local installation

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
cp .env.example .env
python scripts/init_postgres.py
bash scripts/start_local.sh
```

Open `http://127.0.0.1:8000`.

## Neon variables

Use Neon's pooled connection string for runtime requests:

```env
DATABASE_URL=postgresql://...-pooler.../neondb?sslmode=require
```

Use Neon's direct connection string for migrations:

```env
DATABASE_URL_DIRECT=postgresql://......neon.tech/neondb?sslmode=require
```

Then run:

```bash
set -a
source .env
set +a
alembic upgrade head
python scripts/init_postgres.py
```

## Paper-bet migration

When the app is first run at the same local origin used by v0.12.2, the Overview page detects `moneyballPaperAccountV1` in localStorage and offers an **Import bets** button.

The importer:

- sends moneyline bets to the moneyline ledger
- sends strikeout props to the prop ledger
- preserves entry probabilities, prices, edges, statuses, and realized P&L
- keeps the old browser copy untouched as a backup

## Vercel

The FastAPI application is exported from `index.py`. Add these environment variables to Vercel:

```text
DATABASE_URL
DASHBOARD_USERNAME
DASHBOARD_PASSWORD
CRON_SECRET
```

Deploy from the connected GitHub repository. FastAPI is served as one Vercel Function.

## Hobby-plan scheduled jobs

Vercel Hobby cannot run minute-level cron jobs. Use an external HTTP scheduler to call:

```text
GET /api/internal/cron/lineups
GET /api/internal/cron/settle-paper-bets
```

Both require:

```text
Authorization: Bearer <CRON_SECRET>
```

The lineup endpoint implements adaptive cadence internally:

- 6–2 hours before first pitch: every 10 minutes
- 2 hours–15 minutes before first pitch: every 2 minutes
- final 15 minutes: every minute
- after first pitch: stop

The external scheduler may call the endpoint every minute; unnecessary game checks are skipped.

## Testing

```bash
pytest -q
```

Expected for this package:

```text
93 passed
```

## Prop-family selection

Every pitcher/game is represented by one family key:

```text
(game_pk, player_id, PITCHER_STRIKEOUTS)
```

All thresholds use the same expected strikeout mean, lambda:

```text
P(K >= n) = poisson.sf(n - 1, lambda)
```

The engine evaluates both YES and NO at every available threshold. Only one contract can be selected as `BEST_BET`. Other positive contracts remain visible as alternatives but do not receive a paper-bet button.

Betting is disabled when:

- probabilities are non-monotonic
- YES prices are materially inverted
- NO prices are materially inverted
- the opponent lineup is not confirmed
- the proposed stake is not fillable
- edge is below 5%
- expected ROI is below 5%
