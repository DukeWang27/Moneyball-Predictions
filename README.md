# Moneyball Predictions

A research-first MLB sabermetric prediction and paper-trading platform.

The first MVP is intentionally small and auditable. It currently:

- computes dynamic-exponent Pythagorean team strength;
- uses Log5 to produce a matchup probability;
- removes overround from conventional two-way decimal sportsbook odds;
- calculates model edge and expected value for both sides;
- exposes the model through a FastAPI endpoint and a minimal browser dashboard;
- includes automated tests and GitHub Actions CI.

It does **not** yet pull live odds, adjust for starting pitchers, create paper bets, or claim predictive profitability.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m uvicorn moneyball_predictions.api:app --reload
```

Open `http://127.0.0.1:8000`.

Run the tests:

```bash
pytest -q
ruff check .
```

Fetch the canonical MLB team-ID reference file:

```bash
python scripts/fetch_data.py --source mlb-teams
```

The result is written beneath `data/`, which is intentionally ignored by git.

## API example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "team_a": {"name":"Yankees","runs_scored":510,"runs_allowed":430,"games_played":100},
    "team_b": {"name":"Red Sox","runs_scored":470,"runs_allowed":460,"games_played":100},
    "decimal_odds_a":1.75,
    "decimal_odds_b":2.20,
    "stake":10
  }'
```

## Important market-pricing distinction

The current de-vig function is for **sportsbook-style two-way odds**. Polymarket is an order-book prediction market, so its integration should use executable bid/ask prices, spread, depth, and applicable fees rather than blindly applying sportsbook de-vigging to last-trade prices.

## Planned milestones

### MVP 0.1 — included in this commit

- Core Pythagorean + Log5 model
- Two-way de-vig and EV math
- Manual-input dashboard
- Tests and CI

### MVP 0.2 — next

- MLB Stats API team IDs and scheduled games
- Polymarket Gamma market discovery
- CLOB executable-price adapter
- Match MLB games to Polymarket markets
- Show live model probability, market probability, spread/fees, edge, and EV

### v2

- Starting-pitcher FIP adjustment
- Entity matching and unmatched-name review queue
- SQLite paper-trading ledger with $100 starting bankroll
- Automatic resolution from final MLB scores
- Calibration reporting

### v3+

- Fractional Kelly sizing and CLV snapshots
- The Odds API integration for consensus sportsbook lines and props
- Park factors, recency weighting, bullpen fatigue, weather, and small-sample gating
- Walk-forward historical backtesting with strict as-of-date features

## Data and secrets

- Keep downloaded datasets in `data/`.
- Keep API keys in `.env`; never commit them.
- Copy `.env.example` to `.env` when integrations are added.

## Disclaimer

This software is for research and paper trading only. It is not financial advice and should not be treated as evidence of a profitable betting strategy without rigorous out-of-sample validation.
