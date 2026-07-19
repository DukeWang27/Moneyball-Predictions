# Moneyball Predictions v0.3.1

A local MLB sabermetric research dashboard that combines official MLB schedules/scores with executable Polymarket pregame moneyline asks.

## What v0.3.1 adds

- Polymarket-style compact matchup rows with both team prices shown side by side
- Games grouped by local date and sorted by official MLB first pitch, earliest first
- Full Kelly calculation for every side
- Quarter, half, or full Kelly selection in the dashboard
- Paper-bet stake sized from available browser bankroll instead of a fixed dollar input
- Kelly stakes capped by available top-of-book ask depth so the shown price remains executable
- Top recommendation ranked by Kelly-sized expected profit

Quarter Kelly is the default because the current model is not yet validated with starting pitchers, park factors, bullpen usage, weather, or walk-forward backtesting. The page still shows the full-Kelly percentage so the underlying calculation is transparent.

## Upgrade an existing checkout

```bash
cd ~/Downloads/Moneyball-Predictions
unzip -o ~/Downloads/moneyball-polymarket-v0.3.1.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m uvicorn moneyball_predictions.api:app --reload
```

Open `http://127.0.0.1:8000`.

## Kelly formula for a prediction-market share

For model probability `p` and executable buy price `c`, a winning share pays `$1` and a losing share pays `$0`. Full Kelly simplifies to:

```text
full_kelly_fraction = max(0, (p - c) / (1 - c))
selected_stake = bankroll * full_kelly_fraction * selected_multiplier
```

The multiplier is `0.25`, `0.50`, or `1.00`. If the number of shares available at the best ask is smaller than the Kelly order, the paper stake is capped to that top-of-book notional.

## Recommendation thresholds

A side receives `BET` only when both are true at the executable CLOB ask:

- model probability minus ask price is at least 5 percentage points;
- expected ROI is at least 5%.

A positive edge below those thresholds is a `LEAN`. Everything else is `PASS`.

## Tests

```bash
ruff check .
pytest -q
```

## Git workflow

```bash
git status
git add .
git commit -m "Add chronological market board and Kelly sizing"
git push
```

## Disclaimer

Research and paper trading only. Kelly sizing is only as reliable as the probability estimate. A miscalibrated model can make Kelly stakes dangerously large, which is why fractional Kelly is the default.
