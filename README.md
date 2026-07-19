# Moneyball Predictions v0.6.0

A local MLB prediction, Polymarket paper-trading, and leakage-safe model research dashboard.

## What v0.6 changes

The live board and historical test now use the same optimized pregame model when the training data loads successfully.

The optimized model combines:

- prior-regressed Pythagorean/Log5 team strength;
- carried and offseason-regressed Elo ratings;
- last-10 and last-30 run differential;
- last-10 win rate;
- days-of-rest difference;
- L2-regularized logistic regression;
- conservative probability shrinkage chosen on the season before the test season.

For a 2026 backtest, the model uses:

```text
2023: seed priors and Elo
2024: model training
2025: validation and probability-shrinkage selection
2026: untouched target-season evaluation
```

The 2026 result never trains the v0.6 coefficients. Within every season, a game is predicted before its score updates Elo, rolling form, or team totals.

## New model-lab output

The history section displays:

- Optimized v0.6 metrics;
- Enhanced v0.5 metrics;
- Old v0.4 metrics;
- ablation results for Base, Elo, Form, and Full versions;
- standardized feature weights;
- calibration buckets;
- a visible leakage audit;
- recent side-by-side predictions from all three versions.

The live response displays `v0.6.0` when the optimized model is active. If historical model data cannot load, the app explicitly reports `v0.5 fallback` instead of pretending the optimized model ran.

## Upgrade an existing checkout

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
unzip -o ~/Downloads/moneyball-polymarket-v0.6.0.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m uvicorn moneyball_predictions.api:app --reload
```

Open `http://127.0.0.1:8000` and hard-refresh with `Command + Shift + R`.

The first refresh may take longer because the optimized model downloads the three pre-target regular seasons. The trained live context is cached for 20 minutes.

## Recommendation rules

- `BET`: at least 5% edge and 5% expected ROI at the executable ask.
- `LEAN`: positive expected value but below the bet thresholds; do not place a paper bet.
- `PASS`: no positive expected value.

Kelly sizing remains experimental. Quarter Kelly is the default, and the suggested paper stake is capped by available cash and top-of-book depth.

## Tests

```bash
ruff check .
pytest -q
```

Expected:

```text
All checks passed!
41 passed
```

## Git workflow

```bash
git status
git add .
git commit -m "Add leakage-safe Elo and rolling-form model"
git push
```

## Current limitation

v0.6 improves the team-level statistical model but does not yet include historically timestamped starting-pitcher, bullpen, lineup, park, or weather inputs. Those features should only be added when their historical values can be reconstructed as they were known before first pitch.

## Disclaimer

Research and paper trading only. Historical improvement does not guarantee future profitability, and Kelly sizing can be unsafe when probabilities are miscalibrated.
