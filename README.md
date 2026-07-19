# Moneyball Predictions v0.5.0

A local MLB sabermetric research dashboard combining official MLB schedules and scores with executable Polymarket pregame moneyline asks.

## What v0.5.0 changes

The history panel now compares two models on the exact same games:

- **Old v0.4:** current-season runs scored/allowed, Pythagorean expectation, and Log5.
- **Enhanced v0.5:** previous-season regression, league-average shrinkage, a home-field adjustment, and Platt probability calibration trained only on previously completed games.

The page reports Accuracy, Brier, Log loss, calibration buckets, recent predictions, and the improvement or decline versus the old model.

## Leakage controls

Historical games are sorted by exact game timestamp. For each game the code:

1. reads only statistics from earlier completed games;
2. creates and stores the prediction;
3. grades the frozen prediction against the final result;
4. only then adds the current score to running team totals;
5. updates the calibrator only after the prediction has been graded.

Previous-season final data may be used because it was already known before the target season began. The current game's result and every future result are excluded from its prediction.

## Important interpretation

v0.5.0 is an experiment, not a promise that the enhanced model wins. The comparison panel is the acceptance test. Keep the new model only if it improves probability metrics on untouched seasons, especially Brier and Log loss.

The live board now uses previous-season regression plus home field, but it still does not include starting-pitcher quality, bullpen fatigue, lineups, park factors, or weather.

`BET` clears the configured edge and ROI thresholds. `LEAN` is informational only and does not get a paper-bet button.

## Upgrade an existing Desktop checkout

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
unzip -o ~/Downloads/moneyball-polymarket-v0.5.0.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m uvicorn moneyball_predictions.api:app --reload
```

Open `http://127.0.0.1:8000` and hard refresh with `Command + Shift + R`.

The page header must say `v0.5.0`.

## Historical API

```text
http://127.0.0.1:8000/api/v1/backtest/mlb?season=2026&min_games=10
```

The response includes `baseline`, `enhanced`, metric deltas, and `leakage_audit`.

## Tests

```bash
ruff check .
pytest -q
```

Expected result for this package:

```text
All checks passed!
37 passed
```

## Git workflow

```bash
git status
git add .
git commit -m "Add leakage-safe enhanced model comparison"
git push
```

## Disclaimer

Research and paper trading only. Kelly sizing is only as reliable as the probability estimate. Keep stakes small until the enhanced model proves lower out-of-sample Brier and Log loss than both the old model and the market baseline.
