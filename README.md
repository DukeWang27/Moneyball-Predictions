# Moneyball Predictions v0.9.1

Local FastAPI dashboard for MLB pregame prediction, live Polymarket moneyline comparison, browser paper trading, and archived edge-performance research.

## v0.9.1 changes

- Keeps economic edge separate from sabermetric confirmation.
- Grades each side as `A`, `B`, `C`, `LEAN`, or `PASS`.
- `A` means the price thresholds clear and independent team-strength, starter, and material park drivers are confirmed.
- Adds a paper-bet filter that can require sabermetric confirmation.
- Continues preventing duplicate paper bets on the same game.
- Archives model drivers and support labels with every pregame market snapshot.
- Adds an Edge Performance dashboard using honest T-15, T-60, T-6h, or T-24h entry snapshots.
- Reports settled sample size, win rate, flat-stake ROI, profit per $1, Brier score, CLV, positive-CLV rate, and maximum drawdown by edge threshold.
- Compares all economic edges with the stricter sabermetric-confirmed strategy.

Bill James-style Pythagorean expectation and Log5 help estimate fair probability. They do not replace market price: a strong team can still be a bad bet when the contract costs too much.

## Install

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
unzip -o ~/Downloads/moneyball-polymarket-v0.9.1.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Expected test result: `53 passed`.

## Historical pitching cache

```bash
python scripts/fetch_pitching_data.py --seasons 2021 2022 2023 2024 2025 2026
```

The command is resumable. Existing box scores are skipped.

## Run

```bash
python -m uvicorn moneyball_predictions.api:app --reload
```

Open `http://127.0.0.1:8000` and hard refresh with `Command + Shift + R`.

## Edge performance

The app already writes append-only snapshots to `data/market_snapshots/`. Edge Performance selects one snapshot before the requested horizon and the last pregame snapshot as the closing benchmark.

- Entry ask determines whether the strategy would bet and its realized ROI.
- Closing no-vig midpoint is used only for CLV.
- Final score settles the signal.
- Games without an honestly archived snapshot before the requested horizon are omitted.
- Older snapshots without v0.9.1 support labels can appear under `All economic edges`, but not under `Sabermetric confirmed`.

Meaningful conclusions require a much larger settled sample. Do not interpret a few games as proof of profitability.
