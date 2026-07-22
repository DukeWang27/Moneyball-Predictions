# v0.13.0 Prop Selector Display Patch

This patch keeps the full strikeout ladder analysis on the backend but displays only one selected contract per pitcher.

## Changes

- Shows one top option per pitcher rather than every YES/NO threshold.
- Keeps the Paper Bet button restricted to a true `BEST_BET` that clears lineup, pricing, edge, ROI, fillability, and safety gates.
- Shows `WAIT FOR LINEUP`, `SAFETY BLOCK`, `TOP OPTION · NO BET`, or `PASS` when the strongest available contract is not actionable.
- Deduplicates multiple Polymarket markets for the same pitcher, threshold, and side.
- Scans all valid strikeout market types instead of stopping after the first type that returns markets.
- Displays coverage diagnostics so missing props can be traced to unmapped players, unmatched games, started games, or missing order books.

## Verification

- `97 passed`
- JavaScript syntax check passed.
- Python compile check passed.
