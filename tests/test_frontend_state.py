from pathlib import Path


HTML = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
JS = Path("src/moneyball_predictions/static/app.js").read_text(encoding="utf-8")
CSS = Path("src/moneyball_predictions/static/app.css").read_text(encoding="utf-8")


def test_frontend_uses_lazy_tabbed_sections() -> None:
    assert 'data-tab="overview"' in HTML
    assert 'data-tab="moneylines"' in HTML
    assert 'data-tab="props"' in HTML
    assert 'data-tab="portfolio"' in HTML
    assert "/api/v1/dashboard/summary" in JS
    assert "activateTab" in JS
    assert "state.loaded" in JS


def test_frontend_has_two_separate_postgres_ledgers() -> None:
    assert "MONEYLINES" in JS
    assert "PLAYER PROPS" in JS
    assert "Available cash" in JS
    assert "Realized P&amp;L" in JS
    assert "Brier score" in JS
    assert "/api/v1/paper/accounts" in JS
    assert "/api/v1/paper/bets" in JS


def test_frontend_groups_prop_families_and_has_one_best_bet_action() -> None:
    assert "Pitcher strikeout families" in HTML
    assert "familyCard" in JS
    assert "selectedPropContract" in JS
    assert "family.best_market_id" in JS
    assert "prop-bet" in JS
    assert "Shrunk probability" in JS
    assert "expected logarithmic growth" in HTML
    assert "https://securea.mlb.com/mlb/images/players/head_shot/" in JS


def test_frontend_uses_skeletons_fragments_and_lazy_images() -> None:
    assert "skeleton" in HTML
    assert "loading=\"lazy\"" in JS
    assert "DocumentFragment" in JS
    assert "@keyframes shimmer" in CSS


def test_frontend_can_import_legacy_browser_bets() -> None:
    assert "moneyballPaperAccountV1" in JS
    assert "/api/v1/paper/import-browser" in JS
    assert "Import bets" in JS


def test_frontend_displays_only_one_prop_option_per_pitcher() -> None:
    assert "family.contracts.map((contract) => contractRow" not in JS
    assert "Selected from ${lineCount}" in JS
    assert "propCoverageText" in JS
