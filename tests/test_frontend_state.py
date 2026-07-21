from pathlib import Path


def test_frontend_marks_existing_paper_bets_and_hides_duplicate_button() -> None:
    html = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
    assert "function paperBetForGame" in html
    assert "Paper bet placed" in html
    assert "BETTED" in html
    assert "Current BET signals are already covered" in html
    assert "Entry edge" in html
    assert "storedBetEdge" in html
    assert "Avg edge" in html


def test_frontend_has_edge_performance_and_confirmed_strategy_filter() -> None:
    html = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
    assert "Edge performance" in html
    assert "Require sabermetric confirmation" in html
    assert "function renderEdgePerformance" in html
    assert "/api/v1/edge-performance/mlb" in html


def test_frontend_has_prop_board_images_and_release_notes() -> None:
    html = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
    assert "Pitcher strikeout props" in html
    assert "/api/v1/props/strikeouts" in html
    assert "https://www.mlbstatic.com/team-logos/" in html
    assert "https://securea.mlb.com/mlb/images/players/head_shot/" in html
    assert "Release Notes" in html
    assert "/static/changelog.json" in html


def test_frontend_has_prop_paper_betting_and_clear_quote_size_copy() -> None:
    html = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
    assert "Order-book quote size ($)" in html
    assert "This amount is not a bet" in html
    assert "function placePropPaperBet" in html
    assert "function paperBetForProp" in html
    assert "Paper bet $" in html
    assert "/api/v1/props/strikeouts/settlement" in html
    assert "Live and completed games are intentionally excluded" in html
