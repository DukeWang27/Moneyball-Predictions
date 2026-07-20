from pathlib import Path


def test_frontend_marks_existing_paper_bets_and_hides_duplicate_button() -> None:
    html = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
    assert "function paperBetForGame" in html
    assert "Paper bet placed" in html
    assert "BETTED" in html
    assert "Current BET signals are already covered" in html


def test_frontend_has_edge_performance_and_confirmed_strategy_filter() -> None:
    html = Path("src/moneyball_predictions/static/index.html").read_text(encoding="utf-8")
    assert "Edge performance" in html
    assert "Require sabermetric confirmation" in html
    assert "function renderEdgePerformance" in html
    assert "/api/v1/edge-performance/mlb" in html
