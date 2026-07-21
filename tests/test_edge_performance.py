from moneyball_predictions.edge_performance import analyze_archived_snapshots
from moneyball_predictions.mlb import MlbGameState


def _final_game() -> MlbGameState:
    return MlbGameState(
        game_pk=123,
        game_date="2026-07-20T20:00:00Z",
        official_date="2026-07-20",
        away_team="Away Club",
        home_team="Home Club",
        away_score=2,
        home_score=5,
        abstract_state="Final",
        detailed_state="Final",
        coded_state="F",
        current_inning=9,
        inning_state="End",
        inning_ordinal="9th",
        away_probable_pitcher=None,
        home_probable_pitcher=None,
        venue="Test Park",
    )


def test_archived_edge_performance_uses_honest_entry_and_closing_snapshot() -> None:
    rows = [
        {
            "captured_at": "2026-07-20T18:30:00+00:00",
            "game_start": "2026-07-20T20:00:00+00:00",
            "game_pk": 123,
            "team_a": "Home Club",
            "team_b": "Away Club",
            "best_bid_a": 0.49,
            "best_ask_a": 0.50,
            "best_bid_b": 0.50,
            "best_ask_b": 0.51,
            "model_probability_a": 0.58,
            "model_probability_b": 0.42,
            "sabermetric_support_a": "CONFIRMED",
            "sabermetric_support_count_a": 2,
            "sabermetric_support_total_a": 2,
            "sabermetric_support_b": "CONTRARIAN",
            "model_version": "v0.9.2 test",
        },
        {
            "captured_at": "2026-07-20T19:55:00+00:00",
            "game_start": "2026-07-20T20:00:00+00:00",
            "game_pk": 123,
            "team_a": "Home Club",
            "team_b": "Away Club",
            "best_bid_a": 0.54,
            "best_ask_a": 0.56,
            "best_bid_b": 0.44,
            "best_ask_b": 0.46,
            "model_probability_a": 0.58,
            "model_probability_b": 0.42,
            "sabermetric_support_a": "CONFIRMED",
            "sabermetric_support_count_a": 2,
            "sabermetric_support_total_a": 2,
            "sabermetric_support_b": "CONTRARIAN",
            "model_version": "v0.9.2 test",
        },
    ]

    response = analyze_archived_snapshots(
        rows,
        {123: _final_game()},
        entry_horizon_minutes=60,
    )

    assert response.selected_games == 1
    assert response.settled_games == 1
    candidate = response.recent_candidates[0]
    assert candidate.actual_entry_horizon_minutes == 90
    assert candidate.team == "Home Club"
    assert candidate.won is True
    assert candidate.profit_per_dollar == 1.0
    assert candidate.clv is not None
    assert round(candidate.clv, 3) == 0.05

    five_percent = next(
        item
        for item in response.metrics
        if item.strategy == "Sabermetric confirmed" and item.min_edge == 0.05
    )
    assert five_percent.settled_bets == 1
    assert five_percent.win_rate == 1.0
    assert five_percent.flat_stake_roi == 1.0


def test_archived_edge_performance_does_not_use_late_snapshot_for_earlier_horizon() -> None:
    rows = [
        {
            "captured_at": "2026-07-20T19:30:00+00:00",
            "game_start": "2026-07-20T20:00:00+00:00",
            "game_pk": 123,
            "team_a": "Home Club",
            "team_b": "Away Club",
            "best_ask_a": 0.50,
            "best_ask_b": 0.51,
            "model_probability_a": 0.58,
            "model_probability_b": 0.42,
        }
    ]
    response = analyze_archived_snapshots(
        rows,
        {123: _final_game()},
        entry_horizon_minutes=60,
    )
    assert response.selected_games == 0
