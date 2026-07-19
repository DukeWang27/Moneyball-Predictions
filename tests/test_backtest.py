from datetime import date

import pytest

from moneyball_predictions.backtest import evaluate_walk_forward_games
from moneyball_predictions.mlb import MlbGameState


def _final_game(
    game_pk: int,
    game_date: str,
    away: str,
    home: str,
    away_score: int,
    home_score: int,
) -> MlbGameState:
    return MlbGameState(
        game_pk=game_pk,
        game_date=game_date,
        official_date=game_date[:10],
        away_team=away,
        home_team=home,
        away_score=away_score,
        home_score=home_score,
        abstract_state="Final",
        detailed_state="Final",
        coded_state="F",
        current_inning=9,
        inning_state="Bottom",
        inning_ordinal="9th",
        away_probable_pitcher=None,
        home_probable_pitcher=None,
        venue="Test Park",
        game_type="R",
    )


def test_walk_forward_uses_only_prior_games() -> None:
    games = [
        _final_game(1, "2026-04-01T17:00:00Z", "A", "B", 10, 0),
        _final_game(2, "2026-04-02T17:00:00Z", "A", "B", 5, 1),
        _final_game(3, "2026-04-03T17:00:00Z", "B", "A", 1, 4),
    ]
    result = evaluate_walk_forward_games(
        games,
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 4, 3),
        min_games=1,
    )

    assert result.prediction_count == 2
    assert result.accuracy == pytest.approx(1.0)
    assert result.brier_score is not None
    assert result.brier_score < 0.25
    assert all(game.predicted_winner == "A" for game in result.recent_predictions)


def test_future_result_does_not_change_earlier_prediction() -> None:
    first_two = [
        _final_game(1, "2026-04-01T17:00:00Z", "A", "B", 8, 1),
        _final_game(2, "2026-04-02T17:00:00Z", "A", "B", 3, 2),
    ]
    future = _final_game(3, "2026-04-03T17:00:00Z", "A", "B", 0, 25)

    before = evaluate_walk_forward_games(
        first_two,
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 4, 2),
        min_games=1,
    )
    after = evaluate_walk_forward_games(
        [*first_two, future],
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 4, 3),
        min_games=1,
    )

    earlier_before = next(game for game in before.recent_predictions if game.game_pk == 2)
    earlier_after = next(game for game in after.recent_predictions if game.game_pk == 2)
    assert earlier_after.away_probability == pytest.approx(earlier_before.away_probability)


def test_current_result_cannot_change_its_own_probability() -> None:
    history = [
        _final_game(1, "2026-04-01T17:00:00Z", "A", "B", 6, 2),
        _final_game(2, "2026-04-02T17:00:00Z", "B", "A", 3, 5),
    ]
    current_a_wins = _final_game(3, "2026-04-03T17:00:00Z", "A", "B", 10, 0)
    current_b_wins = _final_game(3, "2026-04-03T17:00:00Z", "A", "B", 0, 10)

    result_a = evaluate_walk_forward_games(
        [*history, current_a_wins],
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 4, 3),
        min_games=1,
    )
    result_b = evaluate_walk_forward_games(
        [*history, current_b_wins],
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 4, 3),
        min_games=1,
    )

    prediction_a = next(row for row in result_a.recent_predictions if row.game_pk == 3)
    prediction_b = next(row for row in result_b.recent_predictions if row.game_pk == 3)
    assert prediction_a.home_probability == pytest.approx(prediction_b.home_probability)
    assert prediction_a.away_probability == pytest.approx(prediction_b.away_probability)
    assert prediction_a.winner != prediction_b.winner


def test_backtest_returns_old_and_enhanced_models_with_leakage_audit() -> None:
    games = [
        _final_game(1, "2026-04-01T17:00:00Z", "A", "B", 4, 2),
        _final_game(2, "2026-04-02T17:00:00Z", "B", "A", 2, 5),
        _final_game(3, "2026-04-03T17:00:00Z", "A", "B", 3, 1),
    ]
    result = evaluate_walk_forward_games(
        games,
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 4, 3),
        min_games=1,
    )

    assert result.baseline.label == "Old v0.4"
    assert result.enhanced.label == "Enhanced v0.5"
    assert result.leakage_audit.passed is True
    assert result.leakage_audit.prediction_before_result_update is True
    assert result.prediction_count == result.enhanced.prediction_count
