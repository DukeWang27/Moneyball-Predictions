from datetime import date, timedelta

import pytest

from moneyball_predictions.backtest import evaluate_walk_forward_games
from moneyball_predictions.mlb import MlbGameState
from moneyball_predictions.optimized import (
    FeatureRow,
    RegularizedLogisticModel,
    build_season_rows,
    prepare_optimized_model,
)


def _game(
    game_pk: int,
    day: int,
    away: str,
    home: str,
    away_score: int,
    home_score: int,
    year: int,
) -> MlbGameState:
    game_date = date(year, 4, 1) + timedelta(days=day)
    return MlbGameState(
        game_pk=game_pk,
        game_date=f"{game_date.isoformat()}T17:00:00Z",
        official_date=game_date.isoformat(),
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


def _season(year: int, start_pk: int, games: int = 50) -> list[MlbGameState]:
    teams = ["A", "B", "C", "D"]
    rows: list[MlbGameState] = []
    for index in range(games):
        away = teams[index % 4]
        home = teams[(index + 1) % 4]
        strong_home = home in {"A", "C"}
        home_score = 6 if strong_home else 3
        away_score = 2 if strong_home else 4
        rows.append(
            _game(start_pk + index, index, away, home, away_score, home_score, year)
        )
    return rows


def test_feature_row_is_frozen_before_current_result() -> None:
    history = [
        _game(1, 0, "A", "B", 5, 2, 2026),
        _game(2, 1, "B", "A", 2, 4, 2026),
    ]
    current_a = _game(3, 2, "A", "B", 10, 0, 2026)
    current_b = _game(3, 2, "A", "B", 0, 10, 2026)

    rows_a, _ = build_season_rows(
        [*history, current_a], prior_summary=None, prior_elo=None, min_games=1
    )
    rows_b, _ = build_season_rows(
        [*history, current_b], prior_summary=None, prior_elo=None, min_games=1
    )

    feature_a = next(row.features for row in rows_a if row.game.game_pk == 3)
    feature_b = next(row.features for row in rows_b if row.game.game_pk == 3)
    assert feature_a == pytest.approx(feature_b)


def test_regularized_logistic_model_learns_direction() -> None:
    dummy = _game(99, 0, "A", "B", 1, 2, 2024)
    rows = [
        FeatureRow(dummy, (value, value, 0, 0, 0, 0), 1.0 if value > 0 else 0.0)
        for value in [x / 10 for x in range(-20, 21) if x != 0]
    ]
    model = RegularizedLogisticModel.fit(rows, feature_indices=(0, 1), l2=1.0)
    assert model.predict((2.0, 2.0, 0, 0, 0, 0)) > 0.5
    assert model.predict((-2.0, -2.0, 0, 0, 0, 0)) < 0.5


def test_backtest_exposes_v06_and_ablations() -> None:
    training = {
        2023: _season(2023, 1000),
        2024: _season(2024, 2000),
        2025: _season(2025, 3000),
    }
    target = _season(2026, 4000)
    result = evaluate_walk_forward_games(
        target,
        season=2026,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 7, 1),
        min_games=2,
        prior_games=training[2025],
        training_games_by_season=training,
    )

    assert result.optimized.label == "Optimized v0.6"
    assert result.enhanced.label == "Enhanced v0.5"
    assert result.training_seasons == [2024, 2025]
    assert result.optimized_training_games > 0
    assert len(result.ablations) == 4
    assert result.leakage_audit.coefficients_trained_before_target_season is True


def test_prepare_model_uses_pre_target_seasons_only() -> None:
    artifact, prior_summary, prior_elo = prepare_optimized_model(
        seed_games=_season(2023, 1000),
        training_games=_season(2024, 2000),
        validation_games=_season(2025, 3000),
        min_games=2,
    )
    assert artifact.training_rows > 0
    assert artifact.validation_rows > 0
    assert 0.0 <= artifact.full.shrinkage <= 0.4
    assert prior_summary.teams
    assert prior_elo
