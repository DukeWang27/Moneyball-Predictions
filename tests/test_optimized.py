from datetime import date, timedelta

import pytest

from moneyball_predictions.backtest import evaluate_walk_forward_games
from moneyball_predictions.mlb import MlbBattingLine, MlbGameState, MlbPitchingLine
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
        away_probable_pitcher=f"{away} Starter",
        home_probable_pitcher=f"{home} Starter",
        venue="Test Park",
        game_type="R",
        away_probable_pitcher_id=100 + ord(away),
        home_probable_pitcher_id=100 + ord(home),
        away_first5_runs=min(away_score, 5),
        home_first5_runs=min(home_score, 5),
        away_batting=MlbBattingLine(
            at_bats=34, hits=8, doubles=2, triples=0, home_runs=1,
            walks=3, intentional_walks=0, hit_by_pitch=1,
        ),
        home_batting=MlbBattingLine(
            at_bats=33, hits=9, doubles=2, triples=0, home_runs=2,
            walks=4, intentional_walks=0, hit_by_pitch=0,
        ),
        away_pitching=(
            MlbPitchingLine(
                pitcher_id=100 + ord(away),
                name=f"{away} Starter",
                team=away,
                is_starter=True,
                outs_recorded=15,
                batters_faced=21,
                strikeouts=6,
                walks=2,
                hit_batters=0,
                home_runs=1,
                pitches_thrown=88,
                earned_runs=min(home_score, 4),
            ),
            MlbPitchingLine(
                pitcher_id=500 + ord(away),
                name=f"{away} Reliever",
                team=away,
                is_starter=False,
                outs_recorded=12,
                batters_faced=15,
                strikeouts=4,
                walks=1,
                hit_batters=0,
                home_runs=0,
                pitches_thrown=45,
                earned_runs=max(home_score - 4, 0),
            ),
        ),
        home_pitching=(
            MlbPitchingLine(
                pitcher_id=100 + ord(home),
                name=f"{home} Starter",
                team=home,
                is_starter=True,
                outs_recorded=18,
                batters_faced=23,
                strikeouts=7,
                walks=1,
                hit_batters=0,
                home_runs=0,
                pitches_thrown=91,
                earned_runs=min(away_score, 4),
            ),
            MlbPitchingLine(
                pitcher_id=500 + ord(home),
                name=f"{home} Reliever",
                team=home,
                is_starter=False,
                outs_recorded=9,
                batters_faced=12,
                strikeouts=3,
                walks=1,
                hit_batters=0,
                home_runs=0,
                pitches_thrown=37,
                earned_runs=max(away_score - 4, 0),
            ),
        ),
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
        FeatureRow(dummy, (value, value, 0), 1.0 if value > 0 else 0.0)
        for value in [x / 10 for x in range(-20, 21) if x != 0]
    ]
    model = RegularizedLogisticModel.fit(rows, feature_indices=(0, 1), l2=1.0)
    assert model.predict((2.0, 2.0, 0)) > 0.5
    assert model.predict((-2.0, -2.0, 0)) < 0.5


def test_backtest_exposes_v07_and_ablations() -> None:
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

    assert result.optimized.label.startswith("v0.12 ")
    assert result.enhanced.label == "Enhanced v0.5"
    assert result.training_seasons == [2024]
    assert result.calibration_seasons == [2025]
    assert result.optimized_training_games > 0
    assert len(result.ablations) == 9
    assert result.target_starter_coverage > 0
    assert result.target_component_coverage > 0
    assert result.target_bullpen_coverage > 0
    assert result.leakage_audit.coefficients_trained_before_target_season is True
    assert result.optimized_variant
    assert result.feature_diagnostics
    assert result.calibration_comparison
    assert result.tournament_models
    assert result.tournament_champion


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
    assert artifact.full.calibration_method in {"identity", "shrink", "platt", "isotonic"}
    assert prior_summary.teams
    assert prior_elo == {}
    assert artifact.training_starter_coverage > 0
    assert artifact.training_component_coverage > 0
    assert artifact.training_bullpen_coverage > 0
    assert artifact.champion.label in {
        "ComponentStarter",
        "StarterWorkload",
        "Park",
        "StarterSplit",
        "ParkNeutralSplit",
        "BaseRunsStarter",
    }


def test_current_component_line_cannot_change_its_own_features() -> None:
    history = [
        _game(1, 0, "A", "B", 5, 2, 2026),
        _game(2, 1, "B", "A", 2, 4, 2026),
    ]
    normal = _game(3, 2, "A", "B", 3, 2, 2026)
    extreme = MlbGameState(
        **{
            **normal.__dict__,
            "away_pitching": (
                MlbPitchingLine(
                    pitcher_id=normal.away_probable_pitcher_id or 1,
                    name=normal.away_probable_pitcher,
                    team="A",
                    is_starter=True,
                    outs_recorded=1,
                    batters_faced=20,
                    strikeouts=0,
                    walks=10,
                    hit_batters=3,
                    home_runs=5,
                    pitches_thrown=90,
                    earned_runs=12,
                ),
            ),
        }
    )
    rows_normal, _ = build_season_rows(
        [*history, normal], prior_summary=None, prior_elo=None, min_games=1
    )
    rows_extreme, _ = build_season_rows(
        [*history, extreme], prior_summary=None, prior_elo=None, min_games=1
    )
    features_normal = next(row.features for row in rows_normal if row.game.game_pk == 3)
    features_extreme = next(row.features for row in rows_extreme if row.game.game_pk == 3)
    assert features_normal == pytest.approx(features_extreme)


def test_multifold_calibration_rejects_unstable_isotonic() -> None:
    import numpy as np

    from moneyball_predictions.optimized import (
        ModelVariant,
        OptimizedModelArtifact,
        ProbabilityCalibrator,
        apply_multifold_selection,
    )

    model = RegularizedLogisticModel(
        feature_indices=(0,),
        means=np.zeros(1),
        scales=np.ones(1),
        coefficients=np.zeros(2),
    )

    def artifact(identity: float, isotonic: float) -> OptimizedModelArtifact:
        identity_cal = ProbabilityCalibrator()
        isotonic_cal = ProbabilityCalibrator(
            method="isotonic",
            x_thresholds=(0.1, 0.9),
            y_thresholds=(0.2, 0.8),
        )
        variants = {}
        for key in ("ComponentStarter", "StarterWorkload", "Park", "StarterSplit", "ParkNeutralSplit"):
            variants[key] = ModelVariant(
                key=key.casefold(),
                label=key,
                model=model,
                calibrator=isotonic_cal,
                validation_brier=isotonic,
                calibration_candidates={"identity": identity_cal, "isotonic": isotonic_cal},
                calibration_scores={"identity": identity, "isotonic": isotonic},
            )
        variants["Full"] = variants["ComponentStarter"]
        return OptimizedModelArtifact(
            variants=variants,
            training_rows=1000,
            validation_rows=1000,
            training_starter_coverage=1.0,
            validation_starter_coverage=1.0,
        )

    earlier = artifact(identity=0.2440, isotonic=0.2448)
    final = artifact(identity=0.2440, isotonic=0.2420)
    selected = apply_multifold_selection(final, [earlier], fold_years=(2024, 2025))
    assert selected.champion.calibration_method == "identity"
    assert selected.selection_folds == (2024, 2025)
