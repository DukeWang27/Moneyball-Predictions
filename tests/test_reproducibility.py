from datetime import UTC, datetime
from pathlib import Path

from moneyball_predictions.reproducibility import feature_sha256
from moneyball_predictions.storage import database_summary, insert_prediction


def test_feature_hash_is_order_independent() -> None:
    left = feature_sha256({"b": 2.0, "a": [1, 2, 3]})
    right = feature_sha256({"a": [1, 2, 3], "b": 2.0})
    assert left == right


def test_immutable_horizon_insert_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "research.sqlite3"
    kwargs = dict(
        game_id=123,
        horizon="T1H",
        as_of=datetime(2026, 7, 20, 22, tzinfo=UTC),
        model_version="v0.12 test",
        feature_schema_version="v0.12.0",
        code_commit_sha="abc",
        model_artifact_sha256="model",
        calibration_artifact_sha256=None,
        features={"Base": 0.2, "StarterKRate": 0.01},
        source_snapshot={"game_pk": 123},
        raw_home_probability=0.56,
        calibrated_home_probability=0.54,
        home_team="Home",
        away_team="Away",
        probable_home_pitcher_id=1,
        probable_away_pitcher_id=2,
        lineups_confirmed=True,
        path=path,
    )
    first = insert_prediction(**kwargs)
    second = insert_prediction(**kwargs)
    assert first.created is True
    assert second.created is False
    assert first.prediction_id == second.prediction_id
    assert database_summary(path)["prediction_runs"] == 1
