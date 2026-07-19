import pytest

from fastapi.testclient import TestClient

from moneyball_predictions.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint_returns_two_sides() -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "team_a": {
                "name": "Team A",
                "runs_scored": 500,
                "runs_allowed": 420,
                "games_played": 100,
            },
            "team_b": {
                "name": "Team B",
                "runs_scored": 450,
                "runs_allowed": 470,
                "games_played": 100,
            },
            "decimal_odds_a": 1.8,
            "decimal_odds_b": 2.1,
            "stake": 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["side_a"]["team"] == "Team A"
    assert payload["side_b"]["team"] == "Team B"
    assert payload["side_a"]["model_probability"] + payload["side_b"]["model_probability"] == pytest.approx(1.0)
