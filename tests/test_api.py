import base64
import os

import pytest
from fastapi.testclient import TestClient

from moneyball_predictions.api import app


def _dashboard_auth_headers() -> dict[str, str]:
    """Authenticate test requests when the local private-dashboard setting is active."""
    password = os.environ.get("DASHBOARD_PASSWORD", "")
    if not password:
        return {}
    username = os.environ.get("DASHBOARD_USERNAME", "moneyball")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


client = TestClient(app, headers=_dashboard_auth_headers())


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.13.0"
    assert payload["database"]["status"] == "ok"


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
