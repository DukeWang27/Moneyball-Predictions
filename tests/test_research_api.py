import base64
import os

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


def test_vwap_research_endpoint() -> None:
    response = client.post(
        "/api/v1/research/execution/vwap",
        json={
            "order_book": {
                "asks": [
                    {"price": "0.50", "size": "10"},
                    {"price": "0.52", "size": "100"},
                ]
            },
            "desired_dollars": 10,
            "model_probability": 0.60,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["levels_consumed"] == 2
    assert payload["edge"] > 0


def test_t1_prediction_requires_confirmed_lineups() -> None:
    response = client.post(
        "/api/v1/research/predictions/immutable",
        json={
            "game_id": 1,
            "horizon": "T1H",
            "as_of": "2026-07-20T22:00:00Z",
            "model_version": "test",
            "features": {"Base": 0.1},
            "source_snapshot": {"game": 1},
            "raw_home_probability": 0.55,
            "calibrated_home_probability": 0.54,
            "home_team": "Home",
            "away_team": "Away",
            "lineups_confirmed": False,
        },
    )
    assert response.status_code == 422
