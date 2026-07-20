import json

from moneyball_predictions.market_archive import append_market_snapshots


def test_market_snapshots_are_appended_as_jsonl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MONEYBALL_MARKET_ARCHIVE", str(tmp_path))
    path = append_market_snapshots(
        [
            {
                "game_pk": 123,
                "market_id": "market-1",
                "best_ask_a": 0.51,
                "model_probability_a": 0.56,
            }
        ]
    )
    assert path is not None
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["game_pk"] == 123
    assert payload["best_ask_a"] == 0.51
    assert "captured_at" in payload


def test_market_snapshots_can_be_read_back(tmp_path, monkeypatch) -> None:
    from moneyball_predictions.market_archive import read_market_snapshots

    monkeypatch.setenv("MONEYBALL_MARKET_ARCHIVE", str(tmp_path))
    append_market_snapshots([{"game_pk": 987, "best_ask_a": 0.48}])
    rows = read_market_snapshots()
    assert len(rows) == 1
    assert rows[0]["game_pk"] == 987
