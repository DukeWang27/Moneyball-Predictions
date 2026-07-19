from datetime import UTC, datetime, timedelta

from moneyball_predictions.live import (
    _match_schedule_game,
    _parse_start_time,
    _scoreboard_row,
    _signal_for_sides,
)
from moneyball_predictions.mlb import MlbGameState
from moneyball_predictions.polymarket import MlbMoneylineMarket
from moneyball_predictions.schemas import LiveMarketSide


def test_parse_start_time_accepts_zulu_time() -> None:
    parsed = _parse_start_time("2026-07-20T23:05:00Z")
    assert parsed == datetime(2026, 7, 20, 23, 5, tzinfo=UTC)


def test_parse_start_time_can_be_compared_to_now() -> None:
    future = datetime.now(UTC) + timedelta(hours=1)
    parsed = _parse_start_time(future.isoformat())
    assert parsed is not None
    assert parsed > datetime.now(UTC)


def _side(team: str, edge: float, roi: float) -> LiveMarketSide:
    stake = 10.0
    ask = 0.5
    return LiveMarketSide(
        team=team,
        model_probability=ask + edge,
        market_buy_price=ask,
        edge=edge,
        expected_value=roi * stake,
        expected_roi=roi,
        full_kelly_fraction=max(edge, 0),
        applied_kelly_fraction=max(edge, 0) * 0.25,
        kelly_multiplier=0.25,
        kelly_stake=stake,
        stake=stake,
        shares=20,
        price_source="CLOB best ask",
    )


def test_signal_requires_meaningful_threshold() -> None:
    signal, team, _ = _signal_for_sides(_side("A", 0.02, 0.04), _side("B", -0.02, -0.03))
    assert signal == "LEAN"
    assert team == "A"


def test_signal_marks_strong_candidate_as_bet() -> None:
    signal, team, _ = _signal_for_sides(_side("A", 0.08, 0.12), _side("B", -0.08, -0.10))
    assert signal == "BET"
    assert team == "A"


def _game(state: str = "Preview") -> MlbGameState:
    return MlbGameState(
        game_pk=123,
        game_date="2026-07-19T23:20:00Z",
        official_date="2026-07-19",
        away_team="Los Angeles Dodgers",
        home_team="New York Yankees",
        away_score=3 if state == "Live" else None,
        home_score=2 if state == "Live" else None,
        abstract_state=state,
        detailed_state="In Progress" if state == "Live" else "Scheduled",
        coded_state="I" if state == "Live" else "S",
        current_inning=6 if state == "Live" else None,
        inning_state="Top" if state == "Live" else None,
        inning_ordinal="6th" if state == "Live" else None,
        away_probable_pitcher="Pitcher A",
        home_probable_pitcher="Pitcher B",
        venue="Yankee Stadium",
    )


def _market() -> MlbMoneylineMarket:
    return MlbMoneylineMarket(
        event_id="e1",
        event_title="Dodgers vs. Yankees",
        event_slug="mlb-lad-nyy-2026-07-19",
        market_id="m1",
        start_time="2026-07-19T23:20:00Z",
        team_a="Los Angeles Dodgers",
        team_b="New York Yankees",
        token_a="a",
        token_b="b",
        reference_price_a=0.5,
        reference_price_b=0.5,
        liquidity=100,
        volume=200,
        accepting_orders=True,
    )


def test_schedule_match_uses_team_pair_and_start_time() -> None:
    assert _match_schedule_game(_market(), [_game()]).game_pk == 123  # type: ignore[union-attr]


def test_live_scoreboard_explicitly_disables_live_bet_signal() -> None:
    row = _scoreboard_row(_game("Live"), _market())
    assert row.abstract_state == "Live"
    assert "No live bet signal" in row.live_bet_message
    assert row.away_score == 3
