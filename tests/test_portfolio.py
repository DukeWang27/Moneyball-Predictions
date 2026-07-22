from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from moneyball_predictions.db import Base, MONEYLINE_ACCOUNT, PLAYER_PROP_ACCOUNT, seed_accounts
from moneyball_predictions.portfolio import (
    BetPlacement,
    DuplicateBetError,
    account_metrics,
    place_bet,
    settle_bet,
)


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        seed_accounts(session)
        session.commit()
        yield session


def moneyline_payload() -> BetPlacement:
    return BetPlacement(
        account_type=MONEYLINE_ACCOUNT,
        market_type="MONEYLINE",
        game_pk=1,
        market_id="moneyline-1",
        selection="Dodgers",
        team="Dodgers",
        model_version="test",
        model_probability=0.60,
        entry_price=0.50,
        entry_edge=0.10,
        expected_roi=0.20,
        stake=1.00,
        placed_at=datetime.now(UTC),
    )


def test_accounts_are_strictly_separate(session: Session) -> None:
    place_bet(session, moneyline_payload())
    session.commit()
    moneyline = account_metrics(session, MONEYLINE_ACCOUNT)
    props = account_metrics(session, PLAYER_PROP_ACCOUNT)
    assert moneyline.available_cash == pytest.approx(99.0)
    assert moneyline.open_exposure == pytest.approx(1.0)
    assert props.available_cash == pytest.approx(100.0)
    assert props.open_exposure == pytest.approx(0.0)


def test_brier_uses_probability_of_purchased_side(session: Session) -> None:
    bet = place_bet(session, moneyline_payload())
    settle_bet(session, bet_id=bet.bet_id, status="WON")
    session.commit()
    metrics = account_metrics(session, MONEYLINE_ACCOUNT)
    assert metrics.accuracy == 1.0
    assert metrics.brier_score == pytest.approx((0.60 - 1.0) ** 2)
    assert metrics.realized_pnl == pytest.approx(1.0)


def test_only_one_open_prop_per_pitcher_game(session: Session) -> None:
    first = BetPlacement(
        account_type=PLAYER_PROP_ACCOUNT,
        market_type="STRIKEOUT_PROP",
        game_pk=99,
        market_id="p-3",
        selection="Pitcher 3+ YES",
        player_id=123,
        player_name="Pitcher",
        prop_threshold=3,
        prop_side="YES",
        model_version="test",
        model_probability=0.70,
        entry_price=0.50,
        entry_edge=0.20,
        stake=1.0,
    )
    place_bet(session, first)
    second = BetPlacement(**{**first.__dict__, "market_id": "p-4", "prop_threshold": 4})
    with pytest.raises(DuplicateBetError):
        place_bet(session, second)
