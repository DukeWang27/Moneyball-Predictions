import pytest

from moneyball_predictions.odds import (
    devig_two_way_decimal,
    expected_value,
    kelly_fraction,
    prediction_market_kelly_fraction,
    prediction_market_kelly_stake,
)


def test_devigged_probabilities_sum_to_one() -> None:
    probability_a, probability_b = devig_two_way_decimal(1.91, 1.91)
    assert probability_a + probability_b == pytest.approx(1.0)
    assert probability_a == pytest.approx(0.5)


def test_expected_value_is_positive_when_model_probability_exceeds_break_even() -> None:
    assert expected_value(0.60, 2.0, 10.0) == pytest.approx(2.0)


def test_kelly_is_zero_for_negative_edge() -> None:
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_prediction_market_expected_value() -> None:
    from moneyball_predictions.odds import prediction_market_expected_value

    # A 60% model probability at a 50-cent ask has $2 EV on a $10 stake.
    assert prediction_market_expected_value(0.60, 0.50, 10.0) == pytest.approx(2.0)


def test_prediction_market_kelly_fraction_uses_share_price() -> None:
    # At p=60% and a 50-cent ask, full Kelly is (0.60-0.50)/(1-0.50) = 20%.
    assert prediction_market_kelly_fraction(0.60, 0.50) == pytest.approx(0.20)


def test_quarter_kelly_sizes_from_bankroll() -> None:
    stake, full, applied, capped = prediction_market_kelly_stake(
        model_probability=0.60,
        buy_price=0.50,
        bankroll=100.0,
        multiplier=0.25,
    )
    assert full == pytest.approx(0.20)
    assert applied == pytest.approx(0.05)
    assert stake == pytest.approx(5.0)
    assert capped is False


def test_kelly_stake_is_capped_by_best_ask_depth() -> None:
    stake, _, _, capped = prediction_market_kelly_stake(
        model_probability=0.80,
        buy_price=0.50,
        bankroll=100.0,
        multiplier=1.0,
        top_ask_size=10.0,
    )
    assert stake == pytest.approx(5.0)
    assert capped is True


def test_kelly_can_be_capped_at_one_percent_of_bankroll() -> None:
    stake, full, applied, _ = prediction_market_kelly_stake(
        model_probability=0.80,
        buy_price=0.50,
        bankroll=100.0,
        multiplier=0.25,
        max_bankroll_fraction=0.01,
    )
    assert full == pytest.approx(0.60)
    assert applied == pytest.approx(0.01)
    assert stake == pytest.approx(1.0)
