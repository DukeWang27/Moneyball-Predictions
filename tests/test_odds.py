import pytest

from moneyball_predictions.odds import (
    devig_two_way_decimal,
    expected_value,
    kelly_fraction,
)


def test_devigged_probabilities_sum_to_one() -> None:
    probability_a, probability_b = devig_two_way_decimal(1.91, 1.91)
    assert probability_a + probability_b == pytest.approx(1.0)
    assert probability_a == pytest.approx(0.5)


def test_expected_value_is_positive_when_model_probability_exceeds_break_even() -> None:
    assert expected_value(0.60, 2.0, 10.0) == pytest.approx(2.0)


def test_kelly_is_zero_for_negative_edge() -> None:
    assert kelly_fraction(0.40, 2.0) == 0.0
