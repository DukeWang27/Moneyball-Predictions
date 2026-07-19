import pytest

from moneyball_predictions.model import (
    dynamic_pythagorean_exponent,
    log5_probability,
    pythagorean_expectation,
)


def test_dynamic_exponent_is_positive() -> None:
    assert dynamic_pythagorean_exponent(500, 450, 100) > 0


def test_pythagorean_expectation_is_above_half_for_positive_run_differential() -> None:
    assert pythagorean_expectation(500, 400, 100) > 0.5


def test_log5_is_even_for_equal_teams() -> None:
    assert log5_probability(0.6, 0.6) == pytest.approx(0.5)


def test_log5_probabilities_are_complementary_when_teams_are_swapped() -> None:
    probability_a = log5_probability(0.62, 0.48)
    probability_b = log5_probability(0.48, 0.62)
    assert probability_a + probability_b == pytest.approx(1.0)


def test_home_field_moves_equal_matchup_above_half() -> None:
    from moneyball_predictions.model import add_home_field_advantage

    assert add_home_field_advantage(0.5, 0.54) == pytest.approx(0.54)


def test_regression_reduces_small_sample_extremes() -> None:
    from moneyball_predictions.model import (
        pythagorean_expectation,
        regressed_pythagorean_expectation,
    )

    raw = pythagorean_expectation(30, 5, 5)
    regressed = regressed_pythagorean_expectation(
        runs_scored=30,
        runs_allowed=5,
        games_played=5,
        prior_runs_scored_per_game=4.5,
        prior_runs_allowed_per_game=4.5,
        league_runs_per_team_game=4.5,
        pseudo_games=25,
    )
    assert 0.5 < regressed < raw
