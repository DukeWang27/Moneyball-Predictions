from dataclasses import dataclass

import pytest

from moneyball_predictions.tournament import (
    TournamentCandidate,
    _choose_architecture,
    _poisson_home_win_probability,
)


@dataclass
class DummyPredictor:
    probability: float = 0.5

    def predict(self, features: tuple[float, ...]) -> float:
        return self.probability


def candidate(key: str, brier: float, log_loss: float) -> TournamentCandidate:
    return TournamentCandidate(
        key=key,
        label=key,
        family="test",
        predictor=DummyPredictor(),
        validation_brier=brier,
        validation_log_loss=log_loss,
        calibration_method="identity",
    )


def test_poisson_runs_probability_moves_with_run_advantage() -> None:
    assert _poisson_home_win_probability(3.5, 5.0) > 0.5
    assert _poisson_home_win_probability(5.0, 3.5) < 0.5
    assert _poisson_home_win_probability(4.5, 4.5) > 0.5


def test_tournament_retains_logistic_when_challenger_is_unstable() -> None:
    folds = [
        {
            "logistic": candidate("logistic", 0.2450, 0.6820),
            "boosted_tree": candidate("boosted_tree", 0.2430, 0.6790),
        },
        {
            "logistic": candidate("logistic", 0.2450, 0.6820),
            "boosted_tree": candidate("boosted_tree", 0.2460, 0.6830),
        },
    ]
    champion, _, promoted, _, _ = _choose_architecture(folds)
    assert champion == "logistic"
    assert promoted is False


def test_tournament_promotes_stable_brier_improvement() -> None:
    folds = [
        {
            "logistic": candidate("logistic", 0.2450, 0.6820),
            "ensemble": candidate("ensemble", 0.2440, 0.6815),
        },
        {
            "logistic": candidate("logistic", 0.2460, 0.6840),
            "ensemble": candidate("ensemble", 0.2450, 0.6835),
        },
    ]
    champion, reason, promoted, briers, _ = _choose_architecture(folds)
    assert champion == "ensemble"
    assert promoted is True
    assert "Promoted" in reason
    assert briers["ensemble"] == pytest.approx((0.2440, 0.2450))
