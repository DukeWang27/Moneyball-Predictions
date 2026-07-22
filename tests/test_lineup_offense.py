from moneyball_predictions.lineup_offense import apply_lineup_logit_adjustment
from moneyball_predictions.live import paper_bet_eligibility


def test_lineup_adjustment_moves_probability_in_expected_direction() -> None:
    improved, logit_adjustment = apply_lineup_logit_adjustment(
        0.55,
        0.025,
        coefficient=4.0,
        maximum_logit_adjustment=0.20,
    )
    weakened, _ = apply_lineup_logit_adjustment(
        0.55,
        -0.025,
        coefficient=4.0,
        maximum_logit_adjustment=0.20,
    )
    assert logit_adjustment == 0.10
    assert improved > 0.55
    assert weakened < 0.55


def test_lineup_adjustment_is_bounded() -> None:
    adjusted, logit_adjustment = apply_lineup_logit_adjustment(
        0.50,
        1.0,
        coefficient=10.0,
        maximum_logit_adjustment=0.20,
    )
    assert logit_adjustment == 0.20
    assert 0.54 < adjusted < 0.56


def test_paper_bet_gate_waits_for_confirmed_lineups() -> None:
    allowed, reason = paper_bet_eligibility(
        lineup_status="PENDING",
        lineup_model_used=False,
        starters_confirmed=True,
        signal="BET",
        edge=0.10,
        expected_roi=0.20,
    )
    assert allowed is False
    assert "starting lineups" in reason


def test_paper_bet_gate_unlocks_only_after_repricing() -> None:
    allowed, reason = paper_bet_eligibility(
        lineup_status="CONFIRMED",
        lineup_model_used=True,
        starters_confirmed=True,
        signal="BET",
        edge=0.06,
        expected_roi=0.08,
    )
    assert allowed is True
    assert reason == "Eligible"
