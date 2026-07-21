from datetime import UTC, datetime, timedelta

import pytest

from moneyball_predictions.sabermetrics import (
    BatterEventProjection,
    BattingLine,
    PitchingLine,
    ReliefAppearance,
    availability_from_fatigue,
    base_runs_smyth_v1,
    calculate_fip,
    calculate_fip_constant,
    calculate_xfip,
    expected_pitching_runs_allowed,
    leverage_adjusted_bullpen_fatigue,
    project_lineup,
    pythagenpat_win_pct,
    shrink_rate,
)


def test_lineup_projection_uses_expected_pa_weights() -> None:
    batters = []
    for slot in range(1, 10):
        batters.append(
            BatterEventProjection(
                player_id=slot,
                batting_slot=slot,
                expected_pa=5.0 if slot == 1 else 4.0,
                event_rates={
                    "ubb": 0.10,
                    "hbp": 0.01,
                    "single": 0.14,
                    "double": 0.05,
                    "triple": 0.005,
                    "home_run": 0.04 if slot == 1 else 0.02,
                },
            )
        )
    projection = project_lineup(
        batters,
        woba_weights={
            "ubb": 0.69,
            "hbp": 0.72,
            "single": 0.88,
            "double": 1.25,
            "triple": 1.58,
            "home_run": 2.01,
        },
        run_values={
            "ubb": 0.33,
            "hbp": 0.34,
            "single": 0.47,
            "double": 0.78,
            "triple": 1.09,
            "home_run": 1.40,
        },
    )
    assert projection.expected_plate_appearances == pytest.approx(37.0)
    assert 0.2 < projection.xwoba < 0.5
    assert projection.linear_weight_runs_per_pa > 0


def test_shrink_rate_pulls_small_sample_to_prior() -> None:
    small = shrink_rate(3, 10, prior_rate=0.10, prior_strength=100)
    large = shrink_rate(300, 1000, prior_rate=0.10, prior_strength=100)
    assert abs(small - 0.10) < abs(large - 0.10)


def test_base_runs_and_pythagenpat_are_sensible() -> None:
    strong = BattingLine(1000, 280, 55, 8, 45, 105, 5, 10)
    weak = BattingLine(1000, 220, 35, 3, 20, 70, 3, 6)
    strong_runs = base_runs_smyth_v1(strong)
    weak_runs = base_runs_smyth_v1(weak)
    assert strong_runs > weak_runs
    assert pythagenpat_win_pct(strong_runs, weak_runs, 100) > 0.5


def test_fip_and_xfip_use_outs_not_decimal_innings() -> None:
    constant = calculate_fip_constant(
        league_era=4.20,
        league_home_runs=100,
        league_walks=300,
        league_hit_batters=40,
        league_strikeouts=900,
        league_outs_recorded=2700,
    )
    line = PitchingLine(
        outs_recorded=18,
        strikeouts=7,
        walks=2,
        hit_batters=0,
        home_runs=1,
        fly_balls=8,
    )
    assert calculate_fip(line, fip_constant=constant) == pytest.approx(
        (13 + 6 - 14) / 6 + constant
    )
    assert calculate_xfip(
        line,
        league_hr_per_fly_ball=0.10,
        fip_constant=constant,
    ) < calculate_fip(line, fip_constant=constant)


def test_starter_workload_changes_combined_run_expectation() -> None:
    short = expected_pitching_runs_allowed(
        starter_runs_per_nine=3.0,
        expected_starter_innings=4.0,
        bullpen_runs_per_nine=5.0,
    )
    long = expected_pitching_runs_allowed(
        starter_runs_per_nine=3.0,
        expected_starter_innings=7.0,
        bullpen_runs_per_nine=5.0,
    )
    assert long < short


def test_leverage_adjusted_fatigue_penalizes_high_leverage_recent_work() -> None:
    as_of = datetime(2026, 7, 20, 18, tzinfo=UTC)
    appearances = [
        ReliefAppearance(1, as_of - timedelta(hours=20), 25, 2.0),
        ReliefAppearance(2, as_of - timedelta(hours=20), 25, 0.5),
        ReliefAppearance(1, as_of - timedelta(hours=44), 20, 1.5),
    ]
    fatigue = leverage_adjusted_bullpen_fatigue(appearances, as_of=as_of)
    assert fatigue[1] > fatigue[2]
    assert availability_from_fatigue(fatigue[1]) < availability_from_fatigue(fatigue[2])
