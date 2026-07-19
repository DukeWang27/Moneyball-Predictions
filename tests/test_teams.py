from moneyball_predictions.teams import canonical_team_name


def test_team_aliases_match_common_polymarket_labels() -> None:
    assert canonical_team_name("NYY") == "New York Yankees"
    assert canonical_team_name("LA Dodgers") == "Los Angeles Dodgers"
    assert canonical_team_name("D-backs") == "Arizona Diamondbacks"
    assert canonical_team_name("St Louis Cardinals") == "St. Louis Cardinals"


def test_unknown_team_is_not_silently_guessed() -> None:
    assert canonical_team_name("Completely Fake Club") is None


def test_extract_team_pair_from_matchup_title() -> None:
    from moneyball_predictions.teams import extract_team_pair

    assert extract_team_pair("Oakland A's vs. LA Angels") == (
        "Athletics",
        "Los Angeles Angels",
    )


def test_extract_team_pair_rejects_futures_title() -> None:
    from moneyball_predictions.teams import extract_team_pair

    assert extract_team_pair("MLB World Series Champion 2026") is None
