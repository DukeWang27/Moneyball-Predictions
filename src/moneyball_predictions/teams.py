"""Canonical MLB team aliases used to match external labels safely."""

from __future__ import annotations

import re
import unicodedata


TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "Arizona Diamondbacks": (
        "ARI",
        "Arizona",
        "Diamondbacks",
        "D-backs",
        "Dbacks",
        "DBacks",
        "D-Backs",
    ),
    "Athletics": (
        "ATH",
        "OAK",
        "Athletics",
        "A's",
        "As",
        "Oakland A's",
        "Oakland Athletics",
        "Sacramento A's",
        "Sacramento Athletics",
    ),
    "Atlanta Braves": ("ATL", "Atlanta", "Braves"),
    "Baltimore Orioles": ("BAL", "Baltimore", "Orioles", "O's", "Os"),
    "Boston Red Sox": ("BOS", "Boston", "Red Sox", "Boston Sox"),
    "Chicago Cubs": ("CHC", "Chicago Cubs", "Cubs", "Chi Cubs"),
    "Chicago White Sox": (
        "CWS",
        "CHW",
        "Chicago White Sox",
        "White Sox",
        "Chi White Sox",
        "Chi Sox",
    ),
    "Cincinnati Reds": ("CIN", "Cincinnati", "Reds"),
    "Cleveland Guardians": ("CLE", "Cleveland", "Guardians"),
    "Colorado Rockies": ("COL", "Colorado", "Rockies"),
    "Detroit Tigers": ("DET", "Detroit", "Tigers"),
    "Houston Astros": ("HOU", "Houston", "Astros"),
    "Kansas City Royals": ("KC", "KCR", "Kansas City", "Royals", "KC Royals"),
    "Los Angeles Angels": (
        "LAA",
        "LA Angels",
        "Los Angeles Angels",
        "Angels",
        "Anaheim Angels",
    ),
    "Los Angeles Dodgers": ("LAD", "LA Dodgers", "Los Angeles Dodgers", "Dodgers"),
    "Miami Marlins": ("MIA", "Miami", "Marlins"),
    "Milwaukee Brewers": ("MIL", "Milwaukee", "Brewers"),
    "Minnesota Twins": ("MIN", "Minnesota", "Twins"),
    "New York Mets": ("NYM", "NY Mets", "New York Mets", "Mets"),
    "New York Yankees": ("NYY", "NY Yankees", "New York Yankees", "Yankees"),
    "Philadelphia Phillies": ("PHI", "Philadelphia", "Phillies", "Phils"),
    "Pittsburgh Pirates": ("PIT", "Pittsburgh", "Pirates", "Bucs"),
    "San Diego Padres": ("SD", "SDP", "San Diego", "Padres"),
    "San Francisco Giants": ("SF", "SFG", "San Francisco", "Giants"),
    "Seattle Mariners": ("SEA", "Seattle", "Mariners", "M's", "Ms"),
    "St. Louis Cardinals": (
        "STL",
        "St Louis Cardinals",
        "St. Louis Cardinals",
        "Cardinals",
        "Cards",
    ),
    "Tampa Bay Rays": ("TB", "TBR", "Tampa Bay", "Rays"),
    "Texas Rangers": ("TEX", "Texas", "Rangers"),
    "Toronto Blue Jays": ("TOR", "Toronto", "Blue Jays", "Jays"),
    "Washington Nationals": ("WSH", "WAS", "Washington", "Nationals", "Nats"),
}


def normalize_team_name(value: str) -> str:
    """Normalize punctuation, accents, and whitespace for deterministic matching."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    lowered = ascii_value.casefold().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in TEAM_ALIASES.items():
    for alias in (canonical, *aliases):
        _ALIAS_TO_CANONICAL[normalize_team_name(alias)] = canonical


def canonical_team_name(value: str | None) -> str | None:
    """Return the canonical MLB name for a known external label."""
    if not value:
        return None
    normalized = normalize_team_name(value)
    direct = _ALIAS_TO_CANONICAL.get(normalized)
    if direct:
        return direct

    matches = {
        canonical
        for alias, canonical in _ALIAS_TO_CANONICAL.items()
        if len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", normalized)
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def extract_team_pair(value: str | None) -> tuple[str, str] | None:
    """Extract exactly two MLB teams from an event title, preserving title order."""
    if not value:
        return None
    normalized = normalize_team_name(value)
    mentions: list[tuple[int, int, str]] = []
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if len(alias) < 3:
            continue
        match = re.search(rf"\b{re.escape(alias)}\b", normalized)
        if match:
            mentions.append((match.start(), -len(alias), canonical))

    mentions.sort()
    ordered: list[str] = []
    for _, _, canonical in mentions:
        if canonical not in ordered:
            ordered.append(canonical)
    if len(ordered) == 2:
        return ordered[0], ordered[1]
    return None
