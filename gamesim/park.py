"""Park factors: per-outcome multipliers for the home team's ballpark.

Built from 604K PAs (2021-2023 training split). Each park has a multiplier
for each outcome class (K/BB/1B/2B/3B/HR/out) relative to league average.
Coors inflates hits/XBH, Petco/Tropicana suppress them.
"""
from __future__ import annotations

import json
from pathlib import Path

PARK_FACTORS_PATH = Path("data/run_value/park_factors.json")

# Full team name -> Statcast abbreviation
TEAM_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def load_park_factors() -> dict[str, dict[str, float]]:
    """Load park factors from JSON. Returns {team_abbr: {outcome: multiplier}}."""
    if not PARK_FACTORS_PATH.exists():
        return {}
    with open(PARK_FACTORS_PATH) as f:
        return json.load(f)


def resolve_park_factors(
    home_team: str,
    park_table: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    """Look up park factors for a home team, handling full names and abbreviations."""
    if home_team in park_table:
        return park_table[home_team]
    abbr = TEAM_ABBR.get(home_team)
    if abbr and abbr in park_table:
        return park_table[abbr]
    return None
