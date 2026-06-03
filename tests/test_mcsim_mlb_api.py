"""Tests for mcsim.mlb_api — schedule + roster parsing against canned JSON.

No live network in CI: every test patches mcsim.mlb_api._get_json (the single
network seam) with canned statsapi-shaped dicts.
"""
from __future__ import annotations

import mcsim.mlb_api as mlb_api
from mcsim.matchup_card import BatterSpec, PitcherSpec

_SCHED_JSON = {
    "dates": [{"games": [{
        "gamePk": 776543,
        "teams": {
            "home": {"team": {"id": 139, "name": "Tampa Bay Rays"},
                     "probablePitcher": {"id": 111, "fullName": "Home SP"}},
            "away": {"team": {"id": 116, "name": "Detroit Tigers"},
                     "probablePitcher": {"id": 222, "fullName": "Away SP"}},
        },
    }]}]
}


def test_get_schedule_parses_games(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: _SCHED_JSON)
    games = mlb_api.get_schedule("2026-06-02")
    assert len(games) == 1
    g = games[0]
    assert g.game_pk == 776543
    assert g.home_team_id == 139 and g.away_team_id == 116
    assert g.home_team == "Tampa Bay Rays" and g.away_team == "Detroit Tigers"
    assert g.home_probable_pitcher_id == 111
    assert g.away_probable_pitcher_id == 222


def test_get_schedule_handles_missing_probable(monkeypatch):
    j = {"dates": [{"games": [{
        "gamePk": 1, "teams": {
            "home": {"team": {"id": 1, "name": "H"}},
            "away": {"team": {"id": 2, "name": "A"}},
        }}]}]}
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: j)
    g = mlb_api.get_schedule("2026-06-02")[0]
    assert g.home_probable_pitcher_id is None
    assert g.away_probable_pitcher_id is None
