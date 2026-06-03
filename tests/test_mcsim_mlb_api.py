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


def test_get_schedule_skips_malformed_game(monkeypatch):
    j = {"dates": [{"games": [
        {"gamePk": 1, "teams": {  # good
            "home": {"team": {"id": 10, "name": "H"}},
            "away": {"team": {"id": 20, "name": "A"}}}},
        {"gamePk": 2},  # malformed — no "teams"
    ]}]}
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: j)
    games = mlb_api.get_schedule("2026-06-02")
    assert [g.game_pk for g in games] == [1]  # malformed game skipped, batch survives


def test_get_schedule_empty_returns_empty(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: {"dates": []})
    assert mlb_api.get_schedule("2026-06-02") == []


_ROSTER_JSON = {
    "roster": [
        {"position": {"type": "Pitcher", "abbreviation": "P"},
         "person": {"id": 111, "fullName": "Righty Starter",
                    "pitchHand": {"code": "R"}, "batSide": {"code": "R"}}},
        {"position": {"type": "Pitcher", "abbreviation": "P"},
         "person": {"id": 112, "fullName": "Lefty Reliever",
                    "pitchHand": {"code": "L"}, "batSide": {"code": "L"}}},
        {"position": {"type": "Infielder", "abbreviation": "2B"},
         "person": {"id": 201, "fullName": "Switch Hitter",
                    "pitchHand": {"code": "R"}, "batSide": {"code": "S"}}},
        {"position": {"type": "Outfielder", "abbreviation": "CF"},
         "person": {"id": 202, "fullName": "Lefty Bat",
                    "pitchHand": {"code": "L"}, "batSide": {"code": "L"}}},
    ]
}


def test_get_active_roster_splits_and_handedness(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: _ROSTER_JSON)
    pitchers, hitters = mlb_api.get_active_roster(139, "2026-06-02",
                                                  probable_pitcher_id=111)
    assert [p.id for p in pitchers] == [111, 112]
    assert [h.id for h in hitters] == [201, 202]
    assert all(isinstance(p, PitcherSpec) for p in pitchers)
    assert all(isinstance(h, BatterSpec) for h in hitters)
    # handedness
    assert pitchers[0].throws == "R" and pitchers[1].throws == "L"
    # probable starter flagged
    assert pitchers[0].is_starter is True and pitchers[1].is_starter is False
    # switch hitter resolves to 'L' for v1; explicit-side hitter unchanged
    assert hitters[0].stand == "L"   # was "S"
    assert hitters[1].stand == "L"


def test_get_active_roster_no_probable(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: _ROSTER_JSON)
    pitchers, _ = mlb_api.get_active_roster(139, "2026-06-02")
    assert all(p.is_starter is False for p in pitchers)
