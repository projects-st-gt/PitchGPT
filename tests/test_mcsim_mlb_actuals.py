"""Tests for mcsim.mlb_actuals — parsing the MLB live game feed into a
GameActuals (final score + winner + per-PA events with start context).

No live network in CI: every test patches mcsim.mlb_actuals._get_json with a
canned feed shaped like the real statsapi /feed/live payload (verified against
game_pk 777079). The base-state reconstruction is pinned explicitly because
the convenient-looking matchup.splits.menOnBase field is a stat-split LABEL,
not live state (it reports "RISP" for a leadoff hitter) — the correct source
is runners[].movement.originBase.
"""
from __future__ import annotations

import mcsim.mlb_actuals as actuals


def _pa(*, batter, pitcher, inning, half, outs_start, origin_bases, balls, strikes,
        event, event_type, pitch_codes, scores=()):
    """Build one canned allPlays entry. ``origin_bases`` are the bases occupied
    at PA start; ``scores`` are originBases whose runner scored on the play."""
    runners = []
    for b in origin_bases:
        runners.append({"movement": {"originBase": b,
                                     "end": "score" if b in scores else "2B"}})
    play_events = []
    bs = 0
    st = 0
    for code in pitch_codes:
        play_events.append({
            "isPitch": True,
            "count": {"balls": bs, "strikes": st, "outs": outs_start},
            "details": {"type": {"code": code}},
        })
    return {
        "about": {"inning": inning, "halfInning": half, "isComplete": True},
        "count": {"balls": balls, "strikes": strikes, "outs": outs_start},
        "matchup": {
            "batter": {"id": batter, "fullName": f"B{batter}"},
            "pitcher": {"id": pitcher, "fullName": f"P{pitcher}"},
            "batSide": {"code": "L"}, "pitchHand": {"code": "R"},
            "splits": {"menOnBase": "RISP"},  # deliberately wrong label — must be ignored
        },
        "result": {"event": event, "eventType": event_type, "rbi": len(scores)},
        "runners": runners,
        "playEvents": play_events,
    }


_FEED = {
    "gameData": {
        "status": {"detailedState": "Final", "abstractGameState": "Final"},
        "teams": {"home": {"id": 141, "abbreviation": "TOR"},
                  "away": {"id": 137, "abbreviation": "SF"}},
    },
    "liveData": {
        "linescore": {"teams": {"home": {"runs": 8}, "away": {"runs": 6}}},
        "plays": {"allPlays": [
            _pa(batter=1, pitcher=99, inning=1, half="top", outs_start=0,
                origin_bases=[], balls=1, strikes=2,
                event="Double", event_type="double", pitch_codes=["FF", "FF", "SL"]),
            _pa(batter=2, pitcher=99, inning=1, half="top", outs_start=0,
                origin_bases=["2B"], balls=0, strikes=0,
                event="Single", event_type="single", pitch_codes=["CH"], scores=["2B"]),
            # an incomplete play must be skipped
            {"about": {"inning": 1, "halfInning": "top", "isComplete": False},
             "matchup": {"batter": {"id": 3}, "pitcher": {"id": 99}}, "result": {}},
        ]},
    },
}


def test_get_game_actuals_score_and_winner(monkeypatch):
    monkeypatch.setattr(actuals, "_get_json", lambda url, **kw: _FEED)
    a = actuals.get_game_actuals(777079)
    assert a.is_final is True and a.status == "Final"
    assert a.final_score_home == 8 and a.final_score_away == 6
    assert a.winner == "home"


def test_get_game_actuals_parses_completed_pas_only(monkeypatch):
    monkeypatch.setattr(actuals, "_get_json", lambda url, **kw: _FEED)
    a = actuals.get_game_actuals(777079)
    assert len(a.matchup_events) == 2  # the isComplete=False play is dropped


def test_get_game_actuals_reconstructs_start_context(monkeypatch):
    monkeypatch.setattr(actuals, "_get_json", lambda url, **kw: _FEED)
    a = actuals.get_game_actuals(777079)
    pa0, pa1 = a.matchup_events
    # leadoff: bases empty (NOT "RISP" from the bogus splits label)
    assert pa0["bases_start"] == [] and pa0["outs_start"] == 0
    assert pa0["pitcher_id"] == 99 and pa0["batter_id"] == 1
    assert pa0["event"] == "Double" and pa0["pitch_types"] == ["FF", "FF", "SL"]
    assert pa0["runs_scored"] == 0
    # second PA: runner started on 2B and scored
    assert pa1["bases_start"] == ["2B"]
    assert pa1["runs_scored"] == 1 and pa1["rbi"] == 1
    assert pa1["pitch_types"] == ["CH"]


def test_winner_tie_and_unknown():
    assert actuals._winner(5, 5) == "tie"
    assert actuals._winner(3, 1) == "home"
    assert actuals._winner(1, 3) == "away"
    assert actuals._winner(None, 2) is None
