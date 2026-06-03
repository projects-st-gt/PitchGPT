"""Tests for the MCSim App B actuals CLI (scripts/mcsim/fetch_actuals.py).

Network is fully mocked: get_schedule + get_game_actuals are monkeypatched, so
the test exercises the fetch->persist orchestration and the round-trip through
mcsim.storage without touching statsapi.
"""
from __future__ import annotations

from pathlib import Path

from scripts.mcsim import fetch_actuals as fa
from mcsim.mlb_actuals import GameActuals
from mcsim.mlb_api import GameInfo
from mcsim.storage import init_db, read_actual


def test_fetch_and_store_actuals_persists_final_games(tmp_path, monkeypatch, capsys):
    games = [
        GameInfo(101, 1, 2, "Home1", "Away1", None, None),
        GameInfo(102, 3, 4, "Home2", "Away2", None, None),  # not final -> skipped
    ]
    actuals_by_pk = {
        101: GameActuals(
            game_pk=101, status="Final", is_final=True,
            final_score_home=5, final_score_away=2, winner="home",
            matchup_events=[{"pitcher_id": 9, "batter_id": 7, "event": "Strikeout",
                             "bases_start": [], "outs_start": 1}],
        ),
        102: GameActuals(
            game_pk=102, status="In Progress", is_final=False,
            final_score_home=None, final_score_away=None, winner=None,
            matchup_events=[],
        ),
    }
    monkeypatch.setattr(fa, "get_schedule", lambda date: games)
    monkeypatch.setattr(fa, "get_game_actuals", lambda pk: actuals_by_pk[pk])

    conn = init_db(tmp_path / "a.sqlite")
    stored = fa.fetch_and_store_actuals(conn, date="2025-07-20", game_pks=None)

    assert [a.game_pk for a in stored] == [101]  # only the final game persisted
    got = read_actual(conn, game_pk=101)
    assert got is not None
    assert got["final_score_home"] == 5 and got["final_score_away"] == 2
    assert got["winner"] == "home"
    assert got["matchup_events"][0]["event"] == "Strikeout"
    # the non-final game was not written
    assert read_actual(conn, game_pk=102) is None
    out = capsys.readouterr().out
    assert "101" in out and "not final" in out  # logged both the write and the skip


def test_fetch_and_store_actuals_game_pk_filter(tmp_path, monkeypatch):
    games = [GameInfo(101, 1, 2, "H1", "A1", None, None),
             GameInfo(103, 5, 6, "H3", "A3", None, None)]
    fin = lambda pk: GameActuals(pk, "Final", True, 1, 0, "home", [])
    monkeypatch.setattr(fa, "get_schedule", lambda date: games)
    monkeypatch.setattr(fa, "get_game_actuals", fin)

    conn = init_db(tmp_path / "b.sqlite")
    stored = fa.fetch_and_store_actuals(conn, date="2025-07-20", game_pks=[103])
    assert [a.game_pk for a in stored] == [103]  # 101 filtered out
