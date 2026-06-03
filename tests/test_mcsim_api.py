"""Tests for inference.mcsim_api — read-only MCSim prediction/actuals endpoints.

Two layers:
- the pure overlay helper (_overlay_actuals_on_card), tested directly, and
- the two HTTP endpoints, tested via a standalone FastAPI app + TestClient with
  the get_conn dependency overridden to a temp SQLite db (no model loaded).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import inference.mcsim_api as mcsim_api
from mcsim.storage import init_db, write_actual, write_prediction


# ---- a small synthetic card: 1 pitcher (9) vs 2 batters (7, 8) ----
def _card():
    return {
        "game_pk": 555, "game_date": "2025-07-20",
        "home_team": "TOR", "away_team": "SF",
        "n_cells": 2,
        "rows": [
            {"pitcher_id": 9, "name": "P9", "team": "TOR", "cells": [
                {"batter_id": 7, "batter_name": "B7", "predicted_ops": 0.9},
                {"batter_id": 8, "batter_name": "B8", "predicted_ops": 0.6},
            ]},
        ],
    }


def _actual():
    return {
        "final_score_home": 8, "final_score_away": 6, "winner": "home",
        "matchup_events": [
            {"pitcher_id": 9, "batter_id": 7, "event": "Single", "inning": 1},
            {"pitcher_id": 9, "batter_id": 7, "event": "Strikeout", "inning": 4},  # faced twice
            {"pitcher_id": 9, "batter_id": 999, "event": "Home Run", "inning": 6},  # not in grid
        ],
    }


# ============================================================
# Overlay helper (pure)
# ============================================================


def test_overlay_stamps_cells_and_counts_unmatched():
    card, unmatched = mcsim_api._overlay_actuals_on_card(_card(), _actual())
    cells = card["rows"][0]["cells"]
    # batter 7 faced pitcher twice -> pa_count 2, both events attached
    c7 = next(c for c in cells if c["batter_id"] == 7)
    assert c7["actual"]["pa_count"] == 2
    assert [e["event"] for e in c7["actual"]["events"]] == ["Single", "Strikeout"]
    # batter 8 never came up -> actual None
    c8 = next(c for c in cells if c["batter_id"] == 8)
    assert c8["actual"] is None
    # the (9, 999) PA isn't in the grid -> surfaced as unmatched, not dropped
    assert unmatched == 1


def test_overlay_no_actual_yet_leaves_cells_null():
    card, unmatched = mcsim_api._overlay_actuals_on_card(_card(), None)
    assert unmatched == 0
    for c in card["rows"][0]["cells"]:
        assert c["actual"] is None


def test_overlay_does_not_mutate_input_card():
    original = _card()
    mcsim_api._overlay_actuals_on_card(original, _actual())
    # original cells must NOT have gained an 'actual' key
    assert "actual" not in original["rows"][0]["cells"][0]


# ============================================================
# Endpoints (TestClient + temp db)
# ============================================================


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "api.sqlite"
    conn = init_db(db)
    write_prediction(conn, game_pk=555, prediction_date="2025-07-20",
                     app="matchup_card", payload=_card(), model_ckpt_hash="h1")
    write_actual(conn, game_pk=555, final_score_home=8, final_score_away=6,
                 winner="home", matchup_events=_actual()["matchup_events"])
    conn.commit()

    app = FastAPI()
    app.include_router(mcsim_api.router)
    # point the endpoint dependency at our temp db (no model loaded)
    app.dependency_overrides[mcsim_api.get_conn] = lambda: init_db(db)
    return TestClient(app)


def test_list_predictions_for_date(client):
    r = client.get("/mcsim/predictions", params={"date": "2025-07-20"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    s = body["predictions"][0]
    assert s["game_pk"] == 555 and s["home_team"] == "TOR"
    assert s["n_cells"] == 2 and s["has_actual"] is True


def test_get_prediction_overlays_actuals(client):
    r = client.get("/mcsim/predictions/555", params={"date": "2025-07-20"})
    assert r.status_code == 200
    body = r.json()
    assert body["has_actual"] is True
    assert body["final_score_home"] == 8 and body["winner"] == "home"
    assert body["unmatched_event_count"] == 1
    cells = body["card"]["rows"][0]["cells"]
    c7 = next(c for c in cells if c["batter_id"] == 7)
    assert c7["actual"]["pa_count"] == 2


def test_get_prediction_404_when_missing(client):
    r = client.get("/mcsim/predictions/999999", params={"date": "2025-07-20"})
    assert r.status_code == 404
