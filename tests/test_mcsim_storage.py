"""Tests for ``mcsim.storage`` — SQLite round-trip for predictions + actuals.

Every test uses ``tmp_path`` so the prod DB is never touched. Tests cover:

  - init_db is idempotent (call twice on the same path, no errors)
  - schema version is set on first init
  - prediction round-trip: write, read back identical
  - prediction upsert: re-writing the same key updates instead of inserting
  - listing predictions by date returns multiple rows in deterministic order
  - actuals two-pass write: first call sets line score, second call adds
    matchup_events without nulling out the existing line score
  - winner enum validation
  - model_versions register is idempotent
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mcsim.storage import (
    SCHEMA_VERSION,
    init_db,
    lookup_model_version,
    read_actual,
    read_prediction,
    read_predictions_for_date,
    register_model_version,
    write_actual,
    write_prediction,
)


# ============================================================
# init_db
# ============================================================


def test_init_db_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "mcsim.sqlite"
    c1 = init_db(db_path)
    c1.close()
    c2 = init_db(db_path)  # second open must not error
    # Schema version pin
    cur = c2.execute("PRAGMA user_version")
    assert int(cur.fetchone()[0]) == SCHEMA_VERSION
    c2.close()


def test_init_db_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "mcsim.sqlite"
    conn = init_db(nested)
    assert nested.exists()
    conn.close()


# ============================================================
# Predictions
# ============================================================


def test_prediction_roundtrip(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    payload = {"rows": [{"pitcher_id": 543037, "cells": [{"batter_id": 605141, "rv": -0.012}]}]}
    write_prediction(
        conn, game_pk=776543, prediction_date="2026-05-30", app="matchup_card",
        payload=payload, model_ckpt_hash="abc123",
    )
    got = read_prediction(
        conn, game_pk=776543, prediction_date="2026-05-30", app="matchup_card",
    )
    assert got is not None
    assert got["game_pk"] == 776543
    assert got["prediction_date"] == "2026-05-30"
    assert got["app"] == "matchup_card"
    assert got["model_ckpt_hash"] == "abc123"
    assert got["payload"] == payload  # decoded back to dict, deep equality
    assert got["made_at"]  # ISO timestamp populated by default


def test_prediction_upsert_overwrites_same_key(tmp_path: Path):
    """Re-running the nightly job for the same (game, date, app) must
    UPDATE the existing row, not INSERT a duplicate (the UNIQUE constraint
    would refuse a duplicate insert anyway, but we want overwrite, not
    error)."""
    conn = init_db(tmp_path / "x.sqlite")
    write_prediction(conn, game_pk=1, prediction_date="2026-05-30", app="matchup_card",
                     payload={"v": 1}, model_ckpt_hash="h1", made_at="2026-05-29T20:00:00+00:00")
    write_prediction(conn, game_pk=1, prediction_date="2026-05-30", app="matchup_card",
                     payload={"v": 2}, model_ckpt_hash="h2", made_at="2026-05-30T06:00:00+00:00")
    got = read_prediction(conn, game_pk=1, prediction_date="2026-05-30", app="matchup_card")
    assert got["payload"] == {"v": 2}
    assert got["model_ckpt_hash"] == "h2"
    assert got["made_at"] == "2026-05-30T06:00:00+00:00"
    # Only one row exists
    rows = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
    assert int(rows[0]) == 1


def test_prediction_missing_returns_none(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    got = read_prediction(conn, game_pk=999, prediction_date="2026-05-30", app="matchup_card")
    assert got is None


def test_read_predictions_for_date_orders_by_game_pk(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    for gpk in (300, 100, 200):
        write_prediction(conn, game_pk=gpk, prediction_date="2026-05-30",
                         app="matchup_card", payload={"gpk": gpk}, model_ckpt_hash="h")
    rows = read_predictions_for_date(conn, prediction_date="2026-05-30")
    assert [r["game_pk"] for r in rows] == [100, 200, 300]


def test_read_predictions_for_date_filters_by_app(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    write_prediction(conn, game_pk=1, prediction_date="2026-05-30",
                     app="matchup_card", payload={}, model_ckpt_hash="h")
    write_prediction(conn, game_pk=1, prediction_date="2026-05-30",
                     app="score_prediction", payload={}, model_ckpt_hash="h")
    only_matchup = read_predictions_for_date(conn, prediction_date="2026-05-30", app="matchup_card")
    assert len(only_matchup) == 1
    assert only_matchup[0]["app"] == "matchup_card"


def test_prediction_rejects_unknown_app(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    with pytest.raises(ValueError, match="unknown app"):
        write_prediction(conn, game_pk=1, prediction_date="2026-05-30",
                         app="not_an_app", payload={}, model_ckpt_hash="h")


# ============================================================
# Actuals — two-pass write
# ============================================================


def test_actuals_two_pass_keeps_existing_fields(tmp_path: Path):
    """Pass 1 (right after game ends): set line score, no matchup_events.
    Pass 2 (next morning): add matchup_events, NO line score in this call.
    The final row must have BOTH — pass 2 must not null out pass 1's line score.
    """
    conn = init_db(tmp_path / "x.sqlite")
    # Pass 1 — line score only
    write_actual(conn, game_pk=1, final_score_home=7, final_score_away=3, winner="home")
    # Pass 2 — matchup events only
    matchup = [
        {"pitcher_id": 543037, "batter_id": 605141, "ab_outcomes": ["K", "1B"]},
    ]
    write_actual(conn, game_pk=1, matchup_events=matchup)
    got = read_actual(conn, game_pk=1)
    assert got["final_score_home"] == 7
    assert got["final_score_away"] == 3
    assert got["winner"] == "home"
    assert got["matchup_events"] == matchup


def test_actual_missing_returns_none(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    assert read_actual(conn, game_pk=42) is None


def test_actual_rejects_unknown_winner(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    with pytest.raises(ValueError, match="winner must be"):
        write_actual(conn, game_pk=1, winner="dolphins")


# ============================================================
# Model versions
# ============================================================


def test_register_model_version_idempotent(tmp_path: Path):
    conn = init_db(tmp_path / "x.sqlite")
    register_model_version(conn, ckpt_hash="abc", label="v7", trained_at="2026-05-21")
    register_model_version(conn, ckpt_hash="abc", label="v7 again")  # duplicate hash
    got = lookup_model_version(conn, ckpt_hash="abc")
    assert got["label"] == "v7"  # second call did NOT overwrite
    rows = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()
    assert int(rows[0]) == 1


def test_write_prediction_sanitizes_non_finite_to_null(tmp_path: Path):
    """Non-finite floats (NaN/Inf — e.g. an all-truncated cell's run value)
    must be stored as JSON null, not bare ``NaN``/``Infinity`` (invalid JSON
    that strict parsers like JS ``JSON.parse`` reject). Frontend reads this
    payload, so the stored text must be spec-valid."""
    import json as _json

    conn = init_db(tmp_path / "nan.sqlite")
    payload = {"rows": [{"cell": {"rv": float("nan"),
                                  "p95": float("inf"),
                                  "p05": float("-inf"),
                                  "ok": 0.5}}]}
    write_prediction(conn, game_pk=7, prediction_date="2026-06-03",
                     app="matchup_card", payload=payload, model_ckpt_hash="h")

    raw = conn.execute(
        "SELECT payload_json FROM predictions WHERE game_pk=7"
    ).fetchone()[0]
    assert "NaN" not in raw and "Infinity" not in raw  # spec-valid JSON text

    cell = _json.loads(raw)["rows"][0]["cell"]
    assert cell["rv"] is None and cell["p95"] is None and cell["p05"] is None
    assert cell["ok"] == 0.5  # finite values untouched

    got = read_prediction(conn, game_pk=7, prediction_date="2026-06-03",
                          app="matchup_card")
    assert got["payload"]["rows"][0]["cell"]["rv"] is None
