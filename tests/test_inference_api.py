"""Smoke tests for ``inference/api.py``.

Integration tests: spin up the FastAPI app via ``TestClient``, which
lazy-loads the v7 checkpoint and val 2024 H1 parquets on the first request.
The first test triggers the load (~5–10 s); subsequent tests reuse the
cached state via the module-scoped ``client`` fixture.

Tests cover the demo's critical paths so a regression — bad config field, a
schema drift, a wiring error after a checkpoint switch — fails CI rather
than the demo at runtime:

- ``/health``     liveness + checkpoint path
- ``/games``      game picker
- ``/at-bats``    AB picker (filtered by ``min_pitches``)
- ``/ab-context`` per-AB metadata + arsenal + batter heatmap
- ``/query``      counterfactual rollout (the headline endpoint)
- ``/query``      input validation surface

Requires::

    checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt
    data/augmented/2024/2024-*.parquet
"""
from __future__ import annotations

import pytest
import torch

# MPS miscompiles the AB-outcome gather (pre-existing Issue #2); the API
# defaults to CPU anyway but be defensive in test env so a stray MPS init
# can't break runs on Apple silicon.
torch.backends.mps.is_available = lambda: False

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Import inside the fixture so the model isn't loaded at module import
    # time (it isn't anyway — see AppState lazy load — but be explicit).
    from inference.api import app
    with TestClient(app) as c:
        yield c


# ============================================================
# Liveness
# ============================================================


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "v7" in body["nuisance_checkpoint"], (
        f"expected v7 in checkpoint path, got {body['nuisance_checkpoint']!r}"
    )


# ============================================================
# Picker endpoints
# ============================================================


def test_games_returns_list(client):
    r = client.get("/games?limit=10")
    assert r.status_code == 200
    body = r.json()
    items = body["items"]
    assert len(items) > 0, "no games returned"
    g = items[0]
    # Required picker fields per schemas.GameSummary
    for k in ("game_pk", "game_date", "n_at_bats"):
        assert k in g, f"missing field {k!r} in /games row"


def test_at_bats_for_game(client):
    games = client.get("/games?limit=1").json()["items"]
    assert games, "no games available for at-bats test"
    gpk = games[0]["game_pk"]
    r = client.get(f"/at-bats?game_pk={gpk}&min_pitches=3")
    assert r.status_code == 200
    body = r.json()
    abs_ = body["items"]
    assert len(abs_) > 0, f"no at-bats with min_pitches=3 in game {gpk}"
    ab = abs_[0]
    for k in ("at_bat_number", "pitcher_id", "batter_id", "n_pitches"):
        assert k in ab, f"missing field {k!r} in /at-bats row"
    assert ab["n_pitches"] >= 3


# ============================================================
# Per-AB context (the cached pre-render the frontend reads)
# ============================================================


def test_ab_context_returns_positions_and_arsenal(client):
    gpk = client.get("/games?limit=1").json()["items"][0]["game_pk"]
    ab_n = client.get(
        f"/at-bats?game_pk={gpk}&min_pitches=3"
    ).json()["items"][0]["at_bat_number"]
    r = client.get(f"/ab-context?game_pk={gpk}&at_bat_number={ab_n}")
    assert r.status_code == 200
    body = r.json()
    # ABContextResponse.positions
    assert len(body["positions"]) > 0, "no per-position entries"
    # Arsenal pieces the frontend reads to disable impossible-type buttons
    assert isinstance(body["pitcher_arsenal_pct"], dict)
    assert isinstance(body["pitcher_has_pitch"], dict)
    # Batter heatmap (9 in-zone cells)
    assert len(body["batter_whiff_grid"]) == 9
    assert len(body["batter_swing_grid"]) == 9


# ============================================================
# Counterfactual rollout — the headline endpoint
# ============================================================


def test_query_basic_intervention(client):
    gpk = client.get("/games?limit=1").json()["items"][0]["game_pk"]
    ab_n = client.get(
        f"/at-bats?game_pk={gpk}&min_pitches=3"
    ).json()["items"][0]["at_bat_number"]
    payload = {
        "game_pk": gpk,
        "at_bat_number": ab_n,
        "intervention_position": 1,  # 2nd pitch — first eligible per the API
        "intervention_type": "FF",
        "n_paths": 50,  # small for smoke speed; default is 200
    }
    r = client.post("/query", json=payload)
    assert r.status_code == 200, f"unexpected {r.status_code}: {r.text[:300]}"
    body = r.json()
    # Trust state from positivity gate
    assert body["trust_state"] in ("green", "yellow", "red")
    # Counterfactual block always populated (low-support cases included)
    cf = body["counterfactual"]
    for k in (
        "effect_runs", "ci_lower", "ci_upper",
        "e_value_point", "e_value_ci_limit",
        "support_level", "is_causal_claim",
    ):
        assert k in cf, f"missing field {k!r} in counterfactual block"


def test_query_rejects_unknown_pitch_type(client):
    """Pydantic validation on the request schema (FastAPI returns 422)."""
    gpk = client.get("/games?limit=1").json()["items"][0]["game_pk"]
    ab_n = client.get(
        f"/at-bats?game_pk={gpk}&min_pitches=3"
    ).json()["items"][0]["at_bat_number"]
    payload = {
        "game_pk": gpk,
        "at_bat_number": ab_n,
        "intervention_position": 1,
        "intervention_type": "INVALID",  # not in PITCH_TYPES
        "n_paths": 50,
    }
    r = client.post("/query", json=payload)
    assert r.status_code in (400, 422), (
        f"expected validation error, got {r.status_code}: {r.text[:200]}"
    )
