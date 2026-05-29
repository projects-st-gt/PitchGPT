"""API smoke tests for ``POST /recommend``.

Three tests:
  - happy path (small candidate set + small n_paths for speed)
  - input validation (unknown pitch types in candidates list)
  - input validation (invalid intervention_position)

Integration-style: `TestClient` exercises the same `FastAPI` app object that
`make demo-api` runs under uvicorn. First request triggers lazy-load of
`NuisanceModels` + val parquets (~5–10 s); subsequent requests reuse the
module-scoped fixture.
"""
from __future__ import annotations

import pytest
import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from inference.api import app
    with TestClient(app) as c:
        yield c


def _pick_real_ab(client: TestClient) -> tuple[int, int, int]:
    """Find a real (game_pk, at_bat_number, n_pitches) with ≥ 4 pitches —
    enough to use intervention_position=2 safely."""
    games = client.get("/games?limit=20").json()["items"]
    for g in games:
        abs_ = client.get(f"/at-bats?game_pk={g['game_pk']}&min_pitches=4").json()["items"]
        if abs_:
            ab = abs_[0]
            return g["game_pk"], ab["at_bat_number"], ab["n_pitches"]
    pytest.skip("no AB with ≥ 4 pitches in first 20 games")


def test_recommend_happy_path(client):
    """Two-candidate request returns ranked + refused lists with the
    expected partition and shape."""
    gpk, ab_n, _ = _pick_real_ab(client)
    payload = {
        "game_pk": gpk,
        "at_bat_number": ab_n,
        "intervention_position": 1,
        "n_paths": 20,           # small for smoke speed
        "candidates": ["FF", "SL"],
    }
    r = client.post("/recommend", json=payload)
    assert r.status_code == 200, f"unexpected {r.status_code}: {r.text[:300]}"
    body = r.json()
    # Partition exhaustive over the requested candidates
    seen = {row["pitch_type"] for row in body["ranked"]} | {
        row["pitch_type"] for row in body["refused"]
    }
    assert seen == {"FF", "SL"}, f"got {sorted(seen)}, expected {{FF, SL}}"
    # Required envelope fields
    assert body["intervention_position"] == 1
    assert body["n_paths"] == 20
    assert "timing_seconds" in body
    # Trust states valid
    for row in body["ranked"]:
        assert row["trust_state"] in ("green", "yellow")
        # In-support rows have rollout numbers, not nulls
        assert row["mean_run_value"] is not None
        assert row["ci_lower"] is not None
        assert row["ci_upper"] is not None
    for row in body["refused"]:
        assert row["trust_state"] == "red"
        # Refused rows have None for rollout fields (NaN → null per
        # _ranking_to_schema)
        assert row["mean_run_value"] is None
        assert row["rank"] is None


def test_recommend_rejects_unknown_candidate(client):
    gpk, ab_n, _ = _pick_real_ab(client)
    payload = {
        "game_pk": gpk,
        "at_bat_number": ab_n,
        "intervention_position": 1,
        "n_paths": 20,
        "candidates": ["FF", "BOGUS"],
    }
    r = client.post("/recommend", json=payload)
    # 400 from the app-level check (it produces a clearer error message than
    # Pydantic could without a custom validator)
    assert r.status_code == 400
    assert "BOGUS" in r.text


def test_recommend_rejects_bad_intervention_position(client):
    """Position past the end of the AB → 400."""
    gpk, ab_n, n_pitches = _pick_real_ab(client)
    payload = {
        "game_pk": gpk,
        "at_bat_number": ab_n,
        "intervention_position": n_pitches,  # past the end
        "n_paths": 20,
        "candidates": ["FF"],
    }
    r = client.post("/recommend", json=payload)
    assert r.status_code == 400
    assert "AB length" in r.text or "≥" in r.text
