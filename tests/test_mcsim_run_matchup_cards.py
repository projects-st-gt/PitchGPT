"""Tests for the MCSim App B runner — checkpoint hashing + end-to-end orchestration."""
from __future__ import annotations

from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

import numpy as np
import pandas as pd
import pytest

from causal.g_computation import AB_OUTCOME_NAMES
from causal.nuisance import NuisanceModels
from mcsim.mlb_api import GameInfo
from mcsim.matchup_card import BatterSpec, PitcherSpec
from mcsim.storage import init_db, read_prediction
from scripts.mcsim import run_matchup_cards as runner

V7_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")
VAL_DIR = Path("data/augmented/2024")

requires_v7 = pytest.mark.skipif(
    not V7_CKPT.exists(), reason=f"v7 checkpoint not present at {V7_CKPT}"
)


def test_compute_ckpt_hash_is_stable_and_truncated(tmp_path):
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"hello world")
    h1 = runner.compute_ckpt_hash(f)
    h2 = runner.compute_ckpt_hash(f)
    assert h1 == h2                      # deterministic
    assert len(h1) == 16                 # truncated
    # sha256("hello world") starts with b94d27b9934d3e08...
    assert h1 == "b94d27b9934d3e08"


@requires_v7
def test_run_matchup_cards_end_to_end(tmp_path, monkeypatch, capsys):
    """Monkeypatch the API client so no network is touched; feed 2 real pitcher
    ids + 2 real batter ids (so profile lookups succeed) through the real
    nuisance + compute_matchup_card, and assert a row lands in SQLite.

    Prints a NAMED numerical output (a real cell's median RV + modal-type pi-hat)
    per CLAUDE.md bug-prevention discipline.
    """
    nuisance = NuisanceModels(V7_CKPT, device="cpu")
    df = pd.read_parquet(sorted(VAL_DIR.glob("2024-*.parquet"))[0])
    pids = [int(x) for x in df["pitcher"].drop_duplicates().head(2)]
    bids = [int(x) for x in df["batter"].drop_duplicates().head(2)]

    game = GameInfo(game_pk=776543, home_team_id=139, away_team_id=116,
                    home_team="HOME", away_team="AWAY",
                    home_probable_pitcher_id=pids[0],
                    away_probable_pitcher_id=pids[1])
    rosters = {
        139: ([PitcherSpec(pids[0], "HSP", "R", True), PitcherSpec(pids[1], "HRP", "L")],
              [BatterSpec(bids[0], "HB1", "R"), BatterSpec(bids[1], "HB2", "L")]),
        116: ([PitcherSpec(pids[1], "ASP", "L", True), PitcherSpec(pids[0], "ARP", "R")],
              [BatterSpec(bids[1], "AB1", "L"), BatterSpec(bids[0], "AB2", "R")]),
    }
    monkeypatch.setattr(runner, "get_schedule", lambda date: [game])
    monkeypatch.setattr(runner, "get_active_roster",
                        lambda team_id, date, probable_pitcher_id=None: rosters[team_id])

    conn = init_db(tmp_path / "t.sqlite")
    cards = runner.run_matchup_cards(
        nuisance, conn, date="2026-06-02", game_pks=None,
        n_paths=50, rng_seed=7, model_ckpt_hash="testhash",
    )

    assert len(cards) == 1
    got = read_prediction(conn, game_pk=776543, prediction_date="2026-06-02",
                          app="matchup_card")
    assert got is not None
    assert got["model_ckpt_hash"] == "testhash"   # provenance column round-trips
    card = got["payload"]
    # 2 halves x (2 pitchers x 2 hitters) = 8 cells
    assert card["n_cells"] == 8
    first_cell = card["rows"][0]["cells"][0]
    assert first_cell["predicted_top1_outcome"] in AB_OUTCOME_NAMES
    assert np.isfinite(first_cell["predicted_rv_median"])
    print(f"\n[runner E2E] {card['rows'][0]['name']} vs {first_cell['batter_name']}: "
          f"median RV={first_cell['predicted_rv_median']:+.4f}  "
          f"modal {first_cell['modal_type']} pi-hat={first_cell['p_hat_top_type']:.3f}  "
          f"trust={first_cell['trust_state']}")
    captured = capsys.readouterr()
    assert "median RV" in captured.out


def test_run_matchup_cards_skips_failing_game(monkeypatch, capsys):
    """A game that errors mid-fetch is logged and skipped — it must not abort
    the batch. Fast: roster fetch raises before the model is ever touched, so
    no checkpoint/nuisance is needed (nuisance=None never gets used)."""
    games = [
        GameInfo(1, 10, 20, "H1", "A1", None, None),
        GameInfo(2, 30, 40, "H2", "A2", None, None),
    ]
    monkeypatch.setattr(runner, "get_schedule", lambda date: games)

    def _boom(team_id, date, probable_pitcher_id=None):
        raise RuntimeError("roster fetch failed")

    monkeypatch.setattr(runner, "get_active_roster", _boom)

    cards = runner.run_matchup_cards(
        None, None, date="2026-06-02", game_pks=None,
        n_paths=1, rng_seed=0, model_ckpt_hash="h",
    )
    assert cards == []                       # both games skipped, no crash
    out, err = capsys.readouterr()
    assert "SKIP game_pk=1" in out and "SKIP game_pk=2" in out
    assert "RuntimeError: roster fetch failed" in err  # traceback surfaced to stderr


def test_run_matchup_cards_parallel_requires_ckpt(monkeypatch):
    """n_workers>1 needs a ckpt_path (workers load their own model)."""
    monkeypatch.setattr(runner, "get_schedule", lambda date: [])
    with pytest.raises(ValueError, match="ckpt_path"):
        runner.run_matchup_cards(None, None, date="2026-06-04", game_pks=None,
                                 n_paths=10, rng_seed=0, model_ckpt_hash="h",
                                 n_workers=2, ckpt_path=None)


def test_compute_one_game_fetches_both_rosters(monkeypatch):
    """_compute_one_game fetches home then away roster and forwards them to
    compute_matchup_card with the game's metadata (no model/network)."""
    from mcsim.mlb_api import GameInfo

    fetched = []
    monkeypatch.setattr(
        runner, "get_active_roster",
        lambda team_id, date, probable_pitcher_id=None: (fetched.append(team_id) or ([], [])),
    )
    captured = {}
    monkeypatch.setattr(
        runner, "compute_matchup_card",
        lambda nuisance, **kw: (captured.update(kw) or {"n_cells": 0, "rows": []}),
    )
    g = GameInfo(1, 10, 20, "Home", "Away", 111, 222)
    card = runner._compute_one_game("FAKE_NZ", g, "2026-06-04", n_paths=5, rng_seed=3)
    assert fetched == [10, 20]  # home then away
    assert captured["game_pk"] == 1 and captured["home_team"] == "Home"
    assert captured["away_team"] == "Away"
    assert captured["n_paths"] == 5 and captured["rng_seed"] == 3
    assert card["n_cells"] == 0
