"""Tests for ``mcsim.matchup_card.compute_matchup_card`` (MCSim App B Step 4).

The matchup-card computer loops every (pitcher, batter) cell in a game, rolls
each out in natural mode via ``g_compute(intervention_position=0,
intervention_type=None)``, and packs the App B payload dict.

Integration-style: uses the real v7 checkpoint and real player IDs from the
val set so the profile-cache lookup exercises the full path. Kept small
(2 pitchers × 2 batters per half-grid, n_paths=50) so it stays CI-fast.

Per CLAUDE.md bug-prevention discipline, the headline test prints a NAMED
numerical output — a real cell's predicted median RV and π̂(modal type) — so
a convention slip surfaces immediately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from causal.g_computation import AB_OUTCOME_NAMES
from causal.nuisance import NuisanceModels
from data.dataset import PITCH_TYPES
from mcsim.matchup_card import BatterSpec, PitcherSpec, compute_matchup_card

V7_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")
VAL_DIR = Path("data/augmented/2024")

requires_v7 = pytest.mark.skipif(
    not V7_CKPT.exists(),
    reason=f"v7 checkpoint not present at {V7_CKPT}",
)


@pytest.fixture(scope="module")
def setup():
    """Load nuisance once + pull two real pitcher IDs and two real batter IDs
    from the val set so profile-cache lookups succeed."""
    nuisance = NuisanceModels(V7_CKPT, device="cpu")
    parquets = sorted(VAL_DIR.glob("2024-*.parquet"))
    assert parquets, f"no augmented val parquets under {VAL_DIR}"
    df = pd.read_parquet(parquets[0])
    pitcher_ids = [int(x) for x in df["pitcher"].drop_duplicates().head(2)]
    batter_ids = [int(x) for x in df["batter"].drop_duplicates().head(2)]
    assert len(pitcher_ids) == 2 and len(batter_ids) == 2

    home_pitchers = [
        PitcherSpec(id=pitcher_ids[0], name="HomeSP", throws="R", is_starter=True),
        PitcherSpec(id=pitcher_ids[1], name="HomeRP", throws="L"),
    ]
    away_pitchers = [
        PitcherSpec(id=pitcher_ids[1], name="AwaySP", throws="L", is_starter=True),
        PitcherSpec(id=pitcher_ids[0], name="AwayRP", throws="R"),
    ]
    home_lineup = [
        BatterSpec(id=batter_ids[0], name="HomeBat1", stand="R"),
        BatterSpec(id=batter_ids[1], name="HomeBat2", stand="L"),
    ]
    away_lineup = [
        BatterSpec(id=batter_ids[1], name="AwayBat1", stand="L"),
        BatterSpec(id=batter_ids[0], name="AwayBat2", stand="R"),
    ]
    return nuisance, home_pitchers, away_pitchers, home_lineup, away_lineup


def _card(setup, **kw):
    nuisance, hp, ap, hl, al = setup
    return compute_matchup_card(
        nuisance,
        game_pk=776543,
        game_date="2024-04-01",
        home_team="NYY",
        away_team="BOS",
        home_pitchers=hp,
        away_pitchers=ap,
        home_lineup=hl,
        away_lineup=al,
        n_paths=50,
        rng_seed=42,
        **kw,
    )


# ============================================================
# Structure
# ============================================================


@requires_v7
def test_card_structure_and_cell_count(setup):
    """Card has one row per pitcher across both staffs; each row's cells cover
    the opposing lineup. 2 home + 2 away pitchers × 2 opposing batters = 8."""
    card = _card(setup)
    assert card["game_pk"] == 776543
    assert card["home_team"] == "NYY" and card["away_team"] == "BOS"
    assert len(card["rows"]) == 4, "expected 4 pitcher rows (2 home + 2 away)"
    assert card["n_cells"] == 8, f"expected 8 cells, got {card['n_cells']}"
    for row in card["rows"]:
        assert len(row["cells"]) == 2, "each pitcher faces the 2-batter opposing lineup"
        assert row["team"] in ("NYY", "BOS")
    # Starters surfaced.
    assert card["starter_home"]["name"] == "HomeSP"
    assert card["starter_away"]["name"] == "AwaySP"


@requires_v7
def test_home_pitchers_face_away_lineup(setup):
    """A home-team pitcher row's cells must reference AWAY batters, and vice
    versa — a swap here would be a silent grid-orientation bug."""
    card = _card(setup)
    away_batter_names = {"AwayBat1", "AwayBat2"}
    home_batter_names = {"HomeBat1", "HomeBat2"}
    for row in card["rows"]:
        cell_batters = {c["batter_name"] for c in row["cells"]}
        if row["team"] == "NYY":  # home staff
            assert cell_batters == away_batter_names, (
                f"home pitcher {row['name']} should face away lineup, got {cell_batters}"
            )
        else:
            assert cell_batters == home_batter_names


# ============================================================
# Cell payload validity
# ============================================================


@requires_v7
def test_cell_fields_valid(setup):
    """Every cell has a well-formed RV band, a valid top-1 outcome, a
    normalized outcome dist, a propensity in [0,1], and a legal trust state."""
    card = _card(setup)
    for row in card["rows"]:
        for cell in row["cells"]:
            assert cell["predicted_rv_p05"] <= cell["predicted_rv_median"] <= cell["predicted_rv_p95"]
            assert cell["predicted_top1_outcome"] in AB_OUTCOME_NAMES
            assert cell["modal_type"] in PITCH_TYPES
            assert 0.0 <= cell["p_hat_top_type"] <= 1.0
            assert cell["trust_state"] in ("green", "yellow", "red")
            dist = cell["predicted_outcome_dist"]
            assert set(dist.keys()) == set(AB_OUTCOME_NAMES)
            np.testing.assert_allclose(sum(dist.values()), 1.0, atol=1e-5)
            # top-1 outcome must be the argmax of the distribution
            assert cell["predicted_top1_outcome"] == max(dist, key=dist.get)
            # projected slash line — derived from the same outcome dist
            assert 0.0 <= cell["predicted_obp"] <= 1.0
            assert cell["predicted_slg"] >= 0.0
            np.testing.assert_allclose(
                cell["predicted_ops"],
                cell["predicted_obp"] + cell["predicted_slg"],
                atol=1e-9,
            )


@requires_v7
def test_reproducible_with_seed(setup):
    """Same rng_seed → identical cells. Pins the per-cell seed derivation."""
    c1 = _card(setup)
    c2 = _card(setup)
    m1 = c1["rows"][0]["cells"][0]["predicted_rv_median"]
    m2 = c2["rows"][0]["cells"][0]["predicted_rv_median"]
    assert m1 == m2, f"seeded card not reproducible: {m1} != {m2}"


# ============================================================
# Named numerical output (CLAUDE.md discipline)
# ============================================================


@requires_v7
def test_named_numerical_output(setup):
    """Print a real cell's predicted median RV + π̂(modal type) so a convention
    slip is visible, not hidden behind 'tests pass'."""
    card = _card(setup)
    cell = card["rows"][0]["cells"][0]
    pitcher = card["rows"][0]["name"]
    print(
        f"\nCell [{pitcher} vs {cell['batter_name']}]: "
        f"predicted median RV = {cell['predicted_rv_median']:+.4f} runs "
        f"(p05 {cell['predicted_rv_p05']:+.4f}, p95 {cell['predicted_rv_p95']:+.4f}); "
        f"top-1 outcome = {cell['predicted_top1_outcome']}; "
        f"modal type = {cell['modal_type']} at π̂ = {cell['p_hat_top_type']:.3f}; "
        f"projected slash = {cell['predicted_obp']:.3f}/{cell['predicted_slg']:.3f}/"
        f"{cell['predicted_ops']:.3f} (OBP/SLG/OPS); "
        f"trust = {cell['trust_state']}; n_truncated = {cell['n_truncated']}/{cell['n_paths']}"
    )
    # The π̂ should be non-degenerate — a real first-pitch propensity, not 1/7.
    assert cell["p_hat_top_type"] > 0.15, (
        f"modal π̂ {cell['p_hat_top_type']:.3f} looks degenerate — "
        "is intervention_type_propensity wired correctly?"
    )


# ============================================================
# Slash-line formula (pure, no model needed)
# ============================================================


def test_slash_line_known_distribution():
    """_slash_line computes projected OBP/SLG/OPS from the AB-outcome dist.
    Pinned against a hand-computed distribution so a formula slip is visible."""
    from mcsim.matchup_card import _slash_line

    # 10% single, 5% double, 5% HR, 10% walk, rest K/out (sums to 1.0).
    d = {"1B": 0.10, "2B": 0.05, "3B": 0.0, "HR": 0.05,
         "BB": 0.10, "K": 0.30, "out": 0.40}
    obp, slg, ops = _slash_line(d)

    # OBP = hits + BB = (0.10+0.05+0.05) + 0.10 = 0.30
    assert abs(obp - 0.30) < 1e-9
    # TB = 1*0.10 + 2*0.05 + 4*0.05 = 0.40; AB = 1 - 0.10 = 0.90
    assert abs(slg - (0.40 / 0.90)) < 1e-9
    assert abs(ops - (0.30 + 0.40 / 0.90)) < 1e-9


def test_slash_line_all_walks_guards_zero_ab():
    """All-walk degenerate case: AB=0, SLG must be 0 (no division by zero)."""
    from mcsim.matchup_card import _slash_line

    d = {"1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 1.0, "K": 0.0, "out": 0.0}
    obp, slg, ops = _slash_line(d)
    assert obp == 1.0 and slg == 0.0 and ops == 1.0
