"""Tests for ``mcsim.state.build_synthetic_ab``.

The synthetic-AB builder must produce a DataFrame that:

  1. Contains every column the model's dataset requires
     (``REQUIRED_AUG_COLS``), so it can pass through
     ``PitchGPTAtBatDataset`` without a schema error.
  2. Honors the ``ReferenceContext`` overrides for count/runners/outs/etc.
  3. End-to-end works as an input to ``g_compute(intervention_position=0,
     intervention_type=None)`` — the actual use case in the matchup-card
     computer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

torch.backends.mps.is_available = lambda: False

from data.preprocess_pitchgpt import HANDEDNESS_MAP, ROOF_CLOSED, ROOF_OPEN
from mcsim.state import ReferenceContext, build_synthetic_ab
from model.pitchgpt_dataset import REQUIRED_AUG_COLS

V7_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")


# ============================================================
# Schema invariants
# ============================================================


def test_synthetic_ab_has_all_required_columns():
    """A synthetic AB must carry every column ``REQUIRED_AUG_COLS`` demands —
    otherwise the model's dataset will reject it before we ever get to g_compute."""
    ab = build_synthetic_ab(
        pitcher_id=543037, batter_id=605141,
        game_date="2026-05-30",
        pitcher_throws="R", batter_stand="L",
    )
    missing = REQUIRED_AUG_COLS - set(ab.columns)
    assert not missing, f"synthetic AB missing required columns: {sorted(missing)}"
    assert len(ab) == 1, "expected exactly one row"


def test_synthetic_ab_default_context_is_neutral():
    """Default ReferenceContext: 0-0 count, empty bases, 0 outs, tied score."""
    ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2,
        game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
    )
    row = ab.iloc[0]
    assert row["count_state"] == 0       # 0-0
    assert row["runners_state"] == 0     # empty bases
    assert row["outs_state"] == 0
    assert row["pos"] == 0
    # Type / zone / velo / spin_rate / result are PAD — will be overwritten by
    # the rollout (Option C: pitch 0 sampled from last-context-token π̂).
    assert row["type_id"] == 0
    assert row["feature_zone"] == 0


def test_handedness_encoded_via_canonical_map():
    """p_throws_id and stand_id must come from ``HANDEDNESS_MAP`` — sharing
    the dataset's encoding avoids the silent-mismatch failure mode."""
    ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2,
        game_date="2026-05-30",
        pitcher_throws="L", batter_stand="R",
    )
    row = ab.iloc[0]
    assert row["p_throws_id"] == HANDEDNESS_MAP["L"]   # 2
    assert row["stand_id"] == HANDEDNESS_MAP["R"]      # 1


def test_unknown_handedness_falls_back_to_pad():
    """Unknown handedness strings → PAD (0) rather than KeyError."""
    ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2,
        game_date="2026-05-30",
        pitcher_throws="?", batter_stand="?",
    )
    row = ab.iloc[0]
    assert row["p_throws_id"] == 0
    assert row["stand_id"] == 0


# ============================================================
# ReferenceContext bucketing
# ============================================================


def test_count_strikes_encoded_in_count_state():
    """count_state = balls * 3 + strikes. 1-2 count must map to 5."""
    ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2,
        game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(count_balls=1, count_strikes=2),
    )
    assert ab.iloc[0]["count_state"] == 1 * 3 + 2


def test_runners_state_bases_loaded():
    """Bases loaded = 4*1 + 2*1 + 1*1 = 7."""
    ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2,
        game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(
            runners_on_1b=True, runners_on_2b=True, runners_on_3b=True,
        ),
    )
    assert ab.iloc[0]["runners_state"] == 7


def test_roof_state_open_vs_closed():
    open_ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2, game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(roof_closed=False),
    )
    closed_ab = build_synthetic_ab(
        pitcher_id=1, batter_id=2, game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(roof_closed=True),
    )
    assert open_ab.iloc[0]["roof_state"] == ROOF_OPEN
    assert closed_ab.iloc[0]["roof_state"] == ROOF_CLOSED


def test_temp_bucketing_cold_vs_hot():
    cold = build_synthetic_ab(
        pitcher_id=1, batter_id=2, game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(temp_f=35.0),
    )
    hot = build_synthetic_ab(
        pitcher_id=1, batter_id=2, game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(temp_f=92.0),
    )
    # 35°F → "<40" bucket (1); 92°F → "80+" bucket (6).
    assert cold.iloc[0]["temp_bucket"] == 1
    assert hot.iloc[0]["temp_bucket"] == 6


def test_inning_clipping():
    """Inning > 12 must clip to 13 (extras bucket)."""
    extras = build_synthetic_ab(
        pitcher_id=1, batter_id=2, game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(inning=15),
    )
    assert extras.iloc[0]["inning_bucket"] == 13


def test_score_diff_shifted_to_zero_based():
    """score_diff -3 → bucket 2 (after shift by SCORE_DIFF_CLIP=5)."""
    behind = build_synthetic_ab(
        pitcher_id=1, batter_id=2, game_date="2026-05-30",
        pitcher_throws="R", batter_stand="R",
        context=ReferenceContext(score_diff=-3),
    )
    assert behind.iloc[0]["score_diff_bucket"] == -3 + 5


# ============================================================
# End-to-end with g_compute (Option C, intervention_position=0)
# ============================================================


@pytest.mark.skipif(
    not V7_CKPT.exists(),
    reason=f"v7 checkpoint not present at {V7_CKPT}",
)
def test_synthetic_ab_runs_through_g_compute_at_position_zero():
    """The synthetic AB must work as an input to
    ``g_compute(intervention_position=0, intervention_type=None)`` — the
    actual use case for the matchup-card cell.

    Picks a real pitcher_id + batter_id from the val set so the profile
    cache lookup succeeds (debutants on the zero-fallback path are OK too,
    but using a known ID exercises the full path).
    """
    from causal.g_computation import g_compute
    from causal.nuisance import NuisanceModels

    nuisance = NuisanceModels(V7_CKPT, device="cpu")
    val_parquets = sorted(Path("data/augmented/2024").glob("2024-*.parquet"))
    assert val_parquets, "no val parquets"
    df = pd.read_parquet(val_parquets[0])
    sample = df.iloc[0]
    pitcher_id = int(sample["pitcher"])
    batter_id = int(sample["batter"])

    ab = build_synthetic_ab(
        pitcher_id=pitcher_id, batter_id=batter_id,
        game_date="2024-04-01",
        pitcher_throws="R", batter_stand="R",
        ballpark_id=int(sample.get("ballpark_id", 0)),
    )

    result = g_compute(
        nuisance, ab,
        intervention_position=0,
        intervention_type=None,
        n_paths=30,
        rng_seed=42,
    )
    assert result.intervention_position == 0
    assert result.intervention_type is None
    assert np.isfinite(result.mean_run_value)
    # AB outcome distribution sums to 1 and isn't degenerate
    np.testing.assert_allclose(result.ab_outcome_distribution.sum(), 1.0, atol=1e-5)
    assert result.ab_outcome_distribution.max() < 0.95, (
        "all paths collapsed to one outcome — synthetic AB likely producing degenerate "
        "first-pitch propensity. Check the context-token columns are populated."
    )
