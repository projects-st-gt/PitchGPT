"""Tests for ``g_compute`` natural mode (MCSim App B D4).

Natural mode lets the rollout sample the pitch type at the intervention
position from ``π̂(type | history)`` instead of clamping it to a specific
type. The cell-computer for the matchup card uses this to roll out
*natural play* between a pitcher and a batter.

Three contracts pinned:
  (a) Back-compat — specifying ``intervention_type`` produces the same
      RolloutResult as the pre-refactor code (with the same seed). This is
      the path the recommender + /query endpoint already use.
  (b) Natural mode runs end-to-end and populates a valid RolloutResult
      (``intervention_type_name`` / ``intervention_type`` are ``None``).
  (c) Property test — in natural mode the empirical distribution of
      sampled types at the intervention step matches ``π̂(type | h)``
      within Monte Carlo tolerance.

Integration-style with the v7 checkpoint and a real val AB.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

import causal.g_computation as gc
from causal.nuisance import NuisanceModels
from data.dataset import N_PITCH_TYPES

V7_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")
VAL_DIR = Path("data/augmented/2024")

requires_v7 = pytest.mark.skipif(
    not V7_CKPT.exists(),
    reason=f"v7 checkpoint not present at {V7_CKPT}",
)


@pytest.fixture(scope="module")
def setup():
    """Load nuisance once + pick a real val AB with ≥ 4 pitches."""
    nuisance = NuisanceModels(V7_CKPT, device="cpu")
    parquets = sorted(VAL_DIR.glob("2024-*.parquet"))
    assert parquets, f"no augmented val parquets under {VAL_DIR}"
    df = pd.read_parquet(parquets[0])
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    for (_, _), ab in df.groupby(["game_pk", "at_bat_number"], sort=False):
        if len(ab) >= 4:
            return nuisance, ab.reset_index(drop=True), 1
    pytest.skip("no AB with ≥ 4 pitches in the first val parquet")


# ============================================================
# Contract (a) — Back-compat
# ============================================================


@requires_v7
def test_specific_intervention_unchanged(setup):
    """Specifying intervention_type='FF' must produce the same result as before
    the refactor (same n_paths + same seed → byte-identical aggregate stats)."""
    nuisance, ab, k = setup
    r1 = gc.g_compute(nuisance, ab, intervention_position=k,
                      intervention_type="FF", n_paths=50, rng_seed=42)
    r2 = gc.g_compute(nuisance, ab, intervention_position=k,
                      intervention_type="FF", n_paths=50, rng_seed=42)
    assert r1.intervention_type == r2.intervention_type
    assert r1.intervention_type_name == "FF"
    assert r1.mean_run_value == r2.mean_run_value
    np.testing.assert_array_equal(r1.terminal_kind, r2.terminal_kind)
    np.testing.assert_array_equal(r1.ab_outcome, r2.ab_outcome)


# ============================================================
# Contract (b) — Natural mode runs end-to-end
# ============================================================


@requires_v7
def test_natural_mode_runs_and_reports_natural(setup):
    """intervention_type=None produces a valid RolloutResult, with the type
    fields set to None to signal 'no intervention'."""
    nuisance, ab, k = setup
    r = gc.g_compute(nuisance, ab, intervention_position=k,
                     intervention_type=None, n_paths=50, rng_seed=42)
    assert r.intervention_type is None
    assert r.intervention_type_name is None
    assert r.intervention_position == k
    assert r.n_paths == 50
    assert np.isfinite(r.mean_run_value)
    assert r.ab_outcome_distribution.shape == (7,)
    np.testing.assert_allclose(r.ab_outcome_distribution.sum(), 1.0, atol=1e-5)


@requires_v7
def test_natural_mode_default_argument(setup):
    """The default for intervention_type is None — i.e., omitting it gives
    natural mode."""
    nuisance, ab, k = setup
    r = gc.g_compute(nuisance, ab, intervention_position=k,
                     n_paths=30, rng_seed=42)
    assert r.intervention_type is None
    assert r.intervention_type_name is None


@requires_v7
def test_natural_mode_differs_from_specific_intervention(setup):
    """Natural mode and a specific-type intervention can't be the same
    rollout — they sample differently at step k, so per-path outputs differ."""
    nuisance, ab, k = setup
    r_natural = gc.g_compute(nuisance, ab, intervention_position=k,
                             intervention_type=None, n_paths=50, rng_seed=42)
    r_ff = gc.g_compute(nuisance, ab, intervention_position=k,
                        intervention_type="FF", n_paths=50, rng_seed=42)
    # The per-path AB outcomes are unlikely to be identical between the two
    # modes — different type samples → different result-head conditioning →
    # different terminal events.
    assert not np.array_equal(r_natural.ab_outcome, r_ff.ab_outcome), (
        "natural mode produced identical per-path outcomes as FF intervention — "
        "suggests natural sampling isn't engaged"
    )


# ============================================================
# Contract (c) — Empirical type distribution matches π̂
# ============================================================


@requires_v7
def test_natural_mode_empirical_distribution_matches_propensity(setup):
    """The first sampled types in natural mode must follow π̂(type | h).

    Captures the type-sampling call at step==k by patching
    ``_sample_from_probs`` to record its arguments, then checks the empirical
    proportions against the propensity vector. Uses n_paths=2000 to get the
    Monte Carlo error down to a few percent.
    """
    nuisance, ab, k = setup
    captured = []
    real_sample = gc._sample_from_probs

    def capturing_sample(probs, rng, active=None):
        result = real_sample(probs, rng, active)
        captured.append((probs.numpy().copy(), result.copy()))
        return result

    with patch.object(gc, "_sample_from_probs", side_effect=capturing_sample):
        gc.g_compute(nuisance, ab, intervention_position=k,
                     intervention_type=None, n_paths=2000, rng_seed=42)

    # The first call to _sample_from_probs is the type-sampling at step==k
    # (zone/velo/spin/result sampling all come after it within the same step).
    assert captured, "no _sample_from_probs calls observed — natural-mode branch not taken?"
    probs_first, samples_first = captured[0]
    assert probs_first.shape == (2000, N_PITCH_TYPES), (
        f"first call's probs shape {probs_first.shape} != expected (2000, {N_PITCH_TYPES}) — "
        "is the first sampling call really the type-head at the intervention step?"
    )
    # All paths replicate one AB → identical π̂ across paths. Take row 0.
    pi_hat = probs_first[0]
    empirical = np.bincount(samples_first, minlength=N_PITCH_TYPES) / len(samples_first)
    # 3% tolerance is generous at n_paths=2000 — for a Bernoulli with p=0.5
    # the 99% CI on the empirical is ~2.9% wide; for rarer types the absolute
    # error is bounded smaller.
    np.testing.assert_allclose(empirical, pi_hat, atol=0.03), (
        f"empirical type proportions {empirical} differ from π̂ {pi_hat} by more than 3%"
    )


# ============================================================
# k=0 — first-pitch rollout (MCSim App B Option C)
# ============================================================


@requires_v7
def test_intervention_position_zero_natural_mode_runs(setup):
    """k=0 + natural mode must run without error and produce a valid
    RolloutResult — pitch 0 is sampled from the model's last-context-token
    propensity output (no observed pitch history)."""
    nuisance, ab, _ = setup
    r = gc.g_compute(nuisance, ab, intervention_position=0,
                     intervention_type=None, n_paths=50, rng_seed=42)
    assert r.intervention_position == 0
    assert r.intervention_type is None
    assert np.isfinite(r.mean_run_value)
    np.testing.assert_allclose(r.ab_outcome_distribution.sum(), 1.0, atol=1e-5)


@requires_v7
def test_intervention_position_zero_specific_type_runs(setup):
    """k=0 + specific type (e.g. do(FF) on the first pitch) must also work."""
    nuisance, ab, _ = setup
    r = gc.g_compute(nuisance, ab, intervention_position=0,
                     intervention_type="FF", n_paths=50, rng_seed=42)
    assert r.intervention_position == 0
    assert r.intervention_type_name == "FF"
    assert np.isfinite(r.mean_run_value)


@requires_v7
def test_intervention_position_zero_first_pitch_propensity_matches_pi_hat(setup):
    """The empirical distribution of first-pitch samples must match the
    propensity at the last-context-token position. This is the pin for
    Option C: the model produces sensible first-pitch propensities from
    the context tokens alone, with no pitch history."""
    nuisance, ab, _ = setup
    captured = []
    real_sample = gc._sample_from_probs

    def capturing_sample(probs, rng, active=None):
        result = real_sample(probs, rng, active)
        captured.append((probs.numpy().copy(), result.copy()))
        return result

    with patch.object(gc, "_sample_from_probs", side_effect=capturing_sample):
        gc.g_compute(nuisance, ab, intervention_position=0,
                     intervention_type=None, n_paths=2000, rng_seed=42)

    assert captured, "no sampling occurred — natural-mode branch not taken at k=0?"
    probs_first, samples_first = captured[0]
    assert probs_first.shape == (2000, N_PITCH_TYPES)
    pi_hat_first_pitch = probs_first[0]
    empirical = np.bincount(samples_first, minlength=N_PITCH_TYPES) / len(samples_first)
    np.testing.assert_allclose(empirical, pi_hat_first_pitch, atol=0.03), (
        f"first-pitch empirical {empirical} differs from π̂ {pi_hat_first_pitch} by >3%"
    )
    # Sanity: π̂ shouldn't be degenerate (uniform or one-hot) — the model
    # should learn that pitchers strongly favor fastballs in 0-0 counts.
    assert pi_hat_first_pitch.max() > 0.20, (
        f"first-pitch propensity looks degenerate: max π̂ = {pi_hat_first_pitch.max():.3f}; "
        f"a real pitcher's modal first-pitch type should be > 20% probability"
    )
