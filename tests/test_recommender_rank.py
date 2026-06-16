"""Tests for ``recommender.rank.rank_pitch_types``.

Integration-style: load the v7 checkpoint, pick a real val AB, run the
recommender end-to-end. The ``v7_recommender_setup`` fixture is
module-scoped so model + data load once across the suite.

Tests pin the four contracts from ``docs/recommender_brainstorm.md``:

  (a) every candidate ends up in exactly one of {ranked, refused}
  (b) refused candidates have ``p_hat < τ_refuse``; in-support candidates
      have ``trust_state ∈ {"green", "yellow"}``
  (c) the ranked list is sorted by ``mean_run_value`` ascending (lowest =
      best outcome for pitcher)
  (d) tossup flag fires when CIs overlap

Plus a few argument-validation cases.

Speed: tests use small ``n_paths`` (20–40) and a 2-element candidate set
where the ranking can be inspected easily. A typical run is ~30 s
including the one-time model load.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from causal.nuisance import NuisanceModels
from data.dataset import PITCH_TYPES
from recommender.rank import (
    CandidateRanking,
    RankedRecommendations,
    rank_pitch_types,
)

V7_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")
VAL_DIR = Path("data/augmented/2024")

requires_v7 = pytest.mark.skipif(
    not V7_CKPT.exists(),
    reason=f"v7 checkpoint not present at {V7_CKPT}",
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def v7_recommender_setup():
    """Load nuisance + pick a real AB with ≥ 4 pitches.

    Returns ``(nuisance, ab_pitches, intervention_position)``. Intervention
    position is 1 (the second pitch) — that matches the rollout-coherence
    test's choice and is the first eligible value (g_compute requires ≥ 1).
    """
    nuisance = NuisanceModels(V7_CKPT, device="cpu")
    # Pick the first val 2024 parquet and walk for an AB with enough pitches.
    parquets = sorted(VAL_DIR.glob("2024-*.parquet"))
    assert parquets, f"no augmented val parquets under {VAL_DIR}"
    df = pd.read_parquet(parquets[0])
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    for (_, _), ab in df.groupby(["game_pk", "at_bat_number"], sort=False):
        if len(ab) >= 4:
            ab = ab.reset_index(drop=True)
            return nuisance, ab, 1
    pytest.skip("no AB with ≥ 4 pitches in the first val parquet")


# ============================================================
# Core contracts
# ============================================================


@requires_v7
def test_every_candidate_lands_in_exactly_one_bucket(v7_recommender_setup):
    """Contract (a) — partition invariant."""
    nuisance, ab, k = v7_recommender_setup
    res = rank_pitch_types(
        nuisance, ab,
        intervention_position=k,
        candidates=list(PITCH_TYPES),
        n_paths=20,
        rng_seed=42,
    )
    assert isinstance(res, RankedRecommendations)
    total = {r.pitch_type for r in res.ranked} | {r.pitch_type for r in res.refused}
    assert total == set(PITCH_TYPES), (
        f"missing or duplicate candidates: got {sorted(total)}, "
        f"expected {sorted(PITCH_TYPES)}"
    )
    assert len(res.ranked) + len(res.refused) == len(PITCH_TYPES)


@requires_v7
def test_refused_below_tau_in_support_above(v7_recommender_setup):
    """Contract (b) — gate matches p_hat partition. Uses a high τ_refuse so a
    non-trivial refused list is guaranteed even on a peaked propensity."""
    nuisance, ab, k = v7_recommender_setup
    # τ_refuse=0.10 ensures most types fall below; the model's argmax typically
    # has π̂ in the 0.3–0.6 range so 1–2 will still clear gate.
    res = rank_pitch_types(
        nuisance, ab,
        intervention_position=k,
        n_paths=20, rng_seed=42,
        tau_refuse=0.10, tau_green=0.20,
    )
    for r in res.refused:
        assert r.trust_state == "red", f"{r.pitch_type} in refused but state={r.trust_state}"
        assert r.p_hat < 0.10, f"refused {r.pitch_type} has p_hat={r.p_hat} ≥ τ_refuse"
        # Refused candidates were NOT rolled out
        assert r.mean_run_value != r.mean_run_value, f"refused {r.pitch_type} has NaN-violating mean"  # NaN check
    for r in res.ranked:
        assert r.trust_state in ("green", "yellow"), (
            f"in-support {r.pitch_type} has state={r.trust_state}"
        )
        assert r.p_hat >= 0.10, f"in-support {r.pitch_type} has p_hat={r.p_hat} < τ_refuse"


@requires_v7
def test_ranked_sorted_ascending_by_mean_run_value(v7_recommender_setup):
    """Contract (c) — sort order is ascending (lower run value = better for pitcher)."""
    nuisance, ab, k = v7_recommender_setup
    res = rank_pitch_types(nuisance, ab, intervention_position=k, n_paths=20, rng_seed=42)
    if len(res.ranked) >= 2:
        means = [r.mean_run_value for r in res.ranked]
        assert means == sorted(means), f"ranked not sorted ascending: {means}"
    for i, r in enumerate(res.ranked):
        assert r.rank == i, f"rank field mis-set: position {i} has rank={r.rank}"


@requires_v7
def test_tossup_flag_fires_on_overlapping_cis(v7_recommender_setup):
    """Contract (d) — when n_paths is tiny, SE balloons and the top two
    candidates' 95% CIs reliably overlap → tossup. n_paths=20 is enough
    to almost guarantee overlap for any pair of in-support candidates."""
    nuisance, ab, k = v7_recommender_setup
    res = rank_pitch_types(nuisance, ab, intervention_position=k, n_paths=20, rng_seed=42)
    if len(res.ranked) < 2:
        pytest.skip("need ≥ 2 in-support candidates to test tossup logic")
    top = res.ranked[0]
    second = res.ranked[1]
    overlap = second.ci_lower <= top.ci_upper
    assert top.is_tossup == overlap, (
        f"#1 is_tossup={top.is_tossup} but ci overlap={overlap} "
        f"(#1 CI [{top.ci_lower:.3f}, {top.ci_upper:.3f}], "
        f"#2 CI [{second.ci_lower:.3f}, {second.ci_upper:.3f}])"
    )
    assert second.is_tossup == overlap


# ============================================================
# Argument validation
# ============================================================


@requires_v7
def test_invalid_intervention_position_rejected(v7_recommender_setup):
    nuisance, ab, _ = v7_recommender_setup
    with pytest.raises(ValueError, match="intervention_position must be"):
        rank_pitch_types(nuisance, ab, intervention_position=0, n_paths=10)
    # The end-of-AB guard uses the Unicode "≥" in its message (not >=).
    with pytest.raises(ValueError, match="AB length"):
        rank_pitch_types(nuisance, ab, intervention_position=len(ab), n_paths=10)


@requires_v7
def test_unknown_candidate_rejected(v7_recommender_setup):
    nuisance, ab, k = v7_recommender_setup
    with pytest.raises(ValueError, match="unknown pitch types"):
        rank_pitch_types(
            nuisance, ab, intervention_position=k,
            candidates=["FF", "INVALID"], n_paths=10,
        )


@requires_v7
def test_effect_vs_observed_annotation(v7_recommender_setup):
    """When the observed pitch at the position is in-support, every in-support
    candidate (including observed itself) gets an effect/E-value annotation.
    """
    nuisance, ab, k = v7_recommender_setup
    res = rank_pitch_types(nuisance, ab, intervention_position=k, n_paths=20, rng_seed=42)
    if res.observed_type_at_position is None:
        pytest.skip("observed pitch type not extractable from this AB")
    obs_in_ranked = [r for r in res.ranked if r.pitch_type == res.observed_type_at_position]
    if not obs_in_ranked:
        pytest.skip("observed type was refused — no contrast available")
    # All in-support rows should have the annotation populated
    for r in res.ranked:
        assert r.effect_vs_observed is not None, f"{r.pitch_type} missing effect_vs_observed"
        assert r.e_value_point is not None, f"{r.pitch_type} missing e_value_point"
    # Observed candidate's effect_vs_observed should be ~0 by construction
    assert abs(obs_in_ranked[0].effect_vs_observed) < 1e-9
