"""Tests for ADR-013 Decision 2: cross-AB context wiring.

Pins three contracts:
  1. ``cross_ab_context=False`` (default) — forward works without
     ``matchup_profile`` and without ``tto_matchup`` in categorical_context.
     This is the back-compat path; existing tests already cover it broadly.
  2. ``cross_ab_context=True`` — forward requires ``matchup_profile``
     (shape ``(B, matchup_profile_dim)``) AND ``tto_matchup`` in
     ``categorical_context``. Raises a clear error if either is missing.
  3. The matchup MLP + ``tto_matchup`` embedding actually influence the
     output — changing the matchup vector at the same (h, type) changes
     the per-position logits.
"""
from __future__ import annotations

import pytest
import torch

from model.config import PitchGPTConfig, sanity_config
from model.pitchgpt import PitchGPT
from tests.test_pitchgpt_model import _fake_batch, _fake_categorical


def _build_cross_ab_batch(B: int = 2, T: int = 4, cfg: PitchGPTConfig | None = None) -> dict:
    """Like ``_fake_batch`` but with matchup_profile and tto_matchup attached."""
    if cfg is None:
        cfg = sanity_config()
        cfg.cross_ab_context = True
    batch = _fake_batch(B, T, cfg)
    # Inject tto_matchup into the categorical_context dict (vocab = n_tto_matchup_buckets)
    batch["categorical_context"]["tto_matchup"] = torch.randint(
        0, cfg.n_tto_matchup_buckets, (B,), dtype=torch.long
    )
    batch["matchup_profile"] = torch.randn(B, cfg.matchup_profile_dim)
    return batch


def test_cross_ab_forward_shapes_when_enabled():
    """End-to-end forward succeeds with cross_ab_context=True + all inputs."""
    cfg = sanity_config()
    cfg.cross_ab_context = True
    model = PitchGPT(cfg)
    B, T = 2, 4
    batch = _build_cross_ab_batch(B, T, cfg)
    out = model(**batch)
    # Same output shapes as the base path — the extra context just adds to
    # the categorical token sum, no sequence-length changes.
    assert out["propensity"]["type"].shape == (B, 3 + T, cfg.n_pitch_types)
    assert out["result"].shape == (B, T, cfg.n_result_logits)


def test_cross_ab_forward_missing_matchup_profile_raises():
    """Useful error if cross_ab_context=True but the dataset forgot to emit
    matchup_profile — guards against the silent-mismatch failure mode."""
    cfg = sanity_config()
    cfg.cross_ab_context = True
    model = PitchGPT(cfg)
    batch = _build_cross_ab_batch(2, 4, cfg)
    batch.pop("matchup_profile")
    with pytest.raises(ValueError, match="matchup_profile is None"):
        model(**batch)


def test_cross_ab_forward_missing_tto_matchup_raises():
    """Mirror of the above for the tto_matchup categorical."""
    cfg = sanity_config()
    cfg.cross_ab_context = True
    model = PitchGPT(cfg)
    batch = _build_cross_ab_batch(2, 4, cfg)
    del batch["categorical_context"]["tto_matchup"]
    with pytest.raises(KeyError, match="tto_matchup"):
        model(**batch)


def test_cross_ab_context_default_false_backcompat():
    """Pre-v7p2 default path: cross_ab_context=False — no matchup needed,
    no extra parameters constructed, no extra categorical key."""
    cfg = sanity_config()
    assert cfg.cross_ab_context is False
    model = PitchGPT(cfg)
    # Neither the matchup MLP nor the tto_matchup embedding should exist
    assert not hasattr(model.context, "matchup_mlp")
    assert not hasattr(model.context, "tto_matchup_emb")
    # Base forward still works with the original _fake_batch (no matchup)
    out = model(**_fake_batch(2, 4, cfg))
    assert out["propensity"]["type"].shape[0] == 2


def test_cross_ab_matchup_vector_actually_changes_output():
    """Sanity: changing matchup_profile changes the output. Confirms the
    matchup MLP is in the gradient path, not a dangling parameter."""
    cfg = sanity_config()
    cfg.cross_ab_context = True
    model = PitchGPT(cfg).eval()
    batch_a = _build_cross_ab_batch(2, 4, cfg)
    batch_b = {**batch_a, "matchup_profile": batch_a["matchup_profile"] + 5.0}
    with torch.no_grad():
        out_a = model(**batch_a)
        out_b = model(**batch_b)
    # Type logits at the first pitch position must differ (matchup vector
    # enters the categorical token, which every pitch position attends to).
    assert not torch.allclose(
        out_a["propensity"]["type"], out_b["propensity"]["type"]
    ), "matchup_profile change did not propagate to the logits"
