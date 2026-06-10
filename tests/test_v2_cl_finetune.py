"""Tether-rule tests for the v1c-cl closed-loop fine-tune (synthetic fixtures).

Pins the substitution conventions from
docs/superpowers/specs/2026-06-10-v1c-cl-finetune-design.md:
  - position 0 (start token) is never substituted
  - p_sub=0 is a no-op; K=full-vocab tether always keeps the real type
  - K=1 substitutes the model's argmax
  - substituted continuous values are physically plausible in raw space
  - targets and the count/result scaffold are untouched
"""
import numpy as np
import pytest
import torch

from model.v2.config import V2Config
from model.v2.model import PitchGPTV2
from scripts.finetune_v2_cl import _CLAMP_HI, _CLAMP_LO, build_substituted_batch


@pytest.fixture
def micro():
    # Legacy 4-dim config (v1c) — the substitution logic must serve both.
    cfg = V2Config(n_layers=1, n_heads=2, d_model=16, d_ff=32,
                   adaln_hidden=8, pitcher_profile_dim=5, batter_profile_dim=3,
                   n_continuous=4)
    model = PitchGPTV2(cfg).eval()
    return cfg, model


def _batch(cfg, B=4, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    type_ids = torch.randint(1, 8, (B, T), generator=g)
    type_ids[:, 0] = 0  # start token
    tgt_type = torch.full((B, T), -100, dtype=torch.long)
    tgt_type[:, :-1] = type_ids[:, 1:]
    return {
        "pitcher_profile": torch.randn(B, cfg.pitcher_profile_dim, generator=g),
        "batter_profile": torch.randn(B, cfg.batter_profile_dim, generator=g),
        "type_ids": type_ids,
        "continuous": torch.randn(B, T, 4, generator=g),  # z-space
        "result_ids": torch.randint(0, 8, (B, T), generator=g),
        "count_state": torch.randint(0, 12, (B, T), generator=g),
        "outs": torch.randint(0, 3, (B, T), generator=g),
        "runners": torch.randint(0, 8, (B, T), generator=g),
        "pitch_number": torch.arange(T).expand(B, T).clone(),
        "padding_mask": torch.ones(B, T, dtype=torch.bool),
        "targets": {"type": tgt_type,
                    "continuous": torch.randn(B, T, 4, generator=g)},
    }


def test_p_sub_zero_is_noop(micro):
    cfg, model = micro
    b = _batch(cfg)
    sub, stats = build_substituted_batch(model, b, cfg, k=3, p_sub=0.0)
    assert torch.equal(sub["type_ids"], b["type_ids"])
    assert torch.equal(sub["continuous"], b["continuous"])
    assert stats["sub_rate"] == 0.0


def test_full_vocab_tether_keeps_real_types(micro):
    cfg, model = micro
    b = _batch(cfg)
    # K = 7 (all real types) -> the real type is always in top-K -> tether
    # hit everywhere -> type_ids unchanged, tether_hit_rate = 1.0.
    sub, stats = build_substituted_batch(model, b, cfg, k=7, p_sub=1.0)
    assert torch.equal(sub["type_ids"], b["type_ids"])
    assert stats["tether_hit_rate"] == pytest.approx(1.0)
    assert stats["sub_rate"] == pytest.approx(1.0)
    # Continuous IS substituted (GMM samples) even on tether hits.
    assert not torch.equal(sub["continuous"][:, 1:], b["continuous"][:, 1:])


def test_k1_substitutes_model_argmax(micro):
    cfg, model = micro
    b = _batch(cfg)
    sub, _ = build_substituted_batch(model, b, cfg, k=1, p_sub=1.0)
    # Recompute the argmax at position 0 over the ORIGINAL prefix (position 0
    # is never substituted, so the t=1 prediction context is unchanged).
    with torch.no_grad():
        out = model(**{kk: b[kk] for kk in (
            "pitcher_profile", "batter_profile", "type_ids", "continuous",
            "result_ids", "count_state", "outs", "runners", "pitch_number",
            "padding_mask")})
    logits = out["type_logits"][:, 0, :].clone()
    logits[:, 0] = -1e9
    expect = logits.argmax(-1)
    assert torch.equal(sub["type_ids"][:, 1], expect), (
        f"K=1 must input the model argmax at position 1: "
        f"got {sub['type_ids'][:, 1].tolist()} vs argmax {expect.tolist()}")
    assert (sub["type_ids"][:, 0] == 0).all(), "start token must stay PAD"


def test_substituted_continuous_physically_plausible(micro):
    cfg, model = micro
    b = _batch(cfg)
    sub, _ = build_substituted_batch(model, b, cfg, k=3, p_sub=1.0)
    n = cfg.n_continuous
    c_mean = torch.tensor(cfg.continuous_means[:n])
    c_std = torch.tensor(cfg.continuous_stds[:n])
    raw = sub["continuous"][:, 1:] * c_std + c_mean
    lo = torch.tensor(_CLAMP_LO[:n])
    hi = torch.tensor(_CLAMP_HI[:n])
    assert torch.isfinite(raw).all()
    assert (raw >= lo - 1e-3).all() and (raw <= hi + 1e-3).all(), (
        f"raw velo range {raw[..., 0].min():.1f}-{raw[..., 0].max():.1f} mph "
        f"must sit inside [60, 110]")


def test_targets_and_scaffold_untouched(micro):
    cfg, model = micro
    b = _batch(cfg)
    sub, _ = build_substituted_batch(model, b, cfg, k=2, p_sub=1.0)
    assert torch.equal(sub["targets"]["type"], b["targets"]["type"])
    assert torch.equal(sub["targets"]["continuous"], b["targets"]["continuous"])
    for key in ("result_ids", "count_state", "outs", "runners", "pitch_number"):
        assert torch.equal(sub[key], b[key]), f"{key} scaffold must stay real"
