"""V2 rollout normalization + temperature conventions.

These tests pin the convention that bit the first rollout integration: the
model trains on z-score-normalized continuous values, so

  1. build_single_ab_batch_v2 must emit NORMALIZED inputs (raw zeros at the
     start token become (0-mean)/std, exactly as in scripts.train_v2), and
  2. GMM samples live in z-score space and must be denormalized before the
     cascade sees mph/rpm/feet, and
  3. NuisanceModelsV2.forward applies the calibrated type temperature
     (logits / T) and refuses uncalibrated checkpoints by default.

Synthetic fixtures only — these are pure-function/convention tests.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from model.v2.config import V2Config
from model.v2.model import PitchGPTV2
from causal.nuisance_v2 import (
    NuisanceModelsV2,
    build_single_ab_batch_v2,
    denormalize_continuous,
    normalize_continuous,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def micro_cfg() -> V2Config:
    # Legacy 4-dim config (v1c) — pins back-compat for old checkpoints.
    return V2Config(
        n_layers=1, n_heads=2, d_model=16, d_ff=32,
        adaln_hidden=8, pitcher_profile_dim=5, batter_profile_dim=3,
        n_continuous=4,
    )


def _save_ckpt(tmp_path, cfg, temperatures=None):
    from dataclasses import asdict
    model = PitchGPTV2(cfg)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "size": "micro",
        "fold_id": 0,
        "step": 0,
        "schema_version": 2,
    }
    if temperatures is not None:
        ckpt["temperatures"] = temperatures
    path = tmp_path / "checkpoint.pt"
    torch.save(ckpt, path)
    return path


def _micro_batch(cfg, B=2, T=3):
    g = torch.Generator().manual_seed(0)
    return {
        "pitcher_profile": torch.randn(B, cfg.pitcher_profile_dim, generator=g),
        "batter_profile": torch.randn(B, cfg.batter_profile_dim, generator=g),
        "type_ids": torch.randint(0, 8, (B, T), generator=g),
        "continuous": torch.randn(B, T, 4, generator=g),
        "result_ids": torch.randint(0, 8, (B, T), generator=g),
        "count_state": torch.randint(0, 12, (B, T), generator=g),
        "outs": torch.randint(0, 3, (B, T), generator=g),
        "runners": torch.randint(0, 8, (B, T), generator=g),
        "pitch_number": torch.arange(T).expand(B, T).clone(),
        "padding_mask": torch.ones(B, T, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# normalize / denormalize
# ---------------------------------------------------------------------------

def test_normalize_named_value(micro_cfg):
    # velo 95.0 mph -> (95 - 88.38) / 6.03 = 1.0978...
    raw = np.array([[95.0, 2254.70, 0.04, 2.24]], dtype=np.float32)
    normed = normalize_continuous(raw, micro_cfg)
    assert normed[0, 0] == pytest.approx((95.0 - 88.38) / 6.03, abs=1e-4)
    # The other three dims sit at their means -> exactly 0.
    assert np.allclose(normed[0, 1:], 0.0, atol=1e-4)


def test_denormalize_round_trip(micro_cfg):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(10, 4)).astype(np.float32) * [6.0, 360.0, 0.85, 0.98] + [
        88.4, 2254.7, 0.04, 2.24,
    ]
    back = denormalize_continuous(normalize_continuous(raw, micro_cfg), micro_cfg)
    assert np.allclose(back, raw, atol=1e-3)


# ---------------------------------------------------------------------------
# build_single_ab_batch_v2 emits normalized inputs
# ---------------------------------------------------------------------------

def _stub_nuisance(cfg):
    lookup = lambda *a, **k: {"vector": np.zeros(cfg.pitcher_profile_dim, np.float32)}
    blookup = lambda *a, **k: {"vector": np.zeros(cfg.batter_profile_dim, np.float32)}
    return SimpleNamespace(
        cfg=cfg,
        pitcher_cache=SimpleNamespace(lookup=lookup),
        batter_cache=SimpleNamespace(lookup=blookup),
        standardizer=None,
    )


def _one_pitch_ab(velo=95.0):
    return pd.DataFrame([{
        "game_date": "2024-08-01", "pitcher": 1, "batter": 2,
        "pitch_number": 1, "type_id": 1, "result_id": 1,
        "count_state": 0, "outs_state": 0, "runners_state": 0,
        "release_speed": velo, "release_spin_rate": 2254.70,
        "plate_x": 0.04, "plate_z": 2.24,
    }])


def test_batch_continuous_is_normalized(micro_cfg):
    batch = build_single_ab_batch_v2(_stub_nuisance(micro_cfg), _one_pitch_ab(95.0))
    velo_normed = float(batch["continuous"][0, 1, 0])
    assert velo_normed == pytest.approx((95.0 - 88.38) / 6.03, abs=1e-4), (
        f"velo at seq position 1 should be z-scored; got {velo_normed} "
        f"(raw 95.0 would mean normalization is missing)"
    )


def test_batch_start_token_matches_training(micro_cfg):
    # Training normalizes AFTER the dataset's nan->0 fill, so the start
    # token's raw zeros become (0 - mean)/std, NOT zeros.
    batch = build_single_ab_batch_v2(_stub_nuisance(micro_cfg), _one_pitch_ab())
    start_velo = float(batch["continuous"][0, 0, 0])
    assert start_velo == pytest.approx((0.0 - 88.38) / 6.03, abs=1e-3)


def test_batch_missing_velo_columns_ok(micro_cfg):
    # Synthetic ABs (mcsim build_synthetic_ab) carry no release_speed column;
    # the builder must treat it as missing (-> 0 raw -> normalized), not crash.
    ab = _one_pitch_ab().drop(columns=["release_speed", "release_spin_rate"])
    batch = build_single_ab_batch_v2(_stub_nuisance(micro_cfg), ab)
    velo_normed = float(batch["continuous"][0, 1, 0])
    assert velo_normed == pytest.approx((0.0 - 88.38) / 6.03, abs=1e-3)


# ---------------------------------------------------------------------------
# Temperature application in NuisanceModelsV2.forward
# ---------------------------------------------------------------------------

def test_scale_type_logits_flat(tmp_path, micro_cfg):
    path = _save_ckpt(tmp_path, micro_cfg, temperatures={"type": 2.0})
    nz = NuisanceModelsV2(path, device="cpu")
    batch = _micro_batch(micro_cfg)
    raw = nz.forward(batch)["type_logits"]    # forward returns RAW logits
    last = raw[:, -1, :]
    scaled = nz.scale_type_logits(last)
    assert torch.allclose(scaled, last / 2.0, atol=1e-6)
    # apply_temperatures=False passes through.
    nz_off = NuisanceModelsV2(path, device="cpu", apply_temperatures=False)
    assert torch.allclose(nz_off.scale_type_logits(last), last)


def test_scale_type_logits_per_count(tmp_path, micro_cfg):
    import dataclasses
    from dataclasses import asdict
    model = PitchGPTV2(micro_cfg)
    ckpt = {
        "model_state_dict": model.state_dict(), "config": asdict(micro_cfg),
        "size": "micro", "fold_id": 0, "step": 0, "schema_version": 2,
        "temperatures": {"type": 1.5},
        "count_temperatures": {"type": {"6": 2.0}},   # 2-0 count = id 6
    }
    path = tmp_path / "ckpt_cs.pt"
    torch.save(ckpt, path)
    nz = NuisanceModelsV2(path, device="cpu")
    logits = torch.ones(3, 8)
    logits[:, 1] = 4.0
    counts = np.array([6, 0, 6])   # rows 0,2 at 2-0 -> T=2.0; row 1 -> flat 1.5
    scaled = nz.scale_type_logits(logits, count_ids=counts)
    assert torch.allclose(scaled[0], logits[0] / 2.0, atol=1e-6)
    assert torch.allclose(scaled[1], logits[1] / 1.5, atol=1e-6)
    assert torch.allclose(scaled[2], logits[2] / 2.0, atol=1e-6)
    # Named behavior: T>1 at 2-0 REDUCES the modal type's probability.
    p_flat = torch.softmax(logits[1] / 1.5, dim=-1)[1]
    p_cs = torch.softmax(scaled[0], dim=-1)[1]
    assert p_cs < p_flat, (
        f"T=2.0 at 2-0 should shrink modal mass: {p_cs:.3f} vs flat {p_flat:.3f}")


def test_uncalibrated_checkpoint_refused_by_default(tmp_path, micro_cfg):
    path = _save_ckpt(tmp_path, micro_cfg, temperatures=None)
    with pytest.raises(RuntimeError, match="calibrate_v2"):
        NuisanceModelsV2(path, device="cpu")
    # Escape hatch still works.
    nz = NuisanceModelsV2(path, device="cpu", apply_temperatures=False)
    assert nz.temperatures == {}
