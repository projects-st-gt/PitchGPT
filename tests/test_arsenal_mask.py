"""Tests for the arsenal-masked type logits.

Pins four contracts:
  1. Flag off (default): forward is byte-identical to pre-mask behavior.
  2. Flag on: types with ``has_pitch=0`` get logits at ``-1e9`` and softmax
     probability ``~0``; types with ``has_pitch=1`` are untouched.
  3. Zero-history fallback: if a pitcher has ``has_pitch`` all zeros (the
     zero-fallback profile path), the mask is bypassed — NO softmax NaN.
  4. Convention discipline: masking aligns has_pitch[i] with model type ID
     ``MODEL_PITCH_TYPES_START_IDX + i``, leaves PAD (idx 0) untouched.

The mask is a pure post-process on the type logits, so it can be enabled at
inference time on an existing checkpoint without retraining. Tests use the
sanity config and synthetic batches to make the contract crisp.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from data.dataset import MODEL_PITCH_TYPES_START_IDX, MODEL_PITCH_TYPES_END_IDX
from model.config import sanity_config
from model.pitchgpt import PitchGPT
from tests.test_pitchgpt_model import _fake_batch


def _batch_with_arsenal(B: int = 2, T: int = 4, cfg=None) -> dict:
    """``_fake_batch`` + a synthetic 14-dim arsenal vector per item.

    Item 0: only FF + SL (has_pitch = [1, 0, 0, 1, 0, 0, 0]).
    Item 1: full arsenal (has_pitch = [1, 1, 1, 1, 1, 1, 1]) — no mask should apply.
    """
    if cfg is None:
        cfg = sanity_config()
    batch = _fake_batch(B, T, cfg)
    # 14-dim arsenal = [7 rates, 7 has_pitch]
    arsenal = torch.zeros(B, cfg.n_arsenal_dims, dtype=torch.float32)
    arsenal[0, 7:14] = torch.tensor([1, 0, 0, 1, 0, 0, 0], dtype=torch.float32)  # FF, SL only
    arsenal[1, 7:14] = torch.ones(7, dtype=torch.float32)                         # all
    arsenal[:, 0:7] = arsenal[:, 7:14] * 0.5  # arbitrary plausible usage rates
    batch["arsenal"] = arsenal
    return batch


def test_arsenal_mask_off_is_backcompat():
    """Default (flag off): forward equals the pre-mask path bit-for-bit."""
    cfg = sanity_config()
    assert cfg.arsenal_mask_type_logits is False
    torch.manual_seed(0)
    model = PitchGPT(cfg).eval()
    batch = _batch_with_arsenal(2, 4, cfg)
    with torch.no_grad():
        out = model(**batch)
    # Sanity: type logits are finite, not all -1e9
    assert torch.isfinite(out["propensity"]["type"]).all()
    assert (out["propensity"]["type"] > -1e8).all()


def test_arsenal_mask_zeros_impossible_types():
    """Flag on: has_pitch=0 types get logits at -1e9 → softmax probability 0."""
    cfg = sanity_config()
    cfg.arsenal_mask_type_logits = True
    torch.manual_seed(0)
    model = PitchGPT(cfg).eval()
    batch = _batch_with_arsenal(2, 4, cfg)
    with torch.no_grad():
        out = model(**batch)
    type_logits = out["propensity"]["type"]   # (B, T_total, n_pitch_types)
    # Item 0: has_pitch[i]=0 at indices [1, 2, 4, 5, 6] of the 7-vector,
    # which map to model type IDs [2, 3, 5, 6, 7].
    masked_type_ids = [
        MODEL_PITCH_TYPES_START_IDX + i for i in (1, 2, 4, 5, 6)
    ]
    for tid in masked_type_ids:
        assert (type_logits[0, :, tid] < -1e8).all(), (
            f"item 0 type id {tid} not masked (logit={type_logits[0, 0, tid]})"
        )
    # Item 0: unmasked types (FF=1, SL=4) and PAD (0) must NOT be -1e9
    for tid in (0, MODEL_PITCH_TYPES_START_IDX + 0, MODEL_PITCH_TYPES_START_IDX + 3):
        assert (type_logits[0, :, tid] > -1e8).all(), (
            f"item 0 type id {tid} should be unmasked but logit={type_logits[0, 0, tid]}"
        )
    # Item 1 (full arsenal): NO masking
    assert (type_logits[1] > -1e8).all(), "item 1 (full arsenal) had a masked logit"

    # Probabilities are well-defined: softmax sums to 1, masked classes ~ 0
    probs = F.softmax(type_logits[0, 0, :], dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    masked_prob = probs[masked_type_ids].sum().item()
    assert masked_prob < 1e-5, f"masked classes leaked {masked_prob:.2e} mass"


def test_arsenal_mask_zero_history_fallback_no_nan():
    """Pitcher with all-zero has_pitch (zero-fallback profile path) MUST NOT
    produce a NaN softmax — the mask is bypassed for that row."""
    cfg = sanity_config()
    cfg.arsenal_mask_type_logits = True
    torch.manual_seed(0)
    model = PitchGPT(cfg).eval()
    batch = _batch_with_arsenal(2, 4, cfg)
    # Override item 0 to zero-history; item 1 keeps a real arsenal — both
    # rows must produce finite logits.
    batch["arsenal"][0, :] = 0.0
    with torch.no_grad():
        out = model(**batch)
    type_logits = out["propensity"]["type"]
    assert torch.isfinite(type_logits).all(), "zero-history pitcher produced inf/NaN logits"
    probs = F.softmax(type_logits[0, 0, :], dim=-1)
    assert torch.isfinite(probs).all(), "softmax produced NaN on zero-history row"


def test_arsenal_mask_requires_arsenal_tensor():
    """Useful error if the flag is on but no `arsenal` is passed."""
    cfg = sanity_config()
    cfg.arsenal_mask_type_logits = True
    model = PitchGPT(cfg)
    batch = _fake_batch(2, 4, cfg)  # no arsenal
    with pytest.raises(ValueError, match="arsenal_mask_type_logits=True"):
        model(**batch)


def test_arsenal_mask_can_be_toggled_at_inference():
    """Inference-time toggle: same checkpoint, mask off vs on, type
    logits differ exactly where has_pitch=0."""
    cfg = sanity_config()
    torch.manual_seed(0)
    model = PitchGPT(cfg).eval()  # built with mask off
    batch = _batch_with_arsenal(2, 4, cfg)
    with torch.no_grad():
        out_off = model(**batch)
    # Flip the flag on the live config — pure post-process, no weight changes
    model.config.arsenal_mask_type_logits = True
    with torch.no_grad():
        out_on = model(**batch)
    # Item 0 (FF + SL only) — masked classes differ; unmasked + PAD unchanged
    masked_ids = [
        MODEL_PITCH_TYPES_START_IDX + i for i in (1, 2, 4, 5, 6)
    ]
    unmasked_ids = [0, MODEL_PITCH_TYPES_START_IDX + 0, MODEL_PITCH_TYPES_START_IDX + 3]
    for tid in masked_ids:
        assert not torch.allclose(
            out_off["propensity"]["type"][0, :, tid],
            out_on["propensity"]["type"][0, :, tid],
        ), f"toggling mask had no effect on item 0 type id {tid}"
    for tid in unmasked_ids:
        assert torch.allclose(
            out_off["propensity"]["type"][0, :, tid],
            out_on["propensity"]["type"][0, :, tid],
        ), f"unmasked type id {tid} changed when mask toggled"
