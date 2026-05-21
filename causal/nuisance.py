"""Trained-model wrapper exposing PitchGPT as the (π̂, μ̂) nuisance pair.

The causal-layer skill: PitchGPT *is* the propensity model (next-pitch heads)
and *is* the conditional outcome model (two-stage result head). This module
exposes them as a single object whose API is decoupled from the model's
training-time forward signature — so g-computation, AIPW, and cross-fit code
can talk to the model without re-deriving the model's input plumbing.

Loads a calibrated checkpoint (with stored per-head temperatures from
``scripts.calibrate_pitchgpt``); applies temperatures by default. The
shared-trunk architecture (one model emits both π̂ and μ̂) is per ADR 007 — see
that ADR for why decoupling π̂ and μ̂ into two separate models was rejected.

Reserved language: "π̂" / "propensity" / "next-pitch distribution"; "μ̂" /
"outcome" / "result distribution". The model's role here is *nuisance* — the
causal estimator is elsewhere. Do not put effect-estimation language in this
module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PITCH_FACTOR_COLS_INT,
    CATEGORICAL_CTX_COLS,
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
)

# Heads exposed by π̂. Spin axis is a continuous head (sin/cos parameterization),
# not in the categorical propensity dict — handled separately when sampling.
PROPENSITY_HEADS: tuple[str, ...] = ("type", "zone", "velo", "spin_rate")


@dataclass
class ForwardOut:
    """Decoupled forward-pass output for one batch.

    All tensors live on CPU as float32 to keep downstream causal code
    device-agnostic. Temperatures are applied per-head; ``logits`` are raw
    (pre-temperature), ``probs`` are post-temperature.

    Shapes:
        propensity_logits/probs: each {head: (B, NC+T, K_head)} — CONTEXT
            positions are at indices [0..NC); pitch positions at [NC..NC+T).
        result_logits/probs: (B, T, 7) — pitch positions only.
        ab_outcome_logits/probs: (B, T, 7) — per-pitch; caller selects terminal.
    """

    propensity_logits: dict[str, torch.Tensor]
    propensity_probs: dict[str, torch.Tensor]
    marginal_propensity_probs: dict[str, torch.Tensor]
    result_logits: torch.Tensor
    result_probs: torch.Tensor
    ab_outcome_logits: torch.Tensor
    ab_outcome_probs: torch.Tensor

    @property
    def n_context_tokens(self) -> int:
        return PitchGPT.N_CONTEXT_TOKENS


class NuisanceModels:
    """Wraps a trained, calibrated PitchGPT checkpoint as (π̂, μ̂).

    Construction loads the checkpoint, builds the model, and reads the
    temperature scalars saved by ``scripts.calibrate_pitchgpt``. After
    construction the model is in ``eval()`` mode on the chosen device.

    The single public entry point is :meth:`forward`, which takes a batch
    dict (the same shape as ``model.pitchgpt.PitchGPT.forward`` expects) and
    returns a :class:`ForwardOut` with both raw and temperature-scaled logits
    for every head. Downstream code (g-computation, AIPW) builds its own
    state machines on top.
    """

    def __init__(
        self,
        ckpt_path: Path,
        *,
        device: str | torch.device | None = None,
        apply_temperatures: bool = True,
        profiles_dir: Path = Path("data/profiles"),
        standardize_profiles: bool = True,
    ):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found at {ckpt_path}; "
                f"the calibrated path is conventionally checkpoint_calibrated.pt"
            )
        self.ckpt_path = ckpt_path

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" not in ckpt:
            raise RuntimeError(
                f"{ckpt_path} has no 'config' key; not a PitchGPT training checkpoint"
            )
        self.cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
        self.model = PitchGPT(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.fold_id = int(ckpt.get("fold_id", 0))
        self.size = ckpt.get("size", "unknown")
        self.step = ckpt.get("step", None)
        self.temperatures: dict[str, float] = ckpt.get("temperatures", {})
        if apply_temperatures and not self.temperatures:
            raise RuntimeError(
                f"{ckpt_path} has no 'temperatures' — run `scripts.calibrate_pitchgpt` "
                f"on the raw checkpoint first; pass apply_temperatures=False to skip."
            )
        self.apply_temperatures = bool(apply_temperatures)

        self.profiles_dir = Path(profiles_dir)
        self.standardize_profiles = bool(standardize_profiles)
        self._pitcher_cache: ProfileCache | None = None
        self._batter_cache: ProfileCache | None = None
        self._standardizer: ProfileStandardizer | None = None

    # ---------- profile-cache lazy loaders ----------

    @property
    def pitcher_cache(self) -> ProfileCache:
        if self._pitcher_cache is None:
            self._pitcher_cache = ProfileCache(
                role="pitcher", fold_id=self.fold_id, profiles_dir=self.profiles_dir
            )
        return self._pitcher_cache

    @property
    def batter_cache(self) -> ProfileCache:
        if self._batter_cache is None:
            self._batter_cache = ProfileCache(
                role="batter", fold_id=self.fold_id, profiles_dir=self.profiles_dir
            )
        return self._batter_cache

    @property
    def standardizer(self) -> ProfileStandardizer | None:
        if not self.standardize_profiles:
            return None
        if self._standardizer is None:
            self._standardizer = ProfileStandardizer()
        return self._standardizer

    # ---------- the public forward ----------

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        return x

    def _apply_temp(self, logits: torch.Tensor, head: str) -> torch.Tensor:
        """Return ``logits / T_head`` if temperatures are enabled, else logits."""
        if not self.apply_temperatures:
            return logits
        T = self.temperatures.get(head)
        if T is None or T == 1.0:
            return logits
        return logits / T

    @torch.no_grad()
    def forward(self, batch: dict) -> ForwardOut:
        """One forward pass + per-head temperature scaling.

        ``batch`` is the collate output of ``PitchGPTAtBatDataset`` — same
        contract as ``model.pitchgpt.PitchGPT.forward``. Returns logits and
        post-temperature softmax probabilities for every head, on CPU.
        """
        bd = self._to_device(batch)
        out = self.model(
            pitcher_profile=bd["pitcher_profile"],
            batter_profile=bd["batter_profile"],
            categorical_context=bd["categorical_context"],
            pitch_factors=bd["pitch_factors"],
            intended_actions=bd["intended_actions"],
            padding_mask=bd["padding_mask"],
            arsenal=bd.get("arsenal"),
        )

        prop_logits: dict[str, torch.Tensor] = {}
        prop_probs: dict[str, torch.Tensor] = {}
        for h in PROPENSITY_HEADS:
            raw = out["propensity"][h].detach().to("cpu").float()
            scaled = self._apply_temp(raw, h)
            prop_logits[h] = raw
            prop_probs[h] = F.softmax(scaled, dim=-1)

        result_raw = out["result"].detach().to("cpu").float()
        result_scaled = self._apply_temp(result_raw, "result")
        result_probs = F.softmax(result_scaled, dim=-1)

        ab_raw = out["ab_outcome_per_pos"].detach().to("cpu").float()
        ab_scaled = self._apply_temp(ab_raw, "ab_outcome")
        ab_probs = F.softmax(ab_scaled, dim=-1)

        # Type-marginal execution distributions (ADR-013). For a checkpoint
        # WITHOUT type_conditioned_heads this is identical to prop_probs (the
        # heads were already type-marginal). With it, marginalize:
        #   P(exec | h) = Σ_type π̂(type | h) · P(exec | type, h),
        # where π̂(type) is the type-head softmax RENORMALIZED over the 7 real
        # pitch types — the PAD logit at index 0 is excluded (PAD is not a
        # treatment; including it leaves the marginal ~10% short of 1).
        marginal_probs: dict[str, torch.Tensor] = dict(prop_probs)
        if getattr(self.model.config, "type_conditioned_heads", False):
            from data.dataset import (
                PITCH_TYPES, MODEL_TYPE_ID,
                MODEL_PITCH_TYPES_START_IDX, MODEL_PITCH_TYPES_END_IDX,
            )
            type_p = prop_probs["type"]  # (B, T, 8) post-temperature softmax
            real_w = type_p[..., MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX]
            real_w = real_w / real_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            heads = ("zone", "velo", "spin_rate")
            acc = {h: torch.zeros_like(prop_probs[h]) for h in heads}
            for i, pt in enumerate(PITCH_TYPES):
                tid = MODEL_TYPE_ID[pt]
                cond_logits = self.model.execution_logits_for_type(bd, type_id=tid)
                w = real_w[..., i:i + 1]
                for h in heads:
                    cond = F.softmax(self._apply_temp(cond_logits[h].cpu().float(), h), dim=-1)
                    acc[h] = acc[h] + w * cond
            for h in heads:
                marginal_probs[h] = acc[h]

        return ForwardOut(
            propensity_logits=prop_logits,
            propensity_probs=prop_probs,
            marginal_propensity_probs=marginal_probs,
            result_logits=result_raw,
            result_probs=result_probs,
            ab_outcome_logits=ab_raw,
            ab_outcome_probs=ab_probs,
        )

    # ---------- per-position accessors ----------

    @staticmethod
    def propensity_at_pitch_position(
        forward_out: ForwardOut,
        position_in_ab: int,
        head: str = "type",
    ) -> torch.Tensor:
        """Get π̂(·) for pitch index ``position_in_ab`` of each batch row.

        The model's autoregressive convention: ``propensity_probs[head][b, t, :]``
        at sequence position ``t = N_CONTEXT_TOKENS + position_in_ab`` is the
        prediction OF pitch ``position_in_ab + 1`` GIVEN pitches 0..position_in_ab.
        So to ask "what's π̂ for the next pitch given pitches 0..k", look at
        ``position_in_ab = k`` and read this offset.

        Returns shape (B, K_head).
        """
        if head not in forward_out.propensity_probs:
            raise KeyError(f"head must be one of {PROPENSITY_HEADS}; got {head!r}")
        if position_in_ab < 0:
            raise ValueError(f"position_in_ab must be ≥ 0; got {position_in_ab}")
        seq_idx = forward_out.n_context_tokens + position_in_ab
        probs = forward_out.propensity_probs[head]
        if seq_idx >= probs.shape[1]:
            raise IndexError(
                f"position_in_ab={position_in_ab} (seq_idx={seq_idx}) out of bounds; "
                f"sequence has only {probs.shape[1]} positions"
            )
        return probs[:, seq_idx, :]

    @staticmethod
    def result_at_pitch_position(
        forward_out: ForwardOut,
        position_in_ab: int,
    ) -> torch.Tensor:
        """Get μ̂(·) for pitch index ``position_in_ab`` of each batch row.

        Result-head positions are zero-indexed over pitch positions only (no
        context-token offset, since the result head outputs are at pitch
        positions). Returns shape (B, 7).
        """
        if position_in_ab < 0:
            raise ValueError(f"position_in_ab must be ≥ 0; got {position_in_ab}")
        probs = forward_out.result_probs
        if position_in_ab >= probs.shape[1]:
            raise IndexError(
                f"position_in_ab={position_in_ab} out of bounds; "
                f"result head has only {probs.shape[1]} positions"
            )
        return probs[:, position_in_ab, :]

    def __repr__(self) -> str:
        return (
            f"NuisanceModels(ckpt={self.ckpt_path.name}, fold={self.fold_id}, "
            f"size={self.size}, step={self.step}, device={self.device}, "
            f"apply_temps={self.apply_temperatures})"
        )


# ---------- helpers for building a single-AB rollout batch ----------


def build_single_ab_batch(
    nuisance: NuisanceModels,
    pitches_df: pd.DataFrame,
    n_replicates: int = 1,
) -> dict:
    """Build the forward batch for one AB, replicated ``n_replicates`` times.

    For Monte Carlo rollouts: replicate=N gives N independent "starting points"
    sharing the same history, from which each path can diverge as we sample
    forward-step-by-step.

    Args:
        nuisance: NuisanceModels (gives profile caches + standardizer).
        pitches_df: one AB's pitch rows in chronological order. Must contain
            the columns the dataset class expects (see REQUIRED_AUG_COLS).
        n_replicates: how many copies to stack along the batch dim.

    Returns:
        batch dict ready for ``NuisanceModels.forward(batch)``.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be ≥ 1; got {n_replicates}")
    ds = PitchGPTAtBatDataset(
        pitches=pitches_df,
        pitcher_profile_lookup=nuisance.pitcher_cache.lookup,
        batter_profile_lookup=nuisance.batter_cache.lookup,
        profile_standardizer=nuisance.standardizer,
    )
    if len(ds) != 1:
        raise ValueError(
            f"expected pitches_df to be exactly one AB; got {len(ds)} ABs "
            f"after sort+group. Filter by (game_pk, at_bat_number) before calling."
        )
    items = [ds[0] for _ in range(n_replicates)]
    batch = collate_pitchgpt_at_bats(items)

    # Profile-dim compat shim. The cache schema is appended-only — v4 adds
    # 7 arm_slot dims at the END of the pitcher vector, v3 added 14 batter
    # dims at the end of the batter vector — so the first ``cfg.X_profile_dim``
    # entries of a newer cache are byte-equivalent to an older one. Slice them
    # if the loaded checkpoint expects a shorter profile than the cache produces.
    # This lets a pre-Sprint-0b checkpoint (pitcher_profile_dim=223) load the
    # v4 cache without retraining; later checkpoints just don't trigger the slice.
    expected_p = int(nuisance.cfg.pitcher_profile_dim)
    cache_p = int(batch["pitcher_profile"].shape[-1])
    if cache_p > expected_p:
        batch["pitcher_profile"] = batch["pitcher_profile"][..., :expected_p].contiguous()
    elif cache_p < expected_p:
        raise RuntimeError(
            f"pitcher cache vector dim ({cache_p}) is SMALLER than the checkpoint's "
            f"expected pitcher_profile_dim ({expected_p}). The cache is stale relative "
            f"to the checkpoint — rebuild it with the current PROFILE_SCHEMA_VERSION."
        )
    expected_b = int(nuisance.cfg.batter_profile_dim)
    cache_b = int(batch["batter_profile"].shape[-1])
    if cache_b > expected_b:
        batch["batter_profile"] = batch["batter_profile"][..., :expected_b].contiguous()
    elif cache_b < expected_b:
        raise RuntimeError(
            f"batter cache vector dim ({cache_b}) < expected ({expected_b}); rebuild the cache"
        )
    return batch
