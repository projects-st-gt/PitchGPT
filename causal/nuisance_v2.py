"""Trained-model wrapper exposing PitchGPTV2 as a nuisance model for rollout.

PitchGPTV2 predicts the next pitch's type (via weight-tied softmax) and
continuous properties (via a mixture-of-Gaussians head conditioned on the
sampled type). This module wraps a V2 checkpoint so the rollout code
(g_computation_v2) can call forward() and predict_continuous() without
dealing with the model's input plumbing.

Key differences from the V1 NuisanceModels:
  - No context tokens (V2 uses adaLN conditioning instead of context-token
    positions, so there is no N_CONTEXT_TOKENS offset).
  - No propensity heads dict — V2 has a single type_logits output.
  - No result head — the cascade (hitter model) handles pitch outcomes.
  - Forward returns {"type_logits": (B,T,8), "hidden": (B,T,d)} directly.
  - predict_continuous() returns GMM parameters (log_w, mu, log_std).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.profile_cache_loader import ProfileCache
from model.v2.config import V2Config
from model.v2.model import PitchGPTV2
from model.pitchgpt_dataset import ProfileStandardizer


def normalize_continuous(raw: np.ndarray, cfg: V2Config) -> np.ndarray:
    """Z-score normalize raw continuous values [velo, spin, plate_x, plate_z].

    The model was trained on normalized inputs (scripts.train_v2 normalizes
    after the dataset's nan->0 fill), so every rollout input must go through
    this before forward().
    """
    mean = np.asarray(cfg.continuous_means, dtype=np.float32)
    std = np.asarray(cfg.continuous_stds, dtype=np.float32)
    return (np.asarray(raw, dtype=np.float32) - mean) / std


def denormalize_continuous(normed: np.ndarray, cfg: V2Config) -> np.ndarray:
    """Invert :func:`normalize_continuous` — GMM samples live in z-score space
    and must be mapped back to raw mph/rpm/feet before the cascade sees them."""
    mean = np.asarray(cfg.continuous_means, dtype=np.float32)
    std = np.asarray(cfg.continuous_stds, dtype=np.float32)
    return np.asarray(normed, dtype=np.float32) * std + mean


class NuisanceModelsV2:
    """Wraps a trained PitchGPTV2 checkpoint for rollout use.

    Construction loads the checkpoint, builds PitchGPTV2(cfg) on the chosen
    device in eval mode, and optionally reads stored temperatures. After
    construction, call :meth:`forward` for type_logits + hidden states, then
    :meth:`predict_continuous` for GMM parameters conditioned on a sampled type.
    """

    def __init__(
        self,
        ckpt_path: Path,
        *,
        device: str | torch.device | None = None,
        profiles_dir: Path = Path("data/profiles"),
        standardize_profiles: bool = True,
        apply_temperatures: bool = True,
    ):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")
        self.ckpt_path = ckpt_path

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" not in ckpt:
            raise RuntimeError(
                f"{ckpt_path} has no 'config' key; not a PitchGPTV2 checkpoint"
            )

        # Build V2Config from the checkpoint, filtering to only V2Config fields
        # so extra keys from older checkpoints don't cause TypeError.
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(V2Config)}
        cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_fields}
        self.cfg = V2Config(**cfg_dict)
        self.model = PitchGPTV2(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.fold_id = int(ckpt.get("fold_id", 0))
        self.size = ckpt.get("size", "unknown")
        self.step = ckpt.get("step", None)
        self.schema_version = ckpt.get("schema_version", 2)

        # Temperatures from scripts.calibrate_v2. Same convention as V1
        # NuisanceModels: refuse an uncalibrated checkpoint unless the caller
        # opts out explicitly. count_temperatures (optional) hold one type-head
        # temperature PER COUNT STATE of the predicted pitch — fitted because
        # the flat-T model over-commits to FF at hitter counts (2-0 +13.4pp in
        # rollout marginals, 2026-06-09).
        self.temperatures: dict[str, float] = ckpt.get("temperatures", {})
        self.count_temperatures: dict[str, dict] = ckpt.get("count_temperatures", {})
        if apply_temperatures and not self.temperatures:
            raise RuntimeError(
                f"{ckpt_path} has no 'temperatures' — run `scripts.calibrate_v2` "
                f"on the raw checkpoint first; pass apply_temperatures=False to skip."
            )
        self.apply_temperatures = bool(apply_temperatures)

        # Lazy profile cache loaders (same pattern as v1).
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
                role="pitcher", fold_id=self.fold_id,
                profiles_dir=self.profiles_dir,
            )
        return self._pitcher_cache

    @property
    def batter_cache(self) -> ProfileCache:
        if self._batter_cache is None:
            self._batter_cache = ProfileCache(
                role="batter", fold_id=self.fold_id,
                profiles_dir=self.profiles_dir,
            )
        return self._batter_cache

    @property
    def standardizer(self) -> ProfileStandardizer | None:
        if not self.standardize_profiles:
            return None
        if self._standardizer is None:
            self._standardizer = ProfileStandardizer()
        return self._standardizer

    # ---------- device helper ----------

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        return x

    # ---------- forward ----------

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        """Forward pass through PitchGPTV2.

        Args:
            batch: dict with keys matching PitchGPTV2.forward() signature:
                pitcher_profile, batter_profile, type_ids, continuous,
                result_ids, count_state, outs, runners, pitch_number,
                padding_mask.

        Returns:
            {"type_logits": (B, T, 8), "hidden": (B, T, d_model)}
            Both tensors are on CPU as float32. type_logits are RAW —
            temperature scaling is count-conditional and the count of the
            PREDICTED pitch is not part of this batch, so the caller applies
            :meth:`scale_type_logits` at the point of use (g_compute_v2 does).
        """
        bd = self._to_device(batch)
        out = self.model(
            pitcher_profile=bd["pitcher_profile"],
            batter_profile=bd["batter_profile"],
            type_ids=bd["type_ids"],
            continuous=bd["continuous"],
            result_ids=bd["result_ids"],
            count_state=bd["count_state"],
            outs=bd["outs"],
            runners=bd["runners"],
            pitch_number=bd["pitch_number"],
            padding_mask=bd["padding_mask"],
        )
        return {
            "type_logits": out["type_logits"].detach().cpu().float(),
            "hidden": out["hidden"].detach().cpu().float(),
        }

    # ---------- temperature scaling ----------

    def scale_type_logits(
        self,
        logits: torch.Tensor,
        count_ids=None,
    ) -> torch.Tensor:
        """Apply calibrated temperature(s) to type logits.

        Args:
            logits: (..., 8) raw type logits.
            count_ids: optional int array/tensor of shape logits.shape[:-1] —
                the count state (0..11) of the pitch being PREDICTED. When
                given and the checkpoint has count_temperatures, each row is
                scaled by its count's temperature (flat T as fallback for a
                count missing from the fit). When None, the flat T applies.

        Returns:
            logits / T, same shape. Unchanged if apply_temperatures is False.
        """
        if not self.apply_temperatures:
            return logits
        T_flat = float(self.temperatures.get("type", 1.0))
        ct = self.count_temperatures.get("type") if count_ids is not None else None
        if ct:
            cid = torch.as_tensor(np.asarray(count_ids), dtype=torch.long)
            t_table = torch.full((12,), T_flat, dtype=logits.dtype)
            for k, v in ct.items():
                k = int(k)
                if 0 <= k < 12:
                    t_table[k] = float(v)
            t = t_table[cid.clamp(0, 11)]
            return logits / t.unsqueeze(-1)
        if T_flat != 1.0:
            return logits / T_flat
        return logits

    # ---------- GMM prediction ----------

    @torch.no_grad()
    def predict_continuous(
        self,
        hidden: torch.Tensor,    # (B, T, d_model)
        type_ids: torch.Tensor,  # (B, T) LongTensor — 1-indexed type ids
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GMM prediction conditioned on a sampled pitch type.

        Args:
            hidden: (B, T, d_model) from forward()["hidden"].
            type_ids: (B, T) LongTensor — pitch type indices (1..7, no PAD).

        Returns:
            log_w  : (B, T, K)       log mixture weights
            mu     : (B, T, K, D)    component means (D=4: velo, spin, px, pz)
            log_std: (B, T, K, D)    log standard deviations
            All on CPU as float32.
        """
        h = hidden.to(self.device)
        t = type_ids.to(self.device)
        log_w, mu, log_std = self.model.predict_continuous(h, t)
        return (
            log_w.detach().cpu().float(),
            mu.detach().cpu().float(),
            log_std.detach().cpu().float(),
        )

    # ---------- GMM sampling ----------

    @torch.no_grad()
    def sample_continuous(
        self,
        hidden: torch.Tensor,    # (B, T, d_model)
        type_ids: torch.Tensor,  # (B, T) LongTensor — 1-indexed type ids
    ) -> torch.Tensor:
        """Sample continuous values from the GMM.

        Returns:
            (B, T, D) float32 tensor of sampled [velo, spin, plate_x, plate_z].
        """
        h = hidden.to(self.device)
        t = type_ids.to(self.device)
        log_w, mu, log_std = self.model.predict_continuous(h, t)
        sample = self.model.gmm_head.sample(log_w, mu, log_std)
        return sample.detach().cpu().float()

    def __repr__(self) -> str:
        return (
            f"NuisanceModelsV2(ckpt={self.ckpt_path.name}, fold={self.fold_id}, "
            f"size={self.size}, step={self.step}, device={self.device})"
        )


# ---------- helpers for building a single-AB rollout batch ----------


def build_single_ab_batch_v2(
    nuisance: NuisanceModelsV2,
    ab_df: pd.DataFrame,
    n_replicates: int = 1,
) -> dict:
    """Build the forward batch for one at-bat in V2 format, replicated N times.

    This is the V2 analogue of ``causal.nuisance.build_single_ab_batch``.
    It constructs the sequence manually (prepending the start token, filling
    all factor arrays) instead of going through V2AtBatDataset, since the
    rollout needs mutable tensors that grow step-by-step.

    The V2 convention:
      - Position 0 = "start" token: type=0 (PAD), continuous=zeros, result=0,
        count/outs/runners from the pre-AB game state.
      - Positions 1..T = the real pitches in the at-bat.

    Args:
        nuisance: NuisanceModelsV2 (provides profile caches + standardizer).
        ab_df: one AB's pitch rows in chronological order.
        n_replicates: how many copies along the batch dim (for MC paths).

    Returns:
        batch dict ready for NuisanceModelsV2.forward().
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1; got {n_replicates}")

    ab_df = ab_df.sort_values("pitch_number").reset_index(drop=True)
    T = len(ab_df)  # number of real pitches
    seq_len = T + 1  # start token + T real pitches
    first = ab_df.iloc[0]

    # ---- Profiles ----
    asof_date = pd.Timestamp(first["game_date"])
    asof_game_num = int(first["game_num"]) if "game_num" in first.index else 1

    pitcher_vec = nuisance.pitcher_cache.lookup(
        int(first["pitcher"]), asof_date, asof_game_num,
        as_of_fallback=True,
    )["vector"]
    batter_vec = nuisance.batter_cache.lookup(
        int(first["batter"]), asof_date, asof_game_num,
        as_of_fallback=True,
    )["vector"]

    if nuisance.standardizer is not None:
        pitcher_vec = nuisance.standardizer.apply(pitcher_vec, "pitcher")
        batter_vec = nuisance.standardizer.apply(batter_vec, "batter")

    # Profile-dim compat shim (same as v1: slice if cache is wider than cfg).
    expected_p = int(nuisance.cfg.pitcher_profile_dim)
    if len(pitcher_vec) > expected_p:
        pitcher_vec = pitcher_vec[:expected_p]
    expected_b = int(nuisance.cfg.batter_profile_dim)
    if len(batter_vec) > expected_b:
        batter_vec = batter_vec[:expected_b]

    # ---- Pitch-level arrays ----
    type_ids_raw = ab_df["type_id"].to_numpy(dtype=np.int64)       # 1..7
    result_ids_raw = ab_df["result_id"].to_numpy(dtype=np.int64)   # 1..7
    count_raw = ab_df["count_state"].to_numpy(dtype=np.int64)      # 0..11
    outs_raw = ab_df["outs_state"].to_numpy(dtype=np.int64)        # 0..2
    runners_raw = ab_df["runners_state"].to_numpy(dtype=np.int64)  # 0..7
    pitch_num_raw = ab_df["pitch_number"].to_numpy(dtype=np.int64) # 1..T

    # Continuous: [velo, spin, plate_x, plate_z]. reindex() yields NaN columns
    # when a synthetic AB (mcsim.state.build_synthetic_ab) lacks velo/spin.
    cont_cols = ["release_speed", "release_spin_rate", "plate_x", "plate_z"]
    cont_raw = ab_df.reindex(columns=cont_cols).to_numpy(dtype=np.float32)
    cont_raw = np.nan_to_num(cont_raw, nan=0.0)

    # ---- Build sequences with start token prepended ----
    type_ids = np.zeros(seq_len, dtype=np.int64)
    type_ids[1:] = type_ids_raw

    continuous = np.zeros((seq_len, 4), dtype=np.float32)
    continuous[1:] = cont_raw
    # Z-score normalize the FULL sequence, start token included. Training
    # normalizes after the dataset's nan->0 fill, so the start token's raw
    # zeros became (0-mean)/std — mirror that exactly or the model sees an
    # input distribution it never trained on.
    continuous = normalize_continuous(continuous, nuisance.cfg)

    result_ids = np.zeros(seq_len, dtype=np.int64)
    result_ids[1:] = result_ids_raw

    count_state = np.zeros(seq_len, dtype=np.int64)
    count_state[0] = int(count_raw[0])  # pre-AB count (same as pitch 1)
    count_state[1:] = count_raw

    outs = np.zeros(seq_len, dtype=np.int64)
    outs[0] = int(outs_raw[0])
    outs[1:] = outs_raw

    runners = np.zeros(seq_len, dtype=np.int64)
    runners[0] = int(runners_raw[0])
    runners[1:] = runners_raw

    pitch_number = np.zeros(seq_len, dtype=np.int64)
    pitch_number[0] = 0  # "no pitch yet"
    pitch_number[1:] = pitch_num_raw

    padding_mask = np.ones(seq_len, dtype=bool)

    # ---- Convert to tensors + replicate ----
    N = n_replicates
    pp = torch.as_tensor(pitcher_vec, dtype=torch.float32).unsqueeze(0).expand(N, -1).contiguous()
    bp = torch.as_tensor(batter_vec, dtype=torch.float32).unsqueeze(0).expand(N, -1).contiguous()

    def _rep(arr):
        t = torch.as_tensor(arr)
        return t.unsqueeze(0).expand(N, *t.shape).contiguous()

    return {
        "pitcher_profile": pp,                             # (N, pitcher_dim)
        "batter_profile": bp,                              # (N, batter_dim)
        "type_ids": _rep(type_ids),                        # (N, seq_len)
        "continuous": _rep(continuous),                     # (N, seq_len, 4)
        "result_ids": _rep(result_ids),                    # (N, seq_len)
        "count_state": _rep(count_state),                  # (N, seq_len)
        "outs": _rep(outs),                                # (N, seq_len)
        "runners": _rep(runners),                          # (N, seq_len)
        "pitch_number": _rep(pitch_number),                # (N, seq_len)
        "padding_mask": _rep(padding_mask),                # (N, seq_len) bool
        # Metadata (not tensors) for downstream use.
        "_n_observed": T,                                  # int
        "_seq_len": seq_len,                               # int
    }
