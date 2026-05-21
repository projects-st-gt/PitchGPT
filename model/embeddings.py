"""Factored embeddings + context-token construction for PitchGPT.

Two main modules:

- ``FactorEmbeddings``: sums per-factor embeddings into a single token
  per pitch. Each factor has its own small table; total embedding params
  are ~113 × d_model rather than (35M × d_model) for a flat vocabulary.
- ``ContextTokens``: builds the three "system prompt" tokens prepended to
  each at-bat sequence — pitcher profile, batter profile, categorical
  confounders.

Lock-in decisions per the architecture brainstorm:

- **Spin axis uses sin/cos continuous encoding** (circular by nature),
  projected through ``Linear(2, d_model)`` rather than a categorical
  embedding table. Preserves the topology that 0° == 360°.
- **pos vocab is 15** per the pitchgpt-model skill — pitch index within
  the at-bat. Context tokens DO NOT get a positional embedding; they're
  distinguished by their construction (different feature inputs) and by
  the trunk's attention mask treating them as the head of the sequence.
- **LayerNorm at the end of each context-token construction** puts all
  three context tokens on the same scale as the per-pitch tokens, which
  matters for the trunk's first attention pass to mix them coherently.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import PitchGPTConfig


class FactorEmbeddings(nn.Module):
    """Per-pitch token = sum of factor embeddings + positional embedding.

    Input is a dict of LongTensors, one per factor, each of shape
    ``(batch, n_pitches)``. Spin axis is special: it's a FloatTensor of
    shape ``(batch, n_pitches, 2)`` containing ``[sin(axis), cos(axis)]``
    if ``config.spin_axis_circular`` (the default), else a LongTensor of
    categorical bins.

    Output: ``(batch, n_pitches, d_model)``.
    """

    # The 11 per-pitch factors, in a fixed order (used for concat-then-project).
    FACTOR_ORDER: tuple[str, ...] = (
        "type", "zone", "velo", "spin_rate", "spin_axis", "result",
        "count", "runners", "outs", "pos", "pitcher_fatigue",
    )
    N_FACTORS: int = len(FACTOR_ORDER)  # 11

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # PAD-aware embeddings for type and result (pad_idx=0 zeros out)
        self.type_emb = nn.Embedding(config.n_pitch_types, d, padding_idx=0)
        self.result_emb = nn.Embedding(config.n_result_classes, d, padding_idx=0)

        # Plain embeddings for the other factors. "MISSING" / "PAD" cells
        # are part of the vocab; we don't use padding_idx so they're treated
        # as informative tokens (e.g. MISSING velo is itself a signal).
        self.zone_emb = nn.Embedding(config.n_zones, d)
        self.velo_emb = nn.Embedding(config.n_velo_bins, d)
        self.spin_rate_emb = nn.Embedding(config.n_spin_rate_bins, d)
        self.count_emb = nn.Embedding(config.n_count_states, d)
        self.runners_emb = nn.Embedding(config.n_runner_states, d)
        self.outs_emb = nn.Embedding(config.n_outs, d)
        self.pos_emb = nn.Embedding(config.n_positions, d)
        # ADR 003 Amendment 1 (2026-05-10): pitcher in-game fatigue as a per-pitch
        # bucketed categorical (vocab 12: 0-9, 10-19, ..., 100+, PAD).
        self.pitcher_fatigue_emb = nn.Embedding(config.n_pitcher_fatigue_buckets, d)

        # Spin axis: continuous-circular by default
        if config.spin_axis_circular:
            self.spin_axis_proj = nn.Linear(2, d)
        else:
            self.spin_axis_emb = nn.Embedding(config.n_spin_axis_bins, d)

        # ADR 011 ("fix #1"): concat-then-project instead of summing the 11
        # per-pitch factor embeddings. Each factor's d_model embedding is
        # down-projected to d//11 dims, the 11 are concatenated (each gets a
        # dedicated input sub-space), then a Linear mixes them back to d_model
        # — a *learned* mixing rather than a forced equal sum. The embedding
        # tables stay d_model-dim, so weight-tied heads / the result head are
        # untouched.
        if config.concat_then_project:
            self._d_factor = d // self.N_FACTORS  # e.g. 23 for d=256, 46 for d=512
            # ModuleList (not ModuleDict) — factor names like "type" collide
            # with nn.Module's reserved attribute names. Indexed by FACTOR_ORDER.
            self.factor_down = nn.ModuleList(
                [nn.Linear(d, self._d_factor) for _ in self.FACTOR_ORDER]
            )
            self.factor_mixer = nn.Linear(self.N_FACTORS * self._d_factor, d)

        self._init_weights()

    def _init_weights(self):
        std = self.config.init_std
        for emb in [
            self.type_emb, self.zone_emb, self.velo_emb, self.spin_rate_emb,
            self.result_emb, self.count_emb, self.runners_emb, self.outs_emb,
            self.pos_emb, self.pitcher_fatigue_emb,
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=std)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()
        if self.config.spin_axis_circular:
            nn.init.normal_(self.spin_axis_proj.weight, mean=0.0, std=std)
            nn.init.zeros_(self.spin_axis_proj.bias)
        else:
            nn.init.normal_(self.spin_axis_emb.weight, mean=0.0, std=std)
        if self.config.concat_then_project:
            for lin in list(self.factor_down) + [self.factor_mixer]:
                nn.init.normal_(lin.weight, mean=0.0, std=std)
                nn.init.zeros_(lin.bias)

    def forward(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of factor embeddings + positional embedding.

        Required keys in ``factors``: ``type, zone, velo, spin_rate,
        spin_axis, result, count, runners, outs, pos, pitcher_fatigue``.

        ``spin_axis``:
        - if circular: FloatTensor of shape ``(B, T, 2)`` with ``[sin, cos]``
        - else: LongTensor of shape ``(B, T)`` with bin indices

        All other entries: LongTensor ``(B, T)``.
        """
        # Per-factor d_model embeddings (all 11 — needed for both the sum and
        # the concat-then-project path).
        emb = {
            "type": self.type_emb(factors["type"]),
            "zone": self.zone_emb(factors["zone"]),
            "velo": self.velo_emb(factors["velo"]),
            "spin_rate": self.spin_rate_emb(factors["spin_rate"]),
            "spin_axis": (
                self.spin_axis_proj(factors["spin_axis"])
                if self.config.spin_axis_circular
                else self.spin_axis_emb(factors["spin_axis"])
            ),
            "result": self.result_emb(factors["result"]),
            "count": self.count_emb(factors["count"]),
            "runners": self.runners_emb(factors["runners"]),
            "outs": self.outs_emb(factors["outs"]),
            "pos": self.pos_emb(factors["pos"]),
            "pitcher_fatigue": self.pitcher_fatigue_emb(factors["pitcher_fatigue"]),
        }
        if self.config.concat_then_project:  # ADR 011
            small = [self.factor_down[i](emb[name]) for i, name in enumerate(self.FACTOR_ORDER)]
            return self.factor_mixer(torch.cat(small, dim=-1))
        # Default: sum the 11 factor embeddings.
        x = emb[self.FACTOR_ORDER[0]]
        for name in self.FACTOR_ORDER[1:]:
            x = x + emb[name]
        return x


class ContextTokens(nn.Module):
    """Three context tokens prepended to each AB sequence.

    - token 0: pitcher profile (223-dim) → MLP → d_model → LayerNorm
    - token 1: batter profile (91-dim) → MLP → d_model → LayerNorm
    - token 2: sum of categorical confounder embeddings → LayerNorm

    Output: ``(batch, 3, d_model)``.
    """

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # Profile MLPs (expansion factor 4)
        self.pitcher_mlp = nn.Sequential(
            nn.Linear(config.pitcher_profile_dim, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.batter_mlp = nn.Sequential(
            nn.Linear(config.batter_profile_dim, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

        # Categorical confounder embeddings (summed).
        # Per ADR 003 Amendment 1 (2026-05-10): the derived `leverage` embedding
        # was replaced by raw state components (`inning`, `score_diff`,
        # `inning_half`) — strictly sufficient for adjustment, captures
        # score-sign asymmetry that scalar LI loses.
        self.p_throws_emb = nn.Embedding(config.n_p_throws, d)
        self.stand_emb = nn.Embedding(config.n_stand, d)
        self.ballpark_emb = nn.Embedding(config.n_ballparks, d)
        self.umpire_emb = nn.Embedding(config.n_umpires, d)
        self.catcher_emb = nn.Embedding(config.n_catchers, d)
        self.inning_emb = nn.Embedding(config.n_inning_buckets, d)
        self.score_diff_emb = nn.Embedding(config.n_score_diff_buckets, d)
        self.inning_half_emb = nn.Embedding(config.n_inning_half, d)
        self.days_rest_emb = nn.Embedding(config.n_days_rest_buckets, d)
        self.tto_emb = nn.Embedding(config.n_tto_buckets, d)
        self.temp_emb = nn.Embedding(config.n_temp_buckets, d)
        self.roof_emb = nn.Embedding(config.n_roof, d)

        # Per-context-token LayerNorm
        self.pitcher_ln = nn.LayerNorm(d)
        self.batter_ln = nn.LayerNorm(d)
        self.categorical_ln = nn.LayerNorm(d)

        self._init_weights()

    def _init_weights(self):
        std = self.config.init_std
        for emb in [
            self.p_throws_emb, self.stand_emb, self.ballpark_emb,
            self.umpire_emb, self.catcher_emb,
            self.inning_emb, self.score_diff_emb, self.inning_half_emb,
            self.days_rest_emb, self.tto_emb, self.temp_emb, self.roof_emb,
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=std)
        for m in self.pitcher_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)
        for m in self.batter_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        pitcher_profile: torch.Tensor,    # (B, pitcher_profile_dim)
        batter_profile: torch.Tensor,     # (B, batter_profile_dim)
        categorical: dict[str, torch.Tensor],  # each (B,) LongTensor
    ) -> torch.Tensor:
        """Build the three context tokens. Output shape (B, 3, d_model)."""
        pitcher_tok = self.pitcher_ln(self.pitcher_mlp(pitcher_profile))
        batter_tok = self.batter_ln(self.batter_mlp(batter_profile))

        cat_sum = (
            self.p_throws_emb(categorical["p_throws"])
            + self.stand_emb(categorical["stand"])
            + self.ballpark_emb(categorical["ballpark"])
            + self.umpire_emb(categorical["umpire"])
            + self.catcher_emb(categorical["catcher"])
            + self.inning_emb(categorical["inning"])
            + self.score_diff_emb(categorical["score_diff"])
            + self.inning_half_emb(categorical["inning_half"])
            + self.days_rest_emb(categorical["days_rest"])
            + self.tto_emb(categorical["tto"])
            + self.temp_emb(categorical["temp"])
            + self.roof_emb(categorical["roof"])
        )
        cat_tok = self.categorical_ln(cat_sum)

        # (B, 3, d_model)
        return torch.stack([pitcher_tok, batter_tok, cat_tok], dim=1)
