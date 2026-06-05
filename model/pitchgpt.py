"""Top-level PitchGPT model.

Composes:
  - ContextTokens (3 prepended tokens per AB)
  - FactorEmbeddings (per-pitch sum of factor embeddings)
  - Transformer trunk (pre-norm, causal, cross-AB blockable)
  - PropensityHeads (π̂)
  - ResultHead (μ̂) — receives detached trunk hidden state per ADR 007
  - ABOutcomeHead (terminal pitch only)

The forward signature is intentionally explicit about every input so
that wiring from AtBatDataset → batch → model.forward(...) is
deterministic and shape-audited.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import PitchGPTConfig
from model.embeddings import ContextTokens, FactorEmbeddings
from model.heads import ABOutcomeHead, PropensityHeads, ResultHead
from model.transformer import TransformerBlock, build_attention_mask


class PitchGPT(nn.Module):
    """The model.

    Forward consumes a batch dict (see ``PitchGPTBatch`` schema in
    ``forward()`` docstring) and returns a dict of outputs, plus optional
    intermediates for interpretability hooks.
    """

    N_CONTEXT_TOKENS: int = 3  # pitcher_profile, batter_profile, categorical

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        self.config = config

        self.context = ContextTokens(config)
        self.embed = FactorEmbeddings(config)

        # Per-pitch arsenal feature (ADR 009): project the 14-dim
        # arsenal+has-pitch sub-vector and add it to every pitch token. Small
        # init so it's a meaningful-but-not-dominant contribution at start.
        if config.arsenal_per_pitch:
            self.arsenal_proj = nn.Linear(config.n_arsenal_dims, config.d_model)
            nn.init.normal_(self.arsenal_proj.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(self.arsenal_proj.bias)

        # Optional per-pitch profile injection (see config docstring). Small
        # init so it doesn't swamp the factored pitch embeddings at start.
        if config.inject_profiles_per_pitch:
            self.pitcher_per_pitch_proj = nn.Linear(config.pitcher_profile_dim, config.d_model)
            self.batter_per_pitch_proj = nn.Linear(config.batter_profile_dim, config.d_model)
            nn.init.normal_(self.pitcher_per_pitch_proj.weight, std=config.init_std)
            nn.init.zeros_(self.pitcher_per_pitch_proj.bias)
            nn.init.normal_(self.batter_per_pitch_proj.weight, std=config.init_std)
            nn.init.zeros_(self.batter_per_pitch_proj.bias)

        # Optional head-side profile injection: adds a profile-derived vector
        # to the pitch-position hidden states *before* the propensity head, so
        # the head sees the player fingerprint without depending on the trunk
        # having attended to the context tokens.
        if config.inject_profiles_to_head:
            self.pitcher_head_proj = nn.Linear(config.pitcher_profile_dim, config.d_model)
            self.batter_head_proj = nn.Linear(config.batter_profile_dim, config.d_model)
            nn.init.normal_(self.pitcher_head_proj.weight, std=config.init_std)
            nn.init.zeros_(self.pitcher_head_proj.bias)
            nn.init.normal_(self.batter_head_proj.weight, std=config.init_std)
            nn.init.zeros_(self.batter_head_proj.bias)

        # Optional two-stage propensity TYPE head: small MLP that consumes
        # the trunk hidden plus the CLEAN per-pitch factor embeddings (type,
        # zone, result) of pitch t. Same pattern as the result head — the
        # head doesn't have to extract these from the factored-sum residual.
        if config.propensity_type_two_stage:
            d = config.d_model
            self.type_two_stage_mlp = nn.Sequential(
                nn.Linear(4 * d, d),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(d, config.n_pitch_types),
            )
            for m in self.type_two_stage_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                    nn.init.zeros_(m.bias)

        # Situational two-stage propensity head (ADR 010): a small MLP that
        # fuses the trunk hidden at pitch position t with the CLEAN embeddings
        # of the situation pitch t+1 is thrown in — count[t+1], runners[t+1],
        # outs[t+1] — producing the input the propensity heads (type/zone/velo/
        # spin) read instead of the raw trunk hidden. (count, runners, outs)[t+1]
        # is known at decision time; without this the trunk must re-derive it
        # from (count, result)[t]. Reuses the trunk's count/runners/outs
        # embedding tables (no new embeddings).
        if config.propensity_situational:
            d = config.d_model
            self.situation_fusion = nn.Sequential(
                nn.Linear(4 * d, d),  # [hidden, count_emb, runners_emb, outs_emb]
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(d, d),
            )
            for m in self.situation_fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                    nn.init.zeros_(m.bias)

        # Type-conditioned execution heads (ADR-013). A fusion MLP combines
        # the trunk hidden at position t with the embedding of the NEXT
        # pitch's type, producing the input the execution heads (zone/velo/
        # spin) read. Same pattern as the situational fusion above. The TYPE
        # head is untouched — it reads the raw hidden.
        if config.type_conditioned_heads:
            d = config.d_model
            self.type_fusion = nn.Sequential(
                nn.Linear(2 * d, d),  # [hidden, next_type_emb]
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(d, d),
            )
            for m in self.type_fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                    nn.init.zeros_(m.bias)

        # Full autoregressive execution-head conditioning (ADR-014 Decision 2).
        # Completes ADR-013's type→{zone,velo,spin}-in-parallel into the chain
        #   type → zone|type → velo|type,zone → spin|type,zone,velo → loc|all
        # via small fusion MLPs that mix in each factor's teacher-forced (or, at
        # rollout, sampled) embedding — NOT by re-running the transformer. Each
        # fusion is Linear(d + d → d): [prior-conditioning vector, next factor's
        # d-dim embedding] → next-stage conditioning vector. The zone head itself
        # is left to the ADR-013 type-conditioned path (zone|type), so the AR
        # chain starts at velo.
        if config.autoregressive_exec_heads:
            d = config.d_model
            self.zone_fusion = nn.Linear(d + d, d)   # [hidden_exec(type), zone_emb] -> velo cond
            self.velo_fusion = nn.Linear(d + d, d)   # [velo cond, velo_emb]         -> spin cond
            self.spin_fusion = nn.Linear(d + d, d)   # [spin cond, spin_axis_emb]    -> loc cond
            for lin in (self.zone_fusion, self.velo_fusion, self.spin_fusion):
                nn.init.normal_(lin.weight, mean=0.0, std=config.init_std)
                nn.init.zeros_(lin.bias)

        # Continuous-location MDN head (ADR-014 Decision 1): a mixture of K 2D
        # Gaussians over (plate_x, plate_z), conditioned on the final link of the
        # AR chain. Default OFF so pre-v8 checkpoints reload unchanged.
        if config.location_mdn:
            from model.heads import LocationMDN
            self.location_mdn = LocationMDN(config, d_in=config.d_model)

        # ADR 012 ("fix #2"): FiLM-condition the trunk on the player profile —
        # an MLP maps (pitcher ++ batter) profile → per-layer (gamma, beta),
        # and each transformer block's input is modulated `gamma_l * x + beta_l`,
        # so the profile conditions the whole network (the transformer analogue
        # of the LSTM's profile-in-h0). Initialised to identity (gamma=1,
        # beta=0) so it's a no-op at the start of training.
        if config.profile_film:
            d = config.d_model
            self.profile_film_mlp = nn.Sequential(
                nn.Linear(config.pitcher_profile_dim + config.batter_profile_dim, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, config.n_layers * 2 * d),  # [gamma_0, beta_0, gamma_1, beta_1, ...]
            )
            nn.init.normal_(self.profile_film_mlp[0].weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(self.profile_film_mlp[0].bias)
            nn.init.zeros_(self.profile_film_mlp[2].weight)  # → film_params = bias at init → identity
            with torch.no_grad():
                _b = self.profile_film_mlp[2].bias.view(config.n_layers, 2, d)
                _b[:, 0, :] = 1.0  # gamma = 1
                _b[:, 1, :] = 0.0  # beta = 0

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model)

        # Heads share factor embedding tables with the input embedding layer.
        self.propensity = PropensityHeads(
            config,
            type_emb_weight=self.embed.type_emb.weight,
            zone_emb_weight=self.embed.zone_emb.weight,
        )
        self.result_head = ResultHead(
            config,
            type_emb=self.embed.type_emb,
            zone_emb=self.embed.zone_emb,
            velo_emb=self.embed.velo_emb,
            spin_axis_proj=self.embed.spin_axis_proj if config.spin_axis_circular else None,
            spin_axis_emb=None if config.spin_axis_circular else self.embed.spin_axis_emb,
        )
        # AB-outcome head (ADR-014 Decision 4): redundant with the cascade +
        # RE24 in v8, so it's gated off there. Default ON for v7 back-compat.
        if config.ab_outcome_head:
            self.ab_outcome = ABOutcomeHead(config)

        # GPT-2-style residual init scaling (helps deeper networks)
        self._scale_residual_init()

    def _scale_residual_init(self):
        """Per-GPT-2: scale the output projections by 1/sqrt(2 * n_layers)
        for the residual-stream-affecting weights. Reduces variance growth
        with depth."""
        scale = 1.0 / (2 * self.config.n_layers) ** 0.5
        for block in self.layers:
            # out_proj of attention writes to residual stream
            block.attn.out_proj.weight.data.mul_(scale)
            # fc2 of FFN writes to residual stream
            block.ffn.fc2.weight.data.mul_(scale)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_attention_caching(self, enabled: bool) -> None:
        """Enable/disable per-layer attention weight caching. Off during
        training (memory + speed); on during interpretability runs."""
        for block in self.layers:
            block.attn._cache_attention = enabled

    def forward(
        self,
        pitcher_profile: torch.Tensor,        # (B, pitcher_profile_dim)
        batter_profile: torch.Tensor,         # (B, batter_profile_dim)
        categorical_context: dict[str, torch.Tensor],  # each (B,) LongTensor
        pitch_factors: dict[str, torch.Tensor],  # each (B, T) — see FactorEmbeddings
        intended_actions: dict[str, torch.Tensor],  # for result head, (B, T)
        padding_mask: torch.Tensor | None = None,  # (B, T) True=real, False=pad
        arsenal: torch.Tensor | None = None,  # (B, n_arsenal_dims) — required if config.arsenal_per_pitch
        return_intermediates: bool = False,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass. Returns logits for all heads.

        ``pitch_factors`` keys: type, zone, velo, spin_rate, spin_axis,
            result, count, runners, outs, pos.
        ``intended_actions`` keys: type, zone, velo, spin_axis (the
            factors that the result head conditions on). At training,
            these are the next pitch's actual factors (teacher forcing).
            At counterfactual rollout, they're the intervened values.

        Returns a dict with:
            propensity: dict (B, T_total, ...) for each factor head
            result:    (B, T_total, n_result_logits)
            ab_outcome: (B, n_ab_outcome_classes) — only at terminal pitch
                (caller responsible for terminal selection)
            (if return_intermediates) intermediates: list of hidden states
                after each transformer layer.

        T_total = N_CONTEXT_TOKENS + T (pitches). The first N_CONTEXT_TOKENS
        positions of the output sequences correspond to context tokens;
        callers should slice off these when computing per-pitch losses.
        """
        # 1. Build context tokens (B, 3, d_model)
        ctx_tokens = self.context(pitcher_profile, batter_profile, categorical_context)

        # 2. Build pitch tokens (B, T, d_model)
        pitch_tokens = self.embed(pitch_factors)

        # 2a. Per-pitch arsenal feature (ADR 009): add the projected
        #     arsenal+has-pitch vector to every pitch token, so the pitcher's
        #     pitch mix is in the residual stream at every position.
        if self.config.arsenal_per_pitch:
            if arsenal is None:
                raise ValueError(
                    "PitchGPT.forward: config.arsenal_per_pitch is set but no "
                    f"`arsenal` tensor was passed (expected shape (B, {self.config.n_arsenal_dims}))."
                )
            pitch_tokens = pitch_tokens + self.arsenal_proj(arsenal).unsqueeze(1)  # (B,1,d) → broadcasts over T

        # 2b. Optionally inject the (standardized) player profiles into every
        #     pitch token, so the heads can read the fingerprint directly.
        if self.config.inject_profiles_per_pitch:
            p_inj = self.pitcher_per_pitch_proj(pitcher_profile).unsqueeze(1)  # (B,1,d)
            b_inj = self.batter_per_pitch_proj(batter_profile).unsqueeze(1)    # (B,1,d)
            pitch_tokens = pitch_tokens + p_inj + b_inj

        # 3. Concatenate along sequence dim: (B, 3+T, d_model)
        x = torch.cat([ctx_tokens, pitch_tokens], dim=1)
        T_total = x.shape[1]

        # 4. Build attention mask. Context tokens are always "real" (never
        #    padding), so the padding mask is extended with True at the
        #    front.
        if padding_mask is not None:
            B = padding_mask.shape[0]
            ctx_pad = torch.ones(
                B, self.N_CONTEXT_TOKENS, dtype=torch.bool, device=padding_mask.device
            )
            full_pad = torch.cat([ctx_pad, padding_mask], dim=1)
        else:
            full_pad = None

        attn_mask = build_attention_mask(
            seq_len=T_total,
            padding_mask=full_pad,
            ab_boundaries=None,  # single AB per batch row by default; packing TBD
            device=x.device,
        )

        # 5. Trunk — optionally FiLM-modulated per layer by the player profile (ADR 012).
        if self.config.profile_film:
            film = self.profile_film_mlp(
                torch.cat([pitcher_profile, batter_profile], dim=-1)
            ).view(x.shape[0], self.config.n_layers, 2, self.config.d_model)
        intermediates = []
        for li, block in enumerate(self.layers):
            if self.config.profile_film:
                x = film[:, li, 0, :].unsqueeze(1) * x + film[:, li, 1, :].unsqueeze(1)  # gamma_l * x + beta_l
            x = block(x, attn_mask)
            if return_intermediates:
                intermediates.append(x)
        x = self.ln_final(x)

        # 6. Propensity heads. Optionally modify the pitch-position hidden
        #    states before the heads read them:
        #      (a) ADR 010 `propensity_situational`: fuse with the clean
        #          embeddings of the situation pitch t+1 is thrown in —
        #          count[t+1], runners[t+1], outs[t+1] (known at decision time;
        #          without this the trunk must re-derive it from
        #          (count, result)[t]).
        #      (b) `inject_profiles_to_head` (experimental): add a
        #          profile-derived vector.
        if self.config.propensity_situational or self.config.inject_profiles_to_head:
            NC = self.N_CONTEXT_TOKENS
            x_for_prop = x.clone()

            if self.config.propensity_situational:
                def _shift_left(v: torch.Tensor) -> torch.Tensor:
                    # position t ← value at t+1; last position ← 0 (loss-ignored)
                    return torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], dim=1)
                fused = self.situation_fusion(torch.cat([
                    x_for_prop[:, NC:, :],
                    self.embed.count_emb(_shift_left(pitch_factors["count"])),
                    self.embed.runners_emb(_shift_left(pitch_factors["runners"])),
                    self.embed.outs_emb(_shift_left(pitch_factors["outs"])),
                ], dim=-1))  # (B, T, d)
                x_for_prop[:, NC:, :] = fused

            if self.config.inject_profiles_to_head:
                head_inj = (
                    self.pitcher_head_proj(pitcher_profile)
                    + self.batter_head_proj(batter_profile)
                ).unsqueeze(1)  # (B, 1, d)
                x_for_prop[:, NC:, :] = x_for_prop[:, NC:, :] + head_inj

            propensity_logits = self.propensity(x_for_prop)
        else:
            propensity_logits = self.propensity(x)

        # 6a. Type-conditioned execution heads (ADR-013). Fuse the pitch-
        #     position hidden with the NEXT pitch's type embedding. type[t+1]
        #     is the shift-left of pitch_factors["type"] (last position -> PAD,
        #     loss-ignored). At rollout the caller writes the sampled/intervened
        #     type into pitch_factors["type"], so the same path serves do(.).
        if self.config.type_conditioned_heads:
            NC = self.N_CONTEXT_TOKENS
            def _shift_left_type(v: torch.Tensor) -> torch.Tensor:
                return torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], dim=1)
            next_type = _shift_left_type(pitch_factors["type"])
            next_type_emb = self.embed.type_emb(next_type)
            uses_xprop = self.config.propensity_situational or self.config.inject_profiles_to_head
            hidden_for_heads = x_for_prop if uses_xprop else x
            base_hidden = hidden_for_heads[:, NC:, :]
            hidden_exec_pitch = self.type_fusion(
                torch.cat([base_hidden, next_type_emb], dim=-1)
            )
            hidden_exec_full = hidden_for_heads.clone()
            hidden_exec_full[:, NC:, :] = hidden_exec_pitch
            propensity_logits = self.propensity(hidden_for_heads, hidden_exec=hidden_exec_full)

        # 6b. Optional two-stage propensity TYPE head — overrides the type
        # logits at pitch positions with an MLP that reads (hidden, type_emb,
        # zone_emb, result_emb) of pitch t.
        if self.config.propensity_type_two_stage:
            pitch_hidden_for_prop = x[:, self.N_CONTEXT_TOKENS:, :]  # (B, T, d)
            type_e = self.embed.type_emb(pitch_factors["type"])
            zone_e = self.embed.zone_emb(pitch_factors["zone"])
            result_e = self.embed.result_emb(pitch_factors["result"])
            combined = torch.cat(
                [pitch_hidden_for_prop, type_e, zone_e, result_e], dim=-1
            )  # (B, T, 4*d)
            type_logits_two_stage = self.type_two_stage_mlp(combined)  # (B, T, n_pitch_types)
            # Replace the type logits at pitch positions (leave context-token
            # positions as the weight-tied output; they're ignored by the loss
            # via the per_pitch slicing in compute_losses).
            propensity_logits_full = dict(propensity_logits)
            type_full = propensity_logits_full["type"].clone()
            type_full[:, self.N_CONTEXT_TOKENS:, :] = type_logits_two_stage
            propensity_logits_full["type"] = type_full
            propensity_logits = propensity_logits_full

        # 6c. Full autoregressive execution-head conditioning + location MDN
        #     (ADR-014 Decision 2). Completes the ADR-013 chain:
        #       type → zone|type → velo|type,zone → spin|type,zone,velo
        #              → location-MDN|type,zone,velo,spin.
        #     The zone head stays as the ADR-013 type-conditioned output (zone|
        #     type); this block recomputes velo/spin (and the MDN) so each
        #     conditions on the prior factors of the SAME predicted pitch.
        #
        #     Convention: the propensity heads predict pitch t+1 (targets are the
        #     left-shift of the input factors; see model/pitchgpt_dataset.py and
        #     scripts/train_pitchgpt.compute_losses). So the conditioning factor
        #     embeddings are the LEFT-SHIFT of pitch_factors — zone[t+1] etc., the
        #     other factors of the pitch being predicted — exactly as the ADR-013
        #     path conditions on next_type = shift_left(type). At rollout the
        #     caller writes the sampled/intervened factors into pitch_factors, so
        #     the same path serves do(.). Last position's shifted factor is PAD/0
        #     (loss-ignored).
        #
        #     Precedence: when autoregressive_exec_heads is ON, the AR path takes
        #     precedence over the ADR-013-only velo/spin (it consumes the ADR-013
        #     type-conditioned hidden as its base, then conditions further). When
        #     it is OFF, the ADR-013-only path above is left untouched. If AR is
        #     on but type_conditioned_heads is off, hidden_exec_full does not
        #     exist; we fall back to the plain pitch-position hidden so the chain
        #     still starts from a valid (un-type-conditioned) base.
        spin_cond = None
        if self.config.autoregressive_exec_heads or self.config.location_mdn:
            NC = self.N_CONTEXT_TOKENS

            def _shift_left(v: torch.Tensor) -> torch.Tensor:
                # position t ← value at t+1; last position ← 0 (loss-ignored)
                return torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], dim=1)

            # Base for the chain: the ADR-013 type-conditioned full hidden if it
            # was built, else the plain pitch-position hidden (same selection the
            # ADR-013 path uses for hidden_for_heads).
            if self.config.type_conditioned_heads:
                ar_base_full = hidden_exec_full  # (B, T_total, d) — type-conditioned
            else:
                uses_xprop = (
                    self.config.propensity_situational
                    or self.config.inject_profiles_to_head
                )
                ar_base_full = x_for_prop if uses_xprop else x
            ar_base_pitch = ar_base_full[:, NC:, :]  # (B, T, d)

        if self.config.autoregressive_exec_heads:
            # velo | type, zone : fuse the (type-conditioned) base with zone[t+1].
            ze = self.embed.zone_emb(_shift_left(pitch_factors["zone"]))  # (B, T, d)
            velo_cond = torch.relu(
                self.zone_fusion(torch.cat([ar_base_pitch, ze], dim=-1))
            )  # (B, T, d)
            # spin | type, zone, velo : fuse velo_cond with velo[t+1].
            ve = self.embed.velo_emb(_shift_left(pitch_factors["velo"]))  # (B, T, d)
            spin_cond = torch.relu(
                self.velo_fusion(torch.cat([velo_cond, ve], dim=-1))
            )  # (B, T, d)
            # Overwrite the velo/spin logits at pitch positions; context-token
            # positions are loss-ignored, so we only need pitch positions valid.
            velo_full = propensity_logits["velo"].clone()
            spin_rate_full = propensity_logits["spin_rate"].clone()
            spin_axis_full = propensity_logits["spin_axis"].clone()
            velo_full[:, NC:, :] = self.propensity.velo_proj(velo_cond)
            spin_rate_full[:, NC:, :] = self.propensity.spin_rate_proj(spin_cond)
            spin_axis_full[:, NC:, :] = self.propensity.spin_axis_proj(spin_cond)
            propensity_logits = dict(propensity_logits)
            propensity_logits["velo"] = velo_full
            propensity_logits["spin_rate"] = spin_rate_full
            propensity_logits["spin_axis"] = spin_axis_full

        if self.config.location_mdn:
            # location | type, zone, velo, spin : fuse the spin-conditioning
            # vector (or the base hidden if AR is off) with spin_axis[t+1].
            sa = _shift_left(pitch_factors["spin_axis"])
            sae = (
                self.embed.spin_axis_proj(sa)        # (B, T, d) — Linear(2 -> d)
                if self.config.spin_axis_circular
                else self.embed.spin_axis_emb(sa)    # (B, T, d)
            )
            base = spin_cond if spin_cond is not None else ar_base_pitch
            if hasattr(self, "spin_fusion"):
                loc_cond = torch.relu(
                    self.spin_fusion(torch.cat([base, sae], dim=-1))
                )  # (B, T, d)
            else:
                # AR off but MDN on: no spin_fusion layer; condition the MDN on
                # the base hidden directly (spin_axis embedding unused here).
                loc_cond = base
            log_w, mu, log_std = self.location_mdn(loc_cond)  # pitch positions only
            location_mdn_out = {"log_w": log_w, "mu": mu, "log_std": log_std}
        else:
            location_mdn_out = None

        # 7. Result head — shifted-hidden + intended action (per ADR 007 Amendment).
        #
        # The trunk's hidden state at pitch position ``t`` is computed from a
        # token whose embedding *sums in* ``result_emb(result_t)``, so feeding
        # ``hidden[t]`` to the result head creates a leakage shortcut:
        # ``hidden[t]`` already encodes the answer the head is supposed to
        # predict. Empirically this causes the head to ignore
        # ``intended_action`` and silently breaks counterfactual rollout (the
        # exact failure mode this two-stage architecture was meant to prevent).
        #
        # Fix: at pitch position ``t``, the result head reads ``hidden[t-1]``
        # (history through pitch t-1, which does NOT include result_t in its
        # input embedding) and ``intended_action[t]`` (the action being taken
        # at pitch t). At t=0 there is no hidden[-1]; we use the last context
        # token's hidden state as the "history through no pitches" surrogate.
        pitch_hidden = x[:, self.N_CONTEXT_TOKENS:, :]  # (B, T, d_model)
        ctx_last_hidden = x[:, self.N_CONTEXT_TOKENS - 1: self.N_CONTEXT_TOKENS, :]
        hidden_for_result = torch.cat(
            [ctx_last_hidden, pitch_hidden[:, :-1, :]], dim=1
        )  # (B, T, d_model) — hidden_for_result[t] encodes history through pitch t-1.
        result_logits = self.result_head(
            hidden_for_result.detach(),  # ADR 007: stop-gradient
            intended_actions["type"],
            intended_actions["zone"],
            intended_actions["velo"],
            intended_actions["spin_axis"],
        )

        out = {
            "propensity": propensity_logits,    # dict of (B, 3+T, ...) — but only pitch positions are valid
            "result": result_logits,            # (B, T, n_result_logits) — pitch positions only
        }

        # 8. AB-outcome head: applied to terminal-pitch hidden. The caller knows
        #    which position is terminal (depends on padding); we return logits at
        #    every pitch position, and the caller selects. Optional (ADR-014
        #    Decision 4): dropped in v8 via config.ab_outcome_head=False, in which
        #    case the key is absent from the output.
        if self.config.ab_outcome_head:
            out["ab_outcome_per_pos"] = self.ab_outcome(pitch_hidden)  # (B, T, n_ab_outcome_classes)

        # 9. Location MDN params at pitch positions (ADR-014 Decision 1). Present
        #    only when config.location_mdn is on; the loss/rollout consume these.
        if location_mdn_out is not None:
            out["location_mdn"] = location_mdn_out  # {log_w (B,T,K), mu (B,T,K,2), log_std (B,T,K,2)}

        if return_hidden:
            out["hidden"] = x  # (B, T_total, d_model) — post-ln_final trunk hidden
        if return_intermediates:
            out["intermediates"] = intermediates
        return out

    @torch.no_grad()
    def sample_location_mdn_for_rollout(
        self,
        hidden_at_pos: torch.Tensor,
        type_ids: torch.Tensor,
        zone_ids: torch.Tensor,
        velo_ids: torch.Tensor,
        spin_axis: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run the AR fusion chain + MDN on a single set of sampled factors.

        At rollout time, the trunk hidden comes from the forward pass (captured
        externally), and the factor ids have just been sampled. This runs only
        the small fusion MLPs — no transformer re-forward.

        Args:
            hidden_at_pos: (N, d_model) post-ln_final trunk hidden at the
                prediction position.
            type_ids: (N,) sampled type ids (with TYPE_ID_OFFSET, 1..7).
            zone_ids, velo_ids: (N,) sampled factor ids.
            spin_axis: (N, 2) sin/cos spin axis.
            generator: torch RNG for reproducible sampling.

        Returns:
            (N, 2) sampled (plate_x, plate_z).
        """
        if not self.config.location_mdn:
            raise RuntimeError("sample_location_mdn_for_rollout requires location_mdn")
        te = self.embed.type_emb(type_ids)
        if self.config.type_conditioned_heads:
            h = self.type_fusion(torch.cat([hidden_at_pos, te], -1))
        else:
            h = hidden_at_pos

        if self.config.autoregressive_exec_heads:
            ze = self.embed.zone_emb(zone_ids)
            velo_cond = torch.relu(self.zone_fusion(torch.cat([h, ze], -1)))
            ve = self.embed.velo_emb(velo_ids)
            spin_cond = torch.relu(self.velo_fusion(torch.cat([velo_cond, ve], -1)))
            sae = (self.embed.spin_axis_proj(spin_axis)
                   if self.config.spin_axis_circular
                   else self.embed.spin_axis_emb(spin_axis))
            loc_cond = torch.relu(self.spin_fusion(torch.cat([spin_cond, sae], -1)))
        else:
            loc_cond = h

        log_w, mu, log_std = self.location_mdn(loc_cond.unsqueeze(1))
        sampled = self.location_mdn.sample(log_w, mu, log_std, generator=generator)
        return sampled.squeeze(1)

    @torch.no_grad()
    def execution_logits_for_type(
        self, batch: dict, type_id: int
    ) -> dict[str, torch.Tensor]:
        """Execution-head logits (zone/velo/spin) with the next-pitch type
        clamped to ``type_id`` at every position.

        For inference marginalization: call once per pitch type, weight each
        by π̂(type|h), and sum. ``type_id`` is the model-side type id
        (1..7 = PITCH_TYPES; see data.dataset.MODEL_TYPE_ID). Requires
        ``config.type_conditioned_heads``.
        """
        if not self.config.type_conditioned_heads:
            raise RuntimeError("execution_logits_for_type requires type_conditioned_heads")
        pf = {k: v for k, v in batch["pitch_factors"].items()}
        clamped = torch.full_like(pf["type"], int(type_id))
        # keep PAD positions as PAD (id 0) so shift-left stays well-defined
        clamped = torch.where(pf["type"] == 0, pf["type"], clamped)
        pf["type"] = clamped
        out = self.forward(
            pitcher_profile=batch["pitcher_profile"],
            batter_profile=batch["batter_profile"],
            categorical_context=batch["categorical_context"],
            pitch_factors=pf,
            intended_actions=batch["intended_actions"],
            padding_mask=batch.get("padding_mask"),
            arsenal=batch.get("arsenal"),
        )
        return {
            "zone": out["propensity"]["zone"],
            "velo": out["propensity"]["velo"],
            "spin_rate": out["propensity"]["spin_rate"],
            "spin_axis": out["propensity"]["spin_axis"],
        }
