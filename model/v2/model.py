"""PitchGPTV2 — full model wiring all v2 components together.

Forward pass:
    1. V2InputLayer builds per-pitch tokens from type + continuous + positional.
    2. AdaLNConditioner maps pitcher/batter profiles to per-layer adaLN params.
    3. N stacked V2TransformerBlock layers process the sequence under a causal
       + padding mask, each block's LayerNorms modulated by the conditioner.
    4. A final standard LayerNorm (no adaLN — conditioner influence ends at
       the last block).
    5. TypeHead (weight-tied to the input embedding) emits pitch-type logits.

The GMM head is decoupled from forward() so the caller can first sample a
pitch type and then call predict_continuous() conditioned on that type.  This
mirrors the two-stage generation pattern used in inference and the causal
rollout.

Weight init:
    - All sub-modules initialise themselves (see their respective modules).
    - _scale_residual_init() applies GPT-2-style output-projection scaling:
      each block's attn.out_proj and ffn.fc2 are multiplied by
      1 / sqrt(2 * n_layers) so residual magnitudes stay stable at depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.v2.config import V2Config
from model.v2.adaln import AdaLNConditioner
from model.v2.transformer import V2TransformerBlock, build_causal_mask
from model.v2.embeddings import V2InputLayer
from model.v2.heads import TypeHead, ContinuousGMM


class PitchGPTV2(nn.Module):
    """Full PitchGPT v2 model.

    Args:
        cfg: V2Config — architecture hyper-parameters.

    Inputs (forward):
        pitcher_profile : (B, pitcher_profile_dim)
        batter_profile  : (B, batter_profile_dim)
        type_ids        : (B, T) LongTensor 0..7  — pitch type, PAD=0
        continuous      : (B, T, n_continuous)    — velo, spin, plate_x, plate_z
        result_ids      : (B, T) LongTensor 0..7  — result of each pitch
        count_state     : (B, T) LongTensor 0..11
        outs            : (B, T) LongTensor 0..2
        runners         : (B, T) LongTensor 0..7
        pitch_number    : (B, T) LongTensor 0..14
        padding_mask    : (B, T) bool — True = real pitch, False = padding

    Returns:
        dict with:
            "type_logits" : (B, T, n_pitch_types=8) — raw logits; PAD at idx 0
            "hidden"      : (B, T, d_model)          — final hidden states
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_layer = V2InputLayer(cfg)
        self.adaln_cond  = AdaLNConditioner(cfg)
        self.layers      = nn.ModuleList(
            [V2TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        # Standard LN for the final position — no adaLN needed here since
        # conditioner influence has already been applied in every block.
        self.ln_final = nn.LayerNorm(cfg.d_model)

        # TypeHead reuses the input type embedding matrix (weight tying).
        self.type_head = TypeHead(cfg, self.input_layer.type_emb.weight)

        self.gmm_head = ContinuousGMM(cfg)

        self._scale_residual_init()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _scale_residual_init(self) -> None:
        """GPT-2-style residual scaling.

        Scale the output projections of attention and FFN by
        1 / sqrt(2 * n_layers).  This keeps the residual stream variance
        roughly constant regardless of depth, matching the GPT-2 paper
        (Radford et al., 2019).
        """
        scale = 1.0 / (2 * self.cfg.n_layers) ** 0.5
        for block in self.layers:
            block.attn.out_proj.weight.data.mul_(scale)
            block.ffn.fc2.weight.data.mul_(scale)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pitcher_profile: torch.Tensor,   # (B, pitcher_dim)
        batter_profile: torch.Tensor,    # (B, batter_dim)
        type_ids: torch.Tensor,          # (B, T) LongTensor 0..7
        continuous: torch.Tensor,        # (B, T, n_continuous)
        result_ids: torch.Tensor,        # (B, T) LongTensor 0..7
        count_state: torch.Tensor,       # (B, T) LongTensor 0..11
        outs: torch.Tensor,              # (B, T) LongTensor 0..2
        runners: torch.Tensor,           # (B, T) LongTensor 0..7
        pitch_number: torch.Tensor,      # (B, T) LongTensor 0..14
        padding_mask: torch.Tensor,      # (B, T) bool — True = real pitch
    ) -> dict:
        B, T = type_ids.shape
        device = type_ids.device

        # 1. Build input tokens from type + continuous state + positional.
        x = self.input_layer(
            type_ids, continuous, result_ids, count_state,
            outs, runners, pitch_number,
        )  # (B, T, d_model)

        # 2. Compute per-layer adaLN params from pitcher + batter profiles.
        #    Shape: (B, n_layers, 2, 2, d_model)
        adaln_params = self.adaln_cond(pitcher_profile, batter_profile)

        # 3. Build the combined causal + padding attention mask.
        #    causal: (1, 1, T, T) — lower-triangular bool
        #    pad:    (B, 1, 1, T) — True where key position is a real pitch
        #    Combined: both must be True for attention to be allowed.
        causal = build_causal_mask(T, device)               # (1, 1, T, T)
        pad    = padding_mask.unsqueeze(1).unsqueeze(2)     # (B, 1, 1, T)
        mask   = causal & pad                               # (B, 1, T, T)

        # 4. Run through all transformer layers.
        for i, block in enumerate(self.layers):
            # Slice the conditioner output for this layer.
            # adaln_params[:, i] has shape (B, 2, 2, d_model):
            #   dim[1] — which LN (0=pre-attn, 1=pre-FFN)
            #   dim[2] — 0=gamma, 1=beta
            g1 = adaln_params[:, i, 0, 0, :]   # (B, d_model)  pre-attn gamma
            b1 = adaln_params[:, i, 0, 1, :]   # (B, d_model)  pre-attn beta
            g2 = adaln_params[:, i, 1, 0, :]   # (B, d_model)  pre-FFN gamma
            b2 = adaln_params[:, i, 1, 1, :]   # (B, d_model)  pre-FFN beta
            x  = block(x, mask, g1, b1, g2, b2)

        x = self.ln_final(x)  # (B, T, d_model)

        # 5. Pitch-type logits via weight-tied head.
        type_logits = self.type_head(x)  # (B, T, 8)

        return {"type_logits": type_logits, "hidden": x}

    # ------------------------------------------------------------------
    # Two-stage generation helper
    # ------------------------------------------------------------------

    def predict_continuous(
        self,
        hidden: torch.Tensor,    # (B, T, d_model)
        type_ids: torch.Tensor,  # (B, T) LongTensor — sampled pitch types
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GMM prediction conditioned on a (sampled) pitch type.

        Call this after forward() once you have sampled or chosen a pitch type.
        The type embedding is looked up from the shared input embedding table so
        the GMM sees the same type representation as the trunk.

        Args:
            hidden   : (B, T, d_model) — "hidden" from forward() output dict.
            type_ids : (B, T) LongTensor — pitch type indices (1..7, no PAD).

        Returns:
            log_w  : (B, T, K)       log mixture weights
            mu     : (B, T, K, D)    component means
            log_std: (B, T, K, D)    log standard deviations
        """
        type_emb = self.input_layer.type_emb(type_ids)   # (B, T, d_model)
        return self.gmm_head(hidden, type_emb)            # log_w, mu, log_std
