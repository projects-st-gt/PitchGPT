"""
adaLN (Adaptive Layer Normalization) conditioning for PitchGPT v2.

Based on DiT (Peebles & Xie, ICCV 2023): adaLN-Zero initialization means the
final MLP linear layer has zero weights and biases set so gamma=1, beta=0 at
init. The model therefore starts as a plain transformer and gradually learns
pitcher-specific modulation.

AdaLNConditioner: pitcher + batter profiles → per-layer (gamma, beta) tensors
AdaLayerNorm:     applies external (gamma, beta) instead of learned affine params
"""

import torch
import torch.nn as nn

from model.v2.config import V2Config


class AdaLNConditioner(nn.Module):
    """
    Maps (pitcher_profile, batter_profile) to per-layer adaLN parameters.

    Input:
        pitcher_profile: (B, pitcher_profile_dim)  — e.g. 223 dims
        batter_profile:  (B, batter_profile_dim)   — e.g.  91 dims

    Output:
        (B, n_layers, 2, 2, d_model)
        dim 1 — layer index 0..n_layers-1
        dim 2 — which LN in the block (0=pre-attn, 1=pre-FFN)
        dim 3 — 0=gamma, 1=beta
        dim 4 — d_model feature dimension
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        in_dim = cfg.pitcher_profile_dim + cfg.batter_profile_dim  # 314
        # Output: n_layers * 2 LNs per block * 2 params (gamma, beta) * d_model
        out_dim = cfg.n_layers * 2 * 2 * cfg.d_model

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.adaln_hidden),
            nn.GELU(),
            nn.Linear(cfg.adaln_hidden, out_dim),
        )

        # adaLN-Zero init: zero out the final linear's weights; set bias so
        # gamma=1 and beta=0 at initialization.
        # The output layout (per d_model slice) is:
        #   [gamma_0_pre_attn, gamma_0_pre_ffn, ..., beta_0_pre_attn, ...]
        # After reshape to (n_layers, 2, 2, d_model):
        #   [:, :, 0, :] = gamma → target 1.0
        #   [:, :, 1, :] = beta  → target 0.0
        final_linear: nn.Linear = self.mlp[-1]  # type: ignore[assignment]
        nn.init.zeros_(final_linear.weight)
        with torch.no_grad():
            bias = final_linear.bias  # shape (out_dim,)
            # Reshape to (n_layers, 2, 2, d_model) to set gamma slice to 1.
            bias_view = bias.view(cfg.n_layers, 2, 2, cfg.d_model)
            bias_view[:, :, 0, :] = 1.0   # gamma = 1
            bias_view[:, :, 1, :] = 0.0   # beta  = 0

        self._out_shape = (cfg.n_layers, 2, 2, cfg.d_model)

    def forward(
        self,
        pitcher_profile: torch.Tensor,
        batter_profile: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pitcher_profile: (B, 223)
            batter_profile:  (B, 91)

        Returns:
            (B, n_layers, 2, 2, d_model)
        """
        x = torch.cat([pitcher_profile, batter_profile], dim=-1)  # (B, 314)
        out = self.mlp(x)                                          # (B, out_dim)
        B = out.shape[0]
        return out.view(B, *self._out_shape)


class AdaLayerNorm(nn.Module):
    """
    LayerNorm with external (gamma, beta) conditioning instead of learned affine.

        adaLN(x, gamma, beta) = gamma * LayerNorm(x) + beta

    The underlying nn.LayerNorm uses elementwise_affine=False so it only
    computes the normalized values; scaling and shifting come entirely from
    the conditioner.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     (B, T, d_model)
            gamma: (B, d_model)  — broadcast over T
            beta:  (B, d_model)  — broadcast over T

        Returns:
            (B, T, d_model)
        """
        normed = self.norm(x)                        # (B, T, d_model)
        # gamma/beta are (B, d_model); unsqueeze to (B, 1, d_model) for broadcast
        return gamma.unsqueeze(1) * normed + beta.unsqueeze(1)
