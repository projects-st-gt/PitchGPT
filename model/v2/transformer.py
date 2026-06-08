"""Pre-norm transformer block with adaLN conditioning for PitchGPT v2.

GPT-2-style pre-norm pattern, but LayerNorm is replaced by AdaLayerNorm so
pitcher/batter profile tensors can modulate every block independently.

    x = x + Attn(adaLN(x, γ₁, β₁), mask)
    x = x + FFN(adaLN(x, γ₂, β₂))

γ/β come from AdaLNConditioner, one pair per block per LN position.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2.adaln import AdaLayerNorm
from model.v2.config import V2Config


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular boolean causal mask.

    Returns:
        (1, 1, T, T) BoolTensor where True = position is allowed to attend,
        False = position is blocked (future token).

    Example for T=3:
        [[True,  False, False],
         [True,  True,  False],
         [True,  True,  True ]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def _bool_to_additive(mask: torch.Tensor) -> torch.Tensor:
    """Convert (*, T, T) boolean mask (True=attend) to additive float mask.

    0.0 where allowed, -inf where blocked — matches the convention used
    throughout the v9 transformer trunk.
    """
    additive = torch.zeros_like(mask, dtype=torch.float32)
    return additive.masked_fill(~mask, float("-inf"))


# ---------------------------------------------------------------------------
# Multi-head causal self-attention
# ---------------------------------------------------------------------------

class MultiHeadCausalAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    Accepts an optional boolean causal mask of shape (1, 1, T, T).
    If mask is None, no masking is applied (use only for full-sequence
    non-autoregressive contexts).
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
            )
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, d_model)
        mask: torch.Tensor | None = None,      # (1, 1, T, T) bool, True=attend
    ) -> torch.Tensor:                         # (B, T, d_model)
        B, T, C = x.shape

        qkv = self.qkv(x)                      # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)

        if mask is not None:
            # Convert boolean (1,1,T,T) → additive float; broadcasts to (B, n_heads, T, T)
            scores = scores + _bool_to_additive(mask)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v                           # (B, n_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer GELU MLP with expansion factor d_ff/d_model (default 4x).

    Layout: Linear → GELU → Dropout → Linear → Dropout
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d_model)
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# V2 Transformer block
# ---------------------------------------------------------------------------

class V2TransformerBlock(nn.Module):
    """Pre-norm transformer block with adaLN conditioning.

    Both layer norms are replaced by AdaLayerNorm.  The caller slices the
    conditioner output and passes per-block (gamma, beta) pairs in.

    forward signature::

        out = block(x, mask, gamma_1, beta_1, gamma_2, beta_2)

    where:
        x        : (B, T, d_model)  — sequence
        mask     : (1, 1, T, T) bool causal mask from build_causal_mask()
        gamma_1  : (B, d_model)  — pre-attention adaLN scale
        beta_1   : (B, d_model)  — pre-attention adaLN shift
        gamma_2  : (B, d_model)  — pre-FFN adaLN scale
        beta_2   : (B, d_model)  — pre-FFN adaLN shift
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.ln_1 = AdaLayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalAttention(cfg)
        self.ln_2 = AdaLayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,        # (B, T, d_model)
        mask: torch.Tensor,     # (1, 1, T, T) bool
        gamma_1: torch.Tensor,  # (B, d_model)
        beta_1: torch.Tensor,   # (B, d_model)
        gamma_2: torch.Tensor,  # (B, d_model)
        beta_2: torch.Tensor,   # (B, d_model)
    ) -> torch.Tensor:          # (B, T, d_model)
        x = x + self.attn(self.ln_1(x, gamma_1, beta_1), mask)
        x = x + self.ffn(self.ln_2(x, gamma_2, beta_2))
        return x
