"""Pre-norm transformer trunk + attention mask construction.

GPT-2-style pre-norm: ``y = x + Attn(LN(x))``; ``z = y + FFN(LN(y))``.
Causal multi-head self-attention with explicit padding mask and optional
cross-AB block mask (for packed batches).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PitchGPTConfig


def build_attention_mask(
    seq_len: int,
    padding_mask: torch.Tensor | None = None,
    ab_boundaries: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Combined causal + padding + (optional) cross-AB block mask.

    Returns an additive mask of shape ``(B, 1, T, T)`` where ``0.0`` means
    "attend allowed" and ``-inf`` means "blocked." Suitable for adding to
    pre-softmax attention scores.

    Args:
        seq_len: ``T`` — sequence length per row.
        padding_mask: optional BoolTensor ``(B, T)`` where ``True`` = real
            position, ``False`` = padding. If None, no padding masking.
        ab_boundaries: optional LongTensor ``(B, T)`` where each token's
            value identifies its AB. Tokens cannot attend across boundaries
            (cross-AB block). If None, no cross-AB blocking.
    """
    if device is None:
        device = (
            padding_mask.device if padding_mask is not None
            else (ab_boundaries.device if ab_boundaries is not None
                  else torch.device("cpu"))
        )

    # Start with causal: lower-triangular allowed
    # Shape: (T, T), True = allowed
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # Broadcast to (1, 1, T, T)
    allowed = causal.unsqueeze(0).unsqueeze(0)

    if padding_mask is not None:
        # Block attending TO padded positions
        # padding_mask: (B, T) — True = real
        # We want: allowed[b, q, k] &= padding_mask[b, k]
        pad_k = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        allowed = allowed & pad_k

    if ab_boundaries is not None:
        # Block attention across AB boundaries
        # allowed[b, q, k] &= (ab_boundaries[b, q] == ab_boundaries[b, k])
        ab_q = ab_boundaries.unsqueeze(2)  # (B, T, 1)
        ab_k = ab_boundaries.unsqueeze(1)  # (B, 1, T)
        same_ab = (ab_q == ab_k).unsqueeze(1)  # (B, 1, T, T)
        allowed = allowed & same_ab

    # Convert to additive mask (-inf where blocked)
    additive = torch.zeros_like(allowed, dtype=torch.float32)
    additive = additive.masked_fill(~allowed, float("-inf"))
    return additive


class MultiHeadCausalAttention(nn.Module):
    """Standard multi-head self-attention with externally-supplied mask."""

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Cache for diagnostic / interpretability
        self.last_attention_weights: torch.Tensor | None = None
        self._cache_attention = False

    def forward(
        self,
        x: torch.Tensor,            # (B, T, d_model)
        attention_mask: torch.Tensor,  # (B, 1, T, T) additive
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)
        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with explicit mask
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)
        scores = scores + attention_mask  # broadcast (B, 1, T, T) → (B, n_heads, T, T)
        attn = F.softmax(scores, dim=-1)
        if self._cache_attention:
            self.last_attention_weights = attn.detach()
        attn = self.attn_dropout(attn)
        y = attn @ v  # (B, n_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    """GELU MLP with expansion factor 4, dropout."""

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

    forward(x, mask):
        x = x + self.attn(LayerNorm(x), mask)
        x = x + self.ffn(LayerNorm(x))
    """

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadCausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,             # (B, T, d_model)
        attention_mask: torch.Tensor # (B, 1, T, T)
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.ffn(self.ln_2(x))
        return x
