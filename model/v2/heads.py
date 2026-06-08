"""V2 output heads for PitchGPT base-v1c.

Two heads live here:

TypeHead
    Weight-tied softmax over pitch types. The embedding matrix from
    V2InputLayer.type_emb is reused as the projection matrix (a common
    language-model trick: the embedding and un-embedding share weights,
    reducing parameters and regularising the representation). Output has 8
    logits (PAD at index 0, types FF–FS at indices 1–7). During loss
    computation callers slice [:, :, 1:] to get the 7 real-type logits and
    mask out PAD from the cross-entropy.

ContinuousGMM
    Mixture-of-K diagonal Gaussians over the 4 continuous pitch properties:
    velocity, spin rate, plate_x, plate_z.

    The conditioning vector is [hidden_state || type_embedding], shape
    (B, T, 2*d_model), so the head is explicitly conditioned on the
    predicted/actual pitch type. This mirrors the LocationMDN pattern in
    the v9 trunk (model/heads.py:LocationMDN).

    Parameterisation per component c:
        log_w_c  : scalar mixture weight (log-softmax normalised over K)
        mu_c     : (D,) = (4,) means
        log_std_c: (D,) = (4,) log standard deviations, clamped to
                    [cfg.gmm_logstd_floor, cfg.gmm_logstd_ceil]

    Total output per position: K * (1 + D + D) = K * 9 dims (for K=5, D=4 → 45).

    NLL: standard diagonal-Gaussian mixture negative log-likelihood.
    Sample: pick component via multinomial(exp(log_w)), then reparameterize.

Weight init: N(0, cfg.init_std) for all weight matrices, zeros for biases.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2.config import V2Config


class TypeHead(nn.Module):
    """Weight-tied pitch-type head.

    Reuses the type embedding matrix from V2InputLayer as the un-embedding
    projection. Adds a learnable bias of size n_pitch_types (8).

    Output: (B, T, 8) logits — index 0 is PAD, indices 1–7 are FF through FS.
    Callers must exclude the PAD logit from cross-entropy (use [:, :, 1:] or
    pass ignore_index=0 to nn.CrossEntropyLoss).
    """

    def __init__(self, cfg: V2Config, type_emb_weight: nn.Parameter) -> None:
        """
        Args:
            cfg             : V2Config — provides n_pitch_types and init_std.
            type_emb_weight : the .weight tensor of V2InputLayer.type_emb,
                              shape (n_pitch_types, d_model) = (8, d_model).
                              We do NOT own this parameter; it lives in the
                              embedding module. Gradients flow through it.
        """
        super().__init__()
        # We store a reference, not a copy. The embedding module's .weight
        # is the authoritative parameter; we just point at it.
        self._type_emb_weight = type_emb_weight
        self.bias = nn.Parameter(torch.zeros(cfg.n_pitch_types))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, T, d_model)

        Returns:
            (B, T, n_pitch_types=8) — raw logits; apply softmax / cross-entropy
            externally.
        """
        # (B, T, d_model) @ (d_model, 8) + (8,) → (B, T, 8)
        return hidden @ self._type_emb_weight.T + self.bias


class ContinuousGMM(nn.Module):
    """Mixture-of-K diagonal Gaussians over D=4 continuous pitch properties.

    Conditioning: hidden_state (d_model) + type_embedding (d_model), so the
    head sees both the sequence context and the pitch type it is generating
    continuous properties for.

    The three tensors returned by forward() are named to match LocationMDN in
    model/heads.py so existing NLL / sample code is easy to port:
        log_w  : (..., K)       log mixture weights (log_softmax over K)
        mu     : (..., K, D)    component means
        log_std: (..., K, D)    component log-std, clamped to [floor, ceil]
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.K = cfg.gmm_components      # 5
        self.D = cfg.n_continuous         # 4
        self._floor = cfg.gmm_logstd_floor
        self._ceil  = cfg.gmm_logstd_ceil

        # Input: [hidden (d) || type_emb (d)] → (K * (1 + D + D)) = K * 9
        in_dim  = 2 * cfg.d_model
        out_dim = self.K * (1 + 2 * self.D)   # K * 9 = 45 for K=5, D=4
        self.proj = nn.Linear(in_dim, out_dim)

        # Init
        nn.init.normal_(self.proj.weight, mean=0.0, std=cfg.init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,    # (B, T, d_model)
        type_emb: torch.Tensor,  # (B, T, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to GMM parameters.

        Returns:
            log_w  : (B, T, K)       log mixture weights
            mu     : (B, T, K, D)    component means
            log_std: (B, T, K, D)    log standard deviations
        """
        x = torch.cat([hidden, type_emb], dim=-1)  # (B, T, 2*d)
        o = self.proj(x)                            # (B, T, K*(1+2D))

        *lead, _ = o.shape
        o = o.view(*lead, self.K, 1 + 2 * self.D)   # (..., K, 9)

        log_w   = F.log_softmax(o[..., 0], dim=-1)  # (..., K)
        mu      = o[..., 1 : 1 + self.D]            # (..., K, D)
        log_std = o[..., 1 + self.D :].clamp(       # (..., K, D)
            min=self._floor, max=self._ceil
        )
        return log_w, mu, log_std

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nll(
        self,
        log_w: torch.Tensor,    # (..., K)
        mu: torch.Tensor,       # (..., K, D)
        log_std: torch.Tensor,  # (..., K, D)
        target: torch.Tensor,   # (..., D)
    ) -> torch.Tensor:
        """Mean mixture negative log-likelihood over the leading dimensions.

        Implements the standard diagonal-Gaussian mixture log-likelihood:

            log p(x) = log Σ_k exp[ log w_k + Σ_d log N(x_d; μ_kd, σ_kd) ]

        Returns:
            scalar — mean NLL over all positions in the batch.
        """
        t = target.unsqueeze(-2)                             # (..., 1, D)
        var = (2 * log_std).exp()                            # (..., K, D)

        # Per-component, per-dimension log Normal density
        log_density_d = (
            -0.5 * (((t - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
        )  # (..., K, D)

        # Sum over D (diagonal ⟹ product of univariate densities)
        log_density = log_density_d.sum(-1)                  # (..., K)

        # Log-sum-exp to get mixture log-probability
        log_prob = torch.logsumexp(log_w + log_density, dim=-1)  # (...)

        return -log_prob.mean()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        log_w: torch.Tensor,    # (..., K)
        mu: torch.Tensor,       # (..., K, D)
        log_std: torch.Tensor,  # (..., K, D)
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one sample per position from the GMM.

        1. Pick a component index via multinomial sampling from exp(log_w).
        2. Draw from the selected diagonal Gaussian via reparameterization.

        Returns:
            (..., D) float tensor of sampled values.
        """
        w = log_w.exp()                          # (..., K)

        # Flatten batch dimensions for multinomial, then restore.
        lead = w.shape[:-1]
        flat_w = w.reshape(-1, self.K)           # (N, K)
        idx = torch.multinomial(flat_w, num_samples=1, generator=generator)  # (N, 1)
        idx = idx.reshape(*lead)                 # (...,)

        # Gather the chosen component's mu and log_std.
        idx_exp = idx[..., None, None].expand(*lead, 1, self.D)  # (..., 1, D)
        mu_sel      = torch.gather(mu,      -2, idx_exp).squeeze(-2)  # (..., D)
        log_std_sel = torch.gather(log_std, -2, idx_exp).squeeze(-2)  # (..., D)

        # Reparameterized sample.
        eps = torch.randn(
            mu_sel.shape, generator=generator,
            device=mu_sel.device, dtype=mu_sel.dtype,
        )
        return mu_sel + eps * log_std_sel.exp()  # (..., D)
