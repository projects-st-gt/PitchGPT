"""Prediction heads: propensity (π̂), result (μ̂), AB-outcome.

Per ADR 007, the result head sees the trunk's hidden state through a
``.detach()`` boundary: gradients from the result loss do NOT flow back
into the trunk. They DO flow into the result-head's own MLP and into the
factor embedding tables (which are shared and also receive propensity-
head gradients).

Per the architecture brainstorm:
- Propensity ``type`` and ``zone`` heads tie weights to the factor
  embedding tables (mild regularization).
- Result head MLP has 2 hidden layers, not 1.
- Spin axis is predicted as 3 outputs ``(mean_sin, mean_cos, log_kappa)``
  parameterizing a von Mises distribution, since the underlying variable
  is circular.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.config import PitchGPTConfig


class PropensityHeads(nn.Module):
    """Multi-factor propensity head π̂(a | h).

    Predicts each pitch factor independently from the hidden state.
    ``type`` and ``zone`` projections are weight-tied to their embedding
    tables (saves params, regularizes coherence between input and output
    representations of these factors).
    """

    def __init__(
        self,
        config: PitchGPTConfig,
        type_emb_weight: nn.Parameter,
        zone_emb_weight: nn.Parameter,
    ):
        super().__init__()
        self.config = config
        d = config.d_model
        # Tied projections (we don't store our own weight; we reuse the
        # embedding tables' weight tensors at forward time).
        self._type_emb_weight = type_emb_weight
        self._zone_emb_weight = zone_emb_weight
        # Bias for the tied projections (matrix-multiply needs a bias)
        self.type_bias = nn.Parameter(torch.zeros(config.n_pitch_types))
        self.zone_bias = nn.Parameter(torch.zeros(config.n_zones))

        # Independent projections for the other factors
        self.velo_proj = nn.Linear(d, config.n_velo_bins)
        self.spin_rate_proj = nn.Linear(d, config.n_spin_rate_bins)

        if config.spin_axis_circular:
            # 3 outputs: mean direction as (a, b) → normalized to unit,
            # plus log_kappa (the concentration)
            self.spin_axis_proj = nn.Linear(d, 3)
        else:
            self.spin_axis_proj = nn.Linear(d, config.n_spin_axis_bins)

        self._init_weights()

    def _init_weights(self):
        std = self.config.init_std
        for m in [self.velo_proj, self.spin_rate_proj, self.spin_axis_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=std)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        hidden_exec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Hidden (B, T, d_model) → dict of logits per factor.

        The TYPE head always reads ``hidden``. The execution heads (zone,
        velo, spin_rate, spin_axis) read ``hidden_exec`` when supplied —
        the type-conditioned fused vector (ADR-013) — else fall back to
        ``hidden`` (pre-v7 behaviour).
        """
        exec_h = hidden if hidden_exec is None else hidden_exec
        type_logits = hidden @ self._type_emb_weight.T + self.type_bias
        zone_logits = exec_h @ self._zone_emb_weight.T + self.zone_bias
        velo_logits = self.velo_proj(exec_h)
        spin_rate_logits = self.spin_rate_proj(exec_h)
        spin_axis_out = self.spin_axis_proj(exec_h)
        return {
            "type": type_logits,             # (B, T, n_pitch_types)
            "zone": zone_logits,             # (B, T, n_zones)
            "velo": velo_logits,             # (B, T, n_velo_bins)
            "spin_rate": spin_rate_logits,   # (B, T, n_spin_rate_bins)
            "spin_axis": spin_axis_out,      # (B, T, 3) if circular else (B, T, n_spin_axis_bins)
        }


class ResultHead(nn.Module):
    """Two-stage μ̂(y | a, h).

    Inputs the trunk's hidden state PLUS the intended action's
    embeddings (type, zone, velo, and spin-axis sin/cos). The action
    embeddings come from the FactorEmbeddings tables (shared with the
    trunk's input). At training time these are the actual factors of
    the next pitch; at counterfactual rollout time they're the
    intervened values.

    Per ADR 007: the hidden state input is detached. Gradients from the
    result loss DO NOT update the trunk.
    """

    def __init__(
        self,
        config: PitchGPTConfig,
        type_emb: nn.Embedding,
        zone_emb: nn.Embedding,
        velo_emb: nn.Embedding,
        spin_axis_proj: nn.Linear | None,
        spin_axis_emb: nn.Embedding | None,
    ):
        super().__init__()
        self.config = config
        # References to factor embeddings (we don't own these; they live
        # in FactorEmbeddings). We DO want gradients from result loss to
        # flow into them — they're shared with the trunk's input layer.
        self.type_emb = type_emb
        self.zone_emb = zone_emb
        self.velo_emb = velo_emb
        self.spin_axis_proj = spin_axis_proj  # used if circular
        self.spin_axis_emb = spin_axis_emb    # used if categorical

        d = config.d_model
        # Input dim: hidden + type + zone + velo + (2 if circular spin axis else d)
        spin_dim = 2 if config.spin_axis_circular else d
        mlp_in = d + d + d + d + spin_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, config.n_result_logits),
        )
        self._init_weights()

    def _init_weights(self):
        std = self.config.init_std
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden: torch.Tensor,           # (B, T, d_model) — caller passes .detach()
        type_id: torch.Tensor,          # (B, T) LongTensor
        zone_id: torch.Tensor,          # (B, T)
        velo_id: torch.Tensor,          # (B, T)
        spin_axis: torch.Tensor,        # (B, T, 2) if circular else (B, T) LongTensor
    ) -> torch.Tensor:
        type_e = self.type_emb(type_id)
        zone_e = self.zone_emb(zone_id)
        velo_e = self.velo_emb(velo_id)
        if self.config.spin_axis_circular:
            spin_e = spin_axis  # shape (B, T, 2), no projection here; the MLP handles it
        else:
            spin_e = self.spin_axis_emb(spin_axis)
        x = torch.cat([hidden, type_e, zone_e, velo_e, spin_e], dim=-1)
        return self.mlp(x)  # (B, T, n_result_logits)


class ABOutcomeHead(nn.Module):
    """Predict AB-level outcome class (K/BB/1B/2B/3B/HR/out) from terminal-pitch hidden."""

    def __init__(self, config: PitchGPTConfig):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.n_ab_outcome_classes)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_terminal: torch.Tensor) -> torch.Tensor:
        """hidden_terminal: (B, d_model) — only the terminal-pitch hidden states."""
        return self.proj(hidden_terminal)


class LocationMDN(nn.Module):
    """Mixture-of-Gaussians head over continuous (plate_x, plate_z).

    Predicts K diagonal 2D Gaussians from a conditioning vector (trunk hidden +
    the sampled type/zone/velo/spin embeddings). Trained by mixture NLL on the
    real (plate_x, plate_z); sampled at rollout time to give the cascade a real
    location (replacing the zone centroid).
    """

    def __init__(self, config: PitchGPTConfig, d_in: int):
        super().__init__()
        self.config = config
        self.K = config.mdn_components
        self.floor = config.mdn_logstd_floor
        self.proj = nn.Linear(d_in, self.K * 5)   # per comp: weight(1)+mean(2)+log_std(2)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor):
        """cond (..., d_in) -> (log_w (...,K), mu (...,K,2), log_std (...,K,2))."""
        o = self.proj(cond)
        *lead, _ = o.shape
        o = o.view(*lead, self.K, 5)
        log_w = torch.log_softmax(o[..., 0], dim=-1)
        mu = o[..., 1:3]
        log_std = o[..., 3:5].clamp(min=self.floor, max=2.0)
        return log_w, mu, log_std

    def nll(self, log_w, mu, log_std, target):
        """Mean mixture NLL. target (...,2). Returns scalar over the leading dims."""
        t = target.unsqueeze(-2)                          # (...,1,2)
        var = (2 * log_std).exp()
        comp = -0.5 * (((t - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi)).sum(-1)
        logp = torch.logsumexp(log_w + comp, dim=-1)      # (...)
        return -logp.mean()

    def sample(self, log_w, mu, log_std, generator=None):
        """Sample (...,2). Picks a component ~ exp(log_w), then a diagonal Gaussian."""
        w = log_w.exp()
        flat_w = w.reshape(-1, self.K)
        idx = torch.multinomial(flat_w, 1, generator=generator).reshape(*w.shape[:-1])
        mu_s = torch.gather(mu, -2, idx[..., None, None].expand(*idx.shape, 1, 2)).squeeze(-2)
        ls_s = torch.gather(log_std, -2, idx[..., None, None].expand(*idx.shape, 1, 2)).squeeze(-2)
        eps = torch.randn(mu_s.shape, generator=generator)
        return mu_s + eps * ls_s.exp()
