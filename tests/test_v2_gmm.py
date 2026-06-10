"""
Smoke tests for PitchGPT v2 ContinuousGMM and V2InputLayer.

Each test prints at least one named numerical value so that convention bugs
(wrong slice, wrong dimension) surface immediately — in line with the
project's bug-prevention discipline (CLAUDE.md §bug-prevention).
"""

from __future__ import annotations

import math

import pytest
import torch

from model.v2.config import V2Config, tiny_v2_config
from model.v2.heads import ContinuousGMM, TypeHead
from model.v2.embeddings import V2InputLayer, state_vec_dim

CONTINUOUS_DIM = state_vec_dim(6)  # v1c.1 default width


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg() -> V2Config:
    return tiny_v2_config()


@pytest.fixture
def gmm(tiny_cfg: V2Config) -> ContinuousGMM:
    return ContinuousGMM(tiny_cfg)


@pytest.fixture
def sample_hidden_type(tiny_cfg: V2Config):
    """Return (hidden, type_emb) tensors with batch=2, seq_len=5."""
    B, T, d = 2, 5, tiny_cfg.d_model
    torch.manual_seed(42)
    hidden   = torch.randn(B, T, d)
    type_emb = torch.randn(B, T, d)
    return hidden, type_emb


# ---------------------------------------------------------------------------
# ContinuousGMM — output shapes
# ---------------------------------------------------------------------------

class TestContinuousGMM:
    def test_output_shapes(
        self, gmm: ContinuousGMM, sample_hidden_type, tiny_cfg: V2Config
    ) -> None:
        hidden, type_emb = sample_hidden_type
        B, T = hidden.shape[:2]
        K = tiny_cfg.gmm_components
        D = tiny_cfg.n_continuous

        log_w, mu, log_std = gmm(hidden, type_emb)

        print(
            f"\nContinuousGMM output shapes — "
            f"log_w={tuple(log_w.shape)}, "
            f"mu={tuple(mu.shape)}, "
            f"log_std={tuple(log_std.shape)}"
        )

        assert log_w.shape   == (B, T, K),    f"log_w: expected ({B},{T},{K}), got {tuple(log_w.shape)}"
        assert mu.shape      == (B, T, K, D), f"mu:    expected ({B},{T},{K},{D}), got {tuple(mu.shape)}"
        assert log_std.shape == (B, T, K, D), f"log_std: expected ({B},{T},{K},{D}), got {tuple(log_std.shape)}"

    def test_nll_is_scalar(
        self, gmm: ContinuousGMM, sample_hidden_type, tiny_cfg: V2Config
    ) -> None:
        hidden, type_emb = sample_hidden_type
        B, T = hidden.shape[:2]
        D = tiny_cfg.n_continuous

        log_w, mu, log_std = gmm(hidden, type_emb)
        target = torch.randn(B, T, D)
        loss = gmm.nll(log_w, mu, log_std, target)

        print(f"\nContinuousGMM NLL — loss={loss.item():.4f} (expect scalar > 0)")

        assert loss.shape == torch.Size([]), f"NLL should be scalar, got shape {tuple(loss.shape)}"
        assert loss.item() > 0, f"NLL should be positive, got {loss.item():.4f}"
        # Sanity ceiling: for K=5, D=4 Gaussians with random init the NLL
        # should be well under 100 nats.
        assert loss.item() < 100.0, f"NLL suspiciously large: {loss.item():.4f}"

    def test_sample_shape(
        self, gmm: ContinuousGMM, sample_hidden_type, tiny_cfg: V2Config
    ) -> None:
        hidden, type_emb = sample_hidden_type
        B, T = hidden.shape[:2]
        D = tiny_cfg.n_continuous

        log_w, mu, log_std = gmm(hidden, type_emb)
        torch.manual_seed(7)
        samples = gmm.sample(log_w, mu, log_std)

        print(
            f"\nContinuousGMM sample — "
            f"shape={tuple(samples.shape)}, "
            f"sample[0,0,0]={samples[0, 0, 0].item():.4f}"
        )

        assert samples.shape == (B, T, D), (
            f"Sample shape: expected ({B},{T},{D}), got {tuple(samples.shape)}"
        )

    def test_log_weights_sum_to_one(
        self, gmm: ContinuousGMM, sample_hidden_type, tiny_cfg: V2Config
    ) -> None:
        hidden, type_emb = sample_hidden_type
        K = tiny_cfg.gmm_components

        log_w, _, _ = gmm(hidden, type_emb)
        w_sum = log_w.exp().sum(dim=-1)   # (B, T) — should be ~1.0

        max_dev = (w_sum - 1.0).abs().max().item()
        print(
            f"\nMixture weights sum check — "
            f"w_sum[0,0]={w_sum[0, 0].item():.6f}, "
            f"max deviation from 1.0: {max_dev:.2e}"
        )

        assert max_dev < 1e-5, (
            f"Mixture weights do not sum to 1.0 (max deviation {max_dev:.2e})"
        )

    def test_log_std_clamped(
        self, gmm: ContinuousGMM, sample_hidden_type, tiny_cfg: V2Config
    ) -> None:
        """log_std values must stay within [floor, ceil] at init."""
        hidden, type_emb = sample_hidden_type
        _, _, log_std = gmm(hidden, type_emb)

        min_ls = log_std.min().item()
        max_ls = log_std.max().item()

        print(
            f"\nlog_std clamp check — "
            f"min={min_ls:.4f} (floor={tiny_cfg.gmm_logstd_floor}), "
            f"max={max_ls:.4f} (ceil={tiny_cfg.gmm_logstd_ceil})"
        )

        assert min_ls >= tiny_cfg.gmm_logstd_floor - 1e-6, (
            f"log_std below floor: {min_ls:.4f} < {tiny_cfg.gmm_logstd_floor}"
        )
        assert max_ls <= tiny_cfg.gmm_logstd_ceil + 1e-6, (
            f"log_std above ceil: {max_ls:.4f} > {tiny_cfg.gmm_logstd_ceil}"
        )

    def test_nll_differentiable(
        self, gmm: ContinuousGMM, sample_hidden_type, tiny_cfg: V2Config
    ) -> None:
        """NLL backward pass should not error and produce finite grads."""
        hidden, type_emb = sample_hidden_type
        hidden   = hidden.requires_grad_(True)
        type_emb = type_emb.requires_grad_(True)

        B, T, D = *hidden.shape[:2], tiny_cfg.n_continuous
        target = torch.randn(B, T, D)

        log_w, mu, log_std = gmm(hidden, type_emb)
        loss = gmm.nll(log_w, mu, log_std, target)
        loss.backward()

        grad_norm = hidden.grad.norm().item()
        print(f"\nNLL backward — grad_norm(hidden)={grad_norm:.4f}")

        assert math.isfinite(grad_norm), f"Gradient norm is not finite: {grad_norm}"
        assert grad_norm > 0, "Gradient norm is zero — backward not flowing"


# ---------------------------------------------------------------------------
# TypeHead
# ---------------------------------------------------------------------------

class TestTypeHead:
    def test_output_shape(self, tiny_cfg: V2Config) -> None:
        """TypeHead should emit (B, T, 8) logits."""
        B, T = 3, 6
        emb = torch.nn.Embedding(tiny_cfg.n_pitch_types, tiny_cfg.d_model)
        head = TypeHead(tiny_cfg, emb.weight)

        hidden = torch.randn(B, T, tiny_cfg.d_model)
        logits = head(hidden)

        print(
            f"\nTypeHead output — "
            f"shape={tuple(logits.shape)}, "
            f"logits[0,0,1]={logits[0, 0, 1].item():.4f} (FF logit)"
        )

        assert logits.shape == (B, T, tiny_cfg.n_pitch_types), (
            f"Expected ({B},{T},{tiny_cfg.n_pitch_types}), got {tuple(logits.shape)}"
        )

    def test_weight_tied_grad(self, tiny_cfg: V2Config) -> None:
        """Gradient from TypeHead must flow into the shared embedding weight."""
        emb = torch.nn.Embedding(tiny_cfg.n_pitch_types, tiny_cfg.d_model)
        head = TypeHead(tiny_cfg, emb.weight)

        hidden = torch.randn(2, 4, tiny_cfg.d_model)
        logits = head(hidden)
        loss = logits.sum()
        loss.backward()

        assert emb.weight.grad is not None, (
            "Gradient did not flow into shared embedding weight"
        )
        grad_norm = emb.weight.grad.norm().item()
        print(f"\nTypeHead tied-weight grad_norm={grad_norm:.4f}")
        assert grad_norm > 0, "Embedding grad norm is zero"


# ---------------------------------------------------------------------------
# V2InputLayer
# ---------------------------------------------------------------------------

class TestV2InputLayer:
    def _make_batch(self, cfg: V2Config, B: int = 2, T: int = 5):
        """Return all forward() inputs for the given batch dims."""
        torch.manual_seed(0)
        type_ids     = torch.randint(1, cfg.n_pitch_types, (B, T))
        continuous   = torch.randn(B, T, cfg.n_continuous)
        result_ids   = torch.randint(0, cfg.n_result_classes, (B, T))
        count_state  = torch.randint(0, cfg.n_count_states,  (B, T))
        outs         = torch.randint(0, cfg.n_outs,          (B, T))
        runners      = torch.randint(0, cfg.n_runner_states, (B, T))
        pitch_number = torch.randint(0, cfg.max_positions,   (B, T))
        return type_ids, continuous, result_ids, count_state, outs, runners, pitch_number

    def test_output_shape(self, tiny_cfg: V2Config) -> None:
        B, T = 2, 5
        layer = V2InputLayer(tiny_cfg)
        inputs = self._make_batch(tiny_cfg, B, T)
        out = layer(*inputs)

        print(
            f"\nV2InputLayer output — "
            f"shape={tuple(out.shape)}, "
            f"out[0,0,0]={out[0, 0, 0].item():.4f}"
        )

        assert out.shape == (B, T, tiny_cfg.d_model), (
            f"Expected ({B},{T},{tiny_cfg.d_model}), got {tuple(out.shape)}"
        )

    def test_continuous_dim_constant(self) -> None:
        """state_vec_dim: 6+8+12+3+8+15 = 52 (v1c.1); legacy 4-dim = 50."""
        assert CONTINUOUS_DIM == 52, (
            f"state_vec_dim(6) expected 52, got {CONTINUOUS_DIM}"
        )
        assert state_vec_dim(4) == 50, (
            f"legacy state_vec_dim(4) expected 50, got {state_vec_dim(4)}"
        )

    def test_pad_type_zero_contribution(self, tiny_cfg: V2Config) -> None:
        """A type_id=0 (PAD) token should have its type embedding zeroed out."""
        layer = V2InputLayer(tiny_cfg)
        # Force type_ids to 0 (PAD) and a non-PAD (1=FF) to compare type_emb rows.
        pad_embed  = layer.type_emb(torch.tensor([[0]]))   # (1,1,d)
        ff_embed   = layer.type_emb(torch.tensor([[1]]))   # (1,1,d)

        pad_norm = pad_embed.norm().item()
        ff_norm  = ff_embed.norm().item()
        print(
            f"\nPAD type embedding norm={pad_norm:.6f} (expect ~0), "
            f"FF norm={ff_norm:.4f} (expect >0)"
        )

        assert pad_norm < 1e-6, f"PAD embedding not zeroed: norm={pad_norm:.2e}"
        assert ff_norm  > 0,    f"FF embedding is zero (unexpected)"

    def test_output_differs_across_sequence(self, tiny_cfg: V2Config) -> None:
        """Different pitch positions should produce different output tokens."""
        layer = V2InputLayer(tiny_cfg)
        inputs = self._make_batch(tiny_cfg, B=1, T=5)
        out = layer(*inputs)  # (1, 5, d)

        max_diff = (out[0, 0] - out[0, 1]).abs().max().item()
        print(f"\nToken diff [pos0 vs pos1] — max_diff={max_diff:.4f}")
        assert max_diff > 1e-4, (
            "All sequence positions produced identical tokens — positional/content encoding not working"
        )
