"""
Smoke tests for PitchGPT v2 foundation: V2Config + adaLN modules.
"""

import torch
import pytest

from model.v2.config import V2Config, tiny_v2_config, small_v2_config
from model.v2.adaln import AdaLNConditioner, AdaLayerNorm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg() -> V2Config:
    return tiny_v2_config()


@pytest.fixture
def small_cfg() -> V2Config:
    return small_v2_config()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_tiny_config_defaults(tiny_cfg: V2Config) -> None:
    assert tiny_cfg.n_layers == 4
    assert tiny_cfg.d_model == 256
    assert tiny_cfg.n_heads == 4
    assert tiny_cfg.d_ff == 1024
    assert tiny_cfg.adaln_hidden == 1024


def test_small_config(small_cfg: V2Config) -> None:
    assert small_cfg.n_layers == 6
    assert small_cfg.d_model == 512
    assert small_cfg.n_heads == 8
    assert small_cfg.d_ff == 2048
    assert small_cfg.adaln_hidden == 2048


# ---------------------------------------------------------------------------
# AdaLNConditioner output shape
# ---------------------------------------------------------------------------

def test_conditioner_output_shape(tiny_cfg: V2Config) -> None:
    B = 3
    conditioner = AdaLNConditioner(tiny_cfg)
    pitcher = torch.randn(B, tiny_cfg.pitcher_profile_dim)
    batter = torch.randn(B, tiny_cfg.batter_profile_dim)

    out = conditioner(pitcher, batter)

    expected = (B, tiny_cfg.n_layers, 2, 2, tiny_cfg.d_model)
    assert out.shape == expected, (
        f"Expected shape {expected}, got {out.shape}"
    )


def test_conditioner_output_shape_small(small_cfg: V2Config) -> None:
    """Same test for the larger config to ensure d_model scaling is correct."""
    B = 2
    conditioner = AdaLNConditioner(small_cfg)
    pitcher = torch.randn(B, small_cfg.pitcher_profile_dim)
    batter = torch.randn(B, small_cfg.batter_profile_dim)

    out = conditioner(pitcher, batter)

    expected = (B, small_cfg.n_layers, 2, 2, small_cfg.d_model)
    assert out.shape == expected, (
        f"Expected shape {expected}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# adaLN-Zero init: gamma ≈ 1, beta ≈ 0
# ---------------------------------------------------------------------------

def test_adaln_zero_init(tiny_cfg: V2Config) -> None:
    """
    At initialization, for any input, the conditioner should output
    gamma ≈ 1.0 and beta ≈ 0.0 for all layers and LN positions.
    (adaLN-Zero property: model starts as a plain transformer.)
    """
    B = 4
    conditioner = AdaLNConditioner(tiny_cfg)
    conditioner.eval()

    pitcher = torch.randn(B, tiny_cfg.pitcher_profile_dim)
    batter = torch.randn(B, tiny_cfg.batter_profile_dim)

    with torch.no_grad():
        out = conditioner(pitcher, batter)
        # out shape: (B, n_layers, 2, 2, d_model)
        gamma = out[:, :, :, 0, :]   # (B, n_layers, 2, d_model)
        beta = out[:, :, :, 1, :]    # (B, n_layers, 2, d_model)

    # Print named values so convention bugs surface immediately.
    print(
        f"\nadaLN-Zero init check — "
        f"gamma[0,layer0,pre-attn,dim0]={gamma[0, 0, 0, 0].item():.6f}, "
        f"beta[0,layer0,pre-attn,dim0]={beta[0, 0, 0, 0].item():.6f}"
    )

    assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-5), (
        f"Expected gamma=1.0 at init, max deviation: "
        f"{(gamma - 1.0).abs().max().item():.2e}"
    )
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-5), (
        f"Expected beta=0.0 at init, max deviation: "
        f"{beta.abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# AdaLayerNorm modulates output
# ---------------------------------------------------------------------------

def test_adaln_modulates(tiny_cfg: V2Config) -> None:
    """
    AdaLayerNorm with gamma=2.0, beta=0.5 should produce output different
    from plain LayerNorm (which implicitly uses gamma=1, beta=0).
    """
    B, T, D = 2, 7, tiny_cfg.d_model
    ada_ln = AdaLayerNorm(D)
    plain_ln = torch.nn.LayerNorm(D, elementwise_affine=False)

    x = torch.randn(B, T, D)

    gamma = torch.full((B, D), 2.0)
    beta = torch.full((B, D), 0.5)

    with torch.no_grad():
        ada_out = ada_ln(x, gamma, beta)   # (B, T, D)
        plain_out = plain_ln(x)            # (B, T, D)

    # Print a named value for convention verification.
    print(
        f"\nAdaLayerNorm modulation check — "
        f"ada_out[0,0,0]={ada_out[0, 0, 0].item():.4f}, "
        f"plain_out[0,0,0]={plain_out[0, 0, 0].item():.4f}"
    )

    assert not torch.allclose(ada_out, plain_out, atol=1e-3), (
        "AdaLayerNorm with gamma=2, beta=0.5 should differ from plain LayerNorm"
    )

    # Verify the math: ada_out should equal 2 * plain_out + 0.5
    expected = 2.0 * plain_out + 0.5
    assert torch.allclose(ada_out, expected, atol=1e-5), (
        f"AdaLayerNorm output does not match 2*LN(x)+0.5. "
        f"Max deviation: {(ada_out - expected).abs().max().item():.2e}"
    )


def test_adaln_identity_at_zero_init(tiny_cfg: V2Config) -> None:
    """
    AdaLayerNorm with gamma=1, beta=0 should reproduce plain LayerNorm exactly.
    """
    B, T, D = 2, 5, tiny_cfg.d_model
    ada_ln = AdaLayerNorm(D)
    plain_ln = torch.nn.LayerNorm(D, elementwise_affine=False)

    x = torch.randn(B, T, D)
    gamma = torch.ones(B, D)
    beta = torch.zeros(B, D)

    with torch.no_grad():
        ada_out = ada_ln(x, gamma, beta)
        plain_out = plain_ln(x)

    print(
        f"\nAdaLayerNorm identity check — "
        f"ada_out[0,0,0]={ada_out[0, 0, 0].item():.6f}, "
        f"plain_out[0,0,0]={plain_out[0, 0, 0].item():.6f}"
    )

    assert torch.allclose(ada_out, plain_out, atol=1e-5), (
        f"With gamma=1, beta=0, AdaLayerNorm should match plain LayerNorm. "
        f"Max deviation: {(ada_out - plain_out).abs().max().item():.2e}"
    )
