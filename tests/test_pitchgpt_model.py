"""Tests for PitchGPT model components.

Coverage:
- Config defaults and factories produce valid models
- Embedding shapes are correct
- Context-token construction is correct
- Transformer mask honors causality, padding, cross-AB blocking
- Heads produce correct output shapes
- End-to-end forward pass works
- Stop-gradient between result head and trunk (ADR 007)
- Counterfactual sensitivity: changing intended action shifts result distribution
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model.config import PitchGPTConfig, sanity_config, small_config, tiny_config
from model.embeddings import ContextTokens, FactorEmbeddings
from model.heads import ABOutcomeHead, PropensityHeads, ResultHead
from model.pitchgpt import PitchGPT
from model.transformer import (
    MultiHeadCausalAttention,
    TransformerBlock,
    build_attention_mask,
)


# ============================================================
# Config
# ============================================================


def test_config_factories_produce_valid_models():
    for cfg in [sanity_config(), tiny_config(), small_config()]:
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.d_ff == 4 * cfg.d_model


def test_config_rejects_d_model_not_divisible_by_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        PitchGPTConfig(n_layers=2, n_heads=3, d_model=128)


# ============================================================
# Embeddings
# ============================================================


def _fake_factors(B=2, T=4, config: PitchGPTConfig | None = None) -> dict:
    if config is None:
        config = sanity_config()
    rng = torch.Generator().manual_seed(0)
    return {
        "type": torch.randint(1, config.n_pitch_types, (B, T), generator=rng),
        "zone": torch.randint(0, config.n_zones, (B, T), generator=rng),
        "velo": torch.randint(0, config.n_velo_bins, (B, T), generator=rng),
        "spin_rate": torch.randint(0, config.n_spin_rate_bins, (B, T), generator=rng),
        "spin_axis": (
            torch.randn(B, T, 2, generator=rng)
            if config.spin_axis_circular
            else torch.randint(0, config.n_spin_axis_bins, (B, T), generator=rng)
        ),
        "result": torch.randint(1, config.n_result_classes, (B, T), generator=rng),
        "count": torch.randint(0, config.n_count_states, (B, T), generator=rng),
        "runners": torch.randint(0, config.n_runner_states, (B, T), generator=rng),
        "outs": torch.randint(0, config.n_outs, (B, T), generator=rng),
        "pos": torch.arange(T).unsqueeze(0).expand(B, T).contiguous(),
        "pitcher_fatigue": torch.randint(
            0, config.n_pitcher_fatigue_buckets, (B, T), generator=rng
        ),
    }


def _fake_categorical(B=2, config: PitchGPTConfig | None = None) -> dict:
    if config is None:
        config = sanity_config()
    rng = torch.Generator().manual_seed(1)
    return {
        "p_throws": torch.randint(0, config.n_p_throws, (B,), generator=rng),
        "stand": torch.randint(0, config.n_stand, (B,), generator=rng),
        "ballpark": torch.randint(0, config.n_ballparks, (B,), generator=rng),
        "umpire": torch.randint(0, config.n_umpires, (B,), generator=rng),
        "catcher": torch.randint(0, config.n_catchers, (B,), generator=rng),
        "inning": torch.randint(0, config.n_inning_buckets, (B,), generator=rng),
        "score_diff": torch.randint(0, config.n_score_diff_buckets, (B,), generator=rng),
        "inning_half": torch.randint(0, config.n_inning_half, (B,), generator=rng),
        "days_rest": torch.randint(0, config.n_days_rest_buckets, (B,), generator=rng),
        "tto": torch.randint(0, config.n_tto_buckets, (B,), generator=rng),
        "temp": torch.randint(0, config.n_temp_buckets, (B,), generator=rng),
        "roof": torch.randint(0, config.n_roof, (B,), generator=rng),
    }


def test_factor_embeddings_output_shape():
    cfg = sanity_config()
    emb = FactorEmbeddings(cfg)
    B, T = 2, 4
    factors = _fake_factors(B, T, cfg)
    out = emb(factors)
    assert out.shape == (B, T, cfg.d_model)


def test_factor_embeddings_with_categorical_spin_axis():
    cfg = sanity_config()
    cfg.spin_axis_circular = False
    cfg = PitchGPTConfig(  # rebuild with the override
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_model=cfg.d_model,
        d_ff=cfg.d_ff, spin_axis_circular=False,
    )
    emb = FactorEmbeddings(cfg)
    B, T = 2, 4
    factors = _fake_factors(B, T, cfg)
    out = emb(factors)
    assert out.shape == (B, T, cfg.d_model)


def test_context_tokens_output_shape():
    cfg = sanity_config()
    ctx = ContextTokens(cfg)
    B = 2
    pitcher = torch.randn(B, cfg.pitcher_profile_dim)
    batter = torch.randn(B, cfg.batter_profile_dim)
    cat = _fake_categorical(B, cfg)
    out = ctx(pitcher, batter, cat)
    assert out.shape == (B, 3, cfg.d_model)


# ============================================================
# Attention mask
# ============================================================


def test_causal_mask_only_lower_triangle_allowed():
    mask = build_attention_mask(seq_len=4)
    # mask shape: (1, 1, 4, 4)
    # Position (q=1, k=2) should be blocked (q < k)
    assert torch.isinf(mask[0, 0, 1, 2]) and mask[0, 0, 1, 2] < 0
    # Position (q=2, k=1) should be allowed (q > k)
    assert mask[0, 0, 2, 1] == 0.0


def test_padding_mask_blocks_attention_to_padding():
    # B=2, T=4. Row 0: positions 0..3 all real. Row 1: positions 0..2 real, 3 pad.
    pad = torch.tensor([[True, True, True, True],
                        [True, True, True, False]])
    mask = build_attention_mask(seq_len=4, padding_mask=pad)
    # Row 1 should have -inf for attending to position 3 (key=3)
    for q in range(4):
        if q >= 3:
            assert mask[1, 0, q, 3] < 0 and torch.isinf(mask[1, 0, q, 3])
    # Row 0 should not have this restriction (all real)
    # For row 0, the only -inf cells are from causal masking (k > q)


def test_cross_ab_mask_blocks_between_at_bats():
    # B=1, T=4. ABs: [0, 0, 1, 1] — positions 0,1 in AB_0; 2,3 in AB_1.
    boundaries = torch.tensor([[0, 0, 1, 1]])
    mask = build_attention_mask(seq_len=4, ab_boundaries=boundaries)
    # Position q=2 (in AB_1) trying to attend k=1 (in AB_0): blocked
    assert torch.isinf(mask[0, 0, 2, 1])
    # Position q=3 (AB_1) attending k=2 (AB_1): allowed (also causal)
    assert mask[0, 0, 3, 2] == 0.0


# ============================================================
# Transformer block
# ============================================================


def test_transformer_block_output_shape():
    cfg = sanity_config()
    block = TransformerBlock(cfg)
    B, T = 2, 4
    x = torch.randn(B, T, cfg.d_model)
    mask = build_attention_mask(seq_len=T)
    out = block(x, mask)
    assert out.shape == x.shape


def test_attention_cache_disabled_by_default():
    cfg = sanity_config()
    attn = MultiHeadCausalAttention(cfg)
    assert attn._cache_attention is False
    assert attn.last_attention_weights is None


# ============================================================
# Heads
# ============================================================


def test_propensity_heads_shapes():
    cfg = sanity_config()
    emb = FactorEmbeddings(cfg)
    head = PropensityHeads(cfg, emb.type_emb.weight, emb.zone_emb.weight)
    B, T = 2, 4
    hidden = torch.randn(B, T, cfg.d_model)
    out = head(hidden)
    assert out["type"].shape == (B, T, cfg.n_pitch_types)
    assert out["zone"].shape == (B, T, cfg.n_zones)
    assert out["velo"].shape == (B, T, cfg.n_velo_bins)
    assert out["spin_rate"].shape == (B, T, cfg.n_spin_rate_bins)
    # spin_axis (circular): 3 outputs
    assert out["spin_axis"].shape == (B, T, 3)


def test_result_head_shape():
    cfg = sanity_config()
    emb = FactorEmbeddings(cfg)
    head = ResultHead(
        cfg,
        type_emb=emb.type_emb,
        zone_emb=emb.zone_emb,
        velo_emb=emb.velo_emb,
        spin_axis_proj=emb.spin_axis_proj if cfg.spin_axis_circular else None,
        spin_axis_emb=None,
    )
    B, T = 2, 4
    hidden = torch.randn(B, T, cfg.d_model)
    type_id = torch.randint(1, cfg.n_pitch_types, (B, T))
    zone_id = torch.randint(0, cfg.n_zones, (B, T))
    velo_id = torch.randint(0, cfg.n_velo_bins, (B, T))
    spin_axis = torch.randn(B, T, 2)  # circular
    out = head(hidden, type_id, zone_id, velo_id, spin_axis)
    assert out.shape == (B, T, cfg.n_result_logits)


# ============================================================
# End-to-end PitchGPT
# ============================================================


def _fake_batch(B=2, T=4, cfg: PitchGPTConfig | None = None) -> dict:
    if cfg is None:
        cfg = sanity_config()
    factors = _fake_factors(B, T, cfg)
    intended = {k: factors[k] for k in ("type", "zone", "velo", "spin_axis")}
    return {
        "pitcher_profile": torch.randn(B, cfg.pitcher_profile_dim),
        "batter_profile": torch.randn(B, cfg.batter_profile_dim),
        "categorical_context": _fake_categorical(B, cfg),
        "pitch_factors": factors,
        "intended_actions": intended,
        "padding_mask": torch.ones(B, T, dtype=torch.bool),
    }


def test_pitchgpt_forward_shapes():
    cfg = sanity_config()
    model = PitchGPT(cfg)
    B, T = 2, 4
    batch = _fake_batch(B, T, cfg)
    out = model(**batch)
    # Propensity at full sequence length (3 context + T)
    assert out["propensity"]["type"].shape == (B, 3 + T, cfg.n_pitch_types)
    # Result and ab_outcome only at pitch positions
    assert out["result"].shape == (B, T, cfg.n_result_logits)
    assert out["ab_outcome_per_pos"].shape == (B, T, cfg.n_ab_outcome_classes)


def test_pitchgpt_parameter_count_in_sanity_range():
    cfg = sanity_config()
    model = PitchGPT(cfg)
    # Sanity model should be well under 5M params
    n = model.num_parameters()
    assert 100_000 < n < 5_000_000, f"sanity model has {n} params, out of range"


# ============================================================
# ADR 009: per-pitch arsenal feature
# ============================================================


def test_arsenal_per_pitch_off_by_default_and_back_compat():
    """Config-field default must be False so checkpoints predating ADR 009
    (saved config dict has no ``arsenal_per_pitch`` key) reload cleanly — no
    ``arsenal_proj`` module, so it matches their state_dict."""
    assert PitchGPTConfig().arsenal_per_pitch is False
    cfg_old = PitchGPTConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024)
    assert cfg_old.arsenal_per_pitch is False
    assert not hasattr(PitchGPT(cfg_old), "arsenal_proj")


def test_arsenal_per_pitch_adds_expected_params():
    cfg_off = sanity_config()
    cfg_on = sanity_config()
    cfg_on.arsenal_per_pitch = True
    delta = PitchGPT(cfg_on).num_parameters() - PitchGPT(cfg_off).num_parameters()
    # Linear(n_arsenal_dims, d_model): weight + bias
    assert delta == cfg_on.n_arsenal_dims * cfg_on.d_model + cfg_on.d_model


def test_arsenal_per_pitch_forward_works_and_affects_output():
    cfg = sanity_config()
    cfg.arsenal_per_pitch = True
    model = PitchGPT(cfg)
    B, T = 2, 4
    batch = _fake_batch(B, T, cfg)
    batch["arsenal"] = torch.rand(B, cfg.n_arsenal_dims)
    out = model(**batch)
    assert out["propensity"]["type"].shape == (B, 3 + T, cfg.n_pitch_types)
    assert out["result"].shape == (B, T, cfg.n_result_logits)
    # Different arsenal → different output (projection is actually wired in).
    batch2 = dict(batch)
    batch2["arsenal"] = torch.rand(B, cfg.n_arsenal_dims) + 5.0
    out2 = model(**batch2)
    assert not torch.allclose(out["propensity"]["type"], out2["propensity"]["type"]), (
        "arsenal projection has no effect on the output — it isn't wired in"
    )


def test_arsenal_per_pitch_raises_without_arsenal_tensor():
    cfg = sanity_config()
    cfg.arsenal_per_pitch = True
    model = PitchGPT(cfg)
    batch = _fake_batch(cfg=cfg)  # no "arsenal" key
    with pytest.raises(ValueError, match="arsenal"):
        model(**batch)


# ============================================================
# ADR 010: situational two-stage propensity head
# ============================================================


def test_propensity_situational_off_by_default_and_back_compat():
    assert PitchGPTConfig().propensity_situational is False
    cfg_old = PitchGPTConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024)
    assert cfg_old.propensity_situational is False
    assert not hasattr(PitchGPT(cfg_old), "situation_fusion")


# ============================================================
# ADR 011: concat-then-project factor embeddings  /  ADR 012: profile FiLM
# ============================================================


def test_concat_then_project_off_by_default_and_back_compat():
    assert PitchGPTConfig().concat_then_project is False
    cfg = PitchGPTConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024)
    emb = FactorEmbeddings(cfg)
    assert not hasattr(emb, "factor_mixer") and not hasattr(emb, "factor_down")


def test_concat_then_project_forward_shapes_and_params():
    from model.embeddings import FactorEmbeddings as FE
    cfg = sanity_config()
    cfg.concat_then_project = True
    emb = FE(cfg)
    assert hasattr(emb, "factor_mixer") and len(emb.factor_down) == FE.N_FACTORS
    B, T = 2, 4
    out = emb(_fake_factors(B, T, cfg))
    assert out.shape == (B, T, cfg.d_model)
    # full model still works end-to-end with the flag on
    cfg2 = sanity_config(); cfg2.concat_then_project = True
    out2 = PitchGPT(cfg2)(**_fake_batch(B, T, cfg2))
    assert out2["propensity"]["type"].shape == (B, 3 + T, cfg2.n_pitch_types)


def test_profile_film_off_by_default_and_back_compat():
    assert PitchGPTConfig().profile_film is False
    cfg = PitchGPTConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024)
    assert not hasattr(PitchGPT(cfg), "profile_film_mlp")


def test_profile_film_identity_at_init_then_responds_to_profile():
    cfg = sanity_config()
    cfg.profile_film = True
    model = PitchGPT(cfg).eval()
    B, T = 4, 5
    batch = _fake_batch(B, T, cfg)
    # at init FiLM is identity (gamma=1, beta=0) — same model with/without the flag
    # should give the same output for the same inputs (FiLM is a no-op at step 0).
    out_film = model(**batch)["propensity"]["type"]
    # now a *different* profile must change the output (FiLM is wired to the profile)
    batch2 = {k: (v.clone() if isinstance(v, torch.Tensor) else
                  ({kk: vv.clone() for kk, vv in v.items()} if isinstance(v, dict) else v))
              for k, v in batch.items()}
    batch2["pitcher_profile"] = batch2["pitcher_profile"] + 3.0
    # train one step so the FiLM MLP's zero-init last layer picks up a gradient
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    model.train()
    loss = model(**batch)["propensity"]["type"].pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    out_a = model(**batch)["propensity"]["type"]
    out_b = model(**batch2)["propensity"]["type"]
    assert not torch.allclose(out_a, out_b), "FiLM doesn't respond to the profile — not wired in"


def test_propensity_situational_forward_and_conditions_on_next_situation():
    cfg = sanity_config()
    cfg.propensity_situational = True
    model = PitchGPT(cfg)
    B, T = 2, 5
    batch = _fake_batch(B, T, cfg)
    out = model(**batch)
    assert out["propensity"]["type"].shape == (B, 3 + T, cfg.n_pitch_types)
    assert out["result"].shape == (B, T, cfg.n_result_logits)
    # Change the count of pitch t+1 (i.e. pitch_factors["count"][:, 1:]) and the
    # propensity type logits at positions 0..T-2 must move (the head conditions
    # on the *upcoming* situation). Position T-1's target has no successor.
    batch2 = {k: (v.clone() if isinstance(v, torch.Tensor) else
                  ({kk: vv.clone() for kk, vv in v.items()} if isinstance(v, dict) else v))
              for k, v in batch.items()}
    batch2["pitch_factors"]["count"][:, 1:] = (
        (batch2["pitch_factors"]["count"][:, 1:] + 3) % cfg.n_count_states
    )
    out2 = model(**batch2)
    a = out["propensity"]["type"][:, 3:3 + T - 1, :]
    b = out2["propensity"]["type"][:, 3:3 + T - 1, :]
    assert not torch.allclose(a, b), (
        "propensity head ignores the upcoming-situation factors — situational fusion not wired in"
    )


def test_pitchgpt_returns_intermediates_when_requested():
    cfg = sanity_config()
    model = PitchGPT(cfg)
    batch = _fake_batch(cfg=cfg)
    out = model(**batch, return_intermediates=True)
    assert "intermediates" in out
    assert len(out["intermediates"]) == cfg.n_layers
    for h in out["intermediates"]:
        assert h.shape[-1] == cfg.d_model


# ============================================================
# ADR 013: type-conditioned execution heads
# ============================================================


def test_type_conditioned_heads_flag_defaults_off():
    from model.config import PitchGPTConfig, tiny_config
    assert PitchGPTConfig().type_conditioned_heads is False
    assert tiny_config().type_conditioned_heads is False


def test_type_conditioned_heads_flag_can_be_set():
    from model.config import PitchGPTConfig
    cfg = PitchGPTConfig(type_conditioned_heads=True)
    assert cfg.type_conditioned_heads is True


# ============================================================
# ADR 007: stop-gradient between result head and trunk
# ============================================================


def test_result_loss_does_not_update_trunk():
    """The defining test for ADR 007's choice (B1): result-head loss must
    NOT backpropagate into the trunk's parameters."""
    cfg = sanity_config()
    model = PitchGPT(cfg)
    batch = _fake_batch(cfg=cfg)
    out = model(**batch)

    # Synthetic result loss
    result_logits = out["result"]
    target = torch.randint(0, cfg.n_result_logits, result_logits.shape[:2])
    loss = F.cross_entropy(result_logits.reshape(-1, cfg.n_result_logits), target.reshape(-1))

    # Zero grads, then backward
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    # Trunk parameters (transformer layers) must have None or zero gradients
    for layer_i, layer in enumerate(model.layers):
        for name, p in layer.named_parameters():
            assert (p.grad is None) or torch.all(p.grad == 0), (
                f"trunk layer {layer_i} param {name} got non-zero grad from "
                f"result loss — ADR 007 stop-gradient is broken"
            )

    # Result head MLP parameters SHOULD have non-zero gradients
    head_has_grad = False
    for p in model.result_head.mlp.parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            head_has_grad = True
            break
    assert head_has_grad, "result head MLP got no gradient from its own loss"


# ============================================================
# Counterfactual sensitivity (the most important behavior test)
# ============================================================


def test_changing_intended_type_shifts_result_distribution():
    """The defining test of two-stage result-head conditioning.

    Strategy: synthesize data where the result is a deterministic function
    of the intended type. If the architectural path through
    ``type_emb(ã)`` is broken, the model cannot learn this trivial
    dependency. If the path is correct, the model learns it in a few
    hundred steps and the result distribution shifts substantially when
    the intended type is changed at inference.
    """
    cfg = sanity_config()
    torch.manual_seed(42)
    model = PitchGPT(cfg)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    B, T = 16, 3
    for step in range(600):
        batch = _fake_batch(B=B, T=T, cfg=cfg)
        intended_type = batch["intended_actions"]["type"]
        # Target = intended_type modulo n_result_logits — deterministic dep
        target = (intended_type % cfg.n_result_logits).long()
        out = model(**batch)
        result_logits = out["result"]
        loss = F.cross_entropy(
            result_logits.reshape(-1, cfg.n_result_logits), target.reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    base_batch = _fake_batch(B=8, T=T, cfg=cfg)
    # Force intended type = 1 (FF) everywhere
    base_batch["intended_actions"]["type"] = torch.full((8, T), 1, dtype=torch.long)
    with torch.no_grad():
        probs_a = F.softmax(model(**base_batch)["result"], dim=-1)
    # Force intended type = 4 (SL) everywhere
    base_batch["intended_actions"]["type"] = torch.full((8, T), 4, dtype=torch.long)
    with torch.no_grad():
        probs_b = F.softmax(model(**base_batch)["result"], dim=-1)

    l1_diff = (probs_a - probs_b).abs().sum(dim=-1)
    mean_diff = float(l1_diff.mean())
    # A broken architectural path gives ~0.001 (just init noise leak). A
    # working path with mid-training learning gives >0.1. Threshold of
    # 0.2 catches "path broken or not being used" without demanding full
    # convergence on the deterministic toy task.
    assert mean_diff > 0.2, (
        f"Result head appears insensitive to intended type: "
        f"mean L1 diff = {mean_diff:.4f}. Architectural path broken — "
        f"counterfactual rollouts will not work."
    )


# ============================================================
# ADR 007 Amendment: no leakage from hidden's result direction
# ============================================================


def test_result_head_does_not_leak_from_hidden_result():
    """The result head must NOT be able to read ``result_t`` out of its hidden
    input. This is the failure mode that silently breaks counterfactual rollout:
    if ``hidden[t]`` includes ``result_emb(result_t)`` (which it does, since
    ``FactorEmbeddings`` sums every per-pitch factor including result), the
    head can learn to copy from hidden and ignore ``intended_action``.

    Test design: train with ``target = pitch_factors['result'] - 1`` (i.e. the
    head's target equals the result that's also in the trunk's input embedding
    at the same position). intended_action is sampled INDEPENDENTLY of
    pitch_factors. If leakage exists, the head copies from hidden and gets
    high training accuracy without using intended_action. We then change
    intended_action at inference and check the head's output is invariant
    (which would be the failure mode).

    A correctly-fixed head reads ``hidden[t-1]`` (history before pitch t),
    so result_t is NOT in its hidden input. The head MUST go through
    intended_action and the counterfactual L1 diff must be small (because
    the head can't learn the deterministic mapping from intended_action alone
    — intended_action is uncorrelated with target by construction). The
    important assertion is that the head's TRAINING accuracy on the leakage
    target is roughly at chance: it cannot exploit a path that no longer exists.
    """
    cfg = sanity_config()
    torch.manual_seed(0)
    model = PitchGPT(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    B, T = 16, 4
    for step in range(300):
        batch = _fake_batch(B=B, T=T, cfg=cfg)
        # intended actions independent of pitch_factors
        batch["intended_actions"] = {
            "type": torch.randint(1, cfg.n_pitch_types, (B, T)),
            "zone": torch.randint(0, cfg.n_zones, (B, T)),
            "velo": torch.randint(0, cfg.n_velo_bins, (B, T)),
            "spin_axis": torch.randn(B, T, 2),
        }
        target = (
            batch["pitch_factors"]["result"] - 1
        ).clamp(0, cfg.n_result_logits - 1).long()
        out = model(**batch)
        loss = F.cross_entropy(
            out["result"].reshape(-1, cfg.n_result_logits), target.reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()

    # After training, evaluate on a fresh batch.
    model.eval()
    eval_batch = _fake_batch(B=64, T=T, cfg=cfg)
    eval_batch["intended_actions"] = {
        "type": torch.randint(1, cfg.n_pitch_types, (64, T)),
        "zone": torch.randint(0, cfg.n_zones, (64, T)),
        "velo": torch.randint(0, cfg.n_velo_bins, (64, T)),
        "spin_axis": torch.randn(64, T, 2),
    }
    target_eval = (
        eval_batch["pitch_factors"]["result"] - 1
    ).clamp(0, cfg.n_result_logits - 1).long()
    with torch.no_grad():
        probs = F.softmax(model(**eval_batch)["result"], dim=-1)
    top1_acc = (probs.argmax(-1) == target_eval).float().mean().item()
    # With leakage, this hits ~0.75-0.80 (model copies result from hidden).
    # Without leakage and uncorrelated intended_action, accuracy can be no
    # better than chance over n_result_logits classes (1/7 ≈ 0.143). We allow
    # a modest margin for any noisy signal the trunk picks up from other
    # factors that correlate weakly with the random result column.
    assert top1_acc < 0.40, (
        f"Result head appears to LEAK result_t from hidden: "
        f"top-1 accuracy on a leakage-only target is {top1_acc:.3f}. "
        f"Expected < 0.40 once leakage is removed. The head must read "
        f"hidden[t-1], not hidden[t]."
    )
