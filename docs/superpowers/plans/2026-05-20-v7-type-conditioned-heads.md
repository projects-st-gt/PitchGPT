# v7 Step 1 — Type-Conditioned Execution Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the zone / velo / spin_rate / spin_axis propensity heads condition on the pitch type, so sampled and counterfactual pitches are coherent (no curveballs at 96 mph).

**Architecture:** Per ADR-013 Decision 1. When `config.type_conditioned_heads` is set, a fusion MLP combines the trunk hidden state at position `t` with the embedding of the *next* pitch's type (`type[t+1]`, teacher-forced at training, the shifted-left of `pitch_factors["type"]`). The execution heads (zone/velo/spin_rate/spin_axis) read this fused vector; the **type head still reads the raw hidden** (type is not conditioned on itself). This mirrors the existing `propensity_situational` fusion (ADR-010) — lowest-ripple, keeps the weight-tied type/zone projections intact. This plan resolves ADR-013's open question on *how the type embedding enters the heads*: **fusion MLP of `[hidden, type_emb]`**, not FiLM or an added term.

**Tech Stack:** PyTorch, the existing PitchGPT model (`model/`), pytest, Modal (retrain).

**Scope:** This plan is Step 1 of ADR-013 only — the factorization. Cross-AB context (Step 2) and the zone EMD loss (Step 3) get their own plans after this Step's ablation validates.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `model/config.py` | Modify | Add `type_conditioned_heads` flag |
| `model/heads.py` | Modify | `PropensityHeads.forward` accepts a separate `hidden_exec` for the execution factors |
| `model/pitchgpt.py` | Modify | Build the `type_fusion` MLP; wire the shifted next-type into it; expose a reusable execution-head method for inference marginalization |
| `causal/nuisance.py` | Modify | `ForwardOut` / `forward` expose type-conditional + marginalized execution outputs |
| `causal/g_computation.py` | Modify | Rollout samples zone/velo/spin from the type-conditional heads after the type is sampled/clamped |
| `tests/test_pitchgpt_model.py` | Modify | Unit tests for the conditioning behavior + marginalization |
| `tests/test_causal_rollout.py` | Create | Rollout-coherence test (do(type=CU) shifts velo) |

---

## Task 1: Config flag

**Files:**
- Modify: `model/config.py:98` (after the `propensity_situational` field)
- Test: `tests/test_pitchgpt_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pitchgpt_model.py`:

```python
def test_type_conditioned_heads_flag_defaults_off():
    from model.config import PitchGPTConfig, tiny_config
    assert PitchGPTConfig().type_conditioned_heads is False
    assert tiny_config().type_conditioned_heads is False

def test_type_conditioned_heads_flag_can_be_set():
    from model.config import PitchGPTConfig
    cfg = PitchGPTConfig(type_conditioned_heads=True)
    assert cfg.type_conditioned_heads is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_type_conditioned_heads_flag_defaults_off -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument` / `AttributeError`.

- [ ] **Step 3: Add the config field**

In `model/config.py`, immediately after the `propensity_situational: bool = False` field (line ~98), add:

```python
    # Type-conditioned execution heads (ADR-013 Decision 1). When True, the
    # zone/velo/spin_rate/spin_axis propensity heads condition on the NEXT
    # pitch's type: a fusion MLP combines the trunk hidden at position t with
    # the embedding of type[t+1] (teacher-forced at training; the sampled or
    # intervened type at rollout). The TYPE head is unchanged — it still reads
    # the raw hidden, since type is not conditioned on itself. Fixes incoherent
    # sampled pitches in g-computation rollouts (e.g. a curveball at 96 mph).
    # Default OFF so pre-v7 checkpoints reload unchanged.
    type_conditioned_heads: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pitchgpt_model.py -k type_conditioned_heads_flag -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add model/config.py tests/test_pitchgpt_model.py
git commit -m "feat(model): add type_conditioned_heads config flag (ADR-013)"
```

---

## Task 2: PropensityHeads accepts a separate execution-hidden

**Files:**
- Modify: `model/heads.py:71-84` (`PropensityHeads.forward`)
- Test: `tests/test_pitchgpt_model.py`

The type head reads `hidden`; the execution heads read `hidden_exec` when supplied (falls back to `hidden` for back-compat).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pitchgpt_model.py`:

```python
def test_propensity_heads_separate_exec_hidden():
    import torch
    from model.config import tiny_config
    from model.heads import PropensityHeads
    from model.embeddings import FactorEmbeddings

    cfg = tiny_config()
    emb = FactorEmbeddings(cfg)
    heads = PropensityHeads(cfg, emb.type_emb.weight, emb.zone_emb.weight)

    hidden = torch.randn(2, 5, cfg.d_model)
    hidden_exec = torch.randn(2, 5, cfg.d_model)

    out_default = heads(hidden)                       # hidden_exec=None → uses hidden
    out_split = heads(hidden, hidden_exec=hidden_exec)

    # Type head ignores hidden_exec — identical in both calls.
    assert torch.allclose(out_default["type"], out_split["type"])
    # Execution heads differ because hidden_exec differs.
    assert not torch.allclose(out_default["zone"], out_split["zone"])
    assert not torch.allclose(out_default["velo"], out_split["velo"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_propensity_heads_separate_exec_hidden -v`
Expected: FAIL — `forward() got an unexpected keyword argument 'hidden_exec'`.

- [ ] **Step 3: Modify `PropensityHeads.forward`**

Replace `model/heads.py:71-84` (the `forward` method) with:

```python
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
            "type": type_logits,
            "zone": zone_logits,
            "velo": velo_logits,
            "spin_rate": spin_rate_logits,
            "spin_axis": spin_axis_out,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_propensity_heads_separate_exec_hidden -v`
Expected: PASS.

Run the full model test file to confirm no regression:
Run: `uv run pytest tests/test_pitchgpt_model.py -v`
Expected: all pass (the `hidden_exec=None` default preserves old behaviour).

- [ ] **Step 5: Commit**

```bash
git add model/heads.py tests/test_pitchgpt_model.py
git commit -m "feat(model): PropensityHeads accepts separate execution-hidden"
```

---

## Task 3: Build the type-fusion MLP and wire it in `PitchGPT.forward`

**Files:**
- Modify: `model/pitchgpt.py:99-110` (add `type_fusion` module in `__init__`) and `model/pitchgpt.py:302-325` (wire in `forward`)
- Test: `tests/test_pitchgpt_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pitchgpt_model.py` (reuse the existing batch-builder fixture in that file — search for how other tests build a `batch`; the helper there is `_tiny_batch()` or equivalent. If no helper exists, build a batch with `PitchGPTAtBatDataset` as the other tests do):

```python
def test_type_conditioned_heads_change_execution_logits():
    """With the flag on, changing the next-pitch type changes zone/velo/spin
    logits but NOT the type logits."""
    import torch
    from model.config import tiny_config
    from model.pitchgpt import PitchGPT

    cfg = tiny_config()
    cfg.type_conditioned_heads = True
    model = PitchGPT(cfg).eval()

    batch = _tiny_batch(cfg)  # existing helper in this test file
    NC = PitchGPT.N_CONTEXT_TOKENS

    with torch.no_grad():
        out_a = model(**batch)
        # Flip every next-pitch type to FF (model id 1) and re-run.
        batch_b = {**batch, "pitch_factors": {**batch["pitch_factors"]}}
        flipped = batch_b["pitch_factors"]["type"].clone()
        flipped[flipped != 0] = 1  # all non-PAD → FF
        batch_b["pitch_factors"]["type"] = flipped
        out_b = model(**batch_b)

    # Execution heads at pitch positions must respond to the type change.
    za = out_a["propensity"]["zone"][:, NC:, :]
    zb = out_b["propensity"]["zone"][:, NC:, :]
    assert not torch.allclose(za, zb), "zone head ignored the type change"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_type_conditioned_heads_change_execution_logits -v`
Expected: FAIL — `assert not torch.allclose(...)` fails (zone unchanged, because nothing wires type into the heads yet).

- [ ] **Step 3: Add the `type_fusion` module to `__init__`**

In `model/pitchgpt.py`, after the `propensity_situational` block (ends ~line 110), add:

```python
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
```

- [ ] **Step 4: Wire the fusion into `forward`**

In `model/pitchgpt.py`, in the propensity section (step 6, ~line 279-304), replace the `if self.config.propensity_situational or self.config.inject_profiles_to_head:` block's final `propensity_logits = self.propensity(x)` calls so the execution-hidden is computed. Insert this immediately **before** step 6b (`if self.config.propensity_type_two_stage:`), replacing whatever `propensity_logits = self.propensity(...)` was just assigned:

```python
        # 6a. Type-conditioned execution heads (ADR-013). Fuse the pitch-
        #     position hidden with the NEXT pitch's type embedding. type[t+1]
        #     is the shift-left of pitch_factors["type"] (last position → PAD,
        #     loss-ignored). At rollout the caller writes the sampled/intervened
        #     type into pitch_factors["type"], so the same path serves do(·).
        if self.config.type_conditioned_heads:
            NC = self.N_CONTEXT_TOKENS
            def _shift_left_type(v: torch.Tensor) -> torch.Tensor:
                return torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], dim=1)
            next_type = _shift_left_type(pitch_factors["type"])          # (B, T)
            next_type_emb = self.embed.type_emb(next_type)               # (B, T, d)
            base_hidden = x[:, NC:, :] if not (
                self.config.propensity_situational or self.config.inject_profiles_to_head
            ) else x_for_prop[:, NC:, :]
            hidden_exec_pitch = self.type_fusion(
                torch.cat([base_hidden, next_type_emb], dim=-1)
            )                                                            # (B, T, d)
            # Re-run the heads with the conditioned execution-hidden. The type
            # head still reads the (un-fused) hidden used above.
            hidden_for_heads = x_for_prop if (
                self.config.propensity_situational or self.config.inject_profiles_to_head
            ) else x
            hidden_exec_full = hidden_for_heads.clone()
            hidden_exec_full[:, NC:, :] = hidden_exec_pitch
            propensity_logits = self.propensity(hidden_for_heads, hidden_exec=hidden_exec_full)
```

Note: this block runs *after* the existing step-6 `propensity_logits = self.propensity(...)` assignment and overrides it when the flag is on. Leave the existing step-6 block as-is.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_type_conditioned_heads_change_execution_logits -v`
Expected: PASS.

Run the full file:
Run: `uv run pytest tests/test_pitchgpt_model.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add model/pitchgpt.py tests/test_pitchgpt_model.py
git commit -m "feat(model): wire type-conditioned execution heads in forward (ADR-013)"
```

---

## Task 4: Reusable execution-head method for inference marginalization

**Files:**
- Modify: `model/pitchgpt.py` (add a public method `execution_logits_for_type`)
- Test: `tests/test_pitchgpt_model.py`

For inference, the marginal location distribution is `Σ_type P(zone|type,h)·π̂(type|h)`. This needs the execution heads evaluated for each of the 7 types. Expose a method that, given the trunk hidden and a chosen type id, returns the execution logits.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pitchgpt_model.py`:

```python
def test_execution_logits_for_type_marginal_sums_to_one():
    import torch
    import torch.nn.functional as F
    from model.config import tiny_config
    from model.pitchgpt import PitchGPT
    from data.dataset import PITCH_TYPES, MODEL_TYPE_ID

    cfg = tiny_config()
    cfg.type_conditioned_heads = True
    model = PitchGPT(cfg).eval()
    batch = _tiny_batch(cfg)

    with torch.no_grad():
        out = model(**batch, return_intermediates=False)
        type_probs = F.softmax(out["propensity"]["type"], dim=-1)  # (B, T_total, 8)
        # Marginal zone = Σ_type π̂(type) · P(zone | type)
        marginal = torch.zeros_like(out["propensity"]["zone"])
        for pt in PITCH_TYPES:
            tid = MODEL_TYPE_ID[pt]
            cond = model.execution_logits_for_type(batch, type_id=tid)["zone"]
            w = type_probs[..., tid:tid + 1]
            marginal = marginal + w * F.softmax(cond, dim=-1)

    NC = PitchGPT.N_CONTEXT_TOKENS
    s = marginal[:, NC:, :].sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_execution_logits_for_type_marginal_sums_to_one -v`
Expected: FAIL — `AttributeError: 'PitchGPT' object has no attribute 'execution_logits_for_type'`.

- [ ] **Step 3: Add the method**

Add to `model/pitchgpt.py`, as a method of `PitchGPT` (after `forward`):

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_execution_logits_for_type_marginal_sums_to_one -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/pitchgpt.py tests/test_pitchgpt_model.py
git commit -m "feat(model): execution_logits_for_type for inference marginalization"
```

---

## Task 5: Expose conditional + marginal execution outputs in `nuisance.py`

**Files:**
- Modify: `causal/nuisance.py` (`ForwardOut`, `NuisanceModels.forward`)
- Test: `tests/test_pitchgpt_model.py` (or `tests/test_nuisance.py` if it exists — check first)

Add a marginalized zone/velo/spin to `ForwardOut` so downstream causal code and the API get the type-marginal location without re-deriving it. When the checkpoint has `type_conditioned_heads=False`, the marginal equals the plain head output (back-compat).

- [ ] **Step 1: Write the failing test (named-number smoke test)**

Add a test that loads the v6 checkpoint (still `type_conditioned_heads=False`), runs `forward` on a real AB, and asserts the new `marginal_zone_probs` field exists and equals the plain zone probs (since v6 is unconditioned):

```python
def test_nuisance_exposes_marginal_zone_backcompat():
    import numpy as np, pandas as pd
    from pathlib import Path
    from causal.nuisance import NuisanceModels, build_single_ab_batch

    ck = Path("checkpoints_modal/tiny-fold0-v6/checkpoint_calibrated.pt")
    nu = NuisanceModels(ck, device="cpu")
    val = pd.read_parquet("data/augmented/2024/2024-04-01.parquet")
    g = (val.sort_values(["game_pk","at_bat_number","pitch_number"])
            .groupby(["game_pk","at_bat_number"]))
    ab = next(grp for _, grp in g if len(grp) >= 4)
    batch = build_single_ab_batch(nu, ab.reset_index(drop=True))
    out = nu.forward(batch)
    # v6 has no type conditioning → marginal == plain zone head output.
    assert out.marginal_propensity_probs["zone"].shape == out.propensity_probs["zone"].shape
    z = out.marginal_propensity_probs["zone"][0, nu.model.N_CONTEXT_TOKENS + 1]
    print(f"marginal zone @ pitch1 sums to {float(z.sum()):.4f}")
    assert abs(float(z.sum()) - 1.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_nuisance_exposes_marginal_zone_backcompat -v`
Expected: FAIL — `AttributeError: 'ForwardOut' object has no attribute 'marginal_propensity_probs'`.

- [ ] **Step 3: Implement the marginal in `NuisanceModels.forward`**

In `causal/nuisance.py`, add `marginal_propensity_probs: dict[str, torch.Tensor]` to the `ForwardOut` dataclass (after `propensity_probs`). In `forward`, after computing `prop_probs`, add:

```python
        # Type-marginal execution distributions (ADR-013). For a checkpoint
        # WITHOUT type_conditioned_heads this is identical to prop_probs (the
        # heads were already type-marginal). With it, marginalize:
        #   P(exec | h) = Σ_type π̂(type|h) · P(exec | type, h).
        marginal_probs: dict[str, torch.Tensor] = dict(prop_probs)
        if getattr(self.model.config, "type_conditioned_heads", False):
            from data.dataset import PITCH_TYPES, MODEL_TYPE_ID
            type_p = prop_probs["type"]  # (B, T, 8) post-temperature
            for head in ("zone", "velo", "spin_rate"):
                acc = torch.zeros_like(prop_probs[head])
                for pt in PITCH_TYPES:
                    tid = MODEL_TYPE_ID[pt]
                    cond_logits = self.model.execution_logits_for_type(bd, type_id=tid)[head]
                    cond = F.softmax(self._apply_temp(cond_logits.cpu().float(), head), dim=-1)
                    acc = acc + type_p[..., tid:tid + 1] * cond
                marginal_probs[head] = acc
        # then pass marginal_probs into the ForwardOut(...) constructor
```

Add `marginal_propensity_probs=marginal_probs` to the `ForwardOut(...)` return.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pitchgpt_model.py::test_nuisance_exposes_marginal_zone_backcompat -v`
Expected: PASS. Confirm the printed line shows `marginal zone @ pitch1 sums to 1.0000`.

- [ ] **Step 5: Commit**

```bash
git add causal/nuisance.py tests/test_pitchgpt_model.py
git commit -m "feat(causal): expose type-marginal execution distributions in ForwardOut"
```

---

## Task 6: g-computation rollout samples execution from the type-conditional heads

**Files:**
- Modify: `causal/g_computation.py` (the per-step sampling block, ~line 440-492)
- Test: Create `tests/test_causal_rollout.py`

After the type is sampled/clamped at a rollout step, the zone/velo/spin must be sampled from `P(· | sampled_type, h)`, not the marginal. Because the conditioning input is `pitch_factors["type"]` shift-left, writing the sampled type into `full["type"][:, step]` *before* the forward pass that reads zone/velo already conditions the heads — verify the ordering and add the coherence test.

- [ ] **Step 1: Write the failing test**

Create `tests/test_causal_rollout.py`:

```python
"""Rollout coherence: do(type=CU) must yield curveball-like velo."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from causal.nuisance import NuisanceModels
from causal.g_computation import g_compute

CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")

def test_do_curveball_shifts_velo_slow():
    """Under do(type=CU), sampled velo bins should skew slower than under
    do(type=FF). Requires a v7 (type_conditioned_heads) checkpoint."""
    if not CKPT.exists():
        import pytest; pytest.skip("v7 checkpoint not yet trained")
    nu = NuisanceModels(CKPT, device="cpu")
    val = pd.read_parquet("data/augmented/2024/2024-04-01.parquet")
    g = (val.sort_values(["game_pk","at_bat_number","pitch_number"])
            .groupby(["game_pk","at_bat_number"]))
    ab = next(grp for _, grp in g if len(grp) >= 5).reset_index(drop=True)

    r_ff = g_compute(nu, ab, intervention_position=1, intervention_type="FF",
                     n_paths=300, rng_seed=0)
    r_cu = g_compute(nu, ab, intervention_position=1, intervention_type="CU",
                     n_paths=300, rng_seed=0)
    # velo is binned low→high; the intervention pitch's velo bin mean should
    # be lower for CU than FF. (Bins are type-relative deciles; this is a
    # coherence check, not an exact-mph claim.)
    print(f"mean velo bin: FF={r_ff.intervention_velo_bin_mean:.2f} "
          f"CU={r_cu.intervention_velo_bin_mean:.2f}")
    assert r_cu.intervention_velo_bin_mean < r_ff.intervention_velo_bin_mean
```

This test requires `g_compute` / `RolloutResult` to expose `intervention_velo_bin_mean` — the mean sampled velo bin at the intervention position. Add that field.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_causal_rollout.py -v`
Expected: FAIL (or SKIP if no v7 checkpoint) — `AttributeError: 'RolloutResult' object has no attribute 'intervention_velo_bin_mean'`.

- [ ] **Step 3: Record the intervention-position velo in the rollout**

In `causal/g_computation.py`: (a) confirm that at step `k` the sampled type is written into `full["type"][:, step]` **before** the second forward pass that reads zone/velo (it is — the existing code writes `full["type"]` then re-runs forward for the result head; the *next* step's forward then reads the conditioned heads). The conditioning is correct as long as `type_conditioned_heads` is on, because the head reads `pitch_factors["type"]` shift-left. Add a comment noting this. (b) Add `intervention_velo_bin_mean` to `RolloutResult` and populate it: capture `sampled_velo` at `step == k` and store `float(sampled_velo.mean())`.

```python
# in g_compute, near the per-path-state allocation:
intervention_velo_bin_mean = float("nan")
# ... inside the loop, immediately after sampled_velo is computed at step==k:
if step == k:
    intervention_velo_bin_mean = float(sampled_velo[active].mean()) if active.any() else float("nan")
# ... add to the RolloutResult(...) construction:
intervention_velo_bin_mean=intervention_velo_bin_mean,
```

Add the field to the `RolloutResult` dataclass with the others: `intervention_velo_bin_mean: float`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_causal_rollout.py -v`
Expected: SKIP until the v7 checkpoint exists (Task 7); after Task 7, PASS with `CU` velo bin mean below `FF`.

- [ ] **Step 5: Commit**

```bash
git add causal/g_computation.py tests/test_causal_rollout.py
git commit -m "feat(causal): record intervention velo + rollout coherence test"
```

---

## Task 7: Retrain `tiny` fold 0 with the factorization (operational)

**Files:** none — Modal training run.

- [ ] **Step 1: Confirm the training flag is plumbed**

`scripts/train_pitchgpt.py` reads `PitchGPTConfig`. Add a `--type-conditioned-heads` CLI flag mirroring the existing `--concat-then-project` flag (search `train_pitchgpt.py` for `concat_then_project` and copy the argparse + `cfg.X =` pattern). The loss in `compute_losses` does **not** change — head output shapes are unchanged.

- [ ] **Step 2: Kick off the Modal run (detached)**

```bash
modal run --detach modal_app.py::train_remote \
  --size tiny --fold-id 0 --epochs 3 \
  --type-conditioned-heads \
  --run-name tiny-fold0-v7
```

Expected: a detached run id. ~5h on A100. `--detach` is mandatory (roadmap §7 gotcha #1).

- [ ] **Step 3: Pull + calibrate the checkpoint**

After completion, pull to `checkpoints_modal/tiny-fold0-v7/`, then:

```bash
uv run python -m scripts.calibrate_pitchgpt --ckpt checkpoints_modal/tiny-fold0-v7/checkpoint_best.pt
```

Expected: `checkpoint_calibrated.pt` with stored temperatures.

---

## Task 8: Eval + rollout-coherence ablation (operational — the Step-1 gate)

**Files:** none — eval run + comparison.

- [ ] **Step 1: Full eval table vs v6**

```bash
make eval
```

Compare to `eval/results/latest/` for v6. **Gate:** next-pitch type top-1 within ±0.5 pp of v6 (structurally protected); zone/velo/spin_rate head NLL + ECE improve or hold.

- [ ] **Step 2: Rollout-coherence test**

Run: `uv run pytest tests/test_causal_rollout.py -v`
Expected: PASS — `do(type=CU)` velo bin mean below `do(type=FF)`.

- [ ] **Step 3: v6-vs-v7 causal-estimate ablation (ADR-013 mandate)**

Run the population AIPW contrast on the same slice with the v6 and v7 checkpoints; record `τ̂` and CI for both. Document the delta in `docs/decisions/013-v7-model-revision.md` under a new "Step-1 results" section — this quantifies how much the incoherent-rollout v6 estimates were biased.

- [ ] **Step 4: Decide the velo/spin pairwise-term question**

Per ADR-013's open question: if the rollout-coherence test shows residual incoherence (e.g. velo and spin disagree on the pitch identity), note it for the Step-2 plan. Otherwise confirm "execution heads independent given type" is sufficient.

---

## Self-Review

- **Spec coverage:** ADR-013 Decision 1 (type-conditioned execution heads) — Tasks 1-6. Training — Task 7. Eval + the mandated v6/v7 ablation — Task 8. Cross-AB (Decision 2) and EMD (Decision 3) are explicitly out of this plan's scope (separate plans, per the ADR's separable-attribution sequencing).
- **Placeholder scan:** every code step has complete code; operational steps (7-8) have exact commands. The one dependency on an external helper — `_tiny_batch(cfg)` in `tests/test_pitchgpt_model.py` — is flagged in Task 3 Step 1 with a fallback instruction (build via `PitchGPTAtBatDataset` as sibling tests do).
- **Type consistency:** `type_conditioned_heads` (config), `hidden_exec` (PropensityHeads.forward kwarg), `execution_logits_for_type` (PitchGPT method), `marginal_propensity_probs` (ForwardOut field), `intervention_velo_bin_mean` (RolloutResult field) — each defined once and referenced consistently.
- **Known risk carried from ADR-013:** the convention crossing — `pitch_factors["type"]` is 1-indexed (PAD=0, types 1-7) and `embed.type_emb` is sized 8; `MODEL_TYPE_ID` is used in Tasks 4-5 for the marginalization. The implementer must print named per-class checks per CLAUDE.md's bug-prevention discipline.
