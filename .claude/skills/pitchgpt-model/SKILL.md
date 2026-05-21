---
name: pitchgpt-model
description: Use this skill whenever working on the PitchGPT model itself — factored embeddings, the transformer backbone, the two-stage result head, the training loop, attention masking, model sizing, mixed precision, or saving and loading checkpoints. Trigger this any time the user mentions embeddings, factors, transformer, attention mask, training, AdamW, label smoothing, model size, the propensity head, the outcome head, or anything that affects the architecture of the autoregressive pitch model. The model serves dual duty as both π̂ and μ̂ in the causal layer, so changes here have downstream causal-validity implications — read this skill before touching anything in `model/`.
---

# PitchGPT Model

One transformer, multiple heads, dual purpose. The autoregressive next-pitch
head is the propensity model π̂(a | h); the conditional outcome head is μ̂(y | a, h).
The causal layer relies on both being well-calibrated. Changes here ripple into
every causal estimate.

## Architectural commitments

These are not negotiable without an ADR:

- **Factored embeddings, not flat vocabulary.** Each pitch factor has its own
  small embedding table; embeddings are summed before entering the transformer.
- **Two-stage result head.** The result head is conditioned on the predicted
  pitch type, zone, and velocity (via their embeddings), not just the hidden
  state. This is what enables interventional rollouts.
- **Pre-norm transformer** (LayerNorm before attention and FFN, GPT-2 style).
- **Causal attention mask** with explicit blocking across at-bat boundaries
  when sequences are packed.
- **bf16 mixed precision** throughout. fp32 master weights for the optimizer.

## Factored embedding spec

See `model/embeddings.py:FactorEmbeddings`. Each pitch contributes a sum of
these embeddings, projected to `d_model`:

| factor             | vocab | source                                            |
|--------------------|-------|---------------------------------------------------|
| pitch_type         | 8     | 7 canonical types + PAD                           |
| zone               | 26    | 25 zones + MISSING                                |
| velo_bin           | 11    | 10 type-relative deciles + MISSING                |
| spin_rate          | 9     | 8 spin-rate bins + MISSING                        |
| spin_axis          | (sin/cos) | 2D continuous, `Linear(2, d_model)` (circular) |
| result             | 8     | 7 result classes + PAD                            |
| count              | 12    | 12 count states (balls × strikes)                 |
| runners            | 8     | 8 base-occupancy states                           |
| outs               | 3     | 0, 1, 2                                           |
| pos                | 15    | pitch index within at-bat                         |
| pitcher_fatigue    | 12    | per-pitch cumulative count buckets (ADR 003 A1)  |
| arsenal            | 14 (cont.) | `Linear(14, d_model)` over the pitcher's 7 trailing-window usage rates + 7 `has_pitch` flags, raw 0–1 (ADR 009) |

`arsenal` is **not** an embedding-table factor — it's the 14-dim
arsenal+has-pitch sub-vector of the pitcher profile, pulled from the *raw*
(pre-standardisation) profile vector, projected by `Linear(14, d_model)` and
added to every pitch token. Gated on `config.arsenal_per_pitch` (default
`False` so checkpoints predating ADR 009 reload cleanly; `train()` defaults it
`True` for new runs). It exists because the LSTM baseline gets the arsenal as a
direct per-pitch feature while the transformer otherwise only sees it buried in
a context token behind an attention hop — see ADR 009 and the
`statcast-pipeline` / `eval-protocol` skills for the gap analysis.

Total embedding parameters: ~146 × d_model (~131 factor-embedding + ~15 for the
arsenal projection), vs ~35M × d_model for a flat vocabulary. The factored
design is engineering, not novelty — frame it that way in writeups.

Spin axis is circular by default (`Linear(2, d_model)` over `[sin θ, cos θ]`).
A categorical-bin variant (vocab 13) is retained behind
`config.spin_axis_circular = False` for ablation.

## Context tokens

Player profiles and game-context features become "context tokens" prepended
to each at-bat sequence — analogous to a system prompt:

1. `pitcher_profile_vector` (223-dim) → MLP → d_model → context token 0
2. `batter_profile_vector` (91-dim) → MLP → d_model → context token 1
3. Categorical confounders (12 categoricals) summed → d_model → context token 2:
   - `p_throws` (3), `stand` (3)
   - `ballpark` (64), `umpire` (256), `catcher` (384)
   - `inning_bucket` (14), `score_diff_bucket` (11), `inning_half` (3)
     — replaced derived `leverage` per ADR 003 Amendment 1 (2026-05-10)
   - `days_rest` (9), `tto` (5), `temp` (7), `roof` (3)

Three context tokens, then the at-bat pitches. Causal attention lets pitches
attend to context but not vice versa.

## Transformer backbone

GPT-2 architecture, pre-norm. Sizes:

| name  | layers | heads | d_model | params (~) |
|-------|--------|-------|---------|------------|
| tiny  | 4      | 4     | 256     | 6M         |
| small | 6      | 8     | 512     | 25M        |
| base  | 12     | 12    | 768     | 85M        |

**Train tiny and small only by default.** Base is for the ablation table if
small saturates. With ~7M pitches the base model overfits unless regularised
hard, and the compute budget is better spent on cross-fitting (5× training
runs for K=5).

## Multi-head decoder

Per position, the hidden state feeds:

```
hidden → Linear → pitch_type_logits   (vocab 7)
       → Linear → zone_logits         (vocab 25)
       → Linear → velo_logits         (vocab 10)
       → Linear → spin_logits         (vocab 8)
```

The result head is two-stage:

```
# hidden_for_result[t] = trunk_hidden[t-1] (history through pitch t-1).
# At t=0, use the last context-token's hidden as the surrogate.
result_input  = concat(hidden_for_result,
                       type_embed[ã],
                       zone_embed[z̃],
                       velo_embed[ṽ],
                       spin_axis_proj([sin ã, cos ã]))
result_logits = MLP(result_input)     # vocab 7
```

During training, ã, z̃, ṽ are the *teacher-forced* (ground-truth) values.
During interventional rollout, they are the *intervened* values. During
free-running rollout, they are sampled from the predicted distributions.

**Shifted hidden input** (ADR 007 Amendment 1, 2026-05-10): the result head
reads ``hidden[t-1]`` rather than ``hidden[t]``. ``hidden[t]`` would already
contain ``result_emb(result_t)`` via the per-pitch factor sum and create a
silent leakage shortcut — see ADR 007 Amendment 1 for the empirical
demonstration and the fix.

There is also an at-bat outcome head on the last hidden state (terminal
pitch), predicting K, BB, 1B, 2B, 3B, HR, out — used for AB-level run-value
mapping. Run value itself comes through the RE24 table from the
`statcast-pipeline` skill, not as a regression target.

## Attention masking

Causal mask is standard. Two extras that are easy to forget:

- **Padding mask.** Block attention to padded positions (variable AB length).
- **Cross-AB block.** When packing multiple at-bats from the same game into
  one sequence (recommended for compute efficiency), block attention from
  pitch t in AB_j to any position in AB_<j. The mask builder is at
  `model/transformer.py:build_block_mask()`.

If you turn off cross-AB blocking, document it in an ADR — the model
becomes able to see prior at-bats from the same matchup, which is
sometimes-valuable signal but changes the conditional independence
structure that the causal layer assumes.

## Training

Defaults in `configs/train_small.yaml`:

- AdamW, β=(0.9, 0.95), weight_decay=0.1
- lr=3e-4, cosine decay to 3e-5, linear warmup 2000 steps
- batch_size=256 at-bats, gradient_accumulation as needed for memory
- gradient clipping 1.0
- dropout 0.1 on attention and FFN
- label smoothing 0.05 on the pitch_type head only
- per-factor loss weights: type=2.0, zone=2.0, others=1.0; result=1.5

Loss is the sum of per-factor cross-entropies plus the AB-outcome loss.
Validation loss for early stopping is the type+zone+result sum on the val
split, monitored per epoch.

## Calibration — temperature scaling on every head

Raw softmax probabilities from the heads will be miscalibrated. After
training:

1. Freeze the model.
2. On the validation set, fit a single temperature scalar per head by
   minimising NLL.
3. Apply at inference time. Temperatures are saved in the checkpoint.

The causal layer assumes π̂ and μ̂ are calibrated. An uncalibrated propensity
score makes positivity gating unsafe (you'll trust low-probability events that
the model is overconfident on). Do not skip this step.

## Held-out-pitcher generalization

Profile-based encoding promises zero-shot transfer to new pitchers. Verify it.
After training, compute eval metrics on the `held_out_pitchers` cohort
(pitchers debuting in 2024). Report separately in the eval table. If accuracy
collapses on this cohort, the profile encoder is doing more memorization
than generalization and the writeup should say so.

## Checkpoints

Checkpoints are saved to `checkpoints/{run_name}/{step}.pt` and contain:

- model state_dict
- optimizer state_dict
- config dict (full hyperparameters)
- temperature parameters per head
- training-data range (start_date, end_date) — to detect leakage at eval time
- git commit hash

Loading code refuses to load if the training-data range overlaps the current
evaluation split. Do not work around this; if you need to reuse a checkpoint
for a different split, retrain.

## Things to avoid

- **A single giant softmax over compound tokens.** The whole point of
  factored embeddings is to avoid this. Do not collapse them at the output.
- **Result head conditioned only on hidden state.** Breaks interventional
  rollout — you cannot meaningfully change the pitch and have the result
  respond.
- **Random AB shuffling in the dataloader across years.** Use the temporal
  split. Within-split shuffling is fine.
- **Training base before small saturates.** Costs 4× the compute for likely
  worse generalization.
