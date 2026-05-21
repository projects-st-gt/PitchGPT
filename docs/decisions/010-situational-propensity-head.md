# ADR 010 — Situational Two-Stage Propensity Head

**Status:** Accepted (locked 2026-05-12)
**Date:** 2026-05-12

## The question, in plain English

PitchGPT is autoregressive: at pitch position *t* it predicts pitch *t+1*'s
factors (type, zone, velo, spin) from `hidden[t]` — the trunk's compression of
context + pitches 0..t. So the propensity heads, when predicting pitch *t+1*,
have the *situation pitch t was thrown in* (`count[t]`, `runners[t]`,
`outs[t]`, and crucially pitch t's *result*) but **not the situation pitch t+1
is thrown in** (`count[t+1]` etc.). Yet `count[t+1]` is *known* — it's a
deterministic function of `count[t]` and `result[t]` (ball → +1 ball;
strike/whiff/foul-with-<2-strikes → +1 strike; …), and at decision time the
catcher and pitcher obviously know the count before they call the pitch.

So the trunk must *re-derive* `count[t+1]` from `(count, result)[t]` inside its
residual stream — a wasted, latent computation. The profile-aware LSTM
baseline never does this: it's *discriminative* (row t = pitch t, features
include `balls[t], strikes[t]` = the count entering pitch t, target =
`type[t]`), so it gets the current pitch's count as a direct feature for free.
This is a real, fixable asymmetry — not leakage (the count *is* known
pre-pitch), just a worse task formulation on PitchGPT's side.

Should the propensity heads be given the upcoming situation directly?

## Why this matters

- PitchGPT-small (post-ADR-009) plateaus at 0.478 type top-1 vs the LSTM's
  0.691 — a ~21-point gap. A held-out (pitcher × count × prev-pitch) lookup
  gets only 0.41, so the transformer isn't *below* lookup level — it
  generalizes a little (+7) where the LSTM generalizes a lot (+28). One
  concrete piece the transformer is gratuitously missing is the count of the
  pitch it's predicting; this ADR closes that piece.
- It's the cheapest fix on the post-ADR-009 queue (≈ +4·d² params for a small
  fusion MLP; no new embeddings — reuses the trunk's count/runners/outs
  tables; no new model inputs — `pitch_factors` already carries these). It
  isn't expected to single-handedly close the gap (the lookup says count adds
  only ~1.3 pts to *argmax* accuracy) — but it cleanly removes a known
  inefficiency, and it should help calibration and the realism of multi-step
  rollouts (where the propensity head samples the non-intervened pitches) more
  than it helps raw top-1.
- It mirrors a pattern already in the architecture: the two-stage *result*
  head conditions on the (intervened) action's embeddings (ADR 007). This
  does the same for the *propensity* heads w.r.t. the upcoming situation.

## Options

**A. Do nothing; let the trunk derive `count[t+1]`.** Status quo. The
derivation is learnable but latent; the count signal stays buried.

**B. Add `count_emb[t+1] + runners_emb[t+1] + outs_emb[t+1]` to the
pitch-position hidden before the heads (residual-style).** Cheap (no MLP) but
re-introduces additive mixing — the head's input becomes `hidden` summed with
three more embeddings, which is the disentanglement burden ADR-009-adjacent
work is trying to *reduce*.

**C. Fuse via a small MLP: `fused[t] = MLP(concat(hidden[t], count_emb[t+1],
runners_emb[t+1], outs_emb[t+1]))`, then the existing propensity heads read
`fused` instead of `hidden`.** Keeps the upcoming-situation factors
disentangled at the head's input. Same shape as the two-stage result head.

**D. Per-head two-stage MLPs (one per propensity factor).** Maximally
flexible (each head can use the situation differently) but 4× the params and
code; the shared fusion in C is enough — the heads themselves stay unchanged.

## Decision

**Option C.** A `propensity_situational` config flag (default `False` for
checkpoint back-compat — checkpoints predating this ADR have no such key and
reload with the flag off, hence no `situation_fusion` module, matching their
state_dict; `train_pitchgpt.train()` defaults it `True` for new runs, so new
checkpoints save `propensity_situational: True` and reload correctly).

When set, `PitchGPT.forward`:
1. left-shifts `pitch_factors["count" / "runners" / "outs"]` so position *t*
   gets pitch *t+1*'s value (the last position gets 0 — its loss is ignored,
   no successor);
2. embeds them via the trunk's existing `count_emb` / `runners_emb` /
   `outs_emb` tables;
3. `fused = situation_fusion(concat(pitch_hidden, count_e, runners_e, outs_e))`
   — a `Linear(4d, d) → GELU → Dropout → Linear(d, d)` with small init;
4. the propensity heads (`PropensityHeads`) read `fused` at pitch positions
   (context-token positions keep plain `hidden` — their logits are discarded
   in `compute_losses` anyway).

Not meant to be combined with `propensity_type_two_stage` (which reads the
*previous* pitch's factors from the *un*-fused hidden — a different, weaker
idea); the code won't crash if both are set, but the result head's behaviour
is unspecified. The result head (μ̂) is untouched — only the propensity heads
change input.

## Consequences

- **Validation:** the next training runs (`tiny`/`small`, std + arsenal +
  situational) compare against `tiny-v3-arsenal-std` (0.475) / `small-v1-arsenal-std`
  (0.478). Report on the standard 2024-H1 val split + per-head calibration
  (temperature scaling) + ideally the per-position breakdown (this fix should
  flatten the early-position penalty if it works).
- **Causal layer:** strictly improves π̂ — and a better π̂ improves the
  realism of g-computation rollouts (which sample the non-intervened pitches
  from π̂) and the variance of AIPW's IPW correction. It does *not* change
  μ̂, the stop-gradient, or the intervention mechanism, so counterfactual
  rollouts are unaffected in form.
- **Skill update:** `pitchgpt-model` skill — note the propensity heads
  optionally read a situation-fused hidden (per ADR 010) rather than the raw
  trunk hidden.
- **Not in this ADR (queued):** concat-then-project the per-pitch factor
  embeddings (fix #1 — bigger, ripples into the weight-tied type/zone heads);
  FiLM-condition the trunk on the profile / cross-attention to a profile
  memory (fix #2). Order, informed by the LSTM-zero-profile diagnostic.
