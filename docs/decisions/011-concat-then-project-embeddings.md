# ADR 011 — Concat-Then-Project Per-Pitch Factor Embeddings ("fix #1")

**Status:** Accepted (locked 2026-05-12)
**Date:** 2026-05-12

## The question, in plain English

Each pitch token is currently the **sum** of 11 per-pitch factor embeddings
(type, zone, velo, spin_rate, spin_axis, result, count, runners, outs, pos,
pitcher_fatigue) — each a full `d_model`-dim embedding, all added together. The
trunk then has to *disentangle* that sum to recover any individual factor (e.g.
"what was the count?"). Disentangling a sum of 11 vectors is possible if the
embeddings span distinct sub-spaces — but gradient descent doesn't enforce
that, so the count signal can get mixed in with the type signal in directions a
downstream MLP doesn't cleanly isolate. The per-class breakdown shows the model
is weakest on the *context-dependent* pitches (curveball, cutter, changeup) —
exactly the ones whose probability hinges on count / previous pitch, the factors
most prone to being lost in the sum.

Should the factor embeddings be **concatenated** (each in its own sub-space)
rather than summed?

## Why this matters

- Concatenation preserves factor-separability *by construction* (each factor
  lives in a fixed slice of the input vector); summing preserves it only if the
  embeddings happen to stay orthogonal. The LSTM baseline gets its features as a
  *concatenated* vector (count one-hot, arsenal floats, prev-pitch one-hot, …)
  — separately addressable — which is part of why a 100K-param LSTM extracts
  comparable signal to a 23M-param transformer here.
- It's the cheapest of the remaining queued fixes (a few extra `Linear`s in the
  embedding layer; no ripple to the trunk, the heads, or the result head).
- Caveat on impact: with the leak corrected, PitchGPT (0.478) already *beats*
  the leak-clean LSTM (0.449), so this is polish — a couple of points, maybe,
  concentrated on CU/FC/CH — not a fix for a broken thing.

## Options

**A. Status quo: sum the 11 `d_model` embeddings.** Forces equal mixing; the
trunk must disentangle.

**B. Shrink each embedding table to `d_model//11` dims, concat the 11 (≈ d_model
total), use as the token directly.** Maximally separable, but breaks the
weight-tying of the propensity type/zone heads (`hidden @ type_emb.T` needs
`type_emb` to be `d_model`-dim) and changes the two-stage result head's MLP
input dim. Real ripple.

**C. Keep `d_model` embedding tables; down-project each to `d_model//11` for the
concat, then a `Linear(11·(d//11), d_model)` mixer.** Each factor gets a
dedicated `d//11`-dim input sub-space, *and* the mixer is a *learned* mixing
(it can implement the equal sum as a special case, or keep the slices separate,
or anything between — the model gets the *choice*). The embedding tables stay
`d_model`-dim, so weight-tied heads and the result head are untouched.

## Decision

**Option C.** A `concat_then_project` config flag (default `False` for
checkpoint back-compat — checkpoints predating this ADR have no such key and
reload with no `factor_down` / `factor_mixer` modules, matching their
state_dict; `train_pitchgpt.train()` exposes it as a `--concat-then-project`
CLI flag for new runs).

When set, `FactorEmbeddings.forward`:
1. computes the 11 per-factor `d_model` embeddings as usual (tables unchanged);
2. down-projects each via a per-factor `Linear(d_model, d_model//11)`
   (`factor_down`, a `ModuleList` — `ModuleDict` keys like "type" collide with
   `nn.Module`'s reserved attribute names);
3. concatenates the 11 → `(B, T, 11·(d//11))`;
4. `factor_mixer = Linear(11·(d//11), d_model)` → the `(B, T, d_model)` token.

The ADR-009 per-pitch arsenal contribution is still *added* to the token after
this (it's per-AB, not one of the 11 per-pitch factors). Extra params: ~130K
(tiny, d=256) / ~520K (small, d=512) — negligible. Not meant to interact badly
with any other flag; it's orthogonal to the propensity heads.

## Consequences

- **Validation:** train `tiny`/`small` with `--concat-then-project` (std +
  arsenal on, situational *off*, so it's isolated on top of `*-arsenal-std`).
  Compare to `small-v1-arsenal-std` (0.478); watch CU/FC/CH recall specifically.
- **Causal layer:** unaffected in form — only the *input token construction*
  changes; the trunk, π̂ heads, μ̂ head, stop-gradient, intervention mechanism
  are all the same.
- **Skill update:** `pitchgpt-model` skill — note the per-pitch token is either
  the sum (default) or a concat-then-project of the 11 factor embeddings
  (`config.concat_then_project`, ADR 011).
- **If it helps:** combine with whichever of situational (ADR 010) / FiLM
  (ADR 012) also help, in a "kitchen-sink" run.
