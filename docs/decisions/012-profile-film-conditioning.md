# ADR 012 — FiLM-Condition the Trunk on the Player Profile ("fix #2")

**Status:** Accepted (locked 2026-05-12)
**Date:** 2026-05-12

## The question, in plain English

The 314-dim (pitcher ++ batter) profile currently reaches the model as **one
prepended context token**. Pitch tokens get the "who's pitching / batting"
signal only by *attending back* to that single token — competing for attention
with all the other pitch tokens, in every layer, with nothing forcing them to
keep doing it deeper in the stack. The LSTM baseline, by contrast, projects the
profile into its **initial hidden state** (`h0`, `c0`) — so the profile is
baked into how it processes *every* step. (Empirically the LSTM's profile-in-h0
barely moves its accuracy — 0.446 → 0.449 — so this is not a huge lever; but
it's a real architectural asymmetry worth removing.)

Should the transformer get a profile pathway that conditions the *whole*
network, not just one attended-to token?

## Why this matters

- It's the natural "transformer answer to the LSTM's `h0` trick", and it's the
  standard tool for conditioning a transformer on a global vector (FiLM /
  conditional normalisation are used widely in conditional generation).
- The profile reaching every layer (not just being fetchable from one token)
  removes a structural bottleneck — even if, as the LSTM evidence suggests, the
  profile isn't the main thing the model is missing.
- Caveat on impact: PitchGPT (0.478, leak-clean) already beats the leak-clean
  LSTM (0.449); this is polish, not a fix-a-broken-thing.

## Options

**A. Status quo: profile → one context token, fetched via attention.**

**B. Profile → a small set of "prefix" tokens (K=4–8) instead of 1.** More
attention bandwidth, still requires the pitch tokens to attend.

**C. FiLM: an MLP maps the profile to per-layer `(gamma_l, beta_l)`; each
transformer block's input is modulated `gamma_l · x + beta_l`.** The profile
conditions every layer directly, no attention hop. Standard, cheap, and
initialisable to a no-op so it can't destabilise training at the start.

**D. Cross-attention to a small profile/context memory in every block.** Most
expressive, but adds a sublayer per block and is the biggest change.

## Decision

**Option C.** A `profile_film` config flag (default `False` for checkpoint
back-compat — old checkpoints reload with no `profile_film_mlp`, matching their
state_dict; `train_pitchgpt.train()` exposes `--profile-film` for new runs).

When set, `PitchGPT`:
- `profile_film_mlp = Linear(pitcher_dim + batter_dim, 4·d_model) → GELU →
  Linear(4·d_model, n_layers · 2 · d_model)`, the output reshaped to
  `(B, n_layers, 2, d_model)` = per-layer `(gamma_l, beta_l)`;
- **initialised to identity**: the second linear's weight is zeroed and its
  bias set so `gamma_l = 1, beta_l = 0` for all layers — so at step 0 FiLM is a
  no-op (and the zero-init weight still receives gradient, via the input term,
  so the MLP starts learning immediately);
- in the trunk loop, before block `l`: `x ← gamma_l · x + beta_l` (broadcast
  over the sequence dim). Applied to all positions including the context tokens
  (FiLM-ing the profile token by the profile is a harmless redundancy the model
  can ignore).

Extra params: `profile_film_mlp` ≈ `Linear(314, 4d)` + `Linear(4d, 2·n_layers·d)`
≈ ~1.4M (tiny, 4 layers, d=256) / ~9.6M (small, 6 layers, d=512). That's a
notable chunk for `small` (~9.6M on ~23M) — acceptable for an ablation; if it
helps and the param count is a concern, a lower expansion factor or shared
`(gamma, beta)` across layers would shrink it.

## Consequences

- **Validation:** train `tiny`/`small` with `--profile-film` (std + arsenal on,
  situational *off*, so it's isolated on top of `*-arsenal-std`). Compare to
  `small-v1-arsenal-std` (0.478). Given the LSTM-zero-profile evidence, my prior
  is this moves the number ~little — but it's the test that confirms whether the
  profile pathway was a bottleneck at all.
- **Causal layer:** unaffected in form — FiLM modulates the trunk's residual
  stream; the heads, stop-gradient, and intervention mechanism are unchanged.
  (A FiLM'd trunk is still a deterministic function of its inputs, so rollouts
  are unaffected.)
- **Skill update:** `pitchgpt-model` skill — note the trunk is optionally
  FiLM-conditioned on the profile per layer (`config.profile_film`, ADR 012).
