# ADR 009 — Per-Pitch Arsenal Feature

**Status:** Accepted (locked 2026-05-11)
**Date:** 2026-05-11

## The question, in plain English

A pitcher's *arsenal* — which pitch types he throws and how often — is the
single strongest feature for predicting his next pitch type. The profile-aware
LSTM baseline (`eval/baselines/lstm_with_profiles.py`) gets it as a *direct
per-pitch feature*: `compute_pitcher_arsenal_encoding` produces 7 Dirichlet-
smoothed usage-rate columns (`pt_FF_rate … pt_FS_rate`), `build_xgboost_features`
joins them onto every pitch row, and the LSTM *also* projects the full 314-dim
profile (which contains the arsenal) into its initial hidden state. So the
arsenal is front-and-centre on every step.

PitchGPT's factored-embedding design instead folds the arsenal into the 223-dim
pitcher profile (slots `arsenal_FF…arsenal_FS` plus `has_pitch_FF…has_pitch_FS`),
which is squeezed through an MLP into one of three prepended context tokens. To
*use* the arsenal at pitch position *t*, the trunk must attend back to that
context token and recover the arsenal sub-vector from a 256-dim mixture. The
design *assumed* the trunk would learn that read. For a 4.6M-param Tiny model
trained ~5 epochs — and, in the Phase B run, with the profile fed *unstandardised*
(ADR-002-era recipe; standardisation was added to `train_pitchgpt.py` afterward),
so the arsenal slots (~0–1) were numerically swamped in the MLP by `mean_spin_*`
(~2300) and `mean_velo_*` (~93) — the trunk under-learned it. Result: PitchGPT-
Tiny lands at type top-1 ≈ 0.459, ≈ per-pitcher-mode (0.43–0.45) and ~23 points
below the same-data LSTM (re-confirmed 0.691 [0.690, 0.693] on the 2024H1 val
split).

Should the arsenal be given a dedicated, always-present pathway into the model,
the way the LSTM has it?

## Why this matters

- The 23-point gap is the project's current blocker. A weak π̂ is not cosmetic:
  positivity gating thresholds on π̂(a|h), AIPW's IPW correction divides by π̂
  (bad π̂ → high-variance estimates), and g-computation rollouts *sample* the
  non-intervened future pitches from π̂ (bad π̂ → unrealistic trajectories →
  biased counterfactuals). The causal layer can't be trusted on a 0.46 π̂.
- Diagnostics ruled out the cheap explanations: the profile cache is healthy
  (100% per-player hits, median confidence 1.0, arsenal slots vary across
  pitchers); the dataset pipeline is correct; multi-task interference is nil
  (`type_only` ≡ all-heads at 1500 steps); over-regularisation is nil (low-reg
  + high-lr made it *worse*). The remaining explanation is *input encoding* —
  the transformer is starved of the signal the LSTM gets for free.
- This is also the cleanest single change available: it's additive, it touches
  one config field + one projection + one dataset key, and it doesn't disturb
  the trunk, the heads, the two-stage result head, or the splits.

## Options

**A. Do nothing; rely on standardisation alone.** `train_pitchgpt.py` already
defaults `standardize_profiles=True`, which un-drowns the arsenal slots inside
the context-token MLP. Helps (the `std` recipe matches `no-std` at ~15× less
training) but the arsenal is still behind an attention hop and mixed with 209
other profile dims. Likely insufficient on its own.

**B. Arsenal as a dedicated context token.** Pull the arsenal sub-vector out,
give it its own small MLP → a fourth context token. Cleaner than A, but pitch
tokens still have to *attend* to it — the LSTM's advantage is precisely that it
doesn't have to.

**C. Arsenal as a per-pitch additive feature.** Pull the 14-dim arsenal+has-pitch
sub-vector from the (pre-standardisation, raw 0–1) pitcher profile, project via
`Linear(14, d_model)` (small init), and *add it to every pitch token* — so it's
in the residual stream at every position without an attention hop. Mirrors how
the LSTM gets it (per-pitch feature + initial state). ~4K extra params for Tiny.

**D. Match the LSTM exactly — static-over-train arsenal encoding.** Use
`compute_pitcher_arsenal_encoding(train_df)` (a single static per-pitcher vector
over all of ≤2023) instead of the cache's trailing-window arsenal. Apples-to-
apples with the LSTM, but lower-variance-yet-staler, and requires plumbing the
encoding into the augmented-data pipeline or the dataset. Strictly worse than C
on the leakage/recency axis for no implementation saving.

## Decision

**C + (A).** Standardisation stays on (it's the default), *and* the 14-dim
arsenal+has-pitch sub-vector gets a dedicated `Linear(14, d_model)` projection
added to every pitch token.

- **14 dims = 7 `arsenal_*` usage rates + 7 `has_pitch_*` binary flags.** The
  smoothed rate alone never reaches exactly 0 for a never-thrown type
  (`α·league_prior/(n+α) > 0`), so the binary "has he ever thrown this" is a
  sharper constraint worth handing the model explicitly. Both are already in the
  pitcher profile vector — no new data pipeline.
- **Sourced pre-standardisation** (raw 0–1 values) so the `Linear(14, ·)` sees a
  clean, well-scaled input.
- **Trailing-window arsenal, not static-over-train.** It's leak-safe by the same
  rule as the rest of the profile cache (window ends strictly before the at-bat;
  fold-aware per ADR 008), and it adapts to in-season arsenal changes — strictly
  more informative than the LSTM's static encoding. This makes the PitchGPT-vs-
  LSTM comparison *not* perfectly apples-to-apples, in PitchGPT's favour; the
  eval writeup notes it.
- **Small init (std = `config.init_std` = 0.02)** so the arsenal contribution at
  step 0 is ~25% the magnitude of the factored-embedding sum — meaningful but not
  dominant; the model amplifies it from there.

### Config / compat

- New `PitchGPTConfig` fields: `arsenal_per_pitch: bool = False` and
  `n_arsenal_dims: int = 14`. **The config-field default is `False`** so old
  checkpoints (Phase B and earlier — their saved config dict has no such key)
  load without a state-dict mismatch (no `arsenal_proj` module, matching their
  weights).
- `scripts/train_pitchgpt.train()` gets `arsenal_per_pitch: bool = True` (the
  *run* default) which sets `cfg.arsenal_per_pitch`; CLI `--no-arsenal-per-pitch`
  disables it. New checkpoints save `arsenal_per_pitch: True` in their config, so
  they re-load correctly.
- `PitchGPTAtBatDataset` always emits an `"arsenal"` key (14-dim float tensor);
  `collate_pitchgpt_at_bats` stacks it to `(B, 14)`. `PitchGPT.forward` takes
  `arsenal=` and raises if `config.arsenal_per_pitch` is set and it's absent.

### Embedding-parameter accounting

The `pitchgpt-model` skill quotes "~131 × d_model" embedding params for the
factored design. The new `Linear(14, d_model)` adds `14 × d_model + d_model ≈
15 × d_model` — i.e. ~131 → ~146 × d_model. Still negligible vs a flat
vocabulary (~35M × d_model). Frame accordingly; this is engineering, not novelty.

## Consequences

- **What we expect:** combined with standardisation and a `small`-size retrain,
  this should close most of the 23-point gap. If PitchGPT is *still* >10 points
  behind the LSTM after getting the same arsenal signal, that's a surprising
  result and warrants a deeper look (profile-into-trunk-state à la the LSTM's
  h0/c0; concat-vs-sum factored embeddings) — *fixes to* the transformer, not a
  switch away from it. A residual ~2–3 point deficit on raw top-1 is acceptable
  (7M pitches is small for a transformer; RNNs are more sample-efficient at this
  scale) — per `eval-protocol`, the transformer earns its place via the causal
  layer, which the LSTM (a `Linear(hidden → 7 types)` with no action-conditioned
  outcome head) structurally cannot provide.
- **Re-run obligations:** `make eval` after the retrains; per-head calibration
  (temperature scaling) on the new checkpoints; the `pitchgpt-model` skill's
  factored-embedding spec updated to list the arsenal factor.
- **Candidate follow-ups (not in this ADR):** also exposing `mean_velo_*` /
  `mean_spin_*` per-pitch (the rest of the "stuff this pitcher throws hard/with
  spin" signal); profile-into-trunk-initial-state conditioning. Deferred until
  the retrains show whether they're needed.
