# v6 pitcher profile: per-(type×count), per-(type×stand), and per-type movement

**Status:** proposed
**Date:** 2026-05-17
**Author:** sid + claude (brainstorming session)
**Related:** ADR 014 (player profiles), profile schema v4/v5 history in `data/profile_cache.py`

## TL;DR

Bump the pitcher profile schema from v5 (118 dims) to v6 (218 dims) by adding
three new feature blocks that target the diagnosed weaknesses of the v5 model:
CU/FC under-recall and FF argmax over-prediction. The diagnosis is that
the v5 type head's *probabilities* are well-calibrated but the model lacks the
features to distinguish curveballs from sliders in context, and the focal-loss
and class-weight loss-engineering experiments did not help — because they
redistribute gradient rather than add information. The fix is to give the model
explicit conditional structure that baseball analysts already know matters:
arsenal-by-count, arsenal-by-batter-handedness, and per-type ball movement
(pfx_x, pfx_z).

## Diagnosis (why this work is happening)

The May-17 calibration + per-class diagnostic on the v5 baseline
(`checkpoints_modal/tiny-fold0-1778792736`) showed:

- **Mean P(FF) = 0.3086 vs true rate 0.3114** — well-calibrated as a
  probability distribution.
- **Argmax FF rate = 0.3871** — +7.57 pp over-prediction at the *argmax* level,
  driven by winner-take-all on ambiguous pitches.
- **CU recall = 0.2558, FC recall = 0.3272** — minority breaking balls and
  cutters are systematically missed.
- Focal-loss (γ=2.0) experiment made calibration ~7× worse (ECE 0.0077 →
  0.0521) and slightly *increased* FF argmax over-prediction.
- Class-weight (α=0.2) experiment turned out to be a no-op due to a gating
  bug in `scripts/train_pitchgpt.py:135` (class-weight code nested inside
  `if cfg.type_focal_gamma > 0.0`).

The root cause hypothesis: **feature insufficiency, not loss imbalance.** Focal
and class weights redistribute training gradient; they cannot create signal that
isn't already in the input. The model has the unconditional arsenal
(`arsenal_FF` = 0.60) but doesn't have the conditional structure that real
baseball decisions depend on — what this pitcher throws *in this count*, *to
this batter handedness*, and *with what break shape*.

## What this design adds

### A. v6 schema (218 dims)

Retained from v5 (106 dims):

- `arsenal_{pt}` × 7 — unconditional usage fractions
- `mean_velo_{pt}` × 7
- `mean_spin_{pt}` × 7
- `heatmap_{pt}_z{i}` × 63 (7 types × 9 SIS zone cells)
- 5 recent-form scalars (`recent_30d_xwoba`, `recent_30d_n_pitches`,
  `days_since_last_appearance`, `recent_3starts_xwoba`, `recent_3starts_n`)
- `profile_confidence` (1)
- `long_window_span_days`, `long_window_pct_current_season` (2)
- `has_pitch_{pt}` × 7
- `arm_slot_{pt}` × 7 (v4 addition)

Removed from v5 (–12 dims):

- `entropy_b{b}s{s}` × 12 — redundant given the new per-count distribution
  (entropy is computable from the distribution). The
  `pitcher_count_conditional_entropy` function stays defined as a utility but
  is no longer called from the cache build path.

Added in v6 (+112 dims):

- `arsenal_{pt}_b{b}s{s}` × 84 — per-(pitch type × count) usage fraction.
  Type-major order, matching the existing `heatmap_{pt}_z{i}` and
  `arm_slot_{pt}` conventions.
- `arsenal_{pt}_vs{stand}` × 14 — per-(pitch type × batter stand) usage
  fraction.
- `mean_pfx_x_{pt}` × 7 and `mean_pfx_z_{pt}` × 7 — mean horizontal /
  vertical ball break per pitch type.

Schema version: `PROFILE_SCHEMA_VERSION` 5 → 6.

### B. Three new feature functions in `data/player_profiles.py`

All three follow the contract of existing functions
(`pitcher_arm_slot_by_type`, `pitcher_arsenal`): take the pitcher's
trailing-window pitches dataframe (already filtered via `before_asof`),
return a dict that the cache builder writes into named slots.

**B.1 `pitcher_arsenal_by_count(pitches, *, window_pitches=1000)`**

- Required columns: `pitch_type_canonical`, `balls`, `strikes`.
- For each `(balls, strikes)` cell with at least one observation, compute
  the fraction of pitches with each canonical pitch type. The 7 fractions
  sum to 1 within a cell.
- Cells with zero observations produce no entries (slot stays NaN → the
  existing 3-step league-mean fallback in `profile_cache_loader.py:166-194`
  fires automatically).
- Cells with observations but missing a type get `0.0` for that type
  (pitcher *can* throw it, just didn't in this count).

**B.2 `pitcher_arsenal_by_stand(pitches, *, window_pitches=1000)`**

- Required columns: `pitch_type_canonical`, `stand`.
- Same shape as B.1 conditioned on batter handedness (`L`/`R`).

**B.3 `pitcher_movement_by_type(pitches, *, window_pitches=1000)`**

- Required columns: `pitch_type_canonical`, `pfx_x`, `pfx_z`.
- For each pitch type the pitcher threw, mean `pfx_x` and mean `pfx_z`.
- Types with zero observations are omitted (slot stays NaN → league-mean
  fallback).

A small private helper computes the conditional-fraction logic shared
between B.1 and B.2 so the numerator/denominator convention stays
identical across all three "arsenal" features.

## Design decisions and trade-offs

### D1: Drop the 12 entropy dims

Decision: drop. Information is preserved (entropy is computable from the new
84-dim distribution); redundant pre-digested signal encourages the model to
lean on a summary stat when the raw distribution is available.

### D2: NaN / sparse-cell handling — use the existing 3-step fallback

Decision: do nothing special. The profile loader already has a 3-step
fallback chain (per-player slot → league-mean fill → zero-fill). Producing
NaN for sparse cells and letting the chain fire is consistent with how
`arm_slot` works today. No new code needed.

### D3: Handedness encoding — full 7×2 = 14 dims

Decision: full per-(stand × type) usage, not a compact platoon-split delta.
Symmetric with the per-count encoding, information-preserving, and the
14-dim cost is small relative to the 218-dim total.

### D4: Movement features — minimal pfx_x + pfx_z mean

Decision: just horizontal and vertical break means per type (14 dims).
These are *the* canonical CU vs SL disambiguator at a given arm slot
(arm slot is already in v4). Release extension and release_pos are
correlated with arm_angle (already in) and add little marginal info for
this diagnosis.

## Empirical data sufficiency check

Run on 2024-06 + 2024-07 raw (~115K pitches across 632 pitchers, a
*conservative* lower-bound sample since production trailing windows
accumulate more history):

**Per-(type × count) — 84 cells per pitcher (live cells = types thrown):**

| Bucket | Pitchers | Types thrown | Empty live cells | Median obs / filled cell |
|---|---|---|---|---|
| Starter (700+ window) | 108 | 5 | 15.4% | 13 |
| Mid reliever (300–700) | 215 | 4 | 18.8% | 7 |
| Pure reliever (<300) | 309 | 4 | 37.4% | 3 |

**Per-(type × stand) — 14 cells per pitcher:**

| Bucket | Empty live cells | Median obs / cell |
|---|---|---|
| Starter | 3.9% | 79 |
| Mid reliever | 5.2% | 47 |
| Pure reliever | 10.5% | 13 |

**Movement (pfx_x, pfx_z):** 0.00% NaN across all 7 pitch types and
220K+ pitches.

Reading: starters and mid relievers get strong per-pitcher signal. Pure
relievers fall back to league-mean for ~37% of per-count cells — still
strictly more information than v5 (which had only the unconditional
arsenal + 12 entropy summary scalars for the per-count dimension). The
handedness and movement blocks are dense across all buckets. Decision:
proceed as-designed; treat per-(type × count) Bayesian shrinkage to the
pitcher's marginal arsenal as a post-MVP enhancement if eval shows
reliever-specific regressions.

## Code surface — exactly what changes

### Modified files

**`data/player_profiles.py`** (additions only)
- Add `pitcher_arsenal_by_count`, `pitcher_arsenal_by_stand`,
  `pitcher_movement_by_type`. Update module docstring.

**`data/profile_cache.py`**
- Drop `pitcher_count_conditional_entropy` import (line ~39).
- Extend the version-log docstring with a v6 entry.
- Bump `PROFILE_SCHEMA_VERSION` 5 → 6.
- In `PITCHER_FEATURE_NAMES`: remove the 12 `entropy_b{b}s{s}` entries;
  append the 112 new slots in the order
  `arsenal_{pt}_b{b}s{s}`, `arsenal_{pt}_vs{stand}`, `mean_pfx_x_{pt}`,
  `mean_pfx_z_{pt}` (all type-major).
- In `build_pitcher_profile_vector`: replace the entropy-population block
  with three new population blocks (one per new function).

**`scripts/build_profile_cache.py`**
- Add `"pfx_x"`, `"pfx_z"`, `"stand"` to `NEEDED_COLUMNS`.

**`model/config.py`**
- Update the dimension-history comment (line ~123).
- Bump `pitcher_profile_dim: int = 118 → 218`.

**`tests/test_profile_cache.py`**
- Update `expected` arithmetic in
  `test_pitcher_vector_length_matches_documented_size`: remove
  `+ N_COUNT_STATES` (entropy), add the three new feature-block terms.
- Rename `test_schema_version_is_5` → `test_schema_version_is_6`; update
  assertion and docstring.

### Files that do not change

- `data/profile_cache_loader.py` — generic NaN fallback handles new
  features unchanged.
- `model/pitchgpt.py`, `model/embeddings.py` — all references to the
  pitcher profile size go through `config.pitcher_profile_dim`; no
  hardcoded 118 elsewhere.
- `model/pitchgpt_dataset.py`, `data/dataset.py` — flat vector passes
  through.
- `scripts/train_pitchgpt.py` — config-driven, no changes.
- `data/preprocess_pitchgpt.py` — no per-pitch features added.

### What the standardizer expects

`model/pitchgpt_dataset.py:ProfileStandardizer` is generic: it loads
per-dim mean and std from `data/preprocess_artifacts/v1/profile_standardization.npz`.
`scripts/fit_profile_standardizer.py` auto-detects D from the cache,
NaN-aware. After rebuilding the v6 profile cache, the standardizer
**must** be refit before training, or the loader will try to apply
v5-shaped (118-dim) stats to a v6 (218-dim) vector and crash.

## Build, train, and validation plan

### Pipeline

1. **Pre-flight check.** Confirm `pfx_x`, `pfx_z`, `stand` are populated
   in raw parquets for all years 2017–2025. Quick column-presence + NaN
   rate sample.
2. **Code changes.** All edits per the surface map above, one commit.
3. **Small-scale sanity rebuild.** Build v6 cache for fold 0 with
   `--max-games 50`. Verify `schema_version=6`, vector dim = 218, and
   print *named* sample values for one starter and one reliever:
    - `arsenal_FF_b3s0` ≫ `arsenal_FF_b0s2` (FF rate on 3-0 should
      dominate FF rate on 0-2)
    - `mean_pfx_x_SL` negative for RHP (glove-side break)
    - `arsenal_CU_b0s2` ≫ `arsenal_CU_b3s0` (CU on 0-2 ≫ CU on 3-0)
   This step is mandatory — it catches feature-construction bugs
   before paying the full rebuild cost.
4. **Full rebuild.** `python -m scripts.build_profile_cache --role both`
   for folds 0–4.
5. **Refit standardizer.** `python -m scripts.fit_profile_standardizer`.
   Expect `D=218`. Sanity-check mean/std on the new 112 dims (no
   degenerate stds).
6. **Modal training run.** Same hyperparams as
   `tiny-fold0-1778792736` baseline. `γ=0`, `α=0` — let the new features
   do the work.
7. **Calibrate + diagnose.** `scripts/calibrate_pitchgpt.py` +
   `scripts/diagnose_type_perclass.py` on the new checkpoint. Compare
   against the v5 baseline.

### Acceptance criteria

| Metric | v5 baseline | v6 target | Hard floor |
|---|---|---|---|
| Type ECE (before-T) | 0.0077 | ≤ 0.010 | ≤ 0.015 |
| Type NLL (before-T) | 1.2123 | ≤ 1.21 | ≤ 1.22 |
| Type top-1 accuracy | 0.4805 | ≥ 0.482 | ≥ 0.475 |
| **CU recall** | **0.2558** | **≥ 0.286** | **≥ 0.270** |
| FC recall | 0.3272 | ≥ 0.347 | ≥ 0.327 |
| FF over-fire (pred_rate − true_rate) | +7.57 pp | ≤ +6.5 pp | ≤ +7.5 pp |
| Mean P(FF) − true(FF) | −0.0028 | within ±0.01 | within ±0.02 |
| Other heads (zone/result/ab_outcome) | — | no regression | within 1 pp |

CU and FC recall are the headline metrics — they motivated this work.

### Revert triggers

- Type ECE > 1.5% (CLAUDE.md hard rule violation).
- Any single class recall drops > 5 pp.
- Standardizer fit produces std < 1e-6 or std > 1e4 on any new dim
  (data bug).
- Vector dim mismatch error during training (slot misalignment).

## Risks

**R1 — Rebuild cost.** Profile cache rebuild is the slow step
(hours/fold historically). A bug discovered after the full rebuild
forces a re-run. **Mitigation:** Step 3 (small-scale sanity rebuild)
is mandatory before Step 4.

**R2 — Pre-2020 movement coverage.** Statcast coverage of pfx_x/pfx_z
improved over time. Pre-flight check quantifies the NaN rate by year.
The league-mean fallback handles it cleanly, but the magnitude should
be known.

**R2 update (2026-05-17 pre-flight result):** Per-year raw-data NaN
rates ranged from 0.2% (2020, 2026) to 15% (2019, 2025) on raw pitches,
but virtually all NaN rows ALSO have NaN `pitch_type` and are dropped
by `scripts/build_profile_cache.py:_load_corpus` at line 107
(`df.dropna(subset=["pitch_type_canonical"])`). On rows with a valid
pitch_type — the only rows that reach feature computation — pfx_x/pfx_z
NaN rate is **<0.01% across every year 2017-2026**. R2 fully resolved.

**R3 — Shared-helper convention drift.** The three new "arsenal"
features all compute "fraction of pitches per type within a condition."
A shared private helper enforces a single numerator/denominator
convention.

**R4 — Test fixtures with v5 vector content.** Section C names the
assertion updates but does not audit fixture data. Tests that build
fake v5 vectors directly need updating too. **Mitigation:** run
`pytest tests/test_profile_cache.py tests/test_profile_cache_loader.py
tests/test_dataset_with_cache.py` after code changes; failures are
expected and surface fixture issues.

## Out of scope (deferred)

- Bayesian shrinkage of per-(type × count) cells to the pitcher's
  marginal arsenal. Considered; deferred. Will reconsider if
  reliever-specific regressions show up in eval.
- Adding release_extension or release_pos as profile features. Real
  signal but partially correlated with `arm_angle` (already in v4)
  and not targeted at the CU/FC diagnosis.
- Per-(pitch type × TTO) usage. Real signal; defer to a v7 if v6
  doesn't fully close the gap.
- Loss-engineering knobs (focal, class weights). Diagnosed as
  unhelpful given the calibration-focused use case. The
  `train_pitchgpt.py:135` gating bug should still be fixed as a
  hygiene matter, separately from this design.

## References

- `CLAUDE.md` — bug-prevention discipline for model-interfacing code,
  hard rules on calibration and temporal splits.
- `data/profile_cache.py` — schema version history, `PITCHER_FEATURE_NAMES`.
- `data/player_profiles.py` — existing feature-function patterns
  (`pitcher_arsenal`, `pitcher_arm_slot_by_type`).
- `scripts/calibrate_pitchgpt.py` — temperature scaling + ECE.
- `scripts/diagnose_type_perclass.py` — per-class P/R/F1, mean-prob,
  argmax-rate diagnostic added in this session.
- v5 baseline checkpoint: `checkpoints_modal/tiny-fold0-1778792736`.
- Focal-loss experiment: `checkpoints_modal/tiny-fold0-1778979210`.
- α=0.2 weights (no-op due to bug):
  `checkpoints_modal/tiny-fold0-1778979208`.
