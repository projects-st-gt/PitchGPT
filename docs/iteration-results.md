# PitchGPT iteration results — Tier A & Tier B

Running log of the predictive-power iteration roadmap.
Each row = one trained `small` (25M) variant on the 2024-H1 val split, leak-clean (sorted by `pitch_number`
within each at-bat). **π̂ = pitch-type top-1; μ̂ = result top-1.** Per-class breakdown column tracks the weak
spots (CU = curveball, FC = cutter, CH = changeup). Numbers are the in-training-subsample eval unless marked
"(clean)" — clean full-val differs by ~±0.002.

**STOP after Tier B — do not start Tier R until this report has been reviewed.**

## Results

| run | what's new (on top of the previous row) | π̂ top-1 | μ̂ top-1 | CU / FC / CH recall | π̂ ECE | notes |
|---|---|---|---|---|---|---|
| `small-v1-arsenal-std` | baseline — std + per-pitch arsenal feature (ADR 009) | **0.478** (clean) | **0.542** (clean) | 0.24 / 0.30 / 0.36 | 0.021 (post-temp) | the reference for everything below |
| `small-v2-situational` | + situational two-stage propensity head (ADR 010) | 0.478 | ~0.54 | — | — | ≈ flat on `small`; helped `tiny` (+0.4) |
| `small-v3-concat` | + concat-then-project the 11 factor embeddings (ADR 011) | 0.475 | 0.542 | — | — | ≈ −0.3; didn't help |
| `small-v4-film` | + FiLM-condition the trunk on the profile (ADR 012) | 0.472 | 0.542 | — | — | ≈ −0.6; didn't help |
| `small-v5-A1` | + richer batter profile: whiff-on-swing & swing-rate by pitch type (14 new dims; ADR 013) | 0.477 (clean) | 0.540 (clean) | _pending_ | 0.005 (pre-temp) / 0.019 (post-temp) | flat — within ±0.002 of baseline; per-type batter whiff/swing likely redundant with existing chase/zone signals |
| `small-v6-A3` | + A3 pitcher×batter matchup history (new per-pair cache; ~20-dim new context token) | _in progress_ | | | | revised from bundled "v6-allA" after A1's flatness — split for separable ablation; highest-novel-info Tier-A item, gets tested first |
| `small-v7-A4` | + A4 de-bucketed features (continuous fatigue; umpire zone-cal vector) | _planned_ | | | | conditional — different mechanism (μ̂-targeted) so worth trying even if A3 flat |
| `small-v8-A2` | + A2 catcher profile (new role; per-count called-pitch-type rates) | _planned (deferred)_ | | | | lowest prior given A1's flatness; only if A3 or A4 showed any Tier-A signal |
| `small-v9-B2` | + B2 cross-AB packing (times-through-order at pitch granularity) | _planned_ | | | | Tier B; needs ADR |
| `small-v10-B1` | + B1 factor tokenization (one token per factor; output half = R3 factored-autoregressive) | _planned_ | | | | Tier B; needs ADR; only if v6–v9 leave headroom |

## Status / log

- **2026-05-12** — Tier A/B iteration started (approved). Roadmap order: A1 → A2/A3/A4 → B2 → B1 → STOP (Tier R deferred for human review).
- **A1 (in progress)** — adds 14 dims to the batter profile (`whiff_on_swing_{FF,SI,FC,SL,CU,CH,FS}` + `swing_rate_{FF,SI,FC,SL,CU,CH,FS}`): "of the pitches of type X this batter swung at, what fraction did he whiff" and "what fraction of type-X pitches he saw did he swing at" — both over the trailing window, leak-safe (`before_asof`). Targets the model's known weak spot (the offspeed pitches CU/FC/CH) and helps μ̂ directly (whiff-on-type *is* the result). ADR 013.
  - Implementation: `batter_whiff_and_swing_by_type()` helper in `data/player_profiles.py`; new slots in `BATTER_FEATURE_NAMES`; `batter_profile_dim` 91 → 105 in `model/config.py`; per-role schema versions (`PITCHER_SCHEMA_VERSION` stays 2, `BATTER_SCHEMA_VERSION` → 3 so only the *batter* cache needs rebuilding) in `data/profile_cache.py` + propagated to the loader & build script; `scripts/fit_profile_standardizer.py` to refit `batter_mean/std` over the rebuilt cache.
  - Pipeline: rebuild the fold-0 cache (`build_profile_cache --role both --folds 0` — both roles because `PROFILE_SCHEMA_VERSION` is shared; only fold 0 because the `small-v5-A1` retrain uses `--fold 0`, like every prior run; the v2 fold-1..4 caches go stale-but-unused until the eventual K=5 cross-fit) → `fit_profile_standardizer` (new `batter_mean/std`, 105-dim) → upload the 4 rebuilt fold-0 caches + the refitted npz to the Modal volume → retrain `small --fold 0` on Modal (~5 h, ~$10) → `calibrate_pitchgpt.py` + per-class breakdown → fill in the `small-v5-A1` row.
  - **Status:** code done; 111/111 affected tests pass; fold-0 cache rebuilt (pitcher 190,593 entries v3 ~25 min; batter 456,193 entries v3 105-dim ~122 min; + league caches); standardizer refit (`pitcher_mean/std` 223-dim, `batter_mean/std` 105-dim); 4 rebuilt caches + the `.npz` uploaded to the Modal volume; **`small-v5-A1` complete on Modal** (`--size small --fold 0 --epochs 6 --no-propensity-situational` — so it's std + arsenal + A1 only, isolated on top of `small-v1-arsenal-std`; `ap-jZeRWaC2Q4YudHQuv2AJX3`; early-stopped at step 16000, best_val_loss at step 12000). Calibrated against the full 2024-H1 val (440,248 pitches / 113,433 ABs): **π̂ top-1 0.4769, μ̂ top-1 0.5402**, π̂ ECE_before 0.005 / ECE_after 0.019 (the head was already tighter than NLL-optimal T=0.943 wanted; pre-temp is the real calibration). Per-class breakdown deferred — aggregate is flat enough that even a per-class win would be a wash.
  - **Verdict:** A1 is **flat (−0.001 on π̂, −0.002 on μ̂)** — well within the ±0.002 in-training-vs-clean-full-val variance. The hypothesis that per-pitch-type whiff/swing rates would help the offspeed weak-spot didn't materialize on aggregate. Most likely cause: redundancy with the existing `chase_{pt}` (7 dims, out-of-zone-swing per type) + `whiff_z*` (25 in-zone whiff dims) + `swing_z*` (25 in-zone swing dims) — the per-type whiff/swing info was already reachable through those, just not in the most explicit form. Tier-A lesson for what's next: prefer *novel-information* features (matchup history, umpire zone-cal) over features that re-package existing signals.
- **A3 (in progress)** — pitcher × batter matchup history. The original plan bundled A2/A3/A4 into one retrain (`v6-allA`) for compute efficiency, but A1's flatness makes per-feature attribution more valuable than the ~$20/run saving — switching to three separate retrains (`v6-A3` → `v7-A4` → `v8-A2`-deferred) so each Tier-A feature gets a clean ablation. A3 goes first because it's the highest-novel-info item: pitcher × batter matchup state isn't reachable from *any* existing feature (the pitcher/batter profile caches are marginal, not interactional). Spec: ADR 014 — a new `MatchupCache` keyed by `(pitcher_id, batter_id, asof_date, asof_game_num)`, vector includes per-pitch-type pitch mix to this batter, per-type whiff-on-swing vs this pitcher, cumulative K/BB/PAs, last-face xwOBA, days-since-last-face. Wired as a 4th context token; ``N_CONTEXT_TOKENS`` 3 → 4.

## Generalization eval — held-out-pitcher cohort (Sprint 0a)

First time this has been run for PitchGPT. Evaluated `small-v5-A1` calibrated checkpoint against the 2024-H1 val split, split into pitchers who debuted ≤ 2023 ("main") vs ≥ 2024 ("held-out"). The held-out cohort tests whether the profile-based encoder generalizes or just memorizes pitcher IDs (per the eval protocol).

| Cohort | Pitchers | Pitches | π̂ top-1 | μ̂ top-1 | Type ECE | Result ECE |
|---|---|---|---|---|---|---|
| main (debut ≤ 2023) | 701 | 409,107 | **0.4768** | **0.5404** | 0.0185 | 0.0041 |
| held-out (debut 2024+) | 206 | 31,141 | **0.4787** | **0.5383** | 0.0240 | 0.0141 |
| Δ | | | +0.002 | −0.002 | +0.006 | +0.010 |

**Verdict: the encoder generalizes.** Top-1 on both heads is essentially identical across cohorts (within noise — the held-out sample of 31K pitches gives a SE of ~0.003). Calibration degrades modestly on debutants but all heads stay below the 0.03 ECE pass bar for the causal layer.

**Per-class type breakdown — main vs held-out:**

| Pitch | main recall | held-out recall | comment |
|---|---|---|---|
| SI | 0.636 | 0.717 | Easier on debutants (heavy SI users) |
| FC | 0.478 | 0.460 | Stable |
| SL | 0.361 | 0.354 | Stable |
| CU | 0.487 | 0.422 | Mild drop |
| CH | 0.225 | 0.146 | Real drop — changeup usage is pitcher-specific |
| FS | 0.312 | 0.109 | **Big drop** — splitters are even more pitcher-specific |

(FF doesn't appear because the in-zone heatmap collapse means FF is the default predicted class when uncertain; recall is high but "support" is 0 in this script's labeling — needs a follow-up to confirm.)

**Implication for the writeup**: the model's pitcher-profile abstraction (223-dim, soon 230-dim, of arsenal/zone/velo/entropy features) is doing the right thing — it captures transferable structure rather than memorizing identity. The honest qualification is that pitcher-specific *rare* pitches (FS, CH) don't transfer as cleanly as common ones, which is a real limitation worth naming, not hiding.

## Status / log (continued)

- **2026-05-13** — Sprint 0a held-out eval landed. Verdict above. Sprint 0b (arm slot) in progress: pitcher profile schema bumped v3 → v4 (+7 dims `arm_slot_{pt}`, per-pitch-type mean release-side arm angle in degrees). Helper `pitcher_arm_slot_by_type()` added in `data/player_profiles.py`. Smoke test confirmed the vector slot wiring works for both populated (2020+) and missing (2017-2019) arm_angle. Next: pitcher cache rebuild for fold 0 (~25 min), batter cache schema-tag bump (5 sec), standardizer refit, Modal upload, retrain `small-v6-arm`.

(Realistic timeline for all of Tier A + Tier B: ~a couple of days of wall-clock, dominated by ~3–4 profile-cache/data rebuilds and ~4 Modal retrains. Updated as each lands.)
