# ContextSwitcher — Pick up where this session left off

**Last updated**: 2026-06-10 (evening) — **DEFINITIVE n=2400 GATE RUN DONE.**
baseline 1.4719 / lookup 1.4585 / V2 1.4631 @1500 paths. **Paired Δ(V2−lookup)
= +0.0046, 95% CI [−0.0017, +0.0106] — STATISTICAL TIE** (point estimate
halved from n=800's +0.0086; the small sample was unlucky for V2). BB% 8.8 vs
real 8.7 (PASS, essentially exact). Aggregates: out +0.2pp, 1B −0.2, BB +0.1
(excellent); residuals K −1.9pp (the clearest miss), 2B +0.9, HR +0.8 (both
shrank vs n=800 — doubles-light-sample read confirmed). Resolving a ±0.005
gap needs ~10K PAs — diminishing returns. Dists at
`data/backtests/v2_n2400_p1500_seed0.json`. DECISION PENDING (user): declare
the fuel acceptable (tied w/ lookup, beats baseline, walks exact) and
unblock App A, with K −1.9pp + in-play map as the known improvement avenue.

**(2026-06-10 PM)** — **v1c-cl RAN AND IS A MEASURED NEGATIVE.**
Fine-tune executed cleanly (tether 0.80, top-1 cost 0.2pp, teacher-forced 2-0
FF bias 3.1→1.4pp, rollout FF at hitter counts −2-3pp) but gate UNCHANGED:
cl 1.4544 vs base 1.4521 @1500 paths (paired Δ +0.0023, CI [−0.003, +0.007]
— true no-op). Literature's short-horizon caution (He 2021, Duckworth 2019)
confirmed at T=4-6. **KEY STATISTICAL FACT: base-vs-lookup paired Δ =
+0.0086, 95% CI [−0.0025, +0.0188] — the gate is NOT statistically resolved
at n=800.** tiny-v1c-base is in a statistical tie with the lookup, beats
baseline, best aggregates, walks fixed. cl ckpt at
`checkpoints_modal/tiny-v1c-cl/` (archive; base remains the fuel candidate).
DECISION PENDING (user): accept base as fuel now vs one n=2400 run to
resolve the tie sharply vs keep chasing (in-play map era / count-correction).

**(2026-06-10 AM)** — RESEARCH DONE, SPEC WRITTEN for v1c-cl
(closed-loop fine-tune, CAT-K-adapted): see
`docs/superpowers/specs/2026-06-10-v1c-cl-finetune-design.md`. Tiny model
ARCHIVED at `checkpoints_modal/releases/tiny-v1c-base-cal-20260609.pt`
(+ volume `checkpoints/releases/`). Scale verdict: NO base tier, NO 44.8M
small (7-paper evidence; ceiling ~15-20M after low-rank adaLN audit).
Diagnostic chain closed: hitter-count FF drift is real closed-loop at
+4-8pp (matched pitchers + neutral-context-corrected); step-0 = clean
(real pitchers throw +5-6pp more FF at bases-empty/0-out — sim is RIGHT).
NEXT: implement `scripts/finetune_v2_cl.py` per spec → eval ladder → re-gate.

**(2026-06-09 late night)** — DIAGNOSIS CLOSED. V2 @1500 paths
= 1.4521–1.4524 (beats baseline 1.4631, 0.009 behind lookup 1.4435), BB PASS.
Per-count temperatures: implemented, fitted, NO-OP on the gate (teacher-forced
model already calibrated per count — NLL 1.2412→1.2411). **FF over-commit at
hitter counts is CONFIRMED REAL + CLOSED-LOOP** (handedness-controlled: 2-0
+13.3pp, 3-1 +10.3, vs only +3.1pp teacher-forced at 2-0) → drives the
contact-quality bin skew (high-third 0.448 vs 0.333) → 2B/HR up, out down →
the 0.009. Temperature cannot fix closed-loop drift. NEXT = user decision:
(A) accept + ship tiny as fuel, (B) rollout-time count-marginal correction
(~1 day, ml-research first — label-shift/prior-correction family; also a
mechanism test), (C) rollout-aware retraining (pushforward/CAT-K from the
40-paper research; ~2-3 days + 4h GPU; principled exposure-bias fix; CAT-K
CVPR 2025: 7M closed-loop-trained beat 102M). Recommendation: B as fast
mechanism test, C if B confirms. Scale-to-small stays paused.

## ✅ FINAL DIAGNOSIS SUMMARY (2026-06-09 night)

| run | log-loss | note |
|---|---|---|
| V2 @300 paths | 1.4801–1.4858 | original gate protocol — FAIL by 0.037 |
| V2 @300, Laplace-smoothed | 1.4634 | floor artifact removed |
| **V2 @1500 paths** | **1.4521** | zero floor events, smoothing no-op; 558s run (vs 324s @300 — just use 1500) |
| lookup (analytic) | 1.4435 | paths-independent |
| baseline | 1.4631 | V2 BEATS it — v7/v8/v9 never did |

**Remaining real gap (~0.009) decomposed:** (a) K placement — on actual-K PAs
V2 gives 21.7% to K vs baseline's 23.3%; pitcher-K% corr 0.253 vs lookup's
0.281 (batter-side corr both high ~0.7 — cascade-driven); (b) FF over-thrown
at hitter counts in rollout (2-0 +13.4pp, 3-1 +8.7, 0-0 +9.0; pitcher-ahead
counts near-perfect) — CAVEAT: marginals run used stand=R/throws=R for all
(JSON lacks handedness; add to --save-dists). (c) shrinkage λ* @1500 = 0.8
(only −0.0012) — over-spread mostly WAS the MC noise. (d) 2B +2.3pp is mostly
sample doubles-light (real 3.0% vs league ~4.6%) + shared map behavior
(lookup +1.6pp too); in-play out share fine (0.660 vs 0.667).

**Fix list before re-gating (cheap → expensive):**
1. Make n_paths=1500 the gate standard (same wall-clock, removes estimator
   tax; lookup unaffected). Honest framing: under the ORIGINAL 300-path
   protocol V2 fails by 0.037; the protocol conflated model quality with MC
   noise — v8 (1.4762@300) vs V2 (1.4801@300) was never a fair model
   comparison either.
2. Per-count type temperatures (v1 already has `count_temperatures`
   convention) → targets the FF over-commit at hitter counts. Extend
   calibrate_v2 + apply count-conditional T in NuisanceModelsV2/g_compute_v2.
3. Save throws/stand in --save-dists; re-run marginals with real handedness
   to confirm the FF over-commit size.
4. K placement: lookup leads via pitcher-side signal; check putaway-pitch
   (2-strike) sequencing + whiff alignment. Possibly map/cascade era check
   (map=2023, eval=2024H2-2025).
5. Only AFTER 1-4 re-gate; scale to small ONLY on pass (capacity won't fix
   K placement; architecture already beats 27M models on type accuracy).

**Artifacts:** per-PA dists at `data/backtests/v2_n800_p{300,1500}_seed0.json`;
diagnostics `scripts/hitter/diagnose_backtest_dists.py`,
`scripts/hitter/diagnose_v2_rollout_marginals.py` (uses new
`g_compute_v2(step_capture_fn=)` hook, active-masked).

## 🔶🔶🔶 V2 BACKTEST RESULT + DIAGNOSIS (2026-06-09 late)

**Result (800 PAs seed 0, n_paths 300):** BB GATE **PASS** (8.7% vs real
9.25%); log-loss **FAIL** (V2 1.4858 first run / 1.4801 rerun — ±0.005 MC
jitter, now fixed by seeding torch per Modal task — vs same-sample lookup
1.4435, baseline 1.4631). Aggregates: V2 is the MOST calibrated of the three
sources (K −0.9pp, BB −0.5, 1B −0.7, HR +0.3, out −0.6); only 2B +2.3pp, and
the sample itself is doubles-light (real 3.0% vs league ~4.6%; even baseline
is +1.6pp there).

**Gap decomposition** (`scripts/hitter/diagnose_backtest_dists.py` on
`data/backtests/v2_n800_p300_seed0.json` — per-PA dists now saved by the
driver):
1. **Floor artifact ~0.017**: one 3B got 0/300 paths → −log(1e-9) costs 0.026
   alone. Laplace-smoothed (add-one over paths, all sources): V2 1.4801→1.4634,
   lookup 1.4435→1.4443. MC sources structurally pay this; analytic lookup
   can't.
2. **Remaining ~0.019 concentrates on out (+0.019) and K (+0.009)** vs lookup;
   V2 BEATS lookup per-PA on HR (−0.011), BB, 2B. V2 dists: higher entropy
   (1.472 vs 1.428), more cross-matchup spread (std P(out) 0.061 vs 0.050),
   LOWER mean p(actual) (0.295 vs 0.305) → per-matchup variation ≈ noise+
   over-spread, smoothed V2 ≈ baseline.
3. **Mechanism for the out-deficit CONFIRMED (8-PA smoke, full run in
   flight)**: xwOBA-map bin occupancy skewed high — in-play-weighted high-third
   = 0.438 vs uniform 0.333 design (map = quantiles of predicted xwOBA on
   REAL 2023 pitches). High bins → 2B/HR mass up, out mass down → losses on
   the 46%-frequent out class. Open question: is V2's pitch mix genuinely more
   hittable (sequence issue) or is it a cascade feature-distribution mismatch
   (needs Probe-6-style isolation: cascade cq on real vs V2-sampled pitches at
   matched counts)? NOTE the marginals diagnostic used stand=R/throws=R for
   all PAs (saved JSON lacks handedness — add to --save-dists next run).
4. Rollout pitch behavior otherwise CLOSE to real (8-PA smoke): in-zone 0.472
   vs 0.491, ball rate 0.366 vs 0.355 (slightly ball-heavy — consistent with
   walks now right).

**In flight (check /tmp logs):** (a) `/tmp/backtest_v2_p1500.log` — 800 PAs at
n_paths=1500, quantifies MC-noise share of the gap (predict smoothed V2
~1.455-1.460; if ≤1.4443 the gate scoring itself is the issue); (b)
`/tmp/diag_v2_marginals.log` — 60 PAs × 200 paths marginals + bin occupancy.

**Candidate fixes, ordered (post-diagnosis):** (i) variance-honest scoring/
paths for MC sources (Laplace + n_paths≥1000, or Rao-Blackwellize the terminal
step — accumulate fractional outcome mass instead of sampling); (ii) the
in-play split: isolate cascade-vs-sequence cause, then either rebuild
`xwoba_outcome_map` for the V2-era feature distribution or recalibrate
contact_quality; (iii) per-matchup over-spread (cascade compression 2.62× +
strong adaLN) — shrinkage on the cell dists. ml-research gate applies to any
NEW method (e.g. spread shrinkage); map rebuild + scoring fixes are documented
project conventions.

**Tooling added (committed):** driver `--save-dists`; `diagnose_backtest_dists`;
`g_compute_v2(step_capture_fn=)` hook (active-masked);
`diagnose_v2_rollout_marginals` (closed-loop per-count marginals + bin
occupancy); torch seeding in `backtest_v2_remote`.

## 🟡🟡🟡 (superseded) V2 backtest launch notes (2026-06-09 evening)

**Run:** `PYTHONPATH=. caffeinate -i python -m scripts.hitter.run_backtest_v2_modal
--n 800 --n-paths 300 --seed 0 > /tmp/backtest_v2.log` (background). Phases:
sample → local lookup anchor (~35 min) → Modal fan-out `backtest_v2_remote`
(T4, cap 10). The driver prints an explicit `=== GATE ===` verdict at the end
(log-loss vs same-sample lookup + BB% within 1pp of real).

**Done this session (committed on `hitter-swing-model`):**

1. **`scripts/calibrate_v2.py`** (commit `1a0db1d`) — type-head temperature on
   2024H1 val. Results: **T=0.9488**, NLL 1.2423→1.2412, ECE 0.011→0.024
   (model nearly self-calibrated), top-1 0.4683. Named checks: π̂(FF)=0.3146
   vs real 0.3152; first-pitch PAD mass 0.006 (v9's was 0.79); GMM teacher-
   forced velo 89.09 vs real 89.11 mph; |plate_x|>1.1: 0.192 vs real 0.185
   (slight over-dispersion — watch BB% direction). Calibrated ckpt at
   `checkpoints_modal/tiny-v1c-base/checkpoint_calibrated.pt` AND pushed to
   the Modal volume at `checkpoints/tiny-v1c-base/checkpoint_calibrated.pt`.

2. **ROLLOUT NORMALIZATION BUG found + fixed** (same commit) — the committed
   rollout fed RAW continuous values (velo≈88) into a model trained on
   z-scores (≈0), and treated GMM samples (z-score space) as raw mph — the
   clip to [60,110] pinned every velo at 60. Fix: `build_single_ab_batch_v2`
   normalizes (nan→0 BEFORE normalizing, exactly mirroring training);
   `g_compute_v2` denormalizes GMM samples for the cascade and writes clipped
   values back normalized. `NuisanceModelsV2.forward` now applies
   `temperatures["type"]` and refuses uncalibrated ckpts (v1 convention).
   Tests: `tests/test_v2_rollout_norm.py` (7 convention tests, named values).

3. **Backtest wiring** (commit `ed15f47`) — `modal_app.py::backtest_v2_remote`
   (V2 analogue of backtest_remote; synthetic AB + build_cell_step_fn +
   g_compute_v2) + `scripts/hitter/run_backtest_v2_modal.py` (same harness/
   sample/scoring as v1 driver, explicit GATE verdict). Smoke (local, real AB
   746196): π̂(FF)=0.501, velo 90.6 mph, AB len 4.05, 0 truncated, 300 paths
   in 10.3s (~3× faster than v1's ~32s).

**Facts for the scale decision:** adaLN MLP = 59% of params (tiny: 4.5M/7.7M;
small would be 44.8M total, NOT the spec's 38M — implementation conditions 2
LNs/block). Tiny val curve plateaued ~step 11-13K (top-1 ~0.475). Plan
unchanged: gate pass → train small; gate fail → diagnose (per-count marginals,
arsenal slope, walk decomposition) — capacity won't fix structural failures.

## 🟢🟢🟢 base-v1c TRAINED — READY TO CALIBRATE + BACKTEST (2026-06-09)

**Training complete:** `tiny-v1c-base-r2`, 3 epochs, 14,100 steps on Modal L4.
**Checkpoint:** `checkpoints_modal/tiny-v1c-base/checkpoint.pt` (88MB, 7.7M params)
**Best checkpoint:** `checkpoints_modal/tiny-v1c-base/checkpoint_best.pt`

| Metric | V2 tiny (base-v1c) | V9 small | V8 small |
|---|---|---|---|
| Type top-1 (val, OOD) | **0.469** | 0.434 | 0.438 |
| Parameters | 7.7M | 27M | 27M |
| Architecture | adaLN + GMM | context tokens + bins | context tokens + bins + MDN |

**The V2 tiny model beats both v8 and v9 on type accuracy with 1/3 the parameters.** The adaLN conditioning is working — the model is using the pitcher/batter profiles more effectively than the context-token approach.

### What needs to happen next (in order)

1. **Write V2 calibration script** (`scripts/calibrate_v2.py`). The V2 model has different heads than v9 — it needs its own calibration that fits temperatures for the type head. The GMM head may also benefit from calibration but start with type only.

2. **Run calibration:**
   ```bash
   PYTHONPATH=. python -m scripts.calibrate_v2 \
       --ckpt checkpoints_modal/tiny-v1c-base/checkpoint.pt
   ```

3. **Wire V2 into the backtest harness.** The existing `scripts/hitter/run_backtest_modal.py` uses NuisanceModels (v1). It needs to be extended (or a new script created) to use NuisanceModelsV2 + g_compute_v2. Key: the hitter cascade (XGBoost) is unchanged — only the pitch source changes.

4. **Run the backtest:**
   ```bash
   python -m scripts.hitter.run_backtest_v2_modal --n 800 --n-paths 300
   ```
   Gate: log-loss < 1.4435, BB% within 1pp of real (~8-10%).

5. **If gate passes:** Scale to small (6L/8H/d512, ~38M params) for the production checkpoint. Run `modal_app.py::train_v2_remote --size small --epochs 3 --run-name small-v1c-base`.

6. **If gate fails:** Diagnose using the same tools as v9 (per-count type distribution, pitcher arsenal correlation, walk rate decomposition). Then try base-v1a (GIVT-style, spec already written).

### What was built this session (2026-06-08/09)

**New architecture (model/v2/ package):**

| File | What it does |
|---|---|
| `model/v2/__init__.py` | Package init |
| `model/v2/config.py` | V2Config dataclass — adaLN dims, GMM params, continuous normalization constants, tiny/small factories |
| `model/v2/adaln.py` | AdaLNConditioner (MLP: pitcher+batter → 6,144 per-layer knobs) + AdaLayerNorm (replaces standard LN) |
| `model/v2/transformer.py` | V2TransformerBlock with AdaLayerNorm instead of nn.LayerNorm, MultiHeadCausalAttention, FeedForward |
| `model/v2/embeddings.py` | V2InputLayer: type embedding (8 vocab) + continuous projection (50-dim → d_model) + positional encoding |
| `model/v2/heads.py` | TypeHead (weight-tied softmax, 8 classes) + ContinuousGMM (K=5 diagonal Gaussians over 4 dims) |
| `model/v2/model.py` | PitchGPTV2 — wires everything together. forward() → type_logits + hidden; predict_continuous() → GMM params |
| `model/v2/dataset.py` | V2AtBatDataset — position 0 = "before any pitch" start token; emits continuous values (not bins); collate function |

**Training + rollout:**

| File | What it does |
|---|---|
| `scripts/train_v2.py` | Training loop: type CE + GMM NLL loss, z-score normalization of continuous values, noise injection, AdamW cosine schedule |
| `causal/nuisance_v2.py` | NuisanceModelsV2 — loads V2 checkpoint, wraps model for rollout. build_single_ab_batch_v2() for building rollout batches |
| `causal/g_computation_v2.py` | g_compute_v2 — Monte Carlo rollout adapted for V2: samples type (softmax) → continuous (GMM) → cascade. plate_to_zone() for cascade compat |
| `modal_app.py` | Added train_v2_remote function for Modal GPU training |

**Tests (all passing):**

| File | Tests |
|---|---|
| `tests/test_v2_model.py` | 25 tests: config, adaLN, transformer block, full model forward, GMM after type, param count |
| `tests/test_v2_gmm.py` | 12 tests: GMM shapes, NLL, sampling, weight-tying, gradient flow, PAD embedding |
| `tests/test_v2_dataset.py` | 5 tests: position 0 invariants, left-shift targets, padding, real data verification |

**Diagnostic scripts (from v9 investigation, still useful):**

| File | What it does |
|---|---|
| `scripts/hitter/diagnose_rollout_marginals.py` | Per-count type distribution: model vs real data. Run on V2 after calibration. |

**Key design decisions documented:**

| Document | What it covers |
|---|---|
| `docs/superpowers/specs/2026-06-08-base-v1c-design.md` | Full spec: architecture, training, rollout, evaluation |
| `docs/superpowers/specs/2026-06-08-base-v1a-design.md` | Alternative GIVT-style spec (build after v1c is tested) |
| `docs/superpowers/plans/2026-06-08-base-v1c.md` | Implementation plan (8 tasks, all completed) |
| `.claude/skills/ml-research/SKILL.md` | ML literature research skill (hard gate: search before any ML change) |

### How V2 differs from v9 (quick reference)

| Aspect | V9 (PitchGPT) | V2 (PitchGPTV2) |
|---|---|---|
| Pitcher/batter conditioning | Context tokens at positions 0-2 (weak, additive via attention) | adaLN-Zero: 6,144 per-layer scale/shift knobs (strong, multiplicative) |
| First pitch prediction | Position NC-1 = 2 (untrained, 79% PAD mass) | Position 0 (trained, real prediction) |
| Velocity output | 10 bins → bin-to-mph conversion | GMM → exact mph directly |
| Spin output | 8 bins → bin-to-rpm conversion | GMM → exact rpm directly |
| Location output | 13 zones → MDN refines to (x,z) | GMM → exact (x,z) directly |
| Zone head | 13-class softmax | None (location from GMM) |
| Result head | 7-class softmax (detached trunk) | None (cascade handles outcomes) |
| AB-outcome head | Optional 7-class per-pitch | None (cascade + count machine) |
| Break prediction | Not predicted | Not yet (v1.1 — need pfx_x/pfx_z in augmented data) |
| Continuous normalization | None (raw bins) | Z-score: (x - mean) / std per dimension |
| Input representation | Sum of 10+ factor embeddings (all discrete) | Type embedding + 50-dim continuous/state projection |
| Sequence structure | [ctx0, ctx1, ctx2, pitch0, pitch1, ...] | [start, pitch0, pitch1, ...] (no context tokens) |

### Critical conventions for V2

1. **Type IDs are 1-indexed in data (PAD=0, FF=1..FS=7).** The type head outputs 8 logits (including PAD). During rollout, slice indices 1:8 for real types and renormalize. During loss, use cross-entropy with ignore_index=-100.

2. **Continuous values are z-score normalized.** Means and stds are stored in V2Config: `continuous_means = (88.38, 2254.70, 0.04, 2.24)`, `continuous_stds = (6.03, 361.77, 0.85, 0.98)`. The model sees normalized values; the rollout must DENORMALIZE after GMM sampling before passing to the cascade.

3. **Position 0 is the start token.** type_ids=0 (PAD), continuous=zeros, result_ids=0 ("none"), count/outs/runners from the real game state. The model predicts pitch 1 from position 0.

4. **The GMM is conditioned on the sampled type.** After sampling a type from the softmax, the type embedding is concatenated with the hidden state and fed to the GMM head. This means the GMM knows "this will be a slider" before predicting the slider's velocity/spin/location.

5. **The cascade receives exact continuous values.** No bin conversion needed. plate_x/plate_z come directly from GMM sampling. release_speed and release_spin_rate come directly from GMM sampling (after denormalization). in_zone is computed from coordinates: `|plate_x| <= 0.83 and 1.5 <= plate_z <= 3.5`.

6. **Input clamping is applied.** All embedding lookups and one-hot scatters clamp indices to valid ranges to prevent CUDA assertion errors on rare edge-case data.

### v9 training (completed 2026-06-08, FAILED gate, superseded by V2)

See the section below for the full v9 investigation history. Summary: v9 added noise injection + first-pitch training to v8. Type accuracy held (0.434 vs 0.438) but walk rate unchanged (6.4%). Pipeline fixes (velo/spin pass-through, in_zone from coordinates) improved log-loss slightly (1.5014 → 1.4844) but didn't fix walks. Root cause: context-token conditioning too weak (slope 0.84), model regresses toward league average.

### ML research completed (2026-06-07/08/09)

Three research runs using the ml-research skill:

**1. Rollout drift (40 papers):** Noise injection (GNS ICML 2020), pushforward trick (ICLR 2022), CAT-K traffic sim (CVPR 2025), Long Horizon Temperature Scaling (ICML 2023). Applied noise injection in v9 — didn't help walk rate.

**2. Architecture design (13 papers):** GIVT (ECCV 2024), Q-FAT (NeurIPS 2025), DiT adaLN-Zero (ICCV 2023), Decision Transformer (NeurIPS 2021), ScoutGPT (2026). Led to the V2 architecture.

**3. Continuous vs discrete (13 papers):** Stewart et al. (AISTATS 2023), Chronos (Amazon 2024), weather models (GraphCast, Pangu-Weather). Led to GMM heads replacing bins.

## 🔴🔴🔴 ARCHITECTURAL REDESIGN: base-v1c → base-v1a (2026-06-08)

**v7-v9 all failed the backtest gate (log-loss 1.4435).** Exhaustive investigation found:

| Version | Log-loss | BB% | BB real |
|---|---|---|---|
| lookup | 1.4435 | 8.0% | 9.25% |
| v7 | 1.4846 | 5.2% | 9.3% |
| v8 | 1.4762 | 6.4% | 9.3% |
| v9 (noise injection + first-pitch training) | 1.5014 | 6.4% | 9.25% |
| v9 + pipeline fixes (velo/spin/in_zone) | 1.4844 | 6.5% | 9.25% |

**Root causes identified (2026-06-08 session):**

1. **First-pitch prediction is untrained.** Position NC-1 (last context token)
   was never given gradient signal. 79% of probability goes to PAD. Every
   simulated AB starts from garbage. v9 added training here but 1 epoch wasn't
   enough — FF still 22% vs real 36%.

2. **Pitcher conditioning is too weak.** The pitcher profile sits in a context
   token (position 0). The model under-attends to it. Slope = 0.84 — when a
   pitcher throws 60% fastballs, the model predicts ~50%. It hedges toward the
   league average instead of committing to what THIS pitcher does.

3. **Velocity and spin are binned.** 10 velo bins, 8 spin bins. The cascade
   needs continuous mph/rpm. Bin-to-continuous conversion loses information.
   Pipeline fixes helped log-loss slightly but didn't fix walk rate.

4. **Zone is a redundant step.** Model predicts 13 zones, then MDN refines to
   exact coordinates. An unnecessary intermediate discretization.

5. **Cascade gap is noise, not bias.** The cascade's 8.0% vs real 9.25% gap is
   not statistically significant (p=0.22). Population BB% is 8.12%. The cascade
   is fine — the problem is entirely in the pitch generation.

**The fix: new architecture (base-v1c then base-v1a).**

**base-v1c (Hybrid — build FIRST):**
- Type embedding (proven) + continuous projection for velo/spin/break/location
- adaLN-Zero conditioning: pitcher+batter profiles generate per-layer
  scale/shift that modulates every transformer block. 6,144 conditioning
  parameters per matchup. Can't be ignored (multiplicative, not additive).
- First pitch at position 0 (no context tokens, no NC-1 hack)
- GMM output head for all 6 continuous values (velo, spin, h-break, v-break,
  plate_x, plate_z), conditioned on the sampled type
- No zone head, no velo bins, no spin bins
- ~38M params (25M backbone + 13M adaLN MLP)
- Spec: `docs/superpowers/specs/2026-06-08-base-v1c-design.md`

**base-v1a (GIVT-style — build AFTER v1c is tested):**
- Same as v1c but type input is one-hot projected (no embedding lookup)
- Cleaner uniform architecture, all inputs are continuous vectors
- Tests whether the input representation matters
- Spec: `docs/superpowers/specs/2026-06-08-base-v1a-design.md`

**Literature basis (ml-research skill, 2026-06-08):**
- GIVT (Tschannen et al., ECCV 2024): GMM heads on transformers
- Q-FAT (NeurIPS 2025 Spotlight): GMM heads for sequential action prediction
- DiT (Peebles & Xie, ICCV 2023): adaLN-Zero conditioning
- ScoutGPT (2026): player-conditioned sports event transformer
- Decision Transformer / Trajectory Transformer (NeurIPS 2021): mixed output strategies

**What the cascade needs (no retraining):**
The cascade was trained on real Statcast data with real mph, rpm, coordinates.
The new model gives it exact values instead of bin conversions. The cascade
receives BETTER inputs. No cascade changes needed.

**Next step:** Write implementation plan for base-v1c → build → train on
Modal → calibrate → backtest. If v1c passes, ship it. If not, build v1a.

### v9 training (completed 2026-06-08, FAILED gate)

small-v9 trained on Modal (3 epochs, 14,109 steps, ~4h wall-clock):
- `--train-first-pitch` (added NC-1 to type/zone loss)
- `--noise-p 0.2 --noise-ramp-steps 3000` (input perturbation)
- All v8 flags (MDN, AR exec heads, type-conditioned, no AB-outcome)
- Checkpoint: `checkpoints_modal/small-fold0-v9/checkpoint_calibrated.pt`
- Calibration: type ECE 0.018, type top-1 0.434 (matched v8's 0.438)
- Backtest: log-loss 1.5014 (WORSE than v8's 1.4762), BB 6.4% unchanged

Pipeline fixes (velo/spin pass-through, in_zone from coordinates):
- Improved log-loss to 1.4844 but BB still 6.5%

### Diagnostic findings (2026-06-08 session)

**Per-count type distribution diagnostic (diagnose_rollout_marginals.py):**
- Model is OVER-dispersed (mean ΔH = +0.17 nats vs real)
- FF under-predicted by 10-27pp at every count
- Model's top-1 is wrong at 11/12 counts (predicts CH or SL when reality is FF)
- PAD mass at NC-1 = 79% (untrained position)
- PAD mass at trained positions = 0% (model is fine at pitch positions)

**Pitcher profile usage check:**
- Probability wasted on pitches pitcher doesn't throw: 7% at NC-1, 0.4% at NC+0
- Correlation with pitcher's arsenal: r=0.887 at NC-1, r=0.847 at NC+0
- Slope (how much model reacts to arsenal): 0.70 at NC-1, 0.84 at NC+0
- Ideal slope = 1.0 — the model under-reacts to pitcher identity

**Cascade investigation (two sub-agents, 2026-06-08):**
- Cascade math is correct (result probabilities sum to 1.0)
- Foul rates reasonable (0.49-0.54 by count)
- Foul tip bug found (foul tips classified as contact not whiff) — wrong
  direction, inflates walks by ~0.1pp
- Cascade's 8.0% vs 9.25% gap is NOT statistically significant (p=0.22)
- in_zone mismatch confirmed but fixing it alone made things worse (Probe 4)
- Velocity/spin were frozen at per-type means — fixed but minimal impact

### ML research completed (2026-06-07/08)

**Rollout drift research (40 papers, 5 search threads):**
- Noise injection (GNS ICML 2020, MeshGraphNets ICML 2021): proven in physics
- Pushforward trick (ICLR 2022 Spotlight): formalized scheduled sampling for simulation
- CAT-K (CVPR 2025): closest setting match (7M-param traffic sim), beat 102M model
- Scheduled sampling for Transformers: only ~1 BLEU improvement (ACL 2019)
- Long Horizon Temperature Scaling (ICML 2023): sequence-level temperature
- Applied: noise injection in v9 — didn't help walk rate

**Architecture research (13 papers):**
- GIVT (ECCV 2024): GMM heads on transformers for continuous outputs
- Q-FAT (NeurIPS 2025): validates GMM heads for sequential action prediction
- DiT (ICCV 2023): adaLN-Zero for strong identity conditioning
- Decision Transformer vs Trajectory Transformer (NeurIPS 2021): mixed output strategies
- Applied: designed base-v1c and base-v1a architectures

**Continuous vs discrete research (13 papers):**
- Stewart et al. (AISTATS 2023): classification trains better features than regression
- Chronos (Amazon 2024): 4096 bins works for time series
- GIVT outperforms VQ-GAN/MaskGIT for image generation
- Weather models (GraphCast, Pangu-Weather): pure MSE regression works for unimodal
- Applied: decided on GMM heads for multi-modal continuous, softmax for categorical

### New skill created: ml-research

`.claude/skills/ml-research/SKILL.md` — HARD GATE: literature search before
any ML training change, architecture mod, or generation strategy. Searches
arxiv + related fields. Must complete before code.

### Commits this session (hitter-swing-model branch)

- diagnostic script: `scripts/hitter/diagnose_rollout_marginals.py`
- v9 training flags: `--train-first-pitch`, `--noise-p`, `--noise-ramp-steps`
- PAD masking in g_computation.py rollout
- velo/spin bin-to-continuous conversion in nuisance.py
- in_zone coordinate fix in hitter/rollout.py
- ml-research skill + CLAUDE.md skill index update
- base-v1c spec + base-v1a spec

## 🧵 THE THREAD — why we're building small-v8 (read this first)

The whole chain this session (2026-06-05), so a fresh reader knows *why*:

1. **App A** (full-game sim → projected score + win prob, backtested vs real finals)
   needs trustworthy **per-PA fuel** = the **pitchGPT + hitter-cascade** matchup cards
   (the predicted K/BB/1B/2B/3B/HR/out distribution per pitcher×batter).
2. User flagged a suspiciously hot card cell (Jackson **Merrill 1.013 OPS vs Devin
   Williams**). Pressure-testing it surfaced that the cards use the **pitchGPT+cascade**
   path, which had **NEVER been backtested** — only the count-only *lookup* path had
   (1.405 < 1.452 baseline). So we didn't actually know if pitchGPT's pitch selection
   helped.
3. Built + ran the **3-way per-PA backtest** (real held-out 2024H2+2025 PAs, log-loss).
   Result: **pitchGPT+cascade LOST** to both the lookup AND the league baseline — it
   **under-reports walks (5.2% vs 9.3% real)**. (Details in the 🔴 section.)
4. **Root-caused** the walk deficit to the **zone-CENTROID location glue** in the
   simulator — NOT the model. pitchGPT's pitch prediction is well-calibrated on real
   data; the simulator was feeding the cascade each pitch's zone *center* instead of a
   real (x,z), making out-of-zone pitches look borderline → over-swing → too few balls
   → too few walks (compounds to ~half). (Details in the 🟢 section.)
5. **Fix = small-v8:** retrain pitchGPT with a **continuous-location MDN head** so it
   emits a real (x,z) the cascade eats directly (no centroid), + autoregressive pitch
   factorization, drop the unused AB-outcome head, keep result head as light aux, native
   velo/spin into the cascade, rare-type loss tuning, recalibrate. Currently EXECUTING
   (🟡 section; T1–T3 committed, resume at T4).
6. **App A stays PAUSED** until small-v8 passes the backtest GATE (beat lookup **1.4435**
   / baseline **1.4631** on the same harness). Then App A unblocks with trusted fuel.

**Standing user constraints (also in memory):** pitchGPT IS the fuel — NEVER substitute
the count-only lookup. Use very plain language. Pressure-test claims proactively. Keep
THIS doc live. Add progress trackers (i/N + ETA) to every long job.

**What was BUILT this session (all committed, branch `hitter-swing-model`):** the 3-way
backtest harness (`hitter/backtest.py` + tests, `scripts/hitter/run_backtest_modal.py`,
`modal_app.py:backtest_remote`, cap 10); 2 root-cause diagnostics
(`scripts/hitter/diagnose_glue_isolation.py`, `diagnose_pitch_outcomes.py`); the App A
spec; the small-v8 spec + ADR-014 + plan; and small-v8 Tasks 1–3.

---

## 🔴 small-v8 COMPLETE — GATE FAILED, ROOT CAUSE UPDATED (2026-06-07)

**Plan:** `docs/superpowers/plans/2026-06-05-small-v8.md`. **Spec:**
`docs/superpowers/specs/2026-06-05-small-v8-design.md`. **ADR:**
`docs/decisions/014-...md`. Branch `hitter-swing-model`.

**ALL TASKS DONE (T1–T10).** Training completed on Modal (3 epochs, 14,109
steps). Calibrated. Backtested. **Gate FAILED.** The MDN location head works
but the walk deficit persists — and the root cause has been updated.

### small-v8 backtest result

| pitch source | log-loss | BB pred | BB real |
|---|---|---|---|
| lookup + cascade | **1.4435** | 8.0% | 9.3% |
| baseline | 1.4631 | 8.5% | 9.3% |
| **pitchGPT v8 + cascade** | **1.4762** | **6.4%** | **9.3%** |
| pitchGPT v7 + cascade | 1.4846 | 5.2% | 9.3% |

v8 IS better than v7 (1.4762 vs 1.4846), but both fail the gate (must beat
lookup 1.4435). Walks improved from 5.2% to 6.4% but are still ~31% short.

### v8 calibration

| head | ECE | top-1 | temperature |
|---|---|---|---|
| type | 0.017 | 0.438 | 1.01 |
| zone | 0.005 | 0.293 | 1.01 |
| velo | 0.010 | 0.417 | 1.19 |
| spin_rate | 0.009 | 0.553 | 1.03 |
| result | 0.002 | 0.547 | 0.99 |

MDN distributional check: sampled |plate_x|>1.1 = 0.191 (real 0.190) — perfect.

### Diagnosis probes run (2026-06-06/07) — what we tested and learned

**Probe 1 — MDN teacher-forced distributional check (T8):**
Sampled (plate_x) from v8's MDN on val data (326K pitches, teacher-forced).
Result: |plate_x|>1.1 = 19.1% sampled vs 19.0% real, KS stat=0.011.
**Conclusion: MDN reproduces real location distribution perfectly when given
real pitch sequences.**

**Probe 2 — v8 backtest, first run (T10):**
n=800 PAs, n_paths=300, Modal. Result: BB=6.4% (v7 was 5.2%, real 9.3%).
Log-loss 1.4762 — better than v7's 1.4846 but fails gate.
**Conclusion: MDN helps (~23% of walk gap closed) but doesn't fix the problem.**

**Probe 3 — `in_zone` feature mismatch investigation:**
Discovered that the cascade trains `in_zone` from coordinates (`|plate_x| <=
0.83 & 1.5 <= plate_z <= 3.5`) but the rollout computes it from zone_id
(`zone_id < 9`). Measured disagreement: 3.5% of pitches (73 false-in, 83
false-out — roughly symmetric).
**Conclusion: a real bug, but small and symmetric — unlikely to be the
dominant cause.**

**Probe 4 — `in_zone` fix backtest:**
Applied the coordinate-based `in_zone` fix and re-ran the backtest.
Result: BB=6.2% (WORSE than 6.4%), log-loss=1.5003 (WORSE than 1.4762).
Reverted the fix (`27c5366`).
**Conclusion: the cascade was calibrated to the zone_id-based in_zone
definition. Changing it without retraining the cascade broke calibration.
The `in_zone` mismatch is NOT the cause of the walk deficit.**

**Probe 5 — zone coverage analysis (user hypothesis):**
Checked whether pitchGPT's 13 zones can represent "wildly outside" pitches.
Zones 9-12 (out-of-zone) lump everything together: zone 12 contains pitches
from plate_x=0.12 (barely outside) to plate_x=3.30 (three feet wide). 12%
of all pitches land in "no sane batter swings" territory (|x|>1.5 or z
outside [0.5, 4.5]) — real swing rate on these is 6.7% vs 38% on borderline.
**Conclusion: the zones ARE coarse, but the MDN was designed to fix this by
learning the within-zone spread — and it does (Probe 1). The problem is
elsewhere.**

**Probe 6 — CASCADE ISOLATION DIAGNOSTIC (the breakthrough, 2026-06-07):**
Fed the cascade 10K real held-out pitches with three different location
sources, keeping all other features (type, zone, count, profiles) REAL:

| location source | ball rate | swing rate |
|---|---|---|
| ALL REAL | 0.354 | 0.485 |
| CENTROID (v7 bug) | 0.303 | 0.537 |
| **MDN teacher-forced** | **0.372** | **0.451** |

The MDN OVER-CORRECTS — ball rate 0.372 > real 0.354. It fixes **137%** of
the centroid gap. The MDN locations make batters take TOO MUCH, not too little.

**THIS CHANGES THE ROOT CAUSE.** If MDN locations produce MORE balls than
real data, but the rollout produces FEWER walks, the walk deficit is NOT from
the location. It's from the PITCH SEQUENCE that pitchGPT generates — the
type/zone choices the model makes during the rollout produce pitches that
are more swingable on average than real pitches, and this overwhelms the
MDN's over-correction. The MDN is masking the problem, not causing it.

### Updated root cause (2026-06-07)

**ORIGINAL hypothesis (2026-06-05):** zone centroid → borderline location →
over-swing → too few balls → too few walks. **Fix = MDN location head.**

**UPDATED (2026-06-07):** The centroid WAS a problem, and the MDN DOES fix
it (over-fixes it, actually). But there is a SECOND, LARGER problem: **the
pitch sequences pitchGPT generates during rollout produce pitches that are
more hittable than real pitches** — the cascade swings more on model-
generated sequences than on real sequences, even when the locations are
correct. The MDN partially masks this by over-correcting the location, but
the net effect is still too few walks.

**Not yet investigated:** What makes the model-generated sequences more
hittable? Candidates:
- Type distribution: does the model predict too many fastballs / too few
  breaking balls in the rollout?
- Zone distribution: does the model put too many pitches in the strike zone?
- Sequence patterns: does the model fail to reproduce pitch-sequencing
  patterns (e.g., wasting pitches, working counts) that lead to walks?
- Exposure bias: the model was trained on real sequences but generates its
  own — small per-pitch errors compound over a 5-pitch AB.

**Next step:** Compare pitchGPT's rollout type/zone marginals to real data.
If the rollout puts more pitches in-zone than reality, that's the fix target.
If the marginals match but the walk rate is still low, the problem is in the
sequence conditioning (exposure bias) and harder to fix.

### Modal training lessons (2026-06-06)

Three failed attempts to launch a 10h training job:
- Run 1: `modal run` without `--detach` → died at step 700 (Mac sleep)
- Run 2: `modal run --detach modal_app.py` (local entrypoint) → died at step
  50 (local `main()` still blocked; killed on terminal cleanup)
- Run 3: `modal run --detach modal_app.py::train_remote` (direct function
  call) + `nohup` → SURVIVED, completed 14,109 steps.

**Rule:** always use `nohup modal run --detach modal_app.py::train_remote`
for long jobs. Documented in `.claude/skills/modal-training/SKILL.md`.

### Commits this session (hitter-swing-model)

| commit | description |
|---|---|
| `e0b3c2c` | T4: AR exec-head conditioning + LocationMDN wiring |
| `039aa26` | T5: dataset emits (plate_x,plate_z) MDN target |
| `aaddd91` | T6: MDN loss + result reweight + v8 CLI flags |
| `45f6c88` | T7: modal_app v8 flags |
| `9623cd8` | T9: MDN sampling in rollout + native velo/spin glue |
| `e3d357f` | docs: ContextSwitcher T4-T9 done |
| `dd88347` | T8: calibration + MDN distributional check |
| `576cb7d` | modal-training skill |
| `bc8106c` | modal-training skill update (::train_remote lesson) |
| `e73da79` | fix: MDN sample device/dtype for MPS compat |
| `226def2` | fix: nuisance.device typo in MDN rollout |
| `ecbd037` | fix: in_zone from coords (later reverted) |
| `27c5366` | revert: in_zone fix (made things worse) |

## 🟢 ROOT CAUSE FOUND: the zone-CENTROID location glue (2026-06-05)

**Why pitchGPT+cascade lost the backtest (under-reported walks): the simulator
feeds the batter cascade each pitch's location as its ZONE CENTROID, not a real
spot.** Decisive isolation (`scripts/hitter/diagnose_glue_isolation.py`, cascade
on real held-out pitches, swap one glue factor at a time):

| variant (cascade on real pitches) | ball | swing |
|---|---|---|
| ALL REAL features | 0.354 | 0.485 |
| **loc = zone centroid** | **0.303** | **0.537** |
| velo = type-mean | 0.353 | 0.485 |
| spin = blank | 0.350 | 0.490 |

Location-centroid ALONE reproduces the rollout's ball deficit (0.30 vs real 0.35)
and swing excess; velo/spin do nothing. Mechanism: the 4 out-of-zone zones are
COARSE (real |plate_x| spreads 0.83→4 ft, std 0.55; 37% are >1.1 ft out), but the
centroid collapses each to ONE borderline point (~0.9 ft, just off the corner).
So every out-of-zone pitch looks borderline → batter chases → too few takes → too
few balls. A small per-pitch ball deficit COMPOUNDS (walk needs 4 balls):
(0.303/0.354)^4 ≈ 0.54 → 9.3% real walks × 0.54 ≈ 5.0% ≈ the observed 5.2%.

**RULED OUT (each tested):** in-zone over-prediction (~0.48 ≈ real once active-
masked — the "80% in-zone" was a measurement artifact from counting dead at-bats),
drift/self-feedback, spin placeholder, velo means, count-blindness, truncation
(0.04%), ballpark/catcher/umpire context, the type-conditioned zone step. The
pitchGPT zone HEAD is well-calibrated on real data (teacher-forced 49% ≈ real;
zone top-1 25% so no leakage). **The model is fine; the simulator GLUE was wrong.**

**Two measurement bugs I hit (don't repeat):** (1) capturing per-pitch stats over
ALL rollout paths incl. terminated ones inflates rates — mask to ACTIVE; (2)
reading propensity at NCTX vs NC-1 is off-by-one (NC-1 predicts the FIRST pitch).

**FIX (decided): retrain pitchGPT → small-v8 with a CONTINUOUS location head** (so
it generates a real (x,z) the cascade eats directly — no centroid). Also: use the
model's native velo/spin, decide what to do with the now-unused outcome/result
heads, + other improvements. → brainstorm next (see below / new spec).
NOTE a cheap interim hybrid (sample a real within-zone location by zone×type) was
discussed but user chose the principled retrain.

---

## 🔴🔴🔴 pitchGPT+cascade BACKTEST — IT LOSES TO LOOKUP *AND* BASELINE (2026-06-05)

**The pitchGPT pitch source does NOT beat the simple count-only lookup, and is
WORSE than the league-average baseline** on per-PA outcome log-loss. This was the
never-run experiment (commit 50e44b0 had marked it TODO; only lookup had ever
been backtested at 1.405). Now run, three-way, on the SAME 800 held-out 2024H2+
2025 PAs (lower=better):

| pitch source        | log-loss | 95% CI            |
|---------------------|----------|-------------------|
| lookup + cascade    | **1.4435** | [1.387, 1.499]  |
| baseline (league)   | 1.4631   | [1.409, 1.518]    |
| **pitchGPT + cascade** | **1.4846** | [1.415, 1.565] |

Lookup reproduced its known win over baseline (≈0.02) → harness is sound (anchor
n=300 lookup = 1.4073 ≈ the historical 1.405). **pitchGPT is the worst of three.**

**WHY (calibration-in-aggregate, mean pred vs real):** the killer is **walks —
pitchGPT predicts 5.2% vs 9.3% real (≈half)**; also over-predicts K (+2.0pp) and
2B (+2.1pp). **HR (0.043 vs 0.043) and out are well-calibrated in aggregate** — so
it's NOT uniform HR inflation; it's a PLATE-DISCIPLINE problem (pitchGPT throws too
few balls → too few walks, too many K/contact). The earlier "Merrill 1.013 hot OPS"
is the per-matchup face of this.

**Fairness caveat:** the shared xwOBA→outcome map was tuned on the lookup pitch
mix, but the dominant BB/K errors are UPSTREAM of that map (swing/take/called-strike
nodes on pitchGPT's pitch LOCATIONS); the map only affects the hit-type split (the
2B miss). So the loss is a genuine pitchGPT-path problem, not just a map artifact.

**IMPLICATION:** App A is PAUSED — do not build the game sim on this fuel until the
pitchGPT path beats baseline. We do NOT swap to lookup (pitchGPT is the fuel — see
memory). The walk under-prediction was debugged → **root cause = zone-centroid glue
(🟢 section); fix = small-v8 (🟡 section)**. Re-run this backtest at T10 as the GATE.

**The backtest is now a repeatable scorecard:**
`python -m scripts.hitter.run_backtest_modal --n 800 --n-paths 300` (Modal, cap 10;
lookup+baseline local). Pure pieces in `hitter/backtest.py` (+ tests); Modal fn
`modal_app.py:backtest_remote`. Sanity: `--sanity-local` (no Modal). The driver has
live progress trackers (phase banners + i/N + ETA).

---

## ⭐⭐⭐⭐⭐ APP A — FULL-GAME SIM ENGINE (brainstorm, 2026-06-05) — PAUSED (fuel fails backtest)
**Spec written + reviewed:** `docs/superpowers/specs/2026-06-05-app-a-full-game-sim-design.md`.
Engine is fuel-agnostic and buildable, but per the backtest above the pitchGPT
per-PA fuel is miscalibrated — fix the fuel before building/ trusting App A numbers.

**Goal:** simulate a whole game from 0-0 top-1st ~10K times → **projected score +
win probability**, pre-game, backtested against real finals. Drives a daily
prediction site. (NOT live in-game.)

**THE THREE LAYERS (this resolves the recurring "which model does what" confusion):**
- **Per-pitch** (pitchGPT π̂ + hitter cascade μ̂): what pitch, what the batter does to it. ✅ BUILT.
- **Per-AT-BAT** (compose / g_compute hitter mode): chain pitches → the PA *result*
  (K/BB/1B/2B/3B/HR/out). ✅ BUILT — this IS a matchup-card cell's outcome_dist.
- **Per-GAME** (App A): chain PAs → **runs → score**. ⬅️ THE NEW LAYER. Needs a
  base-running model — the pitch/hitter models say "single", they say NOTHING about
  whether the runner on 2nd scores. Converting PA-results → runs is separate.

**KEY FEASIBILITY INSIGHT (why 10K full-game sims are tractable):** do NOT re-run
the ~32s/PA rollout inside the game loop. **PRECOMPUTE** the per-(pitcher,batter)
outcome distribution ONCE (= exactly what a matchup-card cell already holds), then
the 10K game sims just SAMPLE from those precomputed dists through a fast pure-Python
state machine. The matchup cards ARE App A's fuel.

**BASE-RUNNING MODEL — DECIDED (pressure-tested 2026-06-05): empirical base-out
transition matrix.** For each (base_state[8], outs[3], PA_outcome[7]) → real
distribution over (next base_state, runs_scored), computed from Statcast play-by-play
(on_1b/2b/3b + runner movement in data/raw/). Data-grounded, what real sim engines use,
backtestable. Deterministic (single=+1 base) = too crude (under-counts runs); full
event detail (SB/CS/GIDP/errors/first-to-third) = overkill/overfit for v0.
- KNOWN v0 limitations (flag, enrich in v2): league-AVERAGE base-running (no player
  sprint-speed — it's in Statcast for v2); per-PA dist is NEUTRAL-context (no
  pitch-around / 3rd-time-through fade — partly handled by pitching-change logic);
  no individual SB/error events.

**GAME STATE MACHINE — components to build:**
- State: inning, top/bot, outs, base_state (runners 1/2/3), score, per-team lineup
  pointer (1-9, cycles), current pitcher (starter → bullpen).
- Per PA: look up (cur_pitcher, cur_batter) precomputed outcome dist → sample outcome
  → apply base-out transition matrix → advance bases + add runs → next batter.
- Pitching changes (v0 rule-based): starter ~6IP / ~100 pitches → bullpen by
  leverage/role. (Improve later.)
- Loop 9 innings (+extras if tied) → final score. Monte Carlo 10K → score
  distribution → win prob + projected score + run-total.

**VALIDATION (this is how we KNOW it's good — backtest vs real finals):**
- We have actuals via `scripts/mcsim/fetch_actuals.py` (June 4 already overlaid).
- Metrics: (1) win-prob calibration (when we say 60%, does A win ~60%?), (2)
  projected-score / run-total accuracy vs actual, (3) score-distribution calibration.
- Backtest on a set of past completed games BEFORE predicting future ones (D8 discipline).

**REUSES (don't rebuild):** g_compute hitter mode / matchup_card cells (per-PA fuel),
`data/run_value/` (base-out + RE24 tables — start here for the transition matrix),
`mcsim/storage.py` (a new app="game_sim"), `fetch_actuals` (validation), the demo
(a new App-A tab later). Modal: cap is **10 containers** (user's plan) — size runs as
~rows/10 × per-unit time.

**BUILD ORDER (suggested):** (1) build + unit-test the base-out transition matrix from
real play-by-play; (2) pure-Python GameState + step(PA_outcome) → bases/runs (TDD,
hand-checkable cases like "single w/ runner on 2nd"); (3) game Monte Carlo wrapping the
precomputed per-PA dists + pitching changes; (4) backtest harness vs actuals; (5) only
then a daily-prediction path + UI. Follow brainstorming→writing-plans: this section is
the brainstorm; next session should formalize the spec + plan first.

---

## ⭐⭐⭐⭐ pitchGPT + CASCADE SIMULATOR WORKS (2026-06-04)

**`g_compute(outcome_model="hitter", hitter_step_fn=...)` is built + working.**
pitchGPT (small-v7) samples each pitch sequentially (full context + sequence
preserved); the CASCADE decides the batter response (not the transformer's weak
outcome head). Default `"head"` mode byte-unchanged (8 natural-mode tests pass).
E2E: elite hitter vs RHP, 300 sims → OPS 1.24 / K% 25.8% / HR 11.4%.

The pieces (`hitter/rollout.py`, all tested):
- `cascade_to_result_probs` — translator: cascade outputs → 7-class result vocab.
- `build_step_features` — sampled pitch → cascade features (zone→centroid via
  `checkpoints/hitter/zone_centroids.json`, velo/spin from pitcher per-type means,
  prev pitch as the lag → SEQUENCE preserved inside the rollout).
- `make_hitter_step_fn` — per-cell closure bundling cascade + translator.

**NEXT (the actual experiment the user wants):**
1. **Backtest pitchGPT+cascade vs lookup+cascade vs baseline** on real PAs (does
   pitchGPT's batter-aware, sequence-aware pitch selection beat the count-only
   lookup? lookup already = 1.405 < 1.452 baseline). The MC rollout is slow, so
   sample. NOTE: the pitchGPT-path OPS looked hot (1.24) — its in-play mix differs
   from the lookup, so it likely needs its own outcome-map calibration check
   (rebuild the map on predicted-xwoba over PITCHGPT-sampled pitches, or confirm
   the existing map holds). Don't trust the level until calibrated + backtested.
2. **matchup cards**: `mcsim/matchup_card.py` → call g_compute hitter mode; run
   June 4; publish to demo (:8000/:5173 running). Switch DEFAULT_CKPT → small-v7.
3. **App A**: chain hitter-mode PAs through a base-out-inning game state machine.
4. (Future) retrain a PITCH-ONLY transformer (drop outcome heads → free capacity
   for π̂) — test pitch top-1/calibration; only after pitchGPT beats the lookup.

---

## ⭐⭐⭐ HITTER MODEL — CALIBRATED + BEATS BASELINE (2026-06-04)

**Per-PA backtest (the real "does it work" measure) now PASSES.** Cascade +
empirical-lookup pitch source scores **1.405 log-loss vs 1.452 league-average
baseline** on real held-out 2024H2 at-bats (lower=better). It was *worse* (1.460)
until the calibration fix below.

**KEY FIX — the xwOBA→outcome map must bin on PREDICTED xwOBA, not real.**
contact_quality hedges to the mean (pred std ~0.08), so a real-binned map never
fires its high bins → HR came out 5× too low (0.7% vs real 3.3%), singles too
high. Predicted-binning fixes it (HR drift → 3.2% vs 3.3%). Reproducible via
`python -m scripts.hitter.build_outcome_map`. **Re-run this whenever the
contact_quality node is retrained.**

**small-v7 capacity test (DONE, 3 epochs):** type_top1 0.485 (tiny-v7 0.478).
Compression diagnostic (same panel/pitcher, controlled): tiny-v7 5.43×/r0.29,
**small-v7 3.41×/r0.67**, **hitter cascade 2.62×/r0.90** (true-talent 1.37×/r0.94).
Capacity helps spread magnitude but NOT ordering; cascade wins both. Plan:
small-v7 = better π̂, cascade = μ̂. Calibrated ckpt at
`checkpoints_modal/small-fold0-v7/checkpoint_calibrated.pt`.

**contact_outcome multiclass head:** built + saved, but it slightly HURT OPS
spread (2.62→2.81×); xwoba pinned as default `outcome_mode`. Multiclass stays an
option (may help power calibration — untested vs the fixed map).

**OPEN — wire the cards + the pitch-source test:**
- `mcsim/matchup_card.py` still uses g_compute (transformer μ̂). To ship
  cascade-based June 4 cards: add an analytic-compose path (`compose_pa` +
  `build_empirical_pitch_provider`) + run `run_matchup_cards.py` (switch
  DEFAULT_CKPT to small-v7) + publish to the running demo (:8000/:5173).
- **Transformer-vs-empirical pitch-source backtest** (does pitchGPT's batter-aware
  pitch selection beat the count-only lookup): empirical side done (1.405).
  Transformer side blocked: pitchGPT conditions on count via pitch HISTORY, not a
  settable count factor — clean per-count π̂ needs synthetic pitch histories or MC
  rollout extraction (multi-hour). Honest caveat: the analytic engine is count-only
  state, so it mutes sequence regardless of pitch source — count+last-pitch state
  is the upgrade to let sequence show.

---

## ⭐⭐ HITTER MODEL — STEPS 1-4 DONE, PASSES THE GATE (2026-06-03 late)

**The dedicated hitter cascade works.** Compression diagnostic on held-out 2024H2
(12-batter panel, avg over 5 reference pitchers):
- **spread ratio 2.58×** (was **6.8×** for the transformer; target ~1×)
- **Pearson r 0.90** (was 0.66) — near-perfect hitter ordering
- **model_mean 0.76 == real_mean 0.76** (transformer regressed everyone to ~.70)
- robust across ref pitchers (2.44–2.78×, r 0.875–0.925)

**Built + tested (42 hitter tests green), all on real Statcast, no placeholders:**
- `hitter/train.py` — 4 XGBoost nodes trained on FULL 4.74M pitches →
  `checkpoints/hitter/`. Full-data AUC: swing 0.869, called_strike 0.985,
  whiff 0.787; contact_quality (xwOBA reg) RMSE 0.369 / r 0.216. **Honest
  held-out ECE** (not the in-sample ~0): swing 0.009, called_strike 0.004,
  whiff 0.007 — all well-calibrated. **macOS XGBoost segfault root-caused**:
  feed numpy not pandas (columnar adapter crash) + `n_jobs=1`/`OMP_NUM_THREADS=1`
  (libomp race). cat cols via int codes + `feature_types`.
- `hitter/model.py` — `HitterModel.predict_cascade` (round-trip tested).
- `hitter/compose.py` — analytic count-tree solve (absorbing Markov, closed form)
  + real `xwoba_outcome_map` (275K in-play balls) at `checkpoints/hitter/`.
- `hitter/eval.py` — `run_compression_diagnostic` + CLI (`python -m hitter.eval`).
- New CLAUDE.md hard rule **1a**: no placeholder/stub/boilerplate/fabricated
  constants in production paths; synthetic data only in tests/.

**Remaining gap is at the LOW end** (weak hitters .40–.51 real → .61–.70 model,
under-punished). The lever to push 2.58× → ~1× is **contact_quality** (monotonic
xwOBA↑middle constraint, or the {1B/2B/3B/HR/out} multiclass head).

**NEXT (handoff steps 5-6):**
5. Wire into `causal/g_computation.py` behind `outcome_model="head"|"hitter"`
   (cascade = μ̂; keep π̂ from PitchGPT). Re-check AIPW≈g-comp + negative
   control≈0 before using causal language on its outputs.
6. `mcsim/matchup_card.py` `compose="analytic"` option → re-run
   `run_matchup_cards.py` for an in-range date (≤2026-05-08).

**small-v7:** original ephemeral run was CANCELLED at epoch 1/3 (session ended).
RELAUNCHED detached as app **`ap-9gGQmUoG4mbk4DzXcdSWuf`** (full 3 epochs). Pull
when done: `modal volume get pitchgpt-data checkpoints/small-fold0-v7
./checkpoints_modal/` → calibrate → re-run compression diagnostic on it (capacity
test). NOTE: the hitter model already recovers most of the spread, so small-v7 is
now a comparison point, not the fix.

---

## ⭐ START HERE (current state — supersedes older sections below)

- **Active branch: `hitter-swing-model`** (17 ahead of `main`). SUPERSET of
  `mcsim-runner-mp`: contains ALL go-live work (multiprocessing, frontend MCSim
  tab, **as-of profile fallback**, NaN→null fix, SQLite threading fix,
  OPS/AVG/BB%/K% cells, progress logging) **plus** the hitter foundation. Keep
  working here; nothing is merged to `main` yet.
- **Active task: build the hitter/swing model** → see **"▶ NEW-SESSION HANDOFF"**
  at the very BOTTOM of this file (authoritative step-by-step). Design:
  `hitter/MODEL_DESIGN.md`, **APPROVED** — proceed with the §9 leans (xwOBA-on-
  contact regression target; **analytic count-tree composition**, not
  Monte-Carlo; fouls = count-constant in v0; continuous plate_x/z; 5 separate
  XGBoost boosters).
- **Done in `hitter/`:** `labels.py` ✅ + `features.py` ✅ (both tested; features
  incl. the leakage-safe batter-profile join). Remaining: `train.py` →
  `model.py` → `compose.py` → `eval.py` → wire into `causal/g_computation.py`
  (`outcome_model` flag) → matchup-card path.
- **Parallel:** `small-v7` on Modal (app `ap-55acWEObPHd29pPYMZibNh`, L4, ~10h) —
  capacity test for the same compression problem. Pull/calibrate/re-diagnose
  steps in the handoff.
- **Acceptance test:** compression diagnostic — real vs model OPS spread.
  Transformer = **6.8×** (real std 0.261 / model 0.038, r=0.66). Target ≈ **1×**.
- **App A (full-game sim)** is unblocked: `compose.py`'s per-PA engine is its core.
- Sections below ("GO-LIVE SPRINT", Steps 4–8) are HISTORICAL — App B backend is
  done. Trust the handoff for what's live.

---

### (historical) GO-LIVE SPRINT status (2026-06-03) — read this first

On branch **`mcsim-runner-mp`** (NOT merged yet), bundling the "run it for real" work:
- ✅ **Game-level multiprocessing** added to the runner (`--n-workers`; each worker = 1 game at torch threads=1; flat thread-scaling makes this win). Smoke: 2 games/2 workers ~2× concurrency. Committed + unit-tested.
- ✅ **Live-progress logging added** (`compute_matchup_card(progress_every=)` + runner `--progress-every`, default 25). The first n_paths=250 run was opaque (no per-game logging) and ran slow under 9-way contention; killed it and re-ran WITH logging.
- ✅ **PREDICTION RUN DONE for 2026-06-04**: `--n-workers 9 --n-paths 120 --rng-seed 1` → **9/9 cards, 338 cells each, in `data/mcsim.sqlite`** (78 min wall, fully visible via streamed progress). NOTE: n_paths=**120** (not 250) — re-run at higher quality later if desired. Verified via the `/mcsim` API: OPS populated + discriminating (median RV flat at −0.100, as expected). Many 06-04 games had no probable announced → `is_starter=False` (cosmetic).
- ✅ **MCSim frontend tab MVP** built (`frontend/src/MCSimTab.tsx` + tab in `App.tsx` + `/mcsim` client in `api.ts` + types). `tsc -b` clean; **data path verified** through the API. **Browser visual verification STILL PENDING** — `make demo`, open "Matchup cards" tab on date 2026-06-04.

**Next after the run:**
1. Verify frontend renders real 06-04 cards (dev server + screenshot).
2. **2026-06-05:** `python -m scripts.mcsim.fetch_actuals --date 2026-06-04` → actuals overlay appears in the tab.
3. **Calibration eval** (not started): pool real PAs → reliability diagram/ECE; calibration-check OPS magnitude. Needs actuals (06-05) or backfilled past games. The card predicts in NEUTRAL context but real PAs vary — calibration must context-match (re-run model per real PA) or restrict to neutral-context PAs.
4. Merge `mcsim-runner-mp` → main + push.

---

(historical) Step 4 done 2026-06-01; everything below predates the go-live sprint.

This is the handoff doc for a new Claude session (or a VS Code restart). Read it cold; the project state below is everything you need to keep going.

---

## TL;DR — where to resume

**MCSim App B v1 BACKEND IS COMPLETE.** Steps 4–7 are merged+pushed to `main`. **Step 8 (read API endpoints) is DONE on branch `mcsim-app-b-read-api`** (off `main`; NOT yet merged as of 2026-06-03). All five backend pieces exist: storage, synthetic-state builder, matchup-card computer, live MLB-API runner, post-game actuals fetcher, and read endpoints. **Next big piece: the MCSim frontend tab** (date carousel + card grid + result overlay) — separate scope, uses `frontend-system`. Next concrete step:

> **MCSim frontend tab.** Date carousel → per-game matchup-card grid (pitchers × hitters) → result overlay once a game finishes. Wire to the Step 8 endpoints (`/mcsim/predictions?date=...`, `/mcsim/predictions/{game_pk}`). Use `frontend-system` skill (Inter, 3 colors, color+glyph for pitch types). Headline cells on **mean RV or OPS**, not median RV.

> **Step 8 (done) — read API.** `inference/mcsim_api.py`: model-free `APIRouter` mounted on the main app. `GET /mcsim/predictions?date=` → per-game summaries (carousel); `GET /mcsim/predictions/{game_pk}?date=` → full card with each (pitcher,batter) cell stamped with the real PA(s) (`actual.pa_count`+`events`; cells that didn't happen → `null`; PAs outside the grid → `unmatched_event_count`). DB via `get_conn` dependency (override in tests). Tests: `test_mcsim_api.py` (6).

**Step 7 (done) — actuals layer.** `mcsim/mlb_actuals.py::get_game_actuals` parses the MLB live feed (`/api/v1.1/game/{pk}/feed/live`) → final score + winner + per-PA events (pitcher, batter, **start context: bases reconstructed from `runners[].movement.originBase`, outs from first pitch — NOT `matchup.splits.menOnBase`, which is a stat-split label and reports RISP for a leadoff hitter**, verified). `scripts/mcsim/fetch_actuals.py` is the CLI (per date, writes Final games via `storage.write_actual`; non-final skipped). Validated on game 777079 (Giants @ Jays 6–8, 75 PAs). Tests: `test_mcsim_mlb_actuals.py` (4) + `test_mcsim_fetch_actuals.py` (2).

**Key validation insight (drives the future eval step):** a single matchup cell CANNOT be validated — a hitter faces a pitcher only 1–4 times per game. Validate by POOLING thousands of real PAs into a reliability diagram/ECE (calibration is the project's primary metric), and lean on the already-built per-pitch calibration. The 1000-path sim is Monte Carlo to smooth the model's distribution, not the thing being graded.

See sections below for details.

---

## The user's immediate priority for the next session

Continuing App B v1 backend (~3 more focused days):

1. ~~**Step 4 — `mcsim/matchup_card.py`**~~ ✅ done (commit `67f0629`)
2. ~~**Step 5 — CLI runner** (`scripts/mcsim/run_matchup_cards.py`)~~ ✅ done
3. ~~**Step 6 — MLB Stats API client**~~ ✅ done — MERGED into Step 5 as `mcsim/mlb_api.py` (schedule + active-roster; lineups dropped in favour of all-vs-all roster grid)
4. ~~**Step 7 — Post-game actuals fetcher**~~ ✅ done — `mcsim/mlb_actuals.py` + `scripts/mcsim/fetch_actuals.py` (branch `mcsim-app-b-actuals`)
5. ~~**Step 8 — Read API endpoints**~~ ✅ done — `inference/mcsim_api.py` (`GET /mcsim/predictions?date=...` summaries + `GET /mcsim/predictions/{game_pk}` full card with per-cell actuals overlay), mounted on the main app.
6. **App B v1 backend is COMPLETE.** Next big piece: **MCSim frontend tab** (date carousel + per-game card grid + result overlay) — uses `frontend-system` skill. Headline the card on **mean RV or OPS, not median RV** (median is a weak discriminator — see below).
7. (perf, optional) Runner multiprocessing — needed for a full 15-game nightly batch at n_paths=250.
8. (eval) Pooled per-PA calibration: reliability diagram/ECE over real PAs vs the model's per-matchup probabilities (context-matched). The real App-B validation — single cells can't be validated (1–4 real PAs each). Also calibration-check the projected OPS magnitude.

After App B v1 lands: App A (daily score prediction) needs a multi-AB state machine. MCSim brainstorm doc has design notes; that's a separate substantial project.

---

## Current branch: `mcsim-app-b-matchup-card`

Six commits on this branch since branching from main:

| # | Commit | What | Tests |
|---|---|---|:---:|
| 1 | `8a0ce1b` | `docs/mcsim_appB_brainstorm.md` — design decisions D1–D8 with explicit "my lean" + user sign-off recorded in chat | — |
| 2 | `e12438e` | **D4 natural mode** in `causal/g_computation.py` — `intervention_type=None` samples from π̂(type \| h) instead of clamping | 5 |
| 3 | `4da1a1c` | **Storage layer** — `mcsim/storage.py` + SQLite schema (predictions, actuals, model_versions) + 12 round-trip tests | 12 |
| 4 | `d54bd81` | **Option C** — relax `intervention_position >= 1` to `>= 0` (first-pitch rollouts work — propensity at last context-token position) | 3 |
| 5 | `a94210f` | **Synthetic-AB builder** — `mcsim/state.py` with `ReferenceContext` dataclass + `build_synthetic_ab()` | 11 |
| 6 | `67f0629` | **Matchup-card computer** — `mcsim/matchup_card.py` with `compute_matchup_card` + `PitcherSpec`/`BatterSpec`; exposes `RolloutResult.intervention_type_propensity` for the trust gate | 6 |

**Full test suite: 347 pass.** Branch pushed to origin.

The brainstorm doc (`docs/mcsim_appB_brainstorm.md`) is the source of truth for all design decisions. Read it before writing more code.

---

## Open PRs (not on this branch)

| PR # | Branch | Status | Notes |
|---|---|---|---|
| **#1** | `v7-type-conditioned-heads` | **Merged** to main | v7 Part 1 — type-conditioned execution heads (ADR-013 D1) |
| **#4** | `demo-polish` | Open, awaiting review | v7 checkpoint switch + `make demo` + 6 API smoke tests + warm startup hook |
| **#5** | `recommender` | Open, awaiting review | `rank_pitch_types` + `POST /recommend` |

Two other branches with parked work (don't merge):
- `v7-cross-ab-context` — Part 2 of ADR-013. Decided not to ship; flag-default-off, no harm leaving the code on the branch.
- (Arsenal mask experiment) — code committed on `v7-cross-ab-context`; A/B showed it broke NLL (true labels in trailing-window-zero-mass classes). Not shipped.

---

## Step 4 done — `mcsim/matchup_card.py` (commit `67f0629`)

`compute_matchup_card(nuisance, *, game_pk, game_date, home_team, away_team,
home_pitchers, away_pitchers, home_lineup, away_lineup, …, n_paths=1000,
rng_seed=None, context=None) -> dict` is built and tested (6 tests). It loops
every (pitcher, batter) cell across both half-grids, builds a synthetic
reference-state AB, rolls it out in natural mode
(`intervention_position=0, intervention_type=None`), and packs the per-game
payload. Returns the dict; does NOT persist. Key choices that landed:

- RV median/p05/p95 come from the **per-path `run_value` array** (truncated
  paths are NaN and excluded) — not derived from mean/SE.
- Per-cell seed = `rng_seed + cell_index` (reproducible, not identically
  correlated).
- Trust flag reads the exposed `RolloutResult.intervention_type_propensity`
  (the exact π̂(type|h) the rollout sampled from) — no second forward pass.
- Language discipline: cells are **predictive rollouts, not causal
  estimates.** `PositivityGate` is borrowed only for its trust state; its
  causal rationale string is deliberately not surfaced.

Named numerical check from a real cell: `[HomeSP vs AwayBat1]` median RV
= −0.1000 (p05 −0.15, p95 +1.11); top-1 = out; modal type = FS at π̂=0.327;
trust = green; 0/50 truncated.

---

## Next step — Step 5+6 (MERGED): live MLB-API runner (BRAINSTORM IN PROGRESS)

**Decision (user, 2026-06-01):** pull live from the MLB Stats API *now* — this
merges the planned Step 6 (MLB client) into Step 5. And use an **all-vs-all
roster grid**, not the posted 9-batter lineup.

### Why all-vs-all (the lineup problem)
Empirically probed against `statsapi.mlb.com` this session:

| Run timing | Probable pitchers | Real lineups | Actuals |
|---|:---:|:---:|:---:|
| Past date (all 2025) | ✅ | ✅ (9/side) | ✅ |
| Today, day-of (Pre-Game/Warmup) | ✅ | ✅ (~hrs before 1st pitch) | — |
| Tomorrow / future | ✅ (13–14/15) | ❌ **0 games** | — |

So D5's "night-before with probable lineups" is **false** — MLB posts confirmed
lineups only ~2–4h pre-first-pitch. User's fix: skip lineups entirely, grid the
**full active roster** (available night-before via the roster endpoint):
`/api/v1/teams/{id}/roster?rosterType=active&date=…` → 26 players splitting
cleanly into ~13 pitchers + ~13 position players by `position.type=='Pitcher'`.
This is a *better* dugout doc — it helps build a lineup, not just react to one.

### Verified this session
- API reachable; `requests 2.33.1` + `pybaseball 2.2.7` installed; no existing MLB client code.
- statsapi player IDs are MLBAM = same namespace as Statcast `pitcher`/`batter` (no crosswalk). Skenes=694973.
- Roster endpoint returns id/name/position the night before. ✅
- **Per-cell benchmark (v7, CPU, 8 threads): 39.87s @ n_paths=1000, LINEAR in n_paths (~33–40 ms/path).** The old "sublinear" hunch was wrong.

### The compute catch
All-vs-all ≈ **338 cells/game** (2 × ~13 × ~13) vs the old 63. At n_paths=1000
that's **57.7h for 15 games** — NOT nightly-feasible single-process. Linear
knob → n_paths=200 ≈ 12.7h/15. Levers: lower n_paths, multiprocessing across
cores, or run a subset. **v1 plan: build correct + sequential, validate on ONE
real game, measure true wall-clock, THEN tune (per D8 "no cron until manual
end-to-end works").**

### Open flags to verify in implementation
- **Handedness**: `build_synthetic_ab` needs `pitcher_throws`/`batter_stand`;
  roster `person` object may need `hydrate=person` to expose pitchHand/batSide. VERIFY.
- **Missing profiles**: just-called-up players may have no trailing-window
  profile → per-cell try/except + skip-with-log (no crash, no fabrication).
- `compute_matchup_card` itself needs **no change** — it already loops
  pitchers × batters over arbitrary-length lists; runner just feeds full rosters.
- Flag the probable starter via `is_starter=True` inside the all-staff grid.

### Settled mechanics
- Reuse `NuisanceModels(V7_CKPT, device="cpu")`; default ckpt
  `checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt`.
- `model_ckpt_hash`: **no helper exists** — compute it (e.g. truncated sha256 of
  ckpt file); `register_model_version` once per run.
- Persist via `write_prediction(..., app="matchup_card")` (validated key, `storage.py:125`).
- CLI surface (planned): `--date`, `--game-pk` (repeatable filter), `--n-paths`,
  `--rng-seed`, `--ckpt`, `--db-path`, `--max-games`, `--dry-run`.
- New unit: `mcsim/mlb_api.py` (schedule + roster client, its own tests) so the
  HTTP concern is isolated and testable; runner orchestrates.

**STATUS: design + spec + plan APPROVED (2026-06-02). Ready to implement.**
- Spec: `docs/superpowers/specs/2026-06-02-mcsim-appB-step5-live-runner-design.md`
- Plan: `docs/superpowers/plans/2026-06-02-mcsim-appB-step5-live-runner.md` (5 TDD tasks)
- Extra verified finding: **missing profiles do NOT crash** — `ProfileCache.lookup`
  falls back to league-mean then zeros (`data/profile_cache_loader.py:149`). So
  the runner needs only a per-GAME try/except, not per-cell. Debut players get a
  league-mean profile (principled degrade, not fabrication).
- Switch hitters (`batSide=='S'`) resolve to `'L'` in v1 (vs the more common RHP);
  per-pitcher resolution deferred (would need `compute_matchup_card` to vary stand).

---

## Step 5+6 results (DONE — validated on a real game 2026-06-03)

Implemented via subagent-driven TDD (plan: `docs/superpowers/plans/2026-06-02-mcsim-appB-step5-live-runner.md`). Commits on branch:
- `mcsim/mlb_api.py` — `get_schedule` + `get_active_roster` (+ malformed-item hardening on both). Tests: `tests/test_mcsim_mlb_api.py` (7).
- `scripts/mcsim/run_matchup_cards.py` — `compute_ckpt_hash`, `run_matchup_cards`, `main` CLI. Tests: `tests/test_mcsim_run_matchup_cards.py` (3: hash, real-model E2E, fast skip-path).
- **Full suite: 356 passed.**

**Live validation:** `python -m scripts.mcsim.run_matchup_cards --date 2026-06-03 --game-pk 822727 --n-paths 40` → Marlins @ Nationals, **338 cells written to `data/mcsim.sqlite`**, ckpt hash `e67bd67d65a988dd`. Named numbers (row 0, Andrew Alvarez LHP, is_starter=True): vs Christopher Morel median RV −0.1000 (p05 −0.15, p95 +1.40), top1=out, modal FF π̂=0.305, trust=green, 0/40 trunc. All 338 cells green.

**Wall-clock (measured 794.81s / 338 cells @ n_paths=40 = 2.35 s/cell, linear):**

| n_paths | min/game | 15 games single-proc | 15 games 8-core mp |
|---|---|---|---|
| 40 | 13.2 | 3.3h | 0.4h |
| 100 | 33.1 | 8.3h | 1.0h |
| 250 (default) | 82.8 | **20.7h (infeasible)** | 2.6h |

→ For a real 15-game nightly batch at n_paths=250, **multiprocessing is required** (or drop n_paths). Functionally the runner is complete; this is a perf follow-up, not a correctness gap.

**Honest finding — median RV is a weak cross-cell discriminator.** It pinned to −0.1000 for every cell (when >50% of paths end in "out", the median path IS an out → median = out's run value). Batter signal lives in the tails (p05/p95) and the outcome distribution, NOT the median. **Frontend (Step 8+) should headline mean RV or P(reaches base), not median.**

---

## What's built up to this point on `main`

Substantially everything from the v6/v7 era. The model + causal layer + demo backend + frontend are all in place. v7 is the deployed model (PR #1 merged).

### Causal layer (`causal/`)
- `nuisance.py` — loads a calibrated checkpoint, exposes π̂ + μ̂.
- `g_computation.py` — Monte Carlo rollout. **Now supports natural mode (intervention_type=None) and intervention_position=0** (the D4 + Option C changes on this branch).
- `aipw.py` — doubly-robust estimator + influence-function SE.
- `crossfit.py` — K=5 cross-fit dispatcher.
- `positivity.py` — `PositivityGate(tau_refuse=0.01, tau_green=0.05)` (ADR-002).
- `sensitivity.py` — `e_value_for_continuous_effect()`.

### Recommender (`recommender/`, on the open PR #5)
- `rank.py::rank_pitch_types` — wraps g_compute in a ranking loop with positivity gating, tossup flag on overlapping 95% CIs.
- Pinned by 7 integration tests + 3 API tests.

### Inference (`inference/`)
- FastAPI app, 7 endpoints: `/health`, `/games`, `/at-bats`, `/ab-context`, `/query`, `/recommend` (on PR #5), `/pitchers`, `/pitcher/{id}/profile`.
- `AppState` lazy-loads NuisanceModels + val parquets on first request. Demo polish PR #4 adds a startup hook to warm it.

### Frontend (`frontend/`)
- 3 real tabs (~3000 LOC): CounterfactualExplorer, RolloutViewerTab, PitcherProfileTab.
- Design system locked per `frontend-system` skill (Inter, 3 colors, 4 type scales, color + glyph for pitch types).
- Wired to backend via Vite proxy at `/api/*` → `:8000`.

### Frontend tabs NOT YET built
- Recommender Tab — separate scope after PR #5 merges.
- Tipping page — separate substantial project (`tipping-analysis` skill).
- **MCSim Tab — App B's frontend (date carousel + per-game card + result overlay).** Will be a separate scope once App B's backend ships.

---

## How App B fits — pre-game daily batch architecture

User clarified (the framing matters; got it wrong twice before):

- **Score prediction** = simulate the WHOLE game from 0-0 top of 1st, ~10K times per game. Day-before-game prediction. Drives a daily prediction site.
- **Matchup report card** = static document a coach takes into the dugout. ~10K paths per (P, B) cell, ~63 cells per game.

App B is the second one and is being built first because:
1. Reuses single-AB `g_compute` directly — no multi-AB state machine needed.
2. Per-cell cost is ~41s at n_paths=1000 on CPU (measured in this session). 63 cells × 41s ≈ ~43 min per game; ~3h nightly for 4 games. CPU-feasible.

**No GPU needed at user's stated scales** — confirmed empirically by the `g_compute` benchmark in this session.

User requirement: **all predictions stored and re-readable via a date carousel after the game**, with the real result overlaid once the game finishes. Storage layer (commit 3) implements this — SQLite, three tables, COALESCE-on-update for two-pass actuals ingestion.

---

## Design decisions locked for App B (from brainstorm doc)

| # | Decision | Resolved as | Why |
|---|---|---|---|
| D1 | Storage backend | **SQLite** single file | Stdlib, queryable, ~MB-scale; perfect for this size |
| D2 | Reference context per cell | **Marginal** (0-0, no runners, 0 outs, mid-game) for v1 | Apples-to-apples comparable across cells |
| D3 | n_paths per cell | **1000** | Measured ~3h nightly for 4 games — comfortable budget |
| D4 | Natural rollout mode | **Add `intervention_type=None` to g_compute** | Done — commit 2 |
| D5 | Game-day cadence | **Once nightly** (evening before, probable lineups) | Day-of re-run is v2 |
| D6 | Grid scope | **Starter + bullpen (~6-8 arms) × starting 9 batters** (~63 cells) | Spec assumption |
| D7 | Calibration KPIs | **Three**: game-CI hit-rate, winner correctness, per-cell empirical-vs-predicted | All three surface different failure modes |
| D8 | Scheduler | **On-demand for dev, Modal cron for prod** | Don't ship a cron until end-to-end works manually |

Plus the matchup-cell mechanics question (was option A/B/C):

> **Option C — relax `g_compute`'s `intervention_position >= 1` guard.** Pitch 0's propensity lives at the last context-token position (NC-1); the causal mask blocks attention from there to pitch positions, so it works cleanly. Verified empirically: 2000-path natural-mode rollout's empirical type distribution matches π̂ within 3% — and π̂ is non-degenerate (modal first-pitch type > 20% probability).

---

## Storage schema (live, on disk)

`data/mcsim.sqlite` — gitignored. Three tables, `PRAGMA user_version = SCHEMA_VERSION = 1`.

- `predictions(id PK, game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json)` UNIQUE (game_pk, prediction_date, app)
- `actuals(game_pk PK, fetched_at, final_score_home, final_score_away, winner, matchup_events_json)` — COALESCE on UPDATE so two-pass ingest (line score → matchup events) doesn't null earlier fields
- `model_versions(ckpt_hash PK, trained_at, label, notes)`

Public API in `mcsim/storage.py`:
```
init_db                       open + create tables, idempotent
write_prediction              upsert (game_pk, date, app)
read_prediction               fetch one
read_predictions_for_date     date-carousel listing
write_actual                  two-pass COALESCE upsert
read_actual                   fetch one
register_model_version        idempotent provenance
lookup_model_version          fetch by ckpt_hash
```

---

## Critical conventions (read before writing model-interfacing code)

Documented in CLAUDE.md "Bug-prevention discipline." Headlines:

### The PAD-at-0 type vocab gotcha
The propensity TYPE head emits 8 logits where index 0 is PAD and indices 1..7 are PITCH_TYPES (FF..FS). `[:N_PITCH_TYPES]` = `[:7]` slices the WRONG columns. Use named constants from `data/dataset.py`:

```python
MODEL_PITCH_TYPES_START_IDX = 1   # FF lives here
MODEL_PITCH_TYPES_END_IDX = 8     # exclusive end
MODEL_TYPE_ID["FF"] = 1
```

Asymmetry: the RESULT head emits 7 logits with NO PAD column. Only TYPE has the off-by-one trap.

### MPS bug
`compute_losses` AB-outcome gather miscompiles on Apple silicon (Issue #2). CPU + CUDA fine; everything in this session's test infra forces CPU via `torch.backends.mps.is_available = lambda: False`.

### Datetime unit gotcha
Existing helper `_composite_sort_key` in `scripts/build_profile_cache.py` assumes `datetime64[ns]`. Pandas 3.x sometimes produces `[us]`. Tests pin the unit-invariance.

### Print named numerical outputs
Per CLAUDE.md's bug-prevention rule: **before claiming "smoke passed" on any model-interfacing code, print at least one named numerical output (e.g., `π̂(FF) = 0.47` on AB X).** The user explicitly enforces this — "what's π̂(FF) on the first AB?" is the canonical pushback.

---

## What's running in the background

Nothing should be running. If the user's restart was unclean and processes are stranded:

```sh
lsof -i :8000 -i :5173  # uvicorn + Vite — kill if present
ps aux | grep -E "build_profile_cache|build_matchup_cache|train_pitchgpt|run_matchup_cards"
```

---

## Files reference (load-bearing for App B work)

| File | Purpose |
|---|---|
| `docs/MCSim_brainstorm.md` | High-level MCSim design (both apps). Pre-game framing locked. |
| `docs/mcsim_appB_brainstorm.md` | **App B v1 spec.** Design decisions D1–D8. Implementation order. |
| `docs/recommender_brainstorm.md` | Recommender design (PR #5). Reused in App B's matchup-card thinking. |
| `mcsim/__init__.py` | Package docstring. |
| `mcsim/storage.py` | SQLite layer + 8 public functions. |
| `mcsim/state.py` | `ReferenceContext` + `build_synthetic_ab()`. |
| `mcsim/matchup_card.py` | `compute_matchup_card()` + `PitcherSpec`/`BatterSpec`. Step 4. |
| `causal/g_computation.py` | `g_compute(intervention_type=None, intervention_position=0)` both supported now; `RolloutResult.intervention_type_propensity` exposed for the trust gate. |
| `tests/test_g_compute_natural_mode.py` | 8 tests pinning D4 + Option C. |
| `tests/test_mcsim_storage.py` | 12 tests pinning the storage layer. |
| `tests/test_mcsim_state.py` | 11 tests pinning the state builder. |
| `tests/test_mcsim_matchup_card.py` | 6 tests pinning the card computer. |
| `model/pitchgpt_dataset.py` | `REQUIRED_AUG_COLS`, `PITCH_FACTOR_COLS_INT`, `CATEGORICAL_CTX_COLS` — the schema App B's state builder targets. |

---

## How to resume after the VS Code restart

1. **Open terminal in `/Users/sidthakur/Projects/PitchGPT`.**
2. **Confirm branch and clean tree:**
   ```sh
   git branch --show-current   # should print: mcsim-app-b-matchup-card
   git status --short          # should be empty
   ```
3. **Re-read this doc + `docs/mcsim_appB_brainstorm.md` § Implementation order.**
4. **Quick sanity that everything still works:**
   ```sh
   uv run pytest tests/test_mcsim_storage.py tests/test_mcsim_state.py tests/test_mcsim_matchup_card.py -q
   ```
   Should print: `29 passed`.
5. **Start step 5** — write `scripts/mcsim/run_matchup_cards.py` per the contract in "Next concrete step" above. Resolve the open design questions there first.
6. **Discipline:** before claiming anything works, print named numerical output (e.g., a real cell's median RV + π̂(modal type) from an end-to-end run that landed a SQLite row).
7. **When done with step 5**, commit + push + report to user. Don't push past step 5 without checking in.

---

## Key learnings from this session (so the next session doesn't repeat)

- **My ratings on novel ideas are not trustworthy.** Earlier in this session I rated an arsenal-mask experiment 8.5/10; tried it; NLL exploded because the trailing-window `has_pitch=0` flag is "absent from window," not "impossible." User pulled the recommendation privileges. Don't pitch model improvements; only ship what's measured.
- **The user wants you to think actual usage, not demo.** Pre-compute caches for curated demos are theatre; real workflows run nightly batches. SQLite + cron + actual results comparison is the right shape.
- **CPU is enough at the user's stated scales.** Don't reach for GPU prematurely. Measure first.
- **The user reads numbers.** "Smoke-tested" without showing named numerical output is unacceptable. Always print at least one named number per check.
- **Pre-game framing for MCSim.** Both apps are pre-game — score prediction simulates the whole game from 0-0, matchup card is a static doc. NOT live in-game.
- **The brainstorm-then-decisions-then-code pattern works.** Each major piece (recommender, App B) opened with a brainstorm doc, user signed off on decisions, then implementation followed. Don't skip the brainstorm step on big-enough work.
- **Pressure-test before claiming.** When something feels too good (n_paths scales sublinearly!) or too easy (cache will be quick to build!), verify with a measurement. Several wrong estimates in this session got caught only because the user asked for evidence.
- **PR scope discipline.** Each PR is one clean concern: PR #1 v7 Part 1, PR #4 demo polish, PR #5 recommender. App B will be one more. Don't bundle.

---

## End

If anything in this doc contradicts something in the actual codebase, the codebase is right and this doc is stale. Update this doc at the end of every session that materially advances the project state.

---

## small-v7 training — PREPARED (2026-06-03), needs a GPU to run

**Why:** tiny-v7 compresses hitter OPS ~6.8× on real data (real std 0.261 vs
model 0.038, Pearson r=0.66) — it regresses every hitter toward ~.700. Step 1
fix = more capacity (small: 6L/8H/d512, ~25M params vs tiny ~6M).

**Validated command** (smoke-tested end-to-end on CPU, exit 0):
```
PYTHONPATH=. python -m scripts.train_pitchgpt \
  --size small --fold 0 --type-conditioned-heads \
  --epochs 3 --batch-size 256 --run-name small-fold0-v7
```
- `--type-conditioned-heads` = the v7 (ADR-013) change; arsenal_per_pitch +
  propensity_situational are ON by default. (Optional, to fully match v7's EMD
  loss: `--zone-spatial-weight <w>` — needs the zone-centroids file; verify
  whether tiny-v7 used it before relying on it.)
- **Device reality:** this Mac's only GPU path is MPS, which has the AB-outcome
  gather miscompile (Issue #2) → training here would be WRONG (MPS) or far too
  slow (CPU, ~25M params on 7M pitches). **Full run needs CUDA/Modal.** The
  in-repo infra is `scripts/upload_to_modal.py` (upload only) — there is no Modal
  *training* entrypoint checked in; launching the GPU run is a manual/Modal step.

**After training:** `scripts.calibrate_pitchgpt` → `checkpoint_calibrated.pt`,
then re-run the compression diagnostic (real-vs-model OPS spread) to see if
capacity recovered the spread. If not → the dedicated hitter model
(`docs/Hitter_Swing_Model.md`).

## Hitter/swing model brainstorm — `docs/Hitter_Swing_Model.md`
Structural fix for the hitter-compression problem. Key calls: decompose into
swing→whiff→contact-quality nodes; compose a **tabular/tree** hitter model with
the existing pitch transformer inside `g_compute` (behind an `outcome_model`
flag); **a Transformer for the hitter model is most likely overkill** (batter
response to one pitch is tabular, not a long-sequence problem) and a tree is far
more interpretable (serves the interpretability project). Phased: small-v7 →
tabular hitter model → wire into rollout + A/B → (only if needed) sequence-aware.

---

## Hitter/swing model + small-v7 — IN FLIGHT (2026-06-03)

**small-v7 training:** LAUNCHED on Modal, detached (app `ap-55acWEObPHd29pPYMZibNh`,
`modal_app.py` train_remote, L4 GPU, `--size small --type-conditioned-heads
--epochs 3 --run-name small-fold0-v7`, ~10h). Checkpoints land on the
`pitchgpt-data` volume at `/data/checkpoints/small-fold0-v7/`. Pull when done:
`modal volume get pitchgpt-data checkpoints/small-fold0-v7 ./checkpoints_modal/`,
then `scripts.calibrate_pitchgpt`, then re-run the compression diagnostic.
Motivation: tiny-v7 compresses hitter OPS **~6.8× on real data** (real std
0.261 vs model 0.038, r=0.66) — capacity test.

**Hitter/swing model:** branch `hitter-swing-model`, package `hitter/` (design:
`docs/Hitter_Swing_Model.md`). Decision locked: **XGBoost** (spreadsheet-style,
not a Transformer — a batter's response to one pitch is short-range; feed
recent-pitch lag features instead of attention), composed with the pitch
sequence model in `g_compute` behind an `outcome_model` flag.
- ✅ `hitter/labels.py` — per-pitch swing/whiff/fair-contact + in-play outcome
  from `description`/`events`. Validated (swing 0.473, whiff-on-swing 0.248).
- TODO: `features.py` (pitch + count + **recent-pitch lags** + **batter profile
  via ProfileCache** + pitcher) → `train.py` (3 nodes) → `model.py` predict →
  `eval.py` (compression diagnostic, target spread-ratio ≈ 1× vs current 6.8×)
  → wire into `g_compute` → UI tabs.
- Baseline reality (repo's own leak-clean numbers): PitchGPT-small type top-1
  **0.478** > leak-clean LSTM **0.449**; the "0.691 LSTM" was LEAKY. No saved
  XGBoost number — building this produces it. (Transformer wins *pitch
  prediction*; XGBoost's case for the *hitter* model is capacity-focus +
  interpretability, NOT proven accuracy.)

**Also still running:** demo servers (uvicorn :8000 `bizkpwz6m`, vite :5173
`bilyl1h9s`) — the matchup-card tab. NOTE those cards are the n_paths=120
2026-06-04 run that PREDATES the as-of profile fix, so they're player-blind;
re-run a recent in-range date (≤2026-05-08) for meaningful cards.

---

# ▶ NEW-SESSION HANDOFF: build the hitter model + matchup cards (2026-06-03)

Context ran low mid-build. Everything needed to finish is below. Branch:
**`hitter-swing-model`**. Design is locked — read these 3 first:
- `hitter/MODEL_DESIGN.md` — the multi-stage cascade, features, **analytic
  count-tree composition** (no Monte-Carlo), open decisions (all have a "lean").
- `docs/Hitter_Swing_Model.md` — rationale + the ~6.8× compression that motivates it.
- This handoff.

## The goal in one line
A 3-model XGBoost cascade (swing → whiff → contact-quality) for the BATTER side,
composed with PitchGPT (pitch side) over the count tree, to fix the ~6.8×
hitter-OPS compression and power matchup cards / counterfactual / recommender —
and (later) full-game simulation (App A).

## What's DONE
- `hitter/labels.py` ✅ — per-pitch swing/whiff/fair-contact + in-play outcome
  from `description`/`events`. Tested (swing 0.473, whiff-on-swing 0.248).
- `hitter/features.py` ✅ — `build_base_features` (pitch+count+platoon+in_zone+
  within-AB lags) and `attach_batter_profile` (per-(batter,game) ProfileCache
  lookup, `as_of_fallback=True`, leakage-safe). Tested incl. real-data check that
  different batters get different profile vectors. `BASE_FEATURE_COLS` exported.
- `hitter/__init__.py`, `hitter/README.md`, `hitter/MODEL_DESIGN.md` ✅.

## Build steps (in order)
1. ✅ **DONE — `hitter/features.py`** (see above). Profile-join convention used
   (from `model/pitchgpt_dataset.py:244`): per AB `asof_date = first pitch
   game_date`, `asof_game_num = first["game_num"] if present else 1`,
   `ProfileCache(role="batter", fold_id=...).lookup(..., as_of_fallback=True)`.
   TODO when wiring training: also attach the **pitcher "stuff" profile** the same
   way (a `attach_pitcher_profile` mirror), and consider the finer 25-zone
   encoding for interpretability plots.
2. **`hitter/train.py`** — train the nodes (XGBoost), temporal split (train ≤2023,
   val 2024H1). Each node on its conditional population (swing=all; whiff=swings;
   fair=contact; contact-quality=balls-in-play). Monotonic constraints where
   sensible (chase↑ out-of-zone; xwOBA↑ middle). Per-node isotonic calibration.
   **S3 target = xwOBA-on-contact** (`estimated_woba_using_speedangle`) — NOT in
   `data/augmented/`; join from `data/raw/{year}/{date}.parquet`. Fallback for v0:
   discrete {1B/2B/3B/HR/out} from `events` (already in augmented). Save models to
   `checkpoints/hitter/`.
3. **`hitter/model.py`** — `HitterModel.predict(pitch_type, location, count,
   batter_feats, pitcher_feats, ctx) -> {swing, whiff, fair, xwoba/outcome}`.
   Loads the saved nodes; vectorized.
4. **`hitter/compose.py`** — the **analytic count-tree solve**. State = (balls,
   strikes[, last_pitch_type]). For each count, marginalize PitchGPT's
   pitch(type+zone) distribution × the cascade → transition probs over
   {ball, called/swing strike, foul (2-strike=no-op), in-play→out/1B/2B/3B/HR}.
   Build the absorbing-Markov transition matrix; solve for terminal
   distribution (walk, K, out, 1B, 2B, 3B, HR) in closed form → per-PA outcome →
   OPS/AVG/OBP/SLG/K%/BB%. (Keep a Monte-Carlo path as a cross-check.)
5. **`hitter/eval.py`** — THE acceptance test: the compression diagnostic
   (real OPS spread vs model OPS spread across a batter panel; transformer = 6.8×;
   target ≈ 1×). Plus per-node AUC/logloss/ECE, held-out hitters.
6. **Wire into `causal/g_computation.py`** behind `outcome_model="head" | "hitter"`
   — the cascade IS μ̂(y|do(pitch),h); keep π̂ from PitchGPT. AIPW/positivity/
   E-values still apply.
7. **Matchup cards** — `mcsim/matchup_card.py` gets an `outcome_model` /
   `compose="analytic"|"mc"` option → per cell, run the analytic composition →
   the existing payload (OPS/AVG/BB%/K% already shown in `frontend/src/MCSimTab.tsx`).
   Re-run `scripts/mcsim/run_matchup_cards.py` (with multiprocessing) for a recent
   IN-RANGE date (≤2026-05-08, so real profiles + actuals exist).

## small-v7 (capacity test, running in PARALLEL)
- Modal app `ap-55acWEObPHd29pPYMZibNh` (dashboard:
  https://modal.com/apps/siddhartha-thakur/main — find the `pitchgpt` app), L4,
  `--size small --type-conditioned-heads --epochs 3 --run-name small-fold0-v7`,
  ~10h. Check: `modal app list`. When done:
  `modal volume get pitchgpt-data checkpoints/small-fold0-v7 ./checkpoints_modal/`
  → `python -m scripts.calibrate_pitchgpt --ckpt .../checkpoint.pt` →
  re-run the compression diagnostic on it. If small-v7 recovers the hitter spread
  on its own, the dedicated hitter model may be less urgent; if not, the hitter
  model is the fix. (Either way the hitter model is the better long-term answer.)

## App A (full future-game simulation) — yes, this unblocks it
The per-PA outcome engine (steps 4–5) IS the core App A needs. A full game =
chain PAs through a GAME state machine: lineup cycling (1–9), base-out state
(use the RE24 / base-out tables in `data/run_value/`), inning/outs, score, both
bullpens. Backtest on past games (predictions vs real finals — `mcsim` storage +
`fetch_actuals` already exist) before predicting future games. App A is a
separate `mcsim` module on top of the same PitchGPT+hitter engine.

## The compression diagnostic (reuse verbatim as the gate)
Real OPS per batter from held-out `events`; model OPS via the engine; compare
spread (std) + Pearson r. Transformer baseline: real std 0.261 vs model 0.038
(6.8×), r=0.66. Target for the hitter model: ratio ≈ 1×.

## Misc state
- Demo servers may still be running (uvicorn :8000, vite :5173). The 06-04 cards
  in `data/mcsim.sqlite` are PLAYER-BLIND (predate the as-of fix) — re-run a
  ≤2026-05-08 date for meaningful cards.
- Unmerged branches: `mcsim-runner-mp` (mp + frontend + as-of fix + NaN fix + OPS
  + logging — NOT merged to main), `hitter-swing-model` (this work).
