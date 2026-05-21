# ImprovementPlan.md

A running log of improvements deferred from sprints. **Add to this; don't
remove.** When an item ships, mark it `done` and keep the trace — the
reasoning behind why we deferred something is often more useful than the fact
that it shipped.

Statuses: `open` · `in_progress` · `done` · `wontfix` (with reason).

Items are loosely ordered within each section by *(value × leverage) / cost*,
not by recency.

---

## Causal layer

### CAUSAL-01 · Fix Chinn constant in `causal/sensitivity.py`

- **Status:** open
- **What:** Project uses `chinn_constant = 1.81` (Chinn 2000, OR conversion).
  VanderWeele's `EValue` R package uses `0.91` for continuous outcomes via
  `evalues.MD`. The 1.81 inflates reported E-values by ~20% on the same
  effect (1.69 vs 1.42 for SMD=0.1).
- **Why it matters:** Over-stating robustness is the *wrong* direction of
  error for an epistemic-humility demo.
- **Fix:** change default to `0.91`; cross-check against a known
  `EValue::evalues.MD` output; add a unit test pinning the value.
- **Blocker for:** Tab 12 (Sprint 5) — must land before any E-value is shown
  in the UI.

### CAUSAL-02 · Improve μ̂ (outcome model) conditioning on intervention

- **Status:** open
- **What:** AIPW spot-check (Sprint 1) showed μ̂(a, H) varies only ~0.005
  across the 7 pitch types. The IPW correction picks up the slack — the
  doubly-robust estimator is consistent because π̂ is well-calibrated — but
  variance is wider than it could be.
- **Options:**
  - **A · Stronger result-head conditioning.** Architectural change so the
    AB-outcome head reads the intended action more loudly. Risky — could
    regress in-distribution NLL on the result head.
  - **B · Use g-computation rollouts as μ̂.** Replace the AB-outcome head's
    expectation with an N-path rollout per (a, H). ~100× more expensive
    per AIPW unit, but no architectural risk.
- **Recommended diagnostic before fix:** run AIPW on a 1000-AB slice; measure
  Δ(SE of τ̂) vs. the current implementation. If > 30% wider than
  theoretically needed, pursue Option B for population estimands.

### CAUSAL-03 · Per-(base, outs)-conditional RE24 run-value table

- **Status:** open
- **What:** `causal/g_computation.py` uses `DEFAULT_AB_RUN_VALUE`
  (unconditional means: K = −0.15, HR = +1.40, …). The proper version conditions on
  the AB's starting (base, outs) state via `data.run_value` (the RE24 table
  already lives there).
- **Fix:** route the (base, outs) → RE24-conditional run-value lookup into
  `g_compute()` and `compute_aipw_per_unit()`. The unconditional version
  stays as a fallback when state is missing.

### CAUSAL-04 · Cross-fit nuisance models (K=5, folds 1-4)

- **Status:** open (intentionally deferred per roadmap §5)
- **What:** Currently fold 0 only is trained; folds 1-4 needed for
  un-biased AIPW (ADR 006).
- **Cost:** ~5h × 4 = 20h Modal A100 + $$$.
- **Trigger:** unblock when Sprint 4 causal endpoints are stable and the
  architecture is validated end-to-end on single-fit AIPW. Launch command:
  ```bash
  for fold in 1 2 3 4; do
    modal run --detach modal_app.py::train_remote \
      --size tiny --fold-id $fold --epochs 3 \
      --run-name tiny-fold${fold}-v6 &
  done
  ```
  `--detach` is mandatory (roadmap §7 gotcha #1).

### CAUSAL-05 · Negative-control test suite

- **Status:** open
- **What:** Per CLAUDE.md and roadmap §B.5: pick a treatment that *shouldn't*
  have a causal effect (e.g., umpire_id on next-batter outcomes) and
  confirm AIPW centers on zero. Ship as `make negative-controls` in
  `eval/`.
- **Blocker for:** Sprint 4 wrap (acceptance criterion §6, "≥ 3 negative
  controls passing").

---

## Model architecture

### MODEL-01 · Dedicated swing-decision head

- **Status:** open (deferred per roadmap §8)
- **What:** Current marginal P(swing) from result head is 0.7245 OOD vs Ahn
  2026's ~0.78. Factoring swing decision out could close the gap.
- **Trigger:** revisit if/when the causal layer needs swing-decision
  factorization explicitly (e.g., for two-stage interventions: "do(swing=1)
  then do(contact_type=...)").

### MODEL-02 · Hidden-state extraction API

- **Status:** open (raised in Tab 1 brainstorm)
- **What:** Several PitchGPT-derived features need the trunk's final hidden
  state per AB (nearest-neighbor pitchers, hidden-state UMAP, embedding
  diagnostics). Currently buried inside `model.pitchgpt.PitchGPT.forward()`.
- **Fix:** add a `return_hidden_states=True` flag returning a clean
  (B, T, D) tensor; surface via `NuisanceModels.forward`.

### MODEL-04 · Location-resolution ablation (13-cell zone vs continuous)

- **Status:** open (raised in A.3 discussion)
- **What:** The model currently consumes pitch location as the coarse 13-cell
  `feature_zone` factor. Test whether feeding continuous `plate_x`/`plate_z`
  (per-batter-scaled) instead improves next-pitch-type prediction.
- **Pressure-test caveat:** location's signal for next-type is largely
  *mediated* by count and result, which are already factors. The marginal
  gain from finer location resolution is likely modest and could be eaten by
  added measurement noise. Treat as an ablation — train both, compare
  next-type NLL — not an assumed win.

### MODEL-05 · Cross-AB / times-through-order context

- **Status:** open (deferred per roadmap §8)
- **What:** The model is AB-scoped — it does not see what this pitcher threw
  this batter in earlier ABs, or how deep into the game it is (TTO). Pitchers
  measurably change pattern 3rd time through the order. This is the genuine
  feature gap (vs. precise location / per-pitch movement, which are largely
  redundant with existing factors).
- **Why deferred:** roadmap §8 — "would help raw accuracy but adds
  complexity, not on the critical path." The matchup cache
  (`MATCHUP_FEATURE_NAMES`, schema v1) is a partial step; full game-level
  context is the larger lift.

### MODEL-06 · Turn on the zone-head EMD spatial aux loss

- **Status:** open (found while debugging the Tab 3 zone heatmap)
- **What:** The zone head is the model's weakest (acc ~0.24 on 13 classes,
  NLL ~2.33 — roadmap §1 v6 table). Its optional earth-mover's-distance
  spatial-smoothness aux loss is currently off: `zone_spatial_weight = 0.0`
  in the v6 checkpoint config. The EMD term penalizes predictions that are
  spatially far from the truth (a near-miss cell costs less than a wild
  miss), which should sharpen the zone distribution.
- **Fix:** train with `zone_spatial_weight > 0`. Requires precomputed zone
  centroids — `scripts/train_pitchgpt.py:519-527` checks for them and errors
  out otherwise. Run the centroid script first.
- **Why it matters:** the Tab 3 predicted-location heatmap is only as good as
  the zone head. A sharper zone head makes that viz more informative.

### MODEL-03 · Auto-regressive (type, zone) joint sampling

- **Status:** open (called out in `causal/g_computation.py:474`)
- **What:** Currently type and zone are sampled conditionally independent
  given the hidden state at the intervention step. Joint sampling would
  factor type → zone (or vice versa) auto-regressively, matching the way
  pitchers actually decide.
- **Trigger:** revisit when joint (type, zone) interventions become a demo
  query in Sprint 5.

---

## Data pipeline

### DATA-01 · Demo cache for curated ABs

- **Status:** open (roadmap §5 decision point #3)
- **What:** Frontend is slow if every interaction requires live inference.
  Pre-compute ~100 well-known (pitcher, batter, AB) tuples + 5000-path
  rollouts; serve from `inference/cache/curated.json`.
- **Trigger:** before any public-facing demo launch.

### DATA-02 · Modal volume sync target

- **Status:** standing / recurring
- **What:** Roadmap §7 gotcha #4 — `python scripts/upload_to_modal.py --only
  profiles preprocess_artifacts --force` after every profile-cache change.
- **Fix:** wire into a `make profile-rebuild` target so it's hard to forget.

### DATA-03 · Re-publish v5 profile-cache backups when retired

- **Status:** open (cleanup)
- **What:** `data/profiles_v5_backup/` and
  `data/preprocess_artifacts/v1/profile_standardization_v5_backup.npz` are
  retained for v5 baseline comparison. When the v5 baseline retires, delete
  them or move to cold storage.

---

## Frontend / Demo

### FRONT-01 · PitchGPT-derived signals on Tab 1 (parent)

Tab 1 today shows the **profile cache** (hand-crafted features). PitchGPT
itself produces signals the cache doesn't carry. Sub-items below; each is
independently shippable. Loosely grouped by cost.

**Tier A — cheap, read existing forward-pass outputs:**

#### FRONT-01a · Functional arsenal size

- **Status:** open
- **What:** Count of pitch types that reach > 5% in *any* (count, stand)
  cell. Distinguishes "nominal 5-pitch arsenal" from "really uses 3."
- **Source:** derived from `arsenal_by_count` + `arsenal_by_stand` already
  exposed by `/pitcher/{id}/profile`. No new model calls.

#### FRONT-01b · Latent arsenal

- **Status:** open
- **What:** Pitch types where `has_pitch = False` but the model's average
  π̂ (across a sweep of his ABs) still reaches 3-5%. Surfaces the model's
  "he could throw this even though he doesn't" intuition.
- **Source:** val-set forward pass aggregated per pitcher.

#### FRONT-01c · First-pitch tell

- **Status:** open
- **What:** Model's top-3 picks on 0-0 counts vs LHB and vs RHB. The
  pre-game prep question.
- **Source:** average π̂ across his (count=0-0, stand=L|R) ABs in val.

#### FRONT-01d · Behind-in-count tell

- **Status:** open
- **What:** Same as 01c but for 2-0 and 3-1. The counts where a tip hurts
  most.

#### FRONT-01e · Runner-on-base shift

- **Status:** open
- **What:** π̂ mix with runners on vs empty bases. Surfaces strategic
  changes (e.g., more breaking balls with a runner on first to encourage
  the double-play GB).

**Tier B — medium, val-set sweeps + aggregation:**

#### FRONT-01f · Predictability heatmap per (count × stand)

- **Status:** open
- **What:** Entropy of model's π̂ in each (count, batter-stand) cell. Low
  entropy = predictable. This is the tipping signal surfaced as a static
  descriptor (NOT the full T_start metric — see [[TIP-02]]).
- **Source:** val-set sweep across this pitcher's ABs, bucket by cell,
  compute Shannon entropy of average π̂.

#### FRONT-01g · Putaway pitch by handedness

- **Status:** open
- **What:** In 2-strike counts vs each stand, the pitch type with highest
  result-head P(swing_strike | type) — his actual "out pitch."
- **Source:** sweep result head on his 2-strike pitches, condition on type.

#### FRONT-01h · Per-pitcher calibration card

- **Status:** open
- **What:** On his val pitches: accuracy@1, NLL, ECE per head. "Model is
  well-calibrated on this pitcher" vs "model is confused" — earns trust
  before showing predictions.
- **Source:** val-set sweep, standard calibration metrics from `eval/`.

**Tier C — needs prior infrastructure:**

#### FRONT-01i · Nearest-neighbor pitchers

- **Status:** blocked on [[MODEL-02]]
- **What:** Use the trunk's hidden state averaged across this pitcher's
  ABs as an embedding; find 5 nearest-neighbor pitchers in that space.
  "Looks most like Yu Darvish, Tyler Glasnow, …" A learned similarity, not
  a hand-coded one.
- **Source:** requires hidden-state extraction API ([[MODEL-02]]) and a
  one-time embedding sweep across all qualifying pitchers.

### FRONT-05 · "Reason decomposition" panel for surprising pitches

- **Status:** open
- **What:** When the Single AB Rollout Viewer (Tab 3) flags a high-surprisal
  pitch — one the pitcher threw against the model's expectation — give the
  user a panel that helps them reason about *why*, without claiming a cause.
  Show side by side: how well the model knows this pitcher (calibration),
  how common the situation is (support / positivity), and what observable
  context was unusual (runner on, 2 outs, leverage, catcher). The honest
  framing: a surprise is evidence an unmeasured factor was in play — it is
  not a reason the model can name (CLAUDE.md hard rule #5).
- **Depends on:** per-pitcher calibration card ([[FRONT-01h]]).
- **Why deferred:** needs the calibration sweep infrastructure to exist
  first; the surprisal *flag* ships in Tab 3 now, the decomposition panel
  comes after.

### FRONT-02 · About page

- **Status:** open
- **What:** The `frontend-system` skill declares this page "not optional."
  It must state identification assumptions, explain the trust gauge, list
  known limitations, and link to ADRs.
- **Blocker for:** any public-facing demo launch.

### FRONT-03 · Mobile responsiveness audit

- **Status:** open
- **What:** Acceptance criterion §6 — "mobile-responsive." Current tabs
  assume desktop layouts (e.g., 7×12 arsenal-by-count table).

### FRONT-04 · Shared primitives extraction

- **Status:** open (low-priority hygiene)
- **What:** `App.tsx` and `PitcherProfileTab.tsx` both define their own
  `Section`, `SectionLabel`, `Caption`, `PitchBadge`/`PitchGlyph`. As more
  tabs land, factor into `frontend/src/shared/` to keep the design language
  consistent and avoid drift.

---

## Evaluation / methods

### EVAL-01 · Cross-fit-aware reliability diagrams

- **Status:** open
- **What:** Calibration metrics are currently computed on the single-fold
  val set. Once cross-fit lands ([[CAUSAL-04]]), bootstrap reliability
  diagrams across the 5 fold models for a tighter measurement.
- **Blocker for:** Tab 5 (Sprint 3 differentiator).

### EVAL-02 · Per-pitcher held-out generalization cohort

- **Status:** open
- **What:** Per roadmap, hold out a cohort of pitchers (not pitches) from
  training; evaluate model on never-seen pitchers. Tests whether the model
  generalizes the pitcher-profile conditioning vs. memorizes the
  fold-aware cache.

### EVAL-03 · Natural-experiment matching validation

- **Status:** open
- **What:** Per `causal-layer` skill validation hierarchy item #4: find
  pairs of ABs with near-identical π̂ but different observed pitches;
  the difference in their outcomes is a quasi-randomized estimate; compare
  to AIPW prediction. Cheap to wire, strong validation signal.

---

## Tipping detector (Sprint 6+)

### TIP-01 · Batter-observable variant of PitchGPT

- **Status:** open
- **What:** Per `tipping-analysis` skill: retrain a tiny on batter-observable
  features only (no catcher signs, no internal sequencing). The gap between
  this model's π̂ and the full model's π̂ is the "tip."
- **Cost:** ~5h Modal per fold.

### TIP-02 · T_start metric implementation

- **Status:** open
- **What:** First start in a series where pitch type becomes predictable
  from batter-observables (per ADR 005). Requires [[TIP-01]].

### TIP-03 · Four validation runs from `tipping-analysis` skill

- **Status:** open
- **What:** Subsequent-start performance, time-through-order amplification,
  batter behavior shifts, case studies (Yu Darvish 2017).

---

## Bug-class debt

### BUG-01 · Stale comment in `causal/g_computation.py:75-78`

- **Status:** open (cosmetic)
- **What:** Comment says "Model output indices: 0..N_PITCH_TYPES-1 (=0..6)
  for type" but the actual TYPE HEAD outputs 8 logits (PAD at idx 0,
  pitch types at 1..7). The code uses the named constants correctly; only
  the comment is wrong. Easy fix; flagged in Sprint 1 audit.

### BUG-02 · Stale comment in `data/profile_cache.py:100`

- **Status:** open (cosmetic)
- **What:** Comment says "7 × 25 = 175 in-zone heatmap entries" — actual
  is 7 × 9 = 63 since the v5 14-zone migration. Comment didn't get updated.
