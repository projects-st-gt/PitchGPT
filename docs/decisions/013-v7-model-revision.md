# ADR 013 — v7 Model Revision: Type-Conditioned Execution Heads (+ Cross-AB Context, Zone EMD Loss)

**Status:** Proposed (draft — pending review)
**Date:** 2026-05-20
**Consolidates:** `MODEL-03`, `MODEL-05`, `MODEL-06` from `ImprovementPlan.md`
**Related:** ADR 001 (treatment granularity = pitch type), ADR 007 (nuisance decoupling), ADR 011 (factor embeddings)

## The question, in plain English

PitchGPT predicts each pitch as a set of separate heads off the shared trunk —
type, zone, velo, spin_rate, spin_axis — all produced **in parallel** and
**conditionally independent given the trunk hidden state**. None of the
non-type factors is conditioned on the type.

For a single scored forward pass this is just a modeling approximation. But the
causal layer **samples** these heads to build pitches during g-computation
rollouts — and sampling independent factors produces **incoherent pitches**: a
curveball clocked at 96 mph, a slider with fastball spin. When the demo asks
`do(type = SL)`, the rolled-out "slider" gets fastball-like velo, spin, and
location, because the zone/velo/spin heads were never told the type changed.

Should the execution factors (zone, velo, spin) be conditioned on the type?

Two further changes are bundled into the same retrain because they share its
cost and none is worth a retrain alone:

- **Cross-AB context** — the model is AB-scoped; it cannot see what this
  pitcher threw this batter earlier in the game, or how deep into the order it
  is (times-through-order).
- **Zone EMD aux loss** — the zone head is the weakest head (acc ~0.24); its
  optional earth-mover's-distance spatial loss is currently off.

## Why this matters

- **Causal validity, not cosmetics.** `g_computation` clamps `do(type = a)` and
  then samples zone/velo/spin from heads that never saw `a`. Every pitch in the
  rollout — not just the intervention pitch — is built this way, so the
  run-value estimate integrates over incoherent pitches. This is a defect in
  the *estimand*, not the UI. (See the conditional-independence note in
  `causal/g_computation.py`.)
- **The outcome side is already fine.** The result head and AB-outcome head
  read the full `intended_actions` vector (ADR 007), so they evaluate whatever
  action they are handed. The defect is purely in **generating a coherent
  action** to hand them.
- **Type is already the intervention point** (ADR 001 — treatment granularity
  is pitch type). Making type the first factor and conditioning execution on it
  means `do(type = SL)` *by construction* re-samples a coherent slider —
  coherent velo, spin, and location.
- The zone head is the model's weakest; conditioning it on type gives it the
  single most informative covariate it currently lacks.

## Decision 1 — Type-conditioned execution heads (the within-pitch factorization)

Factor each pitch as:

```
P(pitch | h) = P(type | h) · P(zone, velo, spin_rate, spin_axis | type, h)
```

- The execution heads (zone, velo, spin_rate, spin_axis) take a **concrete
  type embedding** as an additional input, alongside the trunk hidden state.
- The execution heads remain **mutually independent given (type, h)** — *not*
  fully autoregressive (type→velo→zone→spin). Rationale: the dominant
  cross-factor correlation (a 96 mph + low-away + tight-spin pitch all
  co-occurring) is *mediated by type*; once type is conditioned on, the
  residual pairwise correlation is small. Full within-pitch autoregression is a
  later option (see Alternatives).
- **Training:** teacher-forced on the **true** type. Every pitch has a
  ground-truth type — there is no "top-k" question at training time.
- **Inference, marginal** ("where does the next pitch go", the predicted-pane
  heatmap): marginalize over all 7 types —
  `P(zone | h) = Σ_type P(zone | type, h) · π̂(type | h)`.
  This is exact, and cheap: the expensive trunk pass runs once; the small
  execution heads are then evaluated 7× with 7 type embeddings. **Not** top-1
  (lossy when the model is uncertain) and **not** top-3 (an arbitrary
  halfway house).
- **Inference, rollout / causal:** feed the **concrete** sampled or intervened
  type. `do(type = a)` → execution heads produce `P(· | a, h)` — coherent.
- The **type head is structurally untouched** — same inputs, same place in the
  graph. The headline next-pitch-type top-1 accuracy therefore cannot regress
  by construction (modulo shared-trunk training dynamics, which are measured).
- **Convention discipline.** Wiring the type factor into the execution heads
  crosses the PAD / 8-class type convention (`data/dataset.py` —
  `MODEL_TYPE_ID`, `MODEL_PITCH_TYPES_START_IDX/END_IDX`). Per the project's
  bug-prevention conventions, the implementation must use the named constants and
  print named per-class checks; this is exactly where a convention bug hides.
- **Back-compat:** gated behind a config flag (e.g. `type_conditioned_heads`),
  default `False`, so pre-v7 checkpoints reload unchanged.

## Decision 2 — Cross-AB context

- Feed the **matchup cache** (`MATCHUP_FEATURE_NAMES`, schema v1 — pitcher×batter
  history) as an additional profile vector, concatenated alongside the existing
  pitcher and batter profile vectors.
- Add an explicit **times-through-order** scalar (how many times this pitcher
  has faced this batter this game).
- This is deliberately the *light* path — a feature concat, not a restructuring
  of the model's sequence into game-level context.
- The matchup cache must obey the same trailing-window, `before_asof`
  no-leakage rule as the pitcher/batter caches (statcast-pipeline skill).
- Expected accuracy-positive (roadmap §8).

## Decision 3 — Zone EMD aux loss on

- Train with `zone_spatial_weight > 0` (the earth-mover's-distance auxiliary
  loss on the zone head — `scripts/train_pitchgpt.py`).
- Requires precomputed zone centroids; run the centroid script first
  (`train_pitchgpt.py` errors out clearly if they are absent).
- Near-free to include since we are retraining regardless; penalizes spatially
  wild misses, should sharpen the weak zone head.

## Retrain & eval plan — separable attribution

The three changes ship together as **v7**, but are implemented and evaluated as
**separable steps**, so an accuracy movement can be attributed to a cause:

1. **Factorization only** → retrain `tiny` fold 0 → `make eval` vs the v6
   baseline. Gate: headline next-type accuracy within noise of v6; zone/velo/spin
   head NLL + calibration improve or hold; a rollout-coherence spot check
   (sample 1000 pitches under `do(type=CU)`, confirm velo/spin distributions
   shift to curveball-like).
2. **+ Cross-AB** → retrain → `make eval`. Gate: accuracy neutral-or-positive.
3. **+ EMD loss** → retrain → `make eval`. Gate: zone head improves; nothing
   else regresses.

Only after all three are measured does v7 become the default. Folds 1–4
(`CAUSAL-04`) are trained afterward, once the architecture is locked.

## Causal-layer impact

- **`g_computation`**: `do(type = a)` now feeds `a` into the execution heads, so
  the counterfactual pitch — and every sampled pitch in the rollout — is
  coherent. This is the core fix.
- **`nuisance.py`**: the forward interface must expose the type-conditional
  execution outputs (or the marginalization helper). `ForwardOut` gains a
  conditional path; the existing marginal path becomes the Σ over types.
- **AIPW / positivity / cross-fitting**: unaffected in form. π̂(type | h) — the
  propensity — is unchanged (the type head is untouched), so the positivity
  gate and treatment definition are stable.
- **Expect a measurable shift in causal estimates.** Coherent rollouts are not
  guaranteed to move the numbers the same direction; this ADR mandates an
  **ablation** — run the population AIPW / a sample of g-computation queries on
  the v6 (incoherent) vs v7 (coherent) rollout and report the delta. If the
  shift is large, that itself is a finding about how much the old estimates
  were biased.

## Alternatives considered

- **Full within-pitch autoregression** (type→velo→zone→spin, each conditioned
  on all prior factors). More expressive, but more complex and slower to
  sample; the cross-factor correlation it captures beyond "conditioned on type"
  is expected to be small. Deferred — revisit if the v7 ablation shows residual
  incoherence.
- **Mixture-input** (feed the soft type distribution as an expected embedding,
  one forward pass). Rejected for the marginal heatmap: `P(zone | E[type_emb])`
  ≠ `E[P(zone | type_emb)]`; the exact Σ over 7 types is cheap, so there is no
  reason to approximate.
- **A separate location model.** Rejected for the same reasons ADR 007 rejected
  decoupling π̂ and μ̂ — the shared trunk is the point.
- **Bundling the swing-decision head (`MODEL-01`).** Rejected — a separate
  architectural change; bundling multiplies risk and destroys attribution.

## Risks

- **Exposure bias.** Execution heads train on the *true* type (teacher-forced)
  but at rollout receive a *sampled* type. A wrong sampled type gives the heads
  wrong conditioning. This does not affect single-step (teacher-forced) eval; it
  is a known autoregressive tradeoff and the rollout was already imperfect.
- **Convention bugs** wiring the type factor into the heads (see Decision 1).
- **Compute.** A full retrain per attribution step; eventually ×5 folds.
- **Matchup-cache leakage** — the `before_asof` rule must hold; a within-game
  leak here would be subtle.

## Consequences

- New v7 checkpoints; v6 retained as the comparison baseline.
- Skill updates: `pitchgpt-model` (the factorization + head structure),
  `causal-layer` (rollout coherence; the v6-vs-v7 ablation), `statcast-pipeline`
  (matchup cache usage + leakage rule).
- `ImprovementPlan.md`: `MODEL-03`, `MODEL-05`, `MODEL-06` are consolidated here.
- Unlocks a UI feature: per-type zone maps for the top-3 most likely types
  (the `P(zone | type, h)` the factorization now exposes).

## Open questions for the implementation plan

- The exact head architecture — how the type embedding enters each execution
  head (concat to the trunk hidden state, FiLM-style modulation à la ADR 012, or
  an added type-embedding term).
- Whether velo/spin need a pairwise term or stay mutually independent given
  type (settle empirically in the Step-1 ablation).
- The times-through-order feature encoding (raw count vs bucketed).
