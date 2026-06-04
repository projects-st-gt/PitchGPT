# Hitter / Swing Model — Modeling Design (brainstorm)

**Status:** brainstorm, 2026-06-03. Detailed modeling spec for the `hitter/`
package. High-level rationale + integration: `docs/Hitter_Swing_Model.md`.
Voice: baseball analyst who knows ML.

---

## 1. What we predict — the plate appearance as a cascade

The atomic unit is a **pitch**. Given a thrown pitch `a_t` and the state, the
batter's response is a **decision cascade** that also drives the count state
machine. Every node is a clean, supervised target from Statcast:

```
pitch a_t thrown in count (b,s)
│
├─ S1  SWING? ──no(take)──► S1b  in-zone & called? ──► CALLED STRIKE (s+1)
│                                                   └─► BALL (b+1; b=4 → WALK*)
│
└─ yes ─► S2  WHIFF? ──yes──► SWINGING STRIKE (s+1)
          │
          └─ no(contact) ─► S2b  FAIR? ──no──► FOUL (s+1 if s<2, else no-op)
                            │
                            └─ yes(in play) ─► S3  CONTACT QUALITY
                                               └─► xwOBA → {out, 1B, 2B, 3B, HR}*  (terminal)
```
`*` = terminal (PA ends): walk, HBP, strikeout (s=3), or ball in play.

This is the multi-stage structure: **first whether they swing; if so whiff vs
contact; if contact, fair vs foul; if fair, how well struck.** Each arrow is a
probability the model produces; the count transitions are deterministic given
the arrow.

---

## 2. Why a cascade (not one flat 7-way head)

Baseball + ML reasons:

1. **Each node isolates a distinct *skill*** — plate discipline (S1), contact
   ability (S2), power/hit-tool (S3) — and each is driven by *different* batter
   features. A flat head has to learn all three entangled; the cascade lets each
   model specialize → **better batter discrimination** (the whole point; the
   monolithic model compressed hitters ~7×).
2. **Right conditional population per node.** The whiff model trains *only on
   swings*; the contact-quality model *only on balls in play*. No dilution by
   irrelevant rows (e.g., learning whiff on pitches nobody swung at).
3. **It speaks the count state machine.** The rollout/recommender need per-pitch
   *transitions* (ball/strike/foul/in-play), not just a PA outcome. The cascade
   produces exactly those.
4. **Interpretable** — separate chase maps, whiff maps, power maps. Directly
   serves the interpretability project (probe each skill independently).
5. **Calibration is checkable per node** (the project's primary metric).

---

## 3. The key ML+baseball insight: compose ANALYTICALLY over the count tree

The matchup card's noise problem came from Monte-Carlo rolling out 120–1000
paths per cell. We don't need to. The count is a tiny **absorbing Markov chain**
(12 ball–strike states + terminals). With:

- pitch-selection probs from the **pitch model** π̂(pitch | count, …), and
- response probs from the **cascade** (S1–S3) for each candidate pitch,

we get, per count, a transition distribution over {ball, strike, foul→same,
in-play-terminal, walk, K}. Marginalize pitch choice, build the 12×12 (+absorb)
transition matrix, and **solve for the terminal distribution in closed form**
(one small linear solve / fundamental matrix). Output: the exact per-PA outcome
distribution (and xwOBA/OPS) **with zero Monte-Carlo noise.**

This is a big win: it kills the noise that forced huge `n_paths`, makes a full
matchup card *instant*, and gives smooth, differentiable cell values. (MC
rollout remains available for sanity-checking and for queries that need full
path distributions.)

---

## 4. Per-node spec

| node | target | trained on | model |
|---|---|---|---|
| **S1 swing** | binary swing/take | all pitches | XGBoost (binary) |
| **S1b called-strike** | binary called-strike \| take | takes | XGBoost (binary) |
| **S2 whiff** | binary whiff \| swing | swings | XGBoost (binary) |
| **S2b fair** | binary fair \| contact | contact (non-whiff swings) | XGBoost (binary) |
| **S3 contact quality** | **xwOBA-on-contact (regression)** + optional {1B/2B/3B/HR/out} multiclass | balls in play | XGBoost (reg + multiclass) |

**S3 target = xwOBA-on-contact, not the actual outcome.** Per ADR-004 /
`statcast-pipeline`: we're scoring the *pitcher's decision*, which controls the
launch parameters of contact, not whether the LF was shifted into the gap.
xwOBA-on-contact (`estimated_woba_using_speedangle`) is the controllable,
lower-variance target; map it to the {1B,2B,3B,HR,out} mix via the per-season
in-play slope already in the run-value layer. (Needs the raw field — it's in
`data/raw/`, dropped from `data/augmented/`; join it for S3.)

Feature emphasis per node (what actually drives each, baseball-wise):
- **S1 swing:** count (king — 3-0 take vs 0-2 expand), location vs the zone,
  **batter chase/zone-swing profile**, pitch type, recent-pitch lag (sitting on
  a pitch). Platoon.
- **S1b called-strike:** location in/out of zone, **count-dependent zone
  expansion**, **umpire**, **catcher framing** (catcher id). Minimal batter signal.
- **S2 whiff:** pitch **stuff** (velo, movement/spin-axis, location — chase-and-
  miss), **batter whiff/contact profile**, count (2-strike), pitch type
  (4-seam up, sweepers miss most).
- **S2b fair:** batter contact quality, pitch location (jam → foul). Lowest
  priority — could simplify (constant foul rate by count) in v0 and refine later.
- **S3 contact quality:** **batter power/hit-tool** (exit velo, hard-hit%,
  launch-angle tendencies), pitch **location×type** (middle-middle → barreled),
  **park/altitude/temp**. Where SLG/power discrimination lives.

---

## 5. Feature catalog

**Common (every node):**
- Pitch: type (7), location (plate_x, plate_z; + in/out-of-zone flag), velo
  (release_speed and/or type-relative velo bin), movement (spin_axis bucket; pfx
  if available).
- Count: balls, strikes (the 12 states) + pitch_number-in-AB.
- Platoon: stand × p_throws (same/opposite).
- **Recent-pitch lags (short-range sequencing):** prev pitch type / zone / velo /
  result (lag-1, opt lag-2); running per-type count seen this AB; "same type as
  last?" / eye-level-change flags. *(This is the sequencing memory — fed
  directly, no attention; see `docs/Hitter_Swing_Model.md §4a`.)*
- **Batter profile** (the discrimination source): the ProfileCache vector —
  zone swing%/whiff%/xBA grids, chase rate by type, exit velo mean & 90th pct,
  K%/BB%/hard-contact. As-of, leakage-safe (reuse the new as-of fallback).
- **Pitcher "stuff" profile:** arsenal, velo/movement quality (so 98 ≠ 91).
- Context: outs, runners, leverage (minor), score (minor).
- Node-specific: **umpire id** (S1b), **catcher id** (S1b framing),
  **park/temp/roof** (S3).

All of these already exist in the pipeline (`statcast-pipeline`); the lift is
assembling per-pitch rows + the profile join, not new extraction.

---

## 6. ML refinements that bake in baseball priors (and aid interpretability)

- **Monotonic constraints (XGBoost supports them):** chase ↑ as the pitch moves
  *out* of the zone; whiff ↑ with velo / movement; xwOBA ↑ toward the middle.
  Enforcing monotonicity injects real baseball priors, improves generalization on
  thin slices, and makes partial-dependence plots clean for the interp project.
- **Per-node probability calibration** (isotonic/Platt) — XGBoost margins aren't
  calibrated out of the box; calibrate so the cascade product is a *calibrated*
  per-pitch outcome distribution (ECE is the project's primary metric).
- **SHAP per node** for the interpretability deliverable: "which batter-profile
  features drive chase on low-away sliders," etc.

---

## 7. Composition with the pitch sequence model

Inside `causal/g_computation.py`, behind `outcome_model="head" | "hitter"`:
- Keep the **pitch model** for π̂(pitch | count, history) — its strength.
- For each pitch, the **cascade** gives the batter response → count transition.
- Compose **analytically** over the count tree (§3) for the per-PA outcome, or
  MC-rollout for full path distributions.
- The cascade *is* μ̂(y | do(pitch), h) — so g-computation / AIPW / positivity /
  E-values still apply; we've swapped in a better μ̂. Causal-language discipline
  unchanged.

---

## 8. Edge cases / baseball nuances to handle

- **2-strike foul = no-op** (count unchanged) — the state machine must special-
  case it (else fouls would falsely end ABs).
- **HBP** — fold into the take branch (rare; a "take → hit-by-pitch" terminal) or
  ignore in v0.
- **IBB, bunts, catcher interference** — drop/ignore for v1.
- **Count-dependent zone (umpire expansion)** in S1b.
- **Platoon is first-order** — never drop stand×p_throws.
- **Park/altitude/temp** materially shift S3 (Coors carry, cold-weather suppress).

---

## 9. Open decisions to lock before coding the trainer

1. **S3 target:** xwOBA-on-contact regression (principled, needs the raw join) vs
   discrete {1B/2B/3B/HR/out} multiclass (simpler, from `events`). *Lean: xwOBA
   regression as primary; discrete as a display fallback.*
2. **Composition:** analytic count-tree solve (fast, noise-free — recommended) vs
   reuse the existing MC rollout (simpler to wire, noisy). *Lean: build analytic;
   keep MC as a cross-check.*
3. **S2b fouls:** model it (S2b node) vs a count-conditional constant in v0.
   *Lean: constant in v0, add the node if it matters.*
4. **Granularity of "pitch" the hitter model sees:** full continuous location
   (plate_x/z) vs the 25-zone bin. *Lean: continuous for the tree (it handles it
   well); zone for interpretability plots.*
5. **One model with a `node` indicator vs five separate boosters.** *Lean: five
   separate (clean populations, separate monotonic constraints, separate
   calibration).*

---

## 10. Validation (gate before shipping)

- **Compression diagnostic** (the motivator): real OPS spread vs cascade-
  predicted spread across a batter panel. Target ratio ≈ 1× (vs the transformer's
  6.8×).
- **Per-node:** AUC / log-loss / **ECE + reliability** (calibration primary).
- **Held-out hitters** cohort (profile-based generalization).
- **Causal coherence:** AIPW≈g-comp with the new μ̂; negative control (next-batter
  PA) ≈ 0.
- **Domain spot-checks:** elite > weak by a *realistic* margin; chase/whiff/power
  maps match scouting; monotonic constraints respected.
