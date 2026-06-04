# Dedicated Hitter / Swing-Decision Model — Brainstorm

**Status:** brainstorm (not a committed design). 2026-06-03.
**Owner:** Sid. Written alongside the `small-v7` training experiment.
**Related:** ADR-013 (v7 type-conditioned heads), `causal-layer`, `eval-protocol`,
`statcast-pipeline` skills; `docs/mcsim_appB_brainstorm.md`.

---

## 1. Why — the problem, with data

The single PitchGPT transformer does double duty: propensity π̂(pitch | history)
**and** outcome μ̂(result | pitch, history). It's good at the first job (pitch
prediction, its design goal). It is **weak at differentiating hitters' outcomes**,
which is exactly what a matchup card / recommender needs.

Measured on held-out 2024 data (`scripts`-level diagnostic, 12 batters, 6 best +
6 worst by real OPS, ≥150 PA, model OPS vs Aaron Nola at an in-range date):

| | spread (max−min) | std |
|---|---|---|
| **Real** OPS | 0.68 | 0.261 |
| **Model** (tiny-v7) OPS | 0.14 | 0.038 |

- **~6.8× compression** of the OPS scale. Pearson r = 0.66.
- Concretely: the best hitter in the sample (real OPS 1.105) is predicted 0.712;
  a .430 hitter is predicted ~0.65. **Everyone regresses toward ~.700.**

So the model gets the *order* partly right (r=0.66) but **crushes the scale**.
For a tool whose whole point is "how does THIS hitter fare vs THIS pitcher," that
is disqualifying. Root cause hypothesis: one model splits capacity between pitch
prediction (easy-ish, high signal) and outcome (hard, noisy), and under-fits the
batter dimension — regressing to the league mean minimizes loss when batter
signal is hard to exploit.

**Two remedies, cheapest first:**
1. **`small-v7`** — same architecture, more capacity (25M vs 6M params). Does
   capacity alone recover the spread? (Training experiment in flight.)
2. **A dedicated hitter / swing model** — this document. The structural fix.

---

## 2. Goal

A model that, **given a pitch and a batter (and context), predicts what the
batter does** — and does so with *calibrated batter discrimination* (recovers
the real OPS spread, not 1/7th of it). It composes with the existing pitch
**sequence** model so the division of labor is clean:

- **Pitch model (existing transformer):** what pitch is thrown, and when —
  sequencing, tipping, propensity. Its strength; keep it.
- **Hitter model (new):** the batter's *response* to a thrown pitch.

This new model becomes the μ̂ used inside `g_compute`'s rollout, so **every
downstream consumer benefits at once**: matchup cards, counterfactual explorer,
recommender, and any future tab.

---

## 3. What the hitter model predicts (the factorization)

Don't predict a flat 7-class AB outcome. Decompose the *plate-appearance
physics*, which is more learnable, more interpretable, and matches how hitting
actually works:

```
given (pitch a_t, count, batter, pitcher-stuff, context):
  1. swing decision:    P(swing | ...)            ← plate discipline / chase
  2a. if take:          P(called strike | ...)     ← (else ball) → count update
  2b. if swing:         P(whiff | ...)             ← contact ability
       if contact:      P(foul | fair)             ← foul → count update
        if fair:        contact quality            ← xwOBA-on-contact / launch
```

Each node is a clean, supervised target straight from Statcast `description` /
`estimated_woba_using_speedangle`. The terminal run value flows through the
existing run-value layer (RE24 + xwOBA-on-contact slope; `statcast-pipeline`).

Why this decomposition:
- **Batter identity lives in 1 and 2b** (chase rate, whiff rate) — exactly where
  hitters differ most and where the monolithic model compresses. A model that
  predicts *these* directly, with rich batter-profile features, will separate
  hitters far better.
- **Interpretable** (see §8): "this hitter chases 38% on low-away sliders" is a
  human-readable, probeable quantity.

---

## 4. Architecture options — and is a Transformer overkill?

| option | what | batter discrimination | interpretability | cost |
|---|---|---|---|---|
| **A. Gradient-boosted trees (XGBoost)** per node | tabular: pitch feats + pitcher arsenal + batter profile + count + context → swing/whiff/contact | likely **strong** (rich tabular batter feats, no capacity split) | **high** (SHAP, partial dependence) | **low** (CPU, minutes) |
| **B. Factored MLP** | shared embeddings → per-node heads | strong | medium (probe embeddings) | low–med (CPU/1 GPU) |
| **C. Transformer (sequence-aware hitter model)** | attends over the in-AB pitch sequence before predicting the response | marginal gain *if* in-AB adjustment matters | low (attention is hard to read) | high (GPU) |
| **D. Hybrid** | pitch **sequence transformer** (existing) + per-pitch **tabular/MLP** hitter model | best of both | high | low–med |

**Is a Transformer overkill for the hitter model? — Most likely yes, and here's
the argument:**

- The thing a Transformer is *for* is long-range sequence dependence. That lives
  on the **pitch-selection** side (a pitcher's sequencing, tipping, setup
  pitches) — which the **existing** transformer already models well.
- A batter's **response to a single thrown pitch** is much closer to a
  *contextual/tabular* prediction: it's dominated by (this pitch's
  type/location/velo, the count, the batter's profile, platoon). The in-AB
  history matters *some* (a hitter sitting on a pitch after seeing two of them),
  but that's a small, low-order effect you can capture with a few summary
  features (prev pitch type, # seen this AB) rather than full self-attention.
- The project's own `eval-protocol` already expects **XGBoost within 2–3 pts of
  the transformer on pitch top-1**, with the transformer's edge being
  *applications*, not raw tabular prediction. For **outcome/batter** prediction —
  a tabular problem with rich engineered batter features — a tree may well
  **beat** the transformer *and* be radically more interpretable.

**Recommendation: start with D (hybrid) using a tabular/tree hitter model (A),
not a Transformer.** Reserve attention for the pitch sequence where it earns its
keep. Only escalate the hitter model to a small sequence model (C) if a measured
"in-AB adjustment" gap justifies it.

### 4a. Sequencing IS real — capture it with recent-pitch features, not attention

Hitters *are* affected by the previous pitch(es) — sequencing/tunneling is a
genuine effect (two fastballs → hitter times the heater → a slider plays better;
high FB then low CU; sitting on a pitch just seen). So the hitter model must NOT
be memoryless. **But that memory is short-range and low-order** — a PA is only
~3–5 pitches, and the relevant context is essentially *the last pitch or two +
the count + the pitch-mix seen so far this AB*. So make these **first-class input
features** to the tabular model:

- previous pitch type / zone / velo / result (lag-1, optionally lag-2),
- count (balls–strikes) and pitch number within the AB,
- running counts of each pitch type seen this AB (cumulative mix),
- (optional) "same type as last pitch?" and "eye-level change" flags.

This captures the sequencing a Transformer would learn, but handed *directly* —
there's no long-range structure to discover over ~4 pitches that lag-features
miss. **Attention earns its keep when context is long and the model must learn
*which* positions matter; here it's the last pitch, which we just feed in.**
Empirical guard: build with lag-features first; only if a held-out gap shows
residual multi-pitch (tunneling-sequence) signal does a small sequence model (C)
become justified. Earn the attention; don't assume it.

---

## 5. How it composes with the pitch sequence model

The rollout (`causal/g_computation.py`) becomes a two-model loop. Interface kept
deliberately small:

```
hitter_model.predict(pitch=a_t, count=c_t, batter=b, pitcher_stuff=p, context=x)
  -> {swing, whiff, called_strike, foul, in_play, xwoba_on_contact}
```

Rollout step:
```
a_t   = pitch_model.sample_pitch(history)        # existing transformer (π̂)
resp  = hitter_model.predict(a_t, count, batter, …)   # NEW (μ̂ for the batter)
count, terminal, run_value = apply_response(resp)      # state machine + RV table
```

This **replaces the monolithic outcome head inside the rollout** with the
dedicated hitter model. Crucially it keeps the **causal-layer contract** intact:
the hitter model *is* a conditional outcome model μ̂(y | do(a), h), so
g-computation / AIPW / positivity / E-values all still apply — we've just swapped
in a better-calibrated μ̂. (Language discipline unchanged: still "model rollout"
unless the full causal machinery is engaged.)

**Behind a flag.** Add `outcome_model="head" | "hitter"` to `g_compute` so we can
A/B the monolithic head vs the hitter model on the same queries.

---

## 6. Where it plugs into the UI / product

Because it lives inside `g_compute`, every consumer benefits with no per-tab work:

- **Matchup cards** — the headline win: hitters finally separate (Judge ≠ bench
  bat). The exact compression diagnostic becomes the acceptance test.
- **Counterfactual explorer** — `do(pitch = slider)` outcomes become
  batter-specific and sharper; the "alternative completion" distributions mean more.
- **Recommender** — ranking pitches by predicted outcome gets sharper signal →
  fewer tossups, more decisive (and honest) recommendations.
- **Tipping** — mostly pitch-side; unaffected, but a cleaner μ̂ helps any
  outcome-weighted tipping metric.
- **(new) "gap" / prescriptive layer** (the earlier brainstorm) — needs a
  batter-sensitive μ̂ to be meaningful; this unblocks it.

---

## 7. Data, targets, training

- **Targets** (per pitch, from Statcast `description` + `estimated_woba_using_speedangle`):
  swing/take, called-strike/ball, whiff/contact, foul/fair, xwOBA-on-contact.
- **Features:** the thrown pitch (type/zone/velo/spin), count, platoon (p_throws ×
  stand), **batter profile** (the existing 1000-PA chase/whiff/xBA grids +
  recent-form 14-day wOBA), pitcher stuff (arsenal/velo profile), context
  (outs/runners/leverage/park/ump). All already produced by the pipeline.
- **Leakage:** identical trailing-window rule (profiles end strictly before the
  AB); reuse the existing leakage-tested profile cache + the new as-of fallback.
- **Splits:** temporal (≤2023 train), same as the main model, so it can join the
  causal eval cleanly.
- **Training cost:** tabular = minutes on CPU (no GPU). This is a big practical
  advantage over retraining the transformer.

---

## 8. Interpretability angle (Sid's interpretability project)

A dedicated, simpler hitter model is **strictly better for interpretability**
than the monolithic transformer:

- **Tabular/tree → SHAP + partial dependence:** directly answer "which
  batter-profile features drive chase / whiff / power vs this pitch?" Human-readable.
- **Modular separation:** the pitch transformer can be probed for *sequencing /
  tipping* (its job) while the hitter model is probed for *plate discipline /
  contact* (its job). No more disentangling two tasks tangled in one network.
- **Swing-decision node is itself a finding:** a calibrated P(swing | pitch,
  count, batter) model is an interpretable artifact (chase maps, count-leverage
  curves) independent of the rollout.
- If a Transformer hitter model is used later, the decomposition (§3) still gives
  named intermediate quantities to probe, vs one opaque 7-way head.

This argues again for **A/B (tabular/MLP) over C (transformer)** as the first
hitter model — interpretability is a first-class goal here, not an afterthought.

---

## 9. Validation (must pass before it ships)

1. **Compression diagnostic** (the one that motivated this): real OPS spread vs
   model OPS spread across a batter panel. Target: ratio ≈ 1× (vs current 6.8×),
   r ≫ 0.66.
2. **Calibration (eval-protocol primary):** ECE / reliability for swing, whiff,
   contact, and the derived OPS/wOBA. Per-node and end-to-end.
3. **Held-out hitters cohort:** does it generalize to batters whose first MLB PA
   is in the test window (profile-based generalization)?
4. **Causal coherence:** AIPW vs g-computation agreement still holds with the new
   μ̂; positivity/E-values still computed; negative control (next-batter PA) still
   ~zero.
5. **Domain spot-checks:** elite hitters > weak hitters by a *realistic* margin;
   chase maps match known scouting.

---

## 10. Risks / open questions

- **Double μ̂.** The transformer already has an outcome head. Do we *replace* it
  in the rollout, *ensemble*, or keep both for different tabs? (Lean: replace
  inside `g_compute` behind a flag; keep the head for the pure sequence model.)
- **Run-value consistency.** The hitter model's outputs must map through the same
  RE24 + xwOBA slope so run-value numbers stay comparable across the app.
- **In-AB adjustment.** Tabular drops full sequence context. Measure whether a
  "prev pitches this AB" feature set recovers it before reaching for a sequence
  model.
- **Calibration vs discrimination tradeoff.** Recovering spread must not break
  calibration (an over-confident model that separates hitters but is miscalibrated
  is worse). Watch both.
- **Coupling for the causal layer.** μ̂ must remain a *conditional* outcome model
  (no leakage of the action into features); keep the do(·) interface clean.

---

## 11. Recommended phased plan

- **Phase 0 (now):** `small-v7` — does capacity in the *same* architecture
  recover the spread? Re-run the compression diagnostic on it. Cheapest possible
  test; may partially solve the problem with zero new architecture.
- **Phase 1:** tabular hitter model (XGBoost) on the §3 decomposition. Validate
  compression recovery + calibration. Fast, CPU, interpretable.
- **Phase 2:** wire it into `g_compute` behind `outcome_model="hitter"`; A/B vs
  the head on matchup cards + recommender; ship if it wins the diagnostic.
- **Phase 3 (only if measured-necessary):** light sequence-aware hitter model for
  in-AB adjustment.

**Bottom line:** the dedicated hitter/swing model is the right structural fix, and
a **tabular/tree** version (not a Transformer) is the right first cut — better
batter discrimination, far better interpretability, and minutes-not-hours to
train. A Transformer here is most likely overkill; keep attention on the pitch
sequence where it already earns its keep.
