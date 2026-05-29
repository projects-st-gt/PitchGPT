# Recommender — Trust-Region-Restricted Causal Recommendation (Brainstorm)

**Status:** brainstorm / not yet started
**Sequencing:** after demo polish (PR #4) merges; before MCSim
**Engine:** wraps `causal/g_computation.py::g_compute()` in a ranking loop — no causal-layer redesign

This is a planning doc, not a spec. The reuse map below comes from a focused inventory pass; net new code is small. The doc's job is to surface the design decisions before any implementation.

---

## What the recommender does (plain English)

Given a pitch-state (pitcher, batter, count, runners, outs, history-so-far in the AB), return a **ranked list of pitch types** the model thinks would be best for the pitcher to throw next — **with the positivity gate refusing to recommend pitch types the data can't confidently support for this state.**

The honest framing in any UX that ships this:

> *"Here are the pitches the model thinks would do best given everything it knows. The greyed-out ones are pitches the model is refusing to recommend because the data doesn't support a confident answer for this state."*

## How it differs from the existing `/query` endpoint

`/query` (built, working): evaluates **one** specified intervention. *"What if pitcher had thrown FF here? → run rollout, return trust state, effect, CI, E-value."*

Recommender (new): evaluates **all** candidate interventions and **ranks** them. *"What should pitcher throw here? → run rollouts for all 7 types (or all in-arsenal types), rank by expected run-value, surface refusals separately."*

So the recommender is, essentially, *"call `/query` 7 times and sort the results"* — with extra work on the ranking and refusal-list semantics.

---

## Reuse map (from the causal-layer inventory)

| Component | Source | What it does | Net new? |
|---|---|---|---|
| Per-candidate rollout | `causal/g_computation.py::g_compute()` → `RolloutResult` | Single intervention sim, returns mean_run_value + se + ab_outcome_distribution + log_weights_per_step | No — call 7× |
| Per-candidate positivity gate | `causal/positivity.py::PositivityGate.gate(p_hat)` → `GateDecision` (green/yellow/red) | ADR-002 thresholds (τ_refuse=0.01, τ_green=0.05) | No |
| Multi-step ESS check | `causal/positivity.py::multi_step_check(log_weights_per_step)` | Refuses rollouts that wander out of support over a multi-step path | No |
| Per-candidate E-value | `causal/sensitivity.py::e_value_for_continuous_effect(effect, se, outcome_sd)` | Robustness to unmeasured confounding | No |
| Batch builder (one batch, 7 replicates) | `causal/nuisance.py::build_single_ab_batch()` | Used by g_compute + aipw; the recommender can reuse it once for all 7 candidates | No |
| Arsenal pre-filter | `data/profile_cache.py::PITCHER_FEATURE_INDEX["has_pitch_{pt}"]` | Optional: exclude types the pitcher has never thrown from the candidate set | No |
| **Ranking logic + refusal list** | — | Sort in-support candidates by mean_run_value; bucket refused candidates separately | **Yes** |
| **API endpoint + Pydantic schemas** | — | `POST /recommend` (or extend `/query`) + response schema | **Yes** |
| **Frontend Tab 5** (later) | — | Recommender UI | **Yes** (separate scope) |

**Bottom line:** the new code is `recommender/__init__.py` + ranking helpers + an API endpoint. ~200–400 LOC, plus tests.

---

## Design decisions to make

### D1 — type-only or type+zone?
- **Type-only (7 candidates):** simplest. ~7 g_compute calls per recommendation. ~70s on CPU at n_paths=200.
- **Type+zone (7×13 = 91 candidates):** much more expensive, ~15 min on CPU. Most cells would be refused by zone positivity (zone τ_refuse=0.003).

**Lean:** **type-only for v1.** Zone recommendation as a v2 deep-dive per recommended type.

### D2 — baseline for "effect"?
Each candidate's expected run-value needs to be compared *to something* for the "this is better than X" framing.

- **(a) vs the observed pitch** — same framing as `/query` ("model thinks SL would be +0.05 runs better than the FF that was actually thrown"). Honest and grounded.
- **(b) vs the best other candidate** — adversarial ("SL beats FF by +0.05, the next-best alternative"). Decision-relevant for choosing.
- **(c) absolute mean_run_value, no contrast** — "throw SL: expected RV = -0.12" with all candidates on the same scale.

**Lean:** **(c) as the primary ranking signal** (absolute mean_run_value) — gives a clean total order. **(a) as a secondary "vs observed" annotation** when the AB has an observed pitch at the intervention position.

### D3 — refusal semantics
When π̂(type | h) < τ_refuse, the candidate gets refused. Two ways to surface:

- **(i) Drop refused candidates from the ranked list entirely.** Cleanest UX — "here are your real options."
- **(ii) Return a separate "refused" list alongside the ranked list.** More transparent — "we considered these and refused them because the data doesn't support it."

**Lean:** **both.** The ranked list shows only the in-support recommendations (green + yellow); the API also returns a `refused` array with `{candidate, p_hat, reason}` so the frontend can show "we refused to recommend 3 pitches: CU (π̂=0.004), FS (π̂=0.001), …" — that's the epistemic-humility flavor the project is built on.

### D4 — arsenal pre-filter
A pitcher who has never thrown a splitter has has_pitch_FS = 0. Two interpretations:

- **(a) Skip arsenal pre-filter — let positivity gate handle it.** The propensity head learns the arsenal signal via the per-pitch arsenal feature (ADR 009); π̂(FS | h) will be tiny for a non-splitter pitcher and the positivity gate will refuse. Honest, fully causal.
- **(b) Hard-filter by has_pitch BEFORE the rollout.** Skip the rollout entirely for impossible types. Cheaper compute, but masks the model's actual prediction.

**Lean:** **(a).** It's more honest, the positivity gate is the right tool. The arsenal-mask experiment showed `has_pitch=0` is "absent from trailing window," not "impossible" — hard-filtering would falsely refuse mid-season-acquired pitches. Show ALL refused candidates in the refusal list (per D3) so the user sees the model's reasoning.

### D5 — CI / "tossup" handling
When the top two candidates have overlapping 95% CIs, the ranking is statistically a tossup. UX rule:

- **(a) Rank by mean anyway** — "SL is #1, FF is #2" — even if the CI overlap is heavy. Simple.
- **(b) Surface a "tossup" group** — "the top 2 candidates (SL, FF) are statistically indistinguishable; either is a defensible call."

**Lean:** **(b).** The honest framing is more aligned with the project's epistemic-humility ethos. Compute the overlap from `se_run_value` and flag pairs whose 95% CIs overlap as tossups.

### D6 — synchronous vs async
The naive implementation (`g_compute` 7×) takes ~70 s on CPU. That's too slow for a synchronous HTTP request.

- **(a) Synchronous, smaller `n_paths`** — `n_paths=100` per candidate → ~35 s. Still too slow.
- **(b) Async with progress.** Return a job ID, poll for completion. More UI plumbing.
- **(c) Pre-compute for curated demo at-bats.** Cache the recommendations for a small set of curated ABs the demo serves. Instant. Trades coverage for speed.
- **(d) GPU.** Single-A100 forward pass is ~50× faster — would make sync feasible at <2s.

**Lean for the *demo*:** **(c) pre-compute for the curated demo ABs.** The demo cache already exists in scaffold form; extend it to include recommender output. Coverage is small but UX is instant. Real recommendations can use (b) async if/when needed.

---

## Suggested implementation order

1. **`recommender/__init__.py` + `recommender/rank.py`** — the core ranking function. Takes `(nuisance, ab_pitches, intervention_position, n_paths)` and returns `RankedRecommendations` (an in-support ranked list + a refused list + per-candidate diagnostics). Pure Python, no API surface yet. **First milestone — testable in isolation.**
2. **Tests** — pin the contracts: (a) ranks all 7 types by mean_run_value, (b) refused candidates have `p_hat < τ_refuse` and don't appear in the ranked list, (c) trust state on each candidate matches the positivity gate decision, (d) tossup flag fires when CIs overlap.
3. **API endpoint** — `POST /recommend` returning `RecommendationResponse` schema. Reuses the same `AppState` plumbing as `/query`.
4. **Pre-compute for the demo cache** — a script that walks the demo's curated ABs and pre-computes recommendations, stored as JSON fixtures. The API can read from the cache when present and fall back to live compute when not.
5. **Frontend Tab 5** — Recommender UI. Separate scope, follow-up PR.

---

## Open questions

- **What "outcome_sd" should we use for E-values?** The existing `/query` uses 0.30 as a Cohen-d-like proxy (`OUTCOME_SD_PROXY` in `inference/api.py`). For the recommender's per-candidate E-values, same value? Or per-state outcome-sd?
- **Should the recommender expose per-candidate AB-outcome distributions?** (P(K)/P(BB)/P(HR) under do(SL) etc.) The data is there from `RolloutResult.ab_outcome_distribution`. Worth surfacing if it helps the manager-decision flavor.
- **Should refusal carry an explanation per candidate?** "Refused FS because the trailing-arsenal flag says pitcher hasn't thrown one recently and π̂ is below τ_refuse" — more informative than just "π̂=0.001."
- **Caching strategy for repeated state queries** (e.g., the same `(pitcher, batter, count, runners, outs)` showing up across multiple games)? Probably overkill for v1 but worth noting.

---

## Things explicitly out of scope (for v1)

- Multi-step / sequential recommendation ("what to throw across the whole AB"). v1 is per-pitch.
- Lineup-level / bullpen-level optimization (that's the MCSim matchup document — separate project).
- Inverse problem ("what should the *batter* do given the predicted pitch?") — different model, not on the roadmap.
- Anything that requires retraining (the recommender is pure inference on the existing v7 checkpoint).
