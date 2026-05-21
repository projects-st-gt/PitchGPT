# ADR 002 — Positivity Threshold

**Status:** Accepted (locked 2026-05-09)
**Date:** 2026-05-08

## The question, in plain English

The demo refuses to estimate counterfactuals when the data can't support them. ("What if Sale had thrown a knuckleball?" — refused, because Sale doesn't throw knuckleballs.)

How rare does the proposed pitch have to be before we say "no"? Picking this number — and how it scales to multi-step rollouts — is what makes the trust gauge work.

## Why this matters

- **Too lenient** → the demo silently extrapolates into regions where the model has almost no data. Estimates are confident-looking nonsense. The "epistemic humility" pitch falls apart.
- **Too strict** → most queries get refused. Demo UX dies. Users learn "the system always says no" instead of "the system says no when the data can't answer."
- **Multi-step blow-up** → a single threshold is wrong for multi-step queries. A 5-step rollout where each step has π̂ = 0.05 has effective coverage of 0.05⁵ ≈ 3×10⁻⁷.

## Options

**A. Single threshold τ = 0.01.** Brainstorm default. Refuse if π̂(a*|H) < 0.01.

**B. Single threshold τ = 0.05.** More conservative. Many demo queries get refused.

**C. Empirically calibrated τ.** After Phase 3, bin held-out pitches by π̂ decile, compute calibration of μ̂ within each decile, set τ where calibration breaks down. Whatever number falls out is τ.

**D. Continuous trust gauge with no hard refusal.** Always show an estimate; let the gauge color convey trust. Risk: people read the number and ignore the gauge.

**Multi-step:** any of A/B/C also need an effective sample size (ESS) check for k-step rollouts. ESS = `(Σ wᵢ)² / Σ wᵢ²` over the N rollouts at step k. If ESS drops below some floor mid-rollout, the rollout has wandered out of support and should be cut off.

## Recommendation

- **Single-step:** start with τ = 0.01 (option A) as a hard refusal floor, and re-calibrate empirically (option C) after the model is trained. The empirical re-calibration is mandatory before the demo ships.
- **Multi-step:** ESS floor of 30 per step (rule of thumb — translates to the IPW weight distribution being "not totally collapsed"). If ESS < 30 at any step k, the rollout is cut and the demo reports "supported answer up to step k − 1, beyond that we don't know."
- **Trust gauge bands** (matches brainstorm):
  - Green: π̂ > 0.05 — confident
  - Yellow: 0.01 < π̂ ≤ 0.05 — extrapolating, take with skepticism
  - Red: π̂ ≤ 0.01 — refused, no point estimate shown

## Divergence from the brainstorm

Substantively aligned. Two additions:

1. **Mandatory empirical re-calibration** of τ after training, with a documented procedure. The brainstorm proposed 0.01 and said "calibrate empirically" without specifying how.
2. **Concrete ESS floor (30) for multi-step rollouts.** The brainstorm mentioned ESS but didn't pick a number.

## Consequences

- We need a calibration script that runs on held-out data after each training run and reports the empirical τ.
- The demo backend rejects queries with `π̂ < 0.01` *before* doing any rollout — saves compute and gives a fast "refused" UX path.
- Multi-step rollout code must track per-step ESS and emit a structured "truncated at step k" response to the frontend.
- For AIPW (population estimands), there's a separate question: trim weights vs. drop observations vs. report unstable. We default to weight trimming at 1/τ = 100, with a note in any reported number that uses trimmed weights.
