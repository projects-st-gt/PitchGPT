---
name: causal-layer
description: Use this skill any time the project touches causal estimation — g-computation rollouts, AIPW estimators, cross-fitting, propensity gating, positivity checks, ESS tracking, E-values, negative controls, or sensitivity analysis. Trigger on mentions of counterfactual, intervention, do-operator, propensity, treatment effect, ATE, doubly-robust, g-formula, identification, ignorability, or "what if." This is the methodological centerpiece — read it before writing any code that produces effect estimates, and before writing any user-facing copy that uses causal language. Most "deep counterfactual" papers skip cross-fitting and positivity; this project does not.
---

# Causal Layer

This is where PitchGPT becomes a causal model rather than a sequence model.
The trained transformer serves as both the propensity model π̂(a | h) (its
next-pitch head) and the conditional outcome model μ̂(y | a, h) (its
two-stage result head). Cross-fitting, positivity gating, and sensitivity
analysis turn it into a defensible estimator.

## What we are estimating

For an individual single-step intervention at pitch index k of an at-bat:

```
τ(a, a' | h) = E[Y | do(A_k = a), H_<k = h]
             - E[Y | do(A_k = a'), H_<k = h]
```

where Y is run value (from the RE24 table; see the `statcast-pipeline` skill)
and H_<k is the full observed history up to pitch k.

For population-level estimands ("what is the average effect of throwing a
slider rather than a fastball in 0–2 counts to LHB league-wide?"), we use
AIPW with cross-fitting.

The treatment space is defined in `docs/decisions/001-treatment-granularity.md`.
Default: pitch type only (7 actions). Type+zone (175 actions) is a stretch
goal; positivity becomes a much harder problem at that granularity.

## Identification assumptions — write these in plain text

Every causal output in the demo and writeup must surface these:

1. **Sequential conditional ignorability** given observed H. We do not see
   the catcher's read of the batter's stance, the in-game scouting
   adjustments, or pre-pitch mechanical breakdown signals. This assumption
   is almost certainly violated to some degree. *Sensitivity analysis
   addresses this.*
2. **Positivity** within a defined trust region. *Enforced.*
3. **Consistency.** Uncontroversial here.
4. **SUTVA at the at-bat level.** Fine for single-AB queries; harder for
   policy-level interventions across a game.

These are reproduced verbatim on the demo's "About" page and in the methods
writeup. Do not soften the language.

## Sequential g-computation via autoregressive rollout

This is the core trick: the autoregressive transformer *is* a learned
sequential propensity score, and its outcome head *is* the conditional
outcome model. Rolling out is g-computation.

```
def g_compute(model, history h, intervention a* at index k, N=1000):
    estimates = []
    for _ in range(N):
        seq = h.clone()
        seq.set_action(k, a*)              # do(A_k = a*)
        for t in range(k+1, max_T):
            r_t  = sample(model.result_head(seq))   # μ̂
            seq.append_result(t-1, r_t)
            if seq.is_terminal(): break
            a_t  = sample(model.policy_head(seq))   # π̂
            seq.append_action(t, a_t)
        estimates.append(run_value(seq))
    return mean(estimates), std(estimates) / sqrt(N)
```

Implementation: `causal/g_computation.py`. N=1000 is the default for the
demo; cached pre-computations use N=5000 for the curated at-bats.

## AIPW for population estimands

For population-level queries, plain g-computation has unmodeled bias if μ̂ is
slightly wrong. The doubly-robust form is consistent if either π̂ or μ̂ is
correct:

```
ψ̂(a) = (1/n) Σ_i { μ̂(a, H_i)
                    + (𝟙[A_i = a] / π̂(a | H_i)) · (Y_i - μ̂(a, H_i)) }
```

Variance via the influence-function-based estimator with cross-fit folds.
Implementation: `causal/aipw.py`.

Use AIPW for:
- Average effect of pitch a vs a' across a defined slice of states
- Recommender validation (per-pitcher, per-state policy values)
- Methods-paper population tables

Use g-computation for:
- Demo single-AB queries (individual counterfactuals)
- Anything conditioned on a specific h (no "average over the population")

## Cross-fitting — K=5, season-stratified, game-blocked

Mandatory for any AIPW or population number reported. Single-fit nuisance
estimates have asymptotic bias (the same data trained the model and is being
evaluated on it).

```
folds = stratified_block_kfold(
    data,
    K=5,
    stratify_by="season",
    block_by="game_pk",
)
for fold_idx, (train_idx, eval_idx) in enumerate(folds):
    π̂_fold, μ̂_fold = train_pitchgpt(data[train_idx])
    aipw_terms[eval_idx] = aipw_per_unit(data[eval_idx], π̂_fold, μ̂_fold)
ψ̂ = mean(aipw_terms)
SE = influence_function_se(aipw_terms)
```

Implementation: `causal/crossfit.py`. Yes, this is 5× the training cost.
Budget for it. The blocking by `game_pk` matters because pitches within a
game share the catcher, umpire, lineup state, and other latent factors —
splitting them across folds leaks.

## Positivity gating — the demo's most important feature

For any individual intervention A_k = a* in state h, before estimating an
effect, check π̂(a* | h). The thresholds:

| π̂(a* | h)    | gauge   | demo behaviour                                    |
|---------------|---------|---------------------------------------------------|
| > 0.05        | green   | show point estimate + CI + E-value                |
| 0.01 – 0.05   | yellow  | show estimate with prominent uncertainty banner    |
| < 0.01        | red     | refuse: "this intervention is outside the data's support" |

For multi-step rollouts, accumulate the inverse-propensity ratio across the
rollout and track effective sample size:

```
ESS = (Σ_i w_i)^2 / Σ_i w_i^2,    w_i = Π_t π̂(a_t,i | h_t,i)^{-1}
```

If ESS drops below max(50, 0.05·N) at any step, the rollout has wandered
out of support. Surface this as "the rollout left the trust region after
pitch X" rather than producing a number.

This is *the* feature that makes the demo honest. Do not soften the
refusals to make queries succeed more often.

## E-values — sensitivity to unmeasured confounding

For every causal estimate reported, also report an E-value: the strength of
association on the risk-ratio scale that an unmeasured confounder would
need to have, with both treatment and outcome, to fully explain away the
observed effect.

For a continuous outcome with effect estimate δ and SE:

```
E_value = δ + sqrt(δ · (δ + 1))    (after rescaling to RR)
```

See VanderWeele & Ding (2017) for the formal mapping; the implementation
is in `causal/sensitivity.py`. Display alongside the effect:

> +0.13 runs (95% CI: 0.04 – 0.22, E-value 2.3)

Tells the user: a hidden confounder would need to confound both the pitch
choice and the outcome at a risk-ratio of 2.3 to nullify this finding.
Calibrate against measured confounders' E-values to give intuition.

## Negative controls

Pick an outcome that should *not* be causally affected by the intervention.
Default: the next batter's PA outcome (assuming no signal-stealing or
mood-contagion path). If your machinery finds a non-zero effect on the
negative control, your model has bias and you cannot trust the primary
estimates.

Run quarterly as part of `make eval`. Failures get a P0 issue, not a footnote.

## Validation hierarchy

Each row is a cell in the methods table:

1. Calibration of μ̂ on held-out at-bats — reliability diagrams, ECE
2. Calibration of π̂ — same
3. AIPW vs g-computation agreement under cross-fitting — should be close;
   gaps indicate misspecified nuisance models
4. **Natural-experiment matching:** find pairs of at-bats with near-identical
   propensity scores but different observed pitches; the difference in their
   outcomes is a quasi-randomized estimate; compare to model prediction
5. Negative controls
6. Domain validity spot-checks (does the model rediscover known wisdom — e.g.,
   low-and-away breaking balls dominate fastballs in the heart in 0-2 counts?)
7. Stability of estimates as N_rollouts grows

If any of 1, 2, 3, 5 fail, the project's causal claims are not defensible.
Stop and diagnose before producing user-facing outputs.

## Language discipline

In code comments, function names, variable names, docstrings, demo copy, and
the writeup:

| forbidden                          | use instead                       |
|------------------------------------|-----------------------------------|
| "what if he had thrown..."         | "the model's expected outcome under do(...)" |
| "counterfactual" (without machinery) | "alternative completion" / "model rollout" |
| "this caused..."                   | "we estimate an effect of..."     |
| "the model predicts X would have happened" | "under our identification assumptions, the AIPW estimate is..." |

The honest version is also the more interesting version. Do not retreat
to wishy-washy language *inside* the causal pipeline either — quantify
uncertainty, be explicit about assumptions, and let users see them.

## Things to avoid

- **Reporting effects without cross-fitting.** Single-fit numbers do not
  go in PRs or the writeup.
- **Silencing positivity refusals.** The refusal *is* the contribution.
- **Aggregating over a slice without checking positivity per-unit first.**
  An ATE with 30% of units violating positivity is meaningless.
- **Causal language in the propensity/outcome model layers.** Those are
  predictive models. Causality lives in `causal/`, not `model/`.
