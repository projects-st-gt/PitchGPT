# ADR 007 — Nuisance Decoupling (π̂ vs μ̂)

**Status:** Accepted (locked 2026-05-09); Amended 2026-05-10 (see Amendment 1 below)
**Date:** 2026-05-08

## The question, in plain English

The whole project rests on one elegant fact: PitchGPT's autoregressive next-pitch head *is* the propensity model π̂(a|h), and the outcome head *is* μ̂(y|a,h). So one trained transformer gives us both nuisance functions for AIPW. No separate propensity network needed.

But there's a catch. AIPW's headline property — the doubly-robust guarantee that the estimate is consistent if *either* π̂ or μ̂ is correct — assumes their errors are uncorrelated. If both heads share a transformer trunk and are trained jointly, their errors are correlated by construction (same parameters, same gradients). The doubly-robust property degrades.

How decoupled do we need π̂ and μ̂ to be?

## Why this matters

- This is the difference between AIPW with the textbook DR property and AIPW that's "IPW with outcome correction." Both are useful estimators, but the methods writeup's pitch is "we did proper sequential causal inference." Decoupling is part of "proper."
- This is the most likely place a careful reviewer pushes back. Pre-empting that pushback in the writeup is much easier than retrofitting it.
- Decoupling can be cheap or expensive depending on which option we pick. Worth deciding before training, not after.

## Options

**A. Shared trunk, separate heads, joint optimization.** Brainstorm default. Cheapest. Errors are coupled at every parameter in the trunk; DR property is at best partial.

**B. Shared trunk, separate heads, stop-gradient between heads at trunk output.** Forward pass shared, backward pass separated. The trunk is trained on a joint loss (the sum or weighted sum), but the propensity head's loss never updates the outcome head's parameters and vice versa. Reduces gradient coupling but not parameter coupling — the trunk is still a single shared representation. Same compute as A.

**C. Two completely separate transformer models.** Strongest decoupling. 2× compute (or 10× combined with K=5 cross-fitting — 10 separate model trainings). Loses the elegance of "one model, both nuisances."

**D. Within K=5 cross-fit, additionally partition heads to folds.** π̂ trained on folds {1,2}, μ̂ on folds {3,4}, evaluated on {5}. Rotate. Same total compute as A, because each pitch is still seen by training only once for each head. Decouples the data each head sees, which is a stronger guarantee than just decoupling parameters. But operationally complex — the training pipeline has to track which fold trains which head in which rotation.

## Recommendation

**Option B as default.** Stop-gradient is a one-line code change, addresses the headline concern (gradient-level coupling), no compute hit, and keeps the elegant "one model" architecture intact for the demo.

The writeup is honest about the residual parameter coupling — the trunk is a shared representation, and that's a real limitation of the approach, just a smaller one than full joint optimization.

### Direction of the gradient block (locked)

The two heads are independent submodules — neither contains parameters of the other, so "gradient between heads" is already absent. The real design question is **what backprops into the trunk.**

Two sub-options:

- **B1. Trunk shaped by both heads.** Each head's loss flows into the head's own parameters AND into the trunk. The trunk receives the sum of gradients from both losses. Combined loss `L = λ_π · L_π + λ_μ · L_μ`.
- **B2. Trunk shaped by one head only; the other reads frozen features.** E.g., trunk learns from L_π only; outcome head consumes `trunk(x).detach()`. Or vice versa.

**Pick: B1.** Both heads contribute to the trunk via their own losses.

**Why B1 over B2:**

- B2 is a strictly dominated middle-ground. It loses the "both nuisances benefit from a shared representation" elegance (the read-only head is just a thin readout from a representation tuned for the other task), without buying full decoupling (the trunk is still a shared parameter set and still has shared initialization). If the goal is full decoupling, train two separate models (option C); if the goal is a shared representation with reduced gradient coupling, B1 is what you want.
- B1 is the standard multi-task setup for deep AIPW work. The published reference points (TARNet, Dragonnet, and the deep-learning-for-causal-inference literature) use shared-trunk + per-head losses summed. Deviating from that without strong reason makes the writeup harder to defend.
- The residual coupling at the trunk level is a real but bounded limitation, addressed by the Phase 9 separate-models sanity check, not by hobbling one head's contribution to learning.

**Concrete training loop:**

```python
shared = trunk(inputs)                       # [B, T, d_model]
logits_pi = propensity_head(shared)          # [B, T, |A|]
logits_mu = outcome_head(shared, action)     # [B, T, |Y|]

L_pi = ce_loss(logits_pi, target_action)
L_mu = ce_loss(logits_mu, target_outcome)

L = lambda_pi * L_pi + lambda_mu * L_mu
L.backward()                                 # gradients flow to trunk + both heads
optimizer.step()
```

There is no `.detach()` on `shared` for either head. Both heads' losses backprop into the trunk; neither head's loss touches the other head's parameters (which is automatic because they're independent modules).

**Loss balancing.** Default `λ_π = λ_μ = 1.0`. Calibration of each head is logged on a held-out batch every K steps. If one head is materially worse-calibrated than the other after warmup (~5k steps), upweight the underperforming head's λ. The methods writeup reports the final λ values used and the calibration trajectory.

### What option C buys, and when

**Option C as a Phase 9 stretch sanity check** on a single estimand. Train two completely separate models, compute the AIPW estimate for one query (e.g., "average effect of throwing a slider vs. fastball in 0-2 counts to LHB"), compare to the shared-trunk B1 estimate.

- If they agree within standard-error: shared-trunk is empirically validated; report B1 as the headline number.
- If they disagree: the parameter coupling matters. Report both, with the divergence as a measured limitation. The writeup gains a quantitative bound on how much trunk-sharing is leaking.

This is the right safety net for the trunk-sharing decision — it converts an a priori concern into an empirical check.

**Option D considered and not chosen.** It's the strongest decoupling story available without paying compute, but the operational complexity in the cross-fit pipeline is high — split rotation logic, fold-to-head assignment, debugging — and the marginal decoupling gain over option B is small. Save the engineering complexity for things that buy more.

## Divergence from the brainstorm

**Yes, this is a real divergence.**

The brainstorm proposed shared trunk with joint optimization (option A): "single shared transformer, multiple heads, both nuisance functions for free." This ADR adds **stop-gradient between heads** as a default, and recommends Option C as a Phase 9 sanity check.

The brainstorm's elegance argument is preserved — the architecture is still "one model, both nuisances" — but the gradient flow between heads is severed.

## Consequences

- Training code: forward pass goes through trunk → both heads see the same trunk output. Each head has its own loss. At backward pass, `outcome_head_loss.backward()` does not update propensity head parameters and vice versa. The trunk receives the sum of gradients from both heads. (Concretely: detach the trunk output for one head's loss-weight backprop path, or use separate optimizers with parameter-group masking.)
- Loss balancing matters: the joint loss is `L = λ_π · L_π + λ_μ · L_μ`. Brainstorm didn't specify λ values; pick them so the two losses are on similar scales after a few thousand training steps. Default: log calibration on a held-out batch every K steps and rebalance if either head is dominating.
- Phase 4 sensitivity work (E-values, negative controls) gets evaluated against this decoupled-head estimate. The Phase 9 separate-models comparison is an additional sanity check, not a different default.
- The methods writeup explicitly addresses the trunk-sharing limitation as a known caveat with the option-C empirical check as supporting evidence.

---

## Amendment 1 — 2026-05-10 (result-head hidden-input position)

### The bug this amendment fixes

The original ADR specifies that the result head reads the trunk's hidden state and the intended action, with a stop-gradient on the hidden input. It does **not** specify *which position's* hidden the head reads.

The initial implementation (pre-amendment) had the result head read ``pitch_hidden[t]`` at every pitch position ``t`` — i.e., the trunk output AT the same position as the pitch whose result is being predicted. But the trunk's input embedding at position ``t`` is the sum of every per-pitch factor at that position, **including** ``result_emb(result_t)`` (the per-pitch ``FactorEmbeddings.result_emb`` table). So ``pitch_hidden[t]`` encodes the answer the result head is supposed to predict.

This is a **silent leakage shortcut**:

- During training (target = actual ``result_t``), the head learns to project the ``result_emb`` direction out of ``hidden[t]`` and ignores ``intended_action``.
- At counterfactual rollout time, the trunk-input embedding at position ``t`` is rebuilt with an alternative pitch (different ``type/zone/velo``, no ``result``). The head's learned "read result from hidden" shortcut breaks — it outputs garbage. The two-stage architecture's headline property — that changing the intended action shifts the result distribution — silently fails.

This was verified empirically on 2026-05-10 with a deterministic test: training to predict ``result_t`` from random pitch factors, with ``intended_action`` sampled independently of ``pitch_factors``. The leaky model reached ~78% top-1 accuracy (vs. ~14% chance over 7 result classes), while the counterfactual L1 sensitivity (between two different ``intended_action`` queries) was only ~0.04.

### The fix

The result head's hidden input is **shifted by one position**: at pitch position ``t`` the head reads ``hidden[t-1]`` (history through pitch ``t-1``, which does NOT contain ``result_t`` in its input embedding) and ``intended_action[t]`` (the action being taken at pitch ``t``).

At ``t = 0`` there is no ``hidden[-1]``. We use the **last context token's** hidden state (the third of the three prepended context tokens — the categorical-context token after all confounders are summed and LayerNorm'd) as the "history through no pitches" surrogate. The trunk has already mixed it with the pitcher- and batter-profile tokens via the first attention block, so it carries the AB-level conditioning information.

Concretely, in ``model/pitchgpt.py:PitchGPT.forward``:

```python
pitch_hidden = x[:, self.N_CONTEXT_TOKENS:, :]           # (B, T, d_model)
ctx_last_hidden = x[:, self.N_CONTEXT_TOKENS - 1: self.N_CONTEXT_TOKENS, :]
hidden_for_result = torch.cat(
    [ctx_last_hidden, pitch_hidden[:, :-1, :]], dim=1
)  # (B, T, d_model) — hidden_for_result[t] = history through pitch t-1
result_logits = self.result_head(
    hidden_for_result.detach(),                          # ADR 007 stop-gradient
    intended_actions["type"],
    intended_actions["zone"],
    intended_actions["velo"],
    intended_actions["spin_axis"],
)
```

### Why this is the right fix

- **No information loss for the propensity head.** The propensity head still reads ``hidden[t]`` (the un-shifted output) and predicts pitch ``t+1``'s factors. ``hidden[t]`` containing ``result_t`` is **correct** for that prediction: a pitcher choosing pitch ``t+1`` knows what happened on pitch ``t``.
- **No information loss for the trunk.** The trunk still sees every factor including result in its input embedding; the autoregressive history is unbroken.
- **The result head now has a clean causal interpretation.** At position ``t`` it predicts ``result_t`` from ``(history-through-(t-1), intended-action-at-t)`` — exactly the conditional we want for the two-stage μ̂(y | a, h).
- **Counterfactual rollout works.** At inference we can swap ``intended_action_t`` and the head's output shifts as expected, because the head was forced to learn the action→outcome mapping during training (it had no alternative path).

### Regression test

``tests/test_pitchgpt_model.py::test_result_head_does_not_leak_from_hidden_result`` trains the model on the leakage target (target = ``pitch_factors["result"] - 1``, intended_action independent of pitch_factors) and asserts top-1 accuracy on a held-out batch stays under 40%. The leaky architecture hits ~78% on this test; the fixed architecture hits ~14% (chance level). The threshold is set at 40% to leave headroom for any weak signal from other factors.

### Status

This is a strict bug fix — no compute or methodology change beyond the one-line ``cat`` in the forward pass. The training script written next will use the fixed architecture. No prior checkpoint exists, so there is no migration concern.
