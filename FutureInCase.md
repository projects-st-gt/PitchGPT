# FutureInCase.md — contingency plan if PitchGPT doesn't close the LSTM gap

**Status (2026-05-11):** PitchGPT-Tiny (Phase B, *no-standardization* recipe) gets
~0.459 type top-1; the profile-aware LSTM gets **0.691** [95% CI 0.690–0.693]
(re-confirmed, leakage-clean) on the same 2024-H1 val split — a **23-point gap**.

Diagnostics *ruled out*: the data/profile cache (healthy — 100% per-player hits,
median confidence 1.0, arsenal varies across pitchers); the dataset pipeline
(factors correct + aligned); multi-task interference (`type_only` ≡ all-7-heads,
0.4468 = 0.4468 at 1500 steps); over-regularization (less reg → *worse*);
over-parameterization (it's underfitting, not overfitting); and the LSTM number
being inflated.

Leading cause: **input-encoding starvation** — the pitcher's arsenal (the single
most predictive feature) was buried as 14 of 223 dims in a profile blob → MLP →
one context token (recoverable only via an attention hop), *and* in Phase B fed
unstandardized so even those slots were numerically swamped by `mean_spin_*`
(~2300) / `mean_velo_*` (~93). The bet: **(a)** profile standardization (already
the default in `train_pitchgpt.py`; Phase B predated it) + **(b)** the per-pitch
arsenal feature (ADR 009 — 14-dim arsenal+has-pitch → `Linear(14, d_model)` →
added to every pitch token) + a `small` (25M) retrain closes most of it. Two L4
runs (`tiny-v3-arsenal-std`, `small-v1-arsenal-std`, 6 epochs, full ≤2023 corpus)
are training now.

**This doc = what to do if that bet underdelivers.** Acceptable landing: mid-0.50s+
(≈ within ~10 pts of the LSTM, well-calibrated) → fine, the gap is then explicable
by generative-vs-discriminative (see "the reframe" below). Still in the 0.40s →
work the ladder below.

---

## Step 0 — re-diagnose before picking a fix

Don't guess; target the confirmed failure mode. On the new checkpoint, re-run:

- per-position accuracy breakdown (`scripts/investigate_phase_b.py`-style) — is
  accuracy flat across pitch positions (= not using within-AB sequence) or rising
  (= using it)?
- baseline comparison (`scripts/run_simple_baselines.py`) — is PitchGPT still
  ≈ per-pitcher-mode? above per-(pitcher×count)-mode yet?
- attention inspection (`model.set_attention_caching(True)`) — are pitch tokens
  actually attending to the pitcher context token / the new arsenal contribution?

Then pick the option below that targets *that*.

---

## Tier 1 — cheap, do first

- **Concat-then-project the per-pitch factors instead of summing them.**
  `pitch_token = Linear(concat(type_emb, zone_emb, …, count_emb, fatigue_emb), d_model)`
  instead of `sum(...)`. Every factor gets its own sub-space; the trunk stops
  disentangling a superposition. Natural extension of ADR 009's (b) (which gave
  the *arsenal* its own sub-space). ~30-line change; needs an ADR (changes the
  embedding spec).
- **More epochs / a flatter LR schedule.** Currently 6 epochs. The early-stop on
  the running L4 jobs reports whether val loss is still descending; if so,
  `--epochs 12–15` is free to try (watch `small` for overfit). Transformers are
  data-hungry; the LSTM converging in 3 epochs doesn't mean the transformer does.
- **More high-signal profile sub-vectors as per-pitch features.** Beyond (b)'s 14
  arsenal+has-pitch dims: `mean_velo_*` / `mean_spin_*` (7 each — "throws hard /
  with spin"), the 12 count-conditional entropy slots ("how predictable per
  count"). Same `Linear(k, d_model)`-into-pitch-tokens trick. Trivial; diminishing
  returns vs Tier 2.
- **Symmetric move for the batter.** The 91-dim batter profile has the same
  "buried in a blob" problem — give the batter's chase-rate / whiff-by-zone its
  own per-pitch pathway. Pitch calling is pitcher × batter; right now both sides
  are squeezed through one context token each.

---

## Tier 2 — real architecture (the "transformer answer to the LSTM's h0 trick")

The LSTM's actual edge: `(h0, c0) = projected profile` — the pitcher fingerprint
*is the starting state*, immediately available at every step, zero attention cost.
Transformer analogs:

- **FiLM-condition the trunk on the profile.** Profile MLP emits `(γ, β)` per
  layer; each layer's activations → `γ ⊙ h + β`. The pitcher conditions *the whole
  network*, not one token others must attend back to. Most principled
  "transformer = LSTM-with-h0" move; standard in conditional generation. Needs an
  ADR.
- **Cross-attention to a profile/context memory.** Add a cross-attention sublayer
  per block attending to a small fixed memory (pitcher token, batter token,
  game-context token, maybe a few more). Keeps causal self-attention over pitch
  tokens, but the context is always-available in *every* layer — the model can't
  "lose" who's pitching deeper in the stack. Perceiver / encoder-decoder pattern.
  Needs an ADR.
- **Pack the pitcher's prior at-bats vs this batter (or this game) into the
  sequence,** with the cross-AB block mask controlling visibility. The model sees
  "he already threw this guy three fastballs last PA" (times-through-order signal).
  Needs an ADR — changes the conditional-independence structure the causal layer
  assumes — but the LSTM doesn't get this either, so it's a place to *beat* the
  LSTM, not just match it.

---

## Tier 3 — pragmatic "make the number good"

- **Distill from the LSTM.** Add a KL term to the LSTM's soft pitch-type
  distribution alongside the hard-label loss. Transfers whatever the LSTM learned
  that the transformer can't extract from data alone. **Fully compatible with the
  counterfactual machinery — see the box below.** Distill from a *cross-fit*
  teacher (fold-k student ← fold-k LSTM) so the student doesn't inherit
  overfitting. Reach for this *after* Tier 1/2 (needing a teacher is a hint the
  inputs are still wrong).
- **LSTM-predictions-as-an-input-feature.** Feed the LSTM's per-pitch 7-float
  type distribution to the transformer as an input. Even simpler than distillation
  (no soft-label training). A hack, but it works.

> ### Does distillation break the generative / counterfactual capability? No — it helps.
> Distillation only trains the `type` propensity head to mimic the LSTM. It does
> *not* touch the result head μ̂(y|a,h), the two-stage interventional architecture
> (result head reads `hidden[t-1].detach()` + the intended action — that's what
> makes "swap the pitch, watch the result move" work), or the AB-outcome head —
> those keep training on their own losses. A counterfactual rollout = set the
> intervened action a' → μ̂ predicts P(y|a',h) → for *subsequent* pitches, sample
> from π̂ to continue the at-bat → map the terminal state to run value.
> Distillation makes that π̂ *better-calibrated to the (more accurate) LSTM*, so
> the non-intervened parts of the rollout get *more* realistic, not less.
> Limitations: (1) the LSTM is type-only, so you can only distill the *type*
> head — zone/velo/spin still train on their own losses (and still matter for
> sampling a realistic pitch in a rollout); (2) for cross-fit validity, distill
> from a cross-fit teacher. No leakage introduced (the LSTM was trained leak-free
> on ≤2023); the doubly-robust setup is preserved.

---

## Tier 4 — the safety net: decouple the nuisances

"One transformer = both π̂ and μ̂" (ADR 007) was an *elegance* choice, not a
*correctness* requirement. If PitchGPT's propensity head stays weak, use the
**best-available π̂** — the LSTM, XGBoost, or the n-gram baseline, whichever
calibrates best — as the propensity model, and keep PitchGPT as μ̂ and for the
rollouts. AIPW is fine with π̂ ≠ μ̂ (they just need to be cross-fit), and
"specialized propensity model + transformer outcome model, cross-fit" is standard
in the deep-AIPW literature — fully defensible. **The causal layer is not held
hostage by the gap.** If everything else underdelivers, the project still ships,
just with a less-elegant nuisance setup. Stand this path up *in parallel* from
the start so the causal layer can proceed regardless of how the prediction race
resolves.

---

## The reframe to keep in your back pocket

PitchGPT's job in this project is *not* to be the best next-pitch predictor — it's
to be a coherent generative model of an at-bat that supports interventions. A
class-conditional generative model is *expected* to lose to a specialized
discriminative classifier on raw classification accuracy — textbook trade-off,
not a failure. The bar isn't "match the LSTM exactly"; it's "close enough that
(i) the causal estimands are trustworthy and (ii) the gap is explicable by
generative-vs-discriminative." 0.46 fails that bar (propensity scores too noisy
for confident positivity gating). Mid-0.50s+ passes it — a ~10-point gap to the
LSTM is then a footnote, not a crisis.

---

## Recommended order (if the retrains land in the 0.40s)

1. Re-diagnose (Step 0).
2. Tier 1: concat-vs-sum embeddings + more epochs.
3. If still short → Tier 2: FiLM or cross-attention (pick based on the diagnostic).
4. *In parallel from day one* → Tier 4: stand up the decoupled-π̂ path so the
   causal layer proceeds regardless.
5. Tier 3 (distill) as a "make the headline number presentable" finisher if
   needed.

If the retrains land in the mid-0.50s+ → done; Tier 1 polish optional, the rest
unnecessary.

---

## Pointers

- ADR 009 (`docs/decisions/009-per-pitch-arsenal-feature.md`) — the (b) change,
  alternatives considered, leakage analysis.
- ADR 007 (`docs/decisions/007-nuisance-decoupling.md`) — the "one model, both
  nuisances" decision (which Tier 4 relaxes).
- Skills: `pitchgpt-model` (architecture conventions, what changes need ADRs),
  `eval-protocol` (what "good enough" means — calibration > top-1, the held-out-
  pitcher cohort), `causal-layer` (what π̂/μ̂ quality the AIPW pipeline needs).
- Diagnostic scripts: `scripts/investigate_phase_b.py` (clean full-val eval +
  per-position breakdown), `scripts/diagnose_propensity.py` (ablation arms),
  `scripts/run_simple_baselines.py` (the baseline floor), `scripts/calibrate_pitchgpt.py`
  (temperature scaling — run on every new checkpoint).
