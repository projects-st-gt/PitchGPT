# v1c-cl: Closed-loop fine-tuning of tiny-v1c-base (CAT-K-adapted)

**Goal:** Remove the genuine closed-loop pitch-type drift in the V2 rollout
(+4–8pp FF at hitter counts after pitcher- and context-matching; zero at the
first pitch) by fine-tuning the existing `tiny-v1c-base` checkpoint on its
own sampled inputs, tethered to the real sequences. No architecture change,
no scale change.

**Gate:** Same harness, n_paths=1500 standard:
`python -m scripts.hitter.run_backtest_v2_modal --n 800 --n-paths 1500`.
Targets: log-loss < 1.4435 (same-sample lookup), BB% within 1pp (currently
PASS — must not regress), and the marginals diagnostic's hitter-count FF
delta < +3pp matched/context-corrected (currently +4–8pp).

**Baseline being improved:** V2@1500 = 1.4521 (beats baseline 1.4631, first
ever; 0.009 behind lookup). Remaining deficit: K/out per-PA placement +
in-play contact-quality skew (xwOBA-map high-third 0.45 vs 0.33), both
plausibly downstream of the hitter-count FF drift.

---

## Evidence base (ml-research run, 2026-06-09/10)

| Pillar | Source | What it gives us |
|---|---|---|
| Closed-loop SFT works on tokenized AR sim models | CAT-K, Zhang et al. CVPR 2025 Oral (arXiv:2412.05334, code: NVlabs/catk) | Detached rollout + top-K-closest-to-GT action selection + CE on recovery actions; SMART-7M+CAT-K (0.7635) beat SMART-101M (0.7614); cost ≈ 24% of pretraining |
| Detach at unroll boundaries; never backprop through sampling | Pushforward trick, Brandstetter et al. ICLR 2022 Spotlight (arXiv:2202.03376); confirmed by CAT-K's design | Detaching is principled even when differentiable; our XGBoost boundary forces it anyway |
| Noise injection cannot fix systematic bias | DART (CoRL 2017), turbulence benchmarking (arXiv:2309.01745) | Explains v9: random perturbation adds variance, doesn't correct directional drift; unrolled training stays stable where noise injection degrades |
| Curriculum unroll is the stable schedule | GraphCast (Science 2023): 1→12-step ramp | We ramp 1→full-AB |
| Exposure-bias gains may be small at short horizons | Duckworth et al. 2019 (length hypothesis), He et al. EMNLP 2021 (self-recovery, no compounding at T=20-100) | Honest expectation-setting: our T=4-6 is below all published regimes; gains are NOT guaranteed |
| Scheduled sampling is improper for distribution matching | Huszár 2015 (arXiv:1511.05101) | Biases toward factorized marginals — exactly wrong for simulation. The tether-to-GT (CAT-K) variant avoids the pathology by keeping targets = the real continuation |

**Known gaps (we are first):** T<15 horizons; mixed softmax+GMM heads; a
black-box ML environment in the loop. All three are implementation
decisions, not method blockers.

---

## Method: tethered top-K self-input fine-tuning

The key adaptation. CAT-K rolls the policy and steers each step to the top-K
candidate closest to ground truth, so the rolled state stays near the real
trajectory while inputs become model-generated. Our analog keeps the REAL
count/result progression as the scaffold (so the cascade is NOT needed in
the training loop) and replaces only the model-controlled inputs (type +
continuous) with the model's own tethered samples:

For each training AB (teacher batch), per step t (left to right):
1. Forward the prefix (positions 0..t-1 with already-substituted inputs).
2. Take the model's top-K type candidates at position t-1's prediction.
   - If the REAL next type is in the top-K → input the REAL type
     (tether hit; the model "would plausibly have thrown it").
   - Else → input the highest-probability candidate that preserves the REAL
     pitch's ball/strike character (in-zone vs out-of-zone by the real
     plate_x/z), falling back to the top-1 candidate. This is the
     count-consistency analog of CAT-K's Euclidean closest-state rule —
     it keeps the real count progression valid.
3. Continuous input at t: sample from the GMM conditioned on the substituted
   type, DETACHED (clamp + renormalize as in g_compute_v2). If the
   substituted type == real type, optionally keep the real continuous values
   (configurable; default = model sample, matching rollout conditions).
4. result_ids / count_state / outs / runners stay REAL (scaffold).
5. Loss at every position: CE against the REAL next type + GMM NLL against
   the REAL next continuous values (teacher targets, rolled inputs).
   Identical loss weights to base training (w_type=2.0, w_continuous=1.0,
   label smoothing 0.05).

All substituted inputs are detached (no gradient through sampling). The loss
gradient flows only through the final forward pass — implement as: one
no-grad pass to build substituted inputs sequentially, then one grad pass on
the substituted batch. (Two-pass scheme, same shape as Mihaylova & Martins
2019 but decoder-only and with the tether rule.)

**Why the real-scaffold variant first** (vs cascade-in-the-loop): keeps
training simple/fast (no XGBoost calls in the loop), keeps targets grounded
(no improper-objective pathology), and directly attacks the measured failure
(model inputs off-distribution at hitter counts). The full cascade-in-loop
variant is the escalation if this under-delivers.

### Hyperparameters

| Param | Value | Basis |
|---|---|---|
| K (top-K tether) | 3 (sweep 2–5 if needed) | CAT-K robust K=5–64 on 1000+ vocab; ours is 7 types so K=3 ≈ same fraction |
| Substitution prob p_sub | ramp 0→1 over first 1000 fine-tune steps | GraphCast curriculum analog |
| Fine-tune steps | ~2000–3000 (≈ 0.5 epoch) | CAT-K used ~16% of pretrain; ours ≈ 15–20% of 14.1K steps |
| LR | 3e-5 flat (= base lr_min) | fine-tune convention |
| Init | tiny-v1c-base checkpoint.pt (NOT calibrated — recalibrate after) | calibration is post-hoc |
| GMM component-collapse monitor | log mixing-weight entropy every eval | GraphCast blurring risk analog |

### Files

| File | Action |
|---|---|
| `scripts/finetune_v2_cl.py` | Create — fine-tune loop (two-pass substitution + standard loss) |
| `model/v2/` | NO changes |
| `modal_app.py` | Add `finetune_v2_cl_remote` (L4, ~1–2h) |
| `scripts/calibrate_v2.py` | Reuse unchanged on the fine-tuned ckpt |
| `tests/test_v2_cl_finetune.py` | Tether rule unit tests (real-in-topK, zone-character fallback, detachment — named numbers) |

### Evaluation ladder (in order; stop on regression)

1. Teacher-forced val: type top-1 must stay ≥ 0.46 (drop > 1pp = abort).
2. Calibrate (flat T; per-count temps optional — measured no-op).
3. Marginals diagnostic (matched pitchers, real handedness): hitter-count FF
   delta target < +3pp.
4. Full gate @1500 paths.

### Expected impact (honest)

Literature gives no short-horizon precedent. Mechanism reasoning: the drift
is per-step small (+3pp teacher-forced at 2-0) and amplified by input
distribution shift; training on tethered self-inputs directly closes that
shift. Plausible outcomes: FF drift → near teacher-forced level; in-play
quality skew partially corrects; gate moves from 1.4521 toward ≤1.4435 but
the K-placement deficit may persist (pitcher-side discrimination, corr 0.253
vs lookup 0.281 — possibly a profile/feature limit, not a training-regime
issue). BB% must be watched for regression in both directions.

### Scale decision (answered by the research)

NO new "base" tier and NO 44.8M small run. Evidence: Muennighoff NeurIPS
2023 (data-constrained allocation; excess-param decay R_N*≈5.3),
"Prescriptive Scaling Laws" 2026 (overfitting penalty zone at our N/U_D),
SMART NeurIPS 2024 (7M→101M = +0.3% on 70× our data; shipped the 7M),
TinyStories 2023 (low-entropy domains plateau small), our own tiny>27M
result. Defensible ceiling ≈ 15–20M, and ONLY after a low-rank conditioning
audit (rank-16/32/64 bottleneck on the 314-dim profile MLP — 59% of params;
LoRA-family evidence says high collinearity → low effective rank). Sequence:
v1c-cl fine-tune (this spec) → re-gate → only if passing AND saturating,
consider a ~15M "tiny-xl" with low-rank adaLN.
