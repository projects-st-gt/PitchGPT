# ADR 014 — small-v8: Continuous-Location MDN Head + Autoregressive Pitch Factorization

**Status:** Proposed (2026-06-05)
**Branch:** `hitter-swing-model`
**Related:** ADR 001 (treatment granularity = pitch type), ADR 007 (nuisance decoupling), ADR 013 (v7 type-conditioned heads)
**Spec:** `docs/superpowers/specs/2026-06-05-small-v8-design.md`

## The question, in plain English

The pitchGPT+cascade simulator lost the per-PA backtest: walk rate was 5.2%
predicted vs 9.3% real, and log-loss was worse than both the lookup (1.4435)
and the baseline (1.4631). The root cause was isolated decisively in
`scripts/hitter/diagnose_glue_isolation.py`: the simulator feeds the batter
cascade each pitch's location as its **zone centroid**, not a real plate
position. Should the model be extended to emit a real, continuous (plate_x,
plate_z) position that the cascade can consume directly?

And while we are retraining anyway: should the pitch factorization from ADR
013 — currently type → (zone, velo, spin in parallel, all conditioned on
type) — be completed to a **full autoregressive chain** so that each factor
conditions on all previously sampled factors, not just the type?

## Why this matters

The out-of-zone zones are coarse. Real `|plate_x|` spreads from 0.83 to 4 ft
(std 0.55 ft), and 37% of out-of-zone pitches land more than 1.1 ft off the
corner. But every zone has only one centroid, which the current simulator
places at roughly 0.9 ft — just off the corner. So every out-of-zone pitch
looks borderline, the batter chases, there are too few takes, too few balls,
and too few walks.

The deficit compounds because a walk needs four consecutive balls. If the per-
pitch ball rate drops from the real 0.354 to the simulated 0.303, the four-
ball joint probability shrinks by (0.303/0.354)^4 ≈ 0.54 — which brings 9.3%
walks down to roughly 5%, matching the observed 5.2% exactly.

**Velo and spin approximations were tested as well and had no effect. Location
is the whole cause.** The pitchGPT zone head is well-calibrated on real data
(no leakage identified). The issue is purely in the simulator's glue: it
discards the zone model's implied within-zone position.

A model that emits a real (plate_x, plate_z) sample eliminates this gap at
the source rather than trying to post-hoc correct the centroids.

## Decision 1 — Add a continuous-location MDN head

Add a **mixture-of-Gaussians location head** (MDN) over `(plate_x, plate_z)`,
as the final factor in the autoregressive chain (Decision 2):

- **K = 5** diagonal-covariance 2D Gaussians. Outputs: K mixture weights,
  K 2D means, K 2D log-stds. Start K small to reduce collapse risk; revisit
  only if the distributional check (Consequence section) shows poor coverage.
- **Trained** by mixture negative-log-likelihood on the real `(plate_x,
  plate_z)` of every pitch. Targets come directly from Statcast — no
  synthetic positions.
- **Conditioned** on all previously sampled factors per the chain in
  Decision 2 (type, zone, velo, spin) as well as the trunk hidden state.
  Because zone is in the conditioning set, the MDN's samples are structurally
  consistent with the discrete zone label the result head and cascade already
  use.
- **At inference:** sample a (plate_x, plate_z) from the MDN; clip to a sane
  physical range (|x| ≤ 2.5 ft, z ∈ [0, 5] ft). Feed the real sampled
  position to the cascade. The zone-centroid lookup is retired.
- **Back-compat:** gated behind a `location_mdn` config flag, default `False`.
  Pre-v8 checkpoints load unchanged.

## Decision 2 — Full autoregressive factorization via head-level conditioning

ADR 013 introduced type-conditioned execution heads: zone, velo, and spin are
each conditioned on the sampled type. Decision 2 extends this to a **full
chain**:

```
type → zone | type → velo | type, zone → spin | type, zone, velo → location-MDN | type, zone, velo, spin
```

Each execution head's MLP reads the **trunk hidden state plus the embeddings
of all already-sampled factors** — not just the type. This supersedes ADR 013's
trunk re-forward for zone conditioning; only the small head MLPs are re-
evaluated as each factor is drawn during rollout, so there is **no full
transformer re-forward per factor** and rollout speed stays comparable to v7.

**Training** remains one forward pass: teacher-forced prior factors are
concatenated into each head's input alongside the trunk hidden state. No
changes to the trunk architecture or the type head.

**Convention discipline:** the same named-constant and print-a-named-number
rules from ADR 013 apply here. Every smoke test must print at least one named
numerical output (e.g., a sampled (x, z) from the MDN on a real AB, or
π̂(FF) = 0.42 on AB X) — "tests pass" without a named number is rejected.

## Decision 3 — Keep the 13-zone head unchanged

The discrete zone head (13 classes, including the 4 out-of-zone quadrants) is
well-calibrated on real data. The result head already conditions on
`zone_embed`; the cascade's `in_zone` feature already derives from it. Nothing
here is broken.

The MDN's role is to fill in **where within the zone** — it does not replace
the zone head's discrete probability or its calibrated uncertainty. Zone is
sampled first (the zone head runs as in v7); the MDN conditions on that
sampled zone and produces a real position consistent with it.

## Decision 4 — Drop the at-bat-outcome head

The `ab_outcome_per_pos` head is **redundant** with the cascade + RE24 run-
value table, which is the production μ̂. Removing it simplifies training,
reduces the loss landscape's dimensionality, and eliminates one source of
gradient competition. The head is removed (config flag `ab_outcome_head`,
default `True` in v7, set to `False` in v8 configs). v7 checkpoints that
include it still load.

## Decision 5 — Keep the per-pitch result head as a light auxiliary

The per-pitch result head (ball / called-strike / swinging-strike / foul / in-
play) is retained, but its loss weight is reduced from **1.5 → 0.3**. The
rationale:

- It likely helps the shared representation — the trunk learns to track count,
  leverage, and batter tendencies partly through the result signal.
- It preserves the transformer-only outcome path as a comparison baseline.
- Reducing its weight keeps it from dominating over the MDN-NLL and the
  discrete heads, which are the model's primary signals.

The **cascade remains the production μ̂**; the light result head is an
auxiliary, not the answer.

## Decision 6 — Enable rare-pitch-type tuning

Enable the existing `type_focal_gamma` / `type_class_weight_alpha` options on
the type head to improve recall on low-frequency types (splitter, eephus,
etc.). These are already implemented in `scripts/train_pitchgpt.py`; this
decision flags them as active for v8. Specific hypervalues are set in the v8
training config and measured in the fold-0 eval.

## Alternatives considered

- **Post-hoc centroid correction** (add a per-zone noise model, tune on real
  plate data). Rejected: it patches the glue, not the model. A trained MDN
  produces a distribution learned from every pitch in the training set, gives
  a principled distributional calibration check, and generalizes to pitcher-
  specific tendencies within a zone.
- **Four-quadrant out-of-zone centroids** (split the single OZ centroid into
  one per OZ quadrant). Tested implicitly by the diagnosis — velo/spin showed
  no effect; the core issue is the continuous spread within each zone, not
  just OZ. Even a finer centroid grid leaves 0.55 ft of std unexplained.
- **Full per-factor transformer re-forward** (the trunk runs once per factor in
  the chain, as in a fully sequential AR model). More expressive but ~5×
  slower at rollout and not needed: the cross-factor correlation beyond "given
  type and zone, how fast?" is small. Head-level conditioning achieves the
  same causal coherence for rollout at much lower cost.
- **A separate standalone location model** (trained independently). Rejected
  for the same reason ADR 007 rejected decoupling π̂ and μ̂ — the shared trunk
  is the point, and a separate model cannot condition on the trunk's
  uncertainty about the current state.

## Retrain plan

This is a retrain. The architecture is locked once this ADR is accepted.

1. Dataset: emit real `(plate_x, plate_z)` as MDN targets
   (`model/pitchgpt_dataset.py`).
2. Model: MDN head + head-level autoregressive conditioning + drop AB head +
   result-head weight (`model/heads.py`, `model/embeddings.py`, config).
   Unit-test MDN NLL and sampling on synthetic data before training.
3. Train **`small-fold0-v8`** on Modal (~10h). Validate before any K=5 cross-
   fitting — fold-0 only until the backtest gate passes.
4. Calibrate: temperature-scale all discrete heads on val (2024H1); for the
   MDN run a distributional check — sampled (x, z) should reproduce real per-
   zone spread and the real per-pitch ball rate when fed to the cascade.
5. Simulator glue: replace centroid lookup with MDN sampling; feed native
   sampled velo/spin to cascade (`hitter/rollout.py`).
6. Verify: local per-pitch ball rate ~0.35 → backtest gate (walk rate ~9%,
   log-loss < lookup 1.4435 and < baseline 1.4631) → standard evals.

Only after the gate passes: regenerate matchup cards and re-examine cross-
fitting (folds 1–4).

## Risks

- **MDN instability** (mode collapse, NaN in the mixture NLL): mitigated by
  K=5 (small), careful init (means spread over the plate, log-std floor),
  gradient clipping, and a unit test on MDN NLL and sampling before training.
- **Exposure bias** in the autoregressive chain: heads train on true prior
  factors but receive sampled prior factors at rollout. Known autoregressive
  tradeoff; the rollout was already imperfect in v7 for this reason. Measured
  in the fold-0 backtest.
- **If the backtest still fails** after v8: location was not the only cause.
  The rule is to stop and reassess — not to pile on fixes.

## Consequences

- The simulator samples a real (plate_x, plate_z) per pitch; out-of-zone
  pitches regain their true spread. The centroid lookup (`zone_centroids.json`)
  is retired in the simulator path.
- All new config flags default to v7 behavior (`location_mdn=False`,
  `ab_outcome_head=True`, `result_loss_weight=1.5`) so existing v7 checkpoints
  reload without code changes.
- The causal "head" mode in g_computation retains the light result head; the
  cascade remains the production μ̂. The at-bat-outcome head is absent from v8
  rollouts.
- Skill updates required: `pitchgpt-model` (MDN head, full factorization chain,
  result-head weight change, dropped AB head), `causal-layer` (rollout now
  samples real positions), `statcast-pipeline` (MDN targets = real
  plate_x/plate_z, leakage rule unchanged).
- A new named numerical check is mandatory for every smoke test: sampled
  (plate_x, plate_z) from the MDN on a real AB, plus the per-pitch ball rate
  from the local backtest — "passes" without those numbers is rejected.
