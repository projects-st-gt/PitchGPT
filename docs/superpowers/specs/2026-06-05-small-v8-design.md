# small-v8 — pitchGPT with a continuous location head (design)

**Date:** 2026-06-05
**Branch:** `hitter-swing-model` (do NOT merge to main yet)
**Status:** spec — pending user review, then `writing-plans`
**Needs an ADR:** yes — `docs/decisions/014-v8-location-mdn-autoregressive-factorization.md`
(model-architecture change per CLAUDE.md workflow rule).

## 1. Why (root cause recap)

The pitchGPT+cascade simulator **lost the per-PA backtest** (BB 5.2% vs 9.3%
real; worse than lookup 1.405 and baseline 1.452). Root cause, isolated
decisively (`scripts/hitter/diagnose_glue_isolation.py`): the simulator feeds
the batter cascade each pitch's location as its **zone centroid**, not a real
spot. The 4 out-of-zone zones are coarse (real |plate_x| spans 0.83→4 ft, std
0.55; 37% are >1.1 ft out) but the centroid collapses each to one borderline
point (~0.9 ft, just off the corner). Every out-of-zone pitch then looks
borderline → the batter chases → too few takes → too few balls → too few walks.
A small per-pitch ball deficit compounds because a walk needs 4 balls:
(0.303/0.354)⁴ ≈ 0.54 → 9.3% × 0.54 ≈ 5.0% ≈ observed 5.2%.

The pitchGPT zone HEAD is fine (well-calibrated on real data, no leakage).
**velo/spin approximations had no effect; location is the whole effect.** The fix
is to make the model emit a real, continuous location the cascade can consume
directly — and, while retraining, complete the autoregressive pitch
factorization and tidy the now-unused heads.

## 2. Goal & success criteria

A pitch-generation model whose *simulated* at-bats reproduce real outcome rates.

- **Primary (the real test):** small-v8 + cascade **beats lookup (1.405) and
  baseline (1.452)** on the per-PA backtest (`scripts/hitter/run_backtest_modal.py`),
  with walk rate restored to ~9% and per-pitch ball rate ~0.35.
- **Secondary:** next-pitch type/zone top-1 **not worse** than v7 (type ~0.485);
  all heads calibrated (ECE reported); held-out-pitcher cohort reported
  separately (per `pitchgpt-model` + `eval-protocol` skills).

Note: top-1 accuracy is explicitly **not** the headline (CLAUDE.md); calibration
and realistic simulated sequences are. Capacity is not the lever — small-v7
(0.485) ≈ tiny-v7 (0.478) on type top-1.

## 3. Architecture changes (the ADR content)

### 3.1 Full autoregressive pitch factorization (head-level conditioning)
Sample order within a pitch:
```
type → zone | type → velo | type,zone → spin | type,zone,velo → location-MDN | type,zone,velo,spin
```
Each execution head's MLP takes the **trunk hidden state + the embeddings of the
already-sampled factors** (the same pattern the two-stage result head already
uses, `model/heads.py`). This **head-level conditioning supersedes ADR-013's
trunk re-forward** for zone: at sampling we recompute only the small head MLPs as
each factor is drawn — no full transformer re-forward per factor — so rollout
stays fast. (Training is one forward: heads read teacher-forced prior factors.)

### 3.2 MDN location head (new)
- A **mixture of K 2D Gaussians** (start K=5, diagonal covariance) over
  `(plate_x, plate_z)`, conditioned per §3.1. Outputs: K mixture weights, K means
  (2D), K log-stds (2D). Trained by **mixture negative-log-likelihood** on the
  real `(plate_x, plate_z)` of each pitch.
- **Keep the 13-zone head unchanged** (well-calibrated; the result head conditions
  on `zone_embed`; the cascade's `in_zone` feature derives from it). The MDN fills
  in *where within the zone*. The MDN conditions on the sampled zone (§3.1), so
  its samples are consistent with the zone.
- **Inference:** sample type → zone → (x,z) from the MDN; clip (x,z) to a sane
  plate region (e.g. |x|≤2.5, z∈[0,5]). Feed the real (x,z) to the cascade.

### 3.3 Head cleanup
- **Drop** the at-bat-outcome head (`ab_outcome_per_pos`) — redundant with the
  cascade + RE24 run-value table.
- **Keep** the per-pitch result head as a **light auxiliary** (low loss weight,
  e.g. 0.3 vs the current 1.5) — likely helps the shared representation and
  preserves the transformer-outcome path as a baseline; the **cascade remains the
  production μ̂**.

## 4. Training changes
- Same temporal split (train ≤ 2023; val 2024H1; test 2024H2+). New config flags
  (all default off so v7 checkpoints reload): `location_mdn`, `mdn_components`,
  `autoregressive_exec_heads`, `ab_outcome_head` (default on; v8 sets off),
  `result_loss_weight`.
- **Rare-type tuning:** enable `type_focal_gamma` / `type_class_weight_alpha`
  (existing options) for better tail (splitter, etc.) recall.
- Loss = type (focal) + zone + velo + spin + **MDN-NLL** + light result; **no AB
  loss**. Per-factor weights tuned in the plan; MDN-NLL scaled to be commensurate.
- **Start fold-0 only** (like v7) to validate before any K=5 cross-fitting.
- Train on Modal (~10h, per `pitchgpt-model`); run name `small-fold0-v8`.

## 5. Calibration
- Temperature-scale type/zone/velo/spin/result on val (existing
  `scripts/calibrate_pitchgpt.py`). For the **MDN**, calibration is distributional:
  validate that **sampled** (x,z) reproduce the real per-zone spread and the real
  per-pitch ball rate when run through the cascade (a sampled-vs-real KS/quantile
  check), and tune a temperature on the mixture if needed.

## 6. Simulator integration (glue — no retrain)
In `hitter/rollout.py:build_step_features` + the `g_compute` hitter path:
- Replace the **zone-centroid lookup** (`zone_centroids.json`) with a **sample
  from the model's MDN** for the cell's sampled pitch.
- Feed the model's **native sampled velo / spin_rate / spin_axis** to the cascade
  instead of per-type means + zero spin axis.
- Keep `in_zone` from the sampled zone.

## 7. Verification (in order)
1. **Local per-pitch** (`diagnose_pitch_outcomes.py`, active-masked): ball rate
   ~0.30 → ~0.35; swing back toward ~0.485.
2. **Backtest on Modal** (`run_backtest_modal.py`, n=800, cap 10): walks ~9%,
   log-loss **< 1.405 (lookup) and < 1.452 (baseline)**. This is the gate.
3. **Standard model evals:** type/zone top-1 vs v7; ECE/reliability per head;
   held-out-pitcher cohort.
4. Only after the gate passes: regenerate matchup cards with small-v8 and
   re-examine the Merrill-type levels.

## 8. Risks & mitigations
- **MDN instability** (mode collapse / NaN): few components (K=5), careful init
  (means spread over the plate, log-std floor), gradient clipping, and a unit
  test on the MDN NLL + sampling on synthetic data before training.
- **Rollout speed:** head-level conditioning avoids per-factor transformer
  re-forwards; sampling cost stays ~v7.
- **Retrain cost** (~10h Modal): fold-0 first; don't cross-fit until the gate
  passes.
- **If the backtest still fails** after v8: location was not the only cause —
  stop and reassess (don't pile on fixes).

## 9. Hard-rule compliance
- Real Statcast only (MDN target = real plate_x/plate_z); no placeholders.
- Temporal splits unchanged; checkpoint records train range.
- Calibration is a primary metric (ECE per head + MDN distributional check).
- Architecture change gated behind an **ADR** (`014-...`), linked in the PR.
- Named-number discipline in every smoke test (print π̂(FF), a sampled (x,z),
  per-pitch ball rate — not "passes").

## 10. Build order (→ writing-plans expands)
1. ADR-014 (architecture decision) — write + commit first.
2. Dataset: emit real `(plate_x, plate_z)` as MDN targets (`model/pitchgpt_dataset.py`).
3. Model: MDN head + head-level autoregressive conditioning + drop AB head +
   result-head weight (`model/heads.py`, `model/embeddings.py`, config). Unit-test
   MDN NLL/sampling on synthetic data.
4. Training: loss wiring + rare-type tuning; train `small-fold0-v8` on Modal.
5. Calibrate (incl. MDN distributional check).
6. Simulator glue: MDN sampling + native velo/spin in `hitter/rollout.py`.
7. Verify: local per-pitch → backtest gate → standard evals.

Each step: TDD where there's a pure function, print named numerical outputs,
commit per step, keep `docs/ContextSwitcher.md` updated live.
