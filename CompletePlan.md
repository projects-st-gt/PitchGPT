# PitchGPT — Complete Plan

End-to-end roadmap for the project. Items are numbered for reference. Each
phase has a clear gate condition before the next one starts. Estimates are
rough wallclock for a single engineer; "compute" is GPU/CPU time when it
matters separately from human work.

This document is the index. Specifics live in `docs/decisions/` (ADRs),
`.claude/skills/` (per-area skills), and `CLAUDE.md` (hard rules).

---

## Phase 0 — Decisions (DONE)

ADRs 001–007 locked in `docs/decisions/`. See those files for the full
reasoning behind each. Summary:

- **001 — Treatment granularity:** pitch type × 5-zone (`up`, `down`, `arm-side`, `glove-side`, `out-of-zone`)
- **002 — Positivity threshold:** τ = 0.01 single-step, ESS ≥ 30 multi-step, empirical re-calibration
- **003 — Confounder set:** baseline state + catcher, fatigue, TTO, umpire, leverage, ballpark, days rest, plus temperature, recent-form windows
- **004 — Run-value definition:** RE24 + count-state value, xwOBA-on-contact, Tango wOBA-to-runs
- **005 — Tipping metric:** batter-observable PitchGPT, T_start = surprisal − 5-start rolling baseline
- **006 — Cross-fit blocks:** K=5, `game_pk`-blocked, season-stratified
- **007 — Nuisance decoupling:** stop-gradient between heads on shared trunk; both heads update trunk; two-separate-models comparison as Phase 9 sanity check

---

## Phase 1 — Data (in progress)

**Gate to next phase:** `make data-sanity` passes; `data/processed/` and
`data/profiles/` exist with leakage tests green.

### Done
- Repo scaffold (pyproject, Makefile, .gitignore, dirs)
- `data/extract_statcast.py` with checkpoint + resume, day-files under `{year}/`
- `data/harmonization.py` (canonical 7 pitch types)
- `data/extract_game_metadata.py` (MLB Stats API scraper for umpire + weather)
- 22/22 unit tests pass
- Statcast extraction running — 2017-03-15 → today, ~5 hour ETA

### Doing tonight
1. **Skill drift cleanup** — update `.claude/skills/statcast-pipeline/SKILL.md` to match what we built: per-day file format under `{year}/`, the umpire-100%-null finding and MLB Stats API as the secondary source, ADR 003 recent-form windows, ADR 004 run-value formula refinement.
2. **Preprocessing layer** — `data/zones.py` (ADR 001 5-zone tagging), `data/preprocess.py` (harmonization + zone tagging + type-relative velocity bins). Consumes completed-day parquets so it does not block on the running pull.
3. **`data/run_value.py`** — RE24 table per ADR 004, computed from training data only, frozen for evaluation.
4. **`data/player_profiles.py`** — pitcher and batter profiles with the strict no-leakage discipline from ADR 003. Synthetic-AB leakage test must be green.

### Blocked on extraction completing
5. `make extract-meta` — runs the staged scraper, ~8–10 hours at 1 req/sec.
6. **`make data-sanity`** — pitch counts vs. Savant published totals (within 1%), pitch-type histogram vs. league (within 0.5pp), no-NaN check on `release_speed` / `plate_x` / `plate_z`, profile-leakage test green, held-out-pitcher cohort non-empty.
7. **`data/dataset.py`** — `AtBatDataset` PyTorch class with packing, attention masking, padding. Per-item dict shape per the `statcast-pipeline` skill.

---

## Phase 2 — Baselines (the discipline gate)

**Gate to next phase:** every baseline has a row in the eval table with top-1
/ top-3 accuracy, ECE, reliability diagram, and held-out-pitcher generalization
numbers. PitchGPT is not allowed to be evaluated until baselines exist.

8. **n-gram baseline** per (pitcher, count) state. Smoothing required.
9. **Retrieval baseline** k-NN on flat-feature state vector.
10. **XGBoost baseline** on harmonized + engineered flat features (the strong baseline; expected to land within 2–3 points of PitchGPT on top-1).
11. **Small LSTM** as the sequence-model baseline.
12. **Calibration tuning** — Platt scaling and isotonic regression for each baseline. Skipping this means the comparison to PitchGPT-with-calibration is unfair.
13. **Eval pipeline** — bootstrapped CIs, ECE, reliability diagrams, held-out-pitcher cohort, top-k tables. Lock these numbers as the floor.

---

## Phase 3 — PitchGPT model

**Gate to next phase:** PitchGPT trained on Tiny and Small sizes, both pass
calibration tuning, eval table extended to include them. No causal layer
work begins until both heads are calibrated on held-out data.

14. **Factored embeddings** (pitch type × zone × velo bin × spin axis × …, additive). Sound engineering, not novelty.
15. **Transformer backbone**, attention masking that blocks cross-at-bat attention when packing.
16. **Two-head architecture** with stop-gradient between heads per ADR 007. Both heads contribute to trunk via their own losses; loss combined as `λ_π·L_π + λ_μ·L_μ` (default λ = 1; rebalance on calibration drift).
17. **Training loop** — AdamW, label smoothing, mixed precision, gradient clipping, cosine LR with warmup.
18. **Tiny size first** as a sanity-check size; **Small only after Tiny calibrates clean.**
19. **Per-head calibration tuning** — temperature scaling on held-out validation.
20. **Three evals:** standard accuracy, calibration (ECE + reliability diagrams), held-out-pitcher generalization (debut-in-2024 cohort).

---

## Phase 4 — Causal layer (methodological centerpiece)

**Time investment:** the largest engineering chunk. Plan accordingly. This is
where the project's contribution actually lives.

**Gate to next phase:** AIPW and g-computation estimates agree within standard
error on a canonical query set; positivity gating refuses out-of-support
queries; sensitivity tools produce E-values; negative-control passes (no
effect detected on outcomes that should not be affected).

21. **Cross-fit orchestration** — K=5, `game_pk`-blocked, season-stratified per ADR 006. 5× training cost — this is the dominant compute cost of the project.
22. **G-computation rollout** — autoregressive sampling through μ̂ and π̂ to terminal AB state, mapped to run value via ADR 004 layer. Monte-Carlo over N rollouts.
23. **AIPW estimator** with cross-fitting, weight trimming at 1/τ = 100.
24. **Positivity gating** — single-step τ = 0.01 floor, multi-step ESS ≥ 30 floor with rollout truncation. Trust gauge bands per ADR 002.
25. **Sensitivity tools** — E-values per estimated effect (single-step only; flag the multi-step extension limit explicitly), negative controls (e.g., predict next batter's PA outcome — should produce null effect).
26. **Validations** — calibration of μ̂ and π̂, AIPW-vs-g-comp agreement, natural-experiment matching (paired ABs with similar π̂ but different observed action), domain spot-checks (does the model rediscover known wisdom: low-and-away breaking balls in 0-2 to LHB beats heart-of-zone fastballs).

---

## Phase 5 — Tipping

**Gate to next phase:** at least 2 of the 4 validations correlate as
predicted; metric definition is locked; failure-mode reporting is honest
(if 0 validations correlate, write that up too).

27. **Batter-observable PitchGPT variant** — same architecture, locked feature mask from ADR 005. Trained as a separate run.
28. **T_start computation** — per pitcher-start, mean batter-observable surprisal minus 5-start rolling baseline. Flag T_start > 1.5σ above 0.
29. **Four validations:**
    - Subsequent-start xwOBA-against and whiff rate (next 1–3 starts).
    - Within-start time-through-order amplification (3rd-time penalty bigger than baseline).
    - Batter behavior (more aggressive early-count swings, better contact on guessed types).
    - Case studies (Darvish 2017 WS and similar should fall in flagged set).

---

## Phase 6 — Recommender

**Gate to next phase:** recommender produces in-support recommendations only;
comparison cells filled (vs actual, vs EV-naïve, vs heuristic).

30. **Trust-region-restricted causal recommender** — argmax over `{a : π̂(a|H) > τ}` of E[run-value | do(A = a), H], with cross-fit CIs.
31. **Comparisons** — actual pitch thrown, XGBoost-EV-naïve recommendation, per-pitcher heuristic. Interesting cells: high π̂ + high μ̂ confidence + recommender disagrees with actual.

---

## Phase 7 — Demo

**Gate to next phase:** every demo query either renders a confident answer
with trust gauge + CI + E-value + sensitivity slider, or refuses with an
explanation. No silent extrapolation. Frontend uses Inter (not SF Pro), and
pitch types are encoded with glyph + color (color-only fails ~8% of male users).

32. **FastAPI backend** + pre-computed cache for the demo at-bats.
33. **React + TS + Tailwind frontend** per the `frontend-system` skill.
34. **"Rewrite the At-Bat" page** — trust gauge, cross-fit CI, E-value, sensitivity slider, supported-alternatives panel, refusal-with-explanation for out-of-support queries.
35. **Tipping page** — three signals, case studies, downstream-correlation evidence.
36. **Recommender page** — top-k in-support recommendations with CIs, divergence-from-actual highlighting.
37. **Strike-zone visualization, pitch timeline, trust gauge** as named components in the design system.

---

## Phase 8 — Writeup

38. **Methods document** — full pipeline, identification assumptions, validation hierarchy, limitations. Frame as a case study in sequential causal inference with autoregressive sequence models. Note transferability to medical sequences, education paths, any domain with sequential decisions and rich observable confounding.
39. **Pre-registration of baselines and primary estimands** before final eval runs (so "discoveries" against pre-registered baselines are real).
40. **Honest limitations section** — every unmeasured confounder named (catcher's pre-pitch read, scouting reports, mechanical state, sign-stealing). E-values quantify the strength of unmeasured confounding that would explain effects away.

---

## Phase 9 — Stretch

41. **Two-separate-models AIPW comparison** for one canonical estimand (ADR 007's empirical decoupling check). Within standard-error agreement = shared-trunk validated; divergence = report both with the gap as a measured limitation.
42. **Causal forest** as a non-parametric sanity check on terminal-pitch ATEs.
43. **Released checkpoints + code + demo.**
44. **"Construct your own state" mode** in the demo for exploration beyond curated queries.

---

## What this project is not — repeated for emphasis

- Not a flat-vocabulary GPT with a "novel architecture" claim.
- Not a "counterfactual" demo unless the causal machinery is actually running.
- Not era-mixing, cross-sport, or video/multimodal work.
- Not chasing top-1 accuracy as the headline metric. Calibration, trust-region behaviour, and the causal estimands are what matter.

The methodological contribution is sequential causal inference done correctly.
The architecture is just the backbone that makes it tractable. Don't hype the
backbone; defend the methodology.
