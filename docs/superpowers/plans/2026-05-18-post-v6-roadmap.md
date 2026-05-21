# Post-v6 Roadmap: Causal Layer + Frontend Tabs + Live Tracker

**Date:** 2026-05-18
**Author:** sid + claude
**Status:** v6 model trained, calibrated, evaluated. Causal-layer skeleton exists; frontend scaffold exists.
**Successor of:** [v6 design](../specs/2026-05-17-v6-profile-features-design.md), [v6 plan](2026-05-17-v6-profile-features.md)

> **For a new session picking this up:** start by reading the "Starting context for a fresh session" section at the bottom of this doc. It has the minimum context needed to be productive on day one.

---

## 1. What v6 actually delivered

V6 is the conditional-arsenal pitcher-profile expansion (118 → 218 dims). Final checkpoint at `checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt`.

### V6 vs V5 — in-distribution validation (2024H1, 326,815 pitches)

| Head | V5 NLL | V6 NLL | V5 ECE | V6 ECE | V5 acc | V6 acc |
|---|---|---|---|---|---|---|
| **type** | 1.2123 | **1.2018** | 0.0077 | **0.0058** | 0.4805 | **0.4848** |
| zone | 2.3325 | 2.3297 | 0.0090 | 0.0115 | 0.2421 | 0.2429 |
| result | 1.2457 | 1.2454 | 0.0034 | 0.0030 | 0.5497 | 0.5497 |

### V6 vs V5 — postseason OOD (25,593 pitches across 86 games)

| Metric | V5 OOD | V6 OOD | Δ | Pi 2018 RNN | Ahn 2026 LLM (3B params) |
|---|---|---|---|---|---|
| 7-class type top-1 | 0.4311 | **0.4904** | +5.93 pp | — | — |
| 7-class NLL | 1.4840 | **1.1639** | -0.32 | — | — |
| 7-class ECE | 0.0260 | **0.0144** | -45% | not reported | not reported |
| Binary FF AUC | 0.6989 | **0.7579** | +5.9 pp | not reported | not reported |
| FF acc @ recall ≈ 0.79 | 0.5743 | **0.6451** | +7.08 pp | 0.633 | 0.637 |
| FF F1 @ recall ≈ 0.79 | 0.5463 | **0.5908** | +4.46 pp | 0.720 | 0.722 |
| Swing acc (marginal P from result head) | 0.7195 | 0.7245 | +0.5 pp | — | ~0.78 |

**Headline result:** V6's accuracy at the matched operating point (recall ≈ 0.79) **beats both Pi 2018 (RNN) and Ahn et al. 2026 (Llama-3.2-3B)** by 0.8-1.2 pp, despite being a 4.5M-parameter model. F1 gap to those baselines remains (-0.13) due to precision differences — likely an evaluation-protocol artifact we can't fully reconstruct without their code.

### Per-class trade-offs (in-dist)
- ✅ SL recall +2.8 pp, CU recall +1.1 pp, FS recall +1.1 pp, SI recall +1.0 pp
- ⚠️ FC recall -1.1 pp, CH recall -1.1 pp (minority class slot regressed slightly)
- ✅ FF over-fire reduced: pred_rate 0.3871 → 0.3795 (+7.57 pp over true → +6.81 pp)

V6 is the new default. The v5 baseline (`checkpoints_modal/tiny-fold0-1778792736/`) stays around for comparisons.

---

## 2. Existing scaffolding (do not rebuild)

These exist with substantial implementations. Read before duplicating effort.

### `causal/` — the causal layer skeleton (~2000 lines)

| File | Lines | Purpose |
|---|---|---|
| `nuisance.py` | 346 | Wraps a calibrated PitchGPT checkpoint as (π̂, μ̂). Applies stored temperatures by default. |
| `g_computation.py` | 630 | Rigorous g-computation forward rollouts. Count state evolves per baseball rules. |
| `aipw.py` | 404 | Doubly-robust per-unit + population AIPW with contrasts. |
| `crossfit.py` | 287 | K=5 cross-fitting, `game_pk`-blocked, season-stratified. |
| `positivity.py` | 217 | τ-banded gates (>0.05 GREEN / 0.01-0.05 YELLOW / <0.01 RED → REFUSE). ESS tracking. |
| `sensitivity.py` | 151 | E-values per VanderWeele & Ding 2017. |

These were almost certainly built against a v5 checkpoint and need a v6-integration pass.

### `inference/api.py` — FastAPI app (662 lines)

Endpoints:
- `GET /health`
- `GET /games?limit=200`
- `GET /at-bats?game_pk=...`
- `GET /ab-context?game_pk=...&at_bat_number=...`
- `POST /query` (causal queries)

Uses a module-level `NuisanceModels` cache. Loads val data lazily.

### `frontend/` — React + Vite + TS + Tailwind

`frontend/src/` has:
- `App.tsx` — root
- `Scoreboard.tsx` — game/AB picker
- `StrikeZone.tsx`, `StrikeZonePreview.tsx` — visualization primitives
- `api.ts`, `types.ts` — backend client

This is partial — additional tabs need to be added.

### `docs/decisions/` — 12 ADRs locked

001-treatment-granularity • 002-positivity-threshold • 003-confounder-set • 004-run-value-definition • 005-tipping-metric • 006-cross-fit-blocks • 007-nuisance-decoupling • 008-profile-cache-fold-awareness • 009-per-pitch-arsenal-feature • 010-situational-propensity-head • 011-concat-then-project-embeddings • 012-profile-film-conditioning.

Read 001, 002, 003, 006 first — they're the load-bearing ones for the causal layer.

### Skills (in `.claude/skills/` or equivalent)

See `CLAUDE.md` skill index. Load on demand:
- `causal-layer` — methodology specs
- `tipping-analysis` — T_start metric
- `eval-protocol` — baselines, calibration
- `frontend-system` — design tokens, motion specs, refusal UX
- `pressure-testing-claims` — pre-claim verification

---

## 3. The three paths

The user's intent: build Path A (frontend tabs 1-5) and Path B (causal layer completion) in parallel, plan for Path C (live tracker) later.

### Path A — Demoable tabs 1-5 (no new ML)

Operates on v6 propensity outputs and the existing profile cache. Builds on frontend + inference scaffold.

| Tab | Description | New backend code | New frontend code |
|---|---|---|---|
| **1. Pitcher profile inspector** | Visualize 218-dim profile for any (pitcher, asof_date). Strike-zone heatmap, arm slot by type, per-(count×type) usage matrix (7×12 grid), per-stand split, ball movement (pfx_x/pfx_z), recent form. Side-by-side compare two pitchers. | `GET /pitcher/{id}/profile?asof_date=...` | new tab w/ heatmap + arsenal grid + radar chart |
| **2. Batter profile inspector** | Same idea, 57-dim: zone heatmaps (swing, whiff, xBA), recent xwOBA, per-type whiff/swing rates. | `GET /batter/{id}/profile?asof_date=...` | parallel structure to tab 1 |
| **3. Single AB rollout viewer** | Pick a real AB, replay pitch-by-pitch with model's top-3 predictions, confidence bar, calibration band, ground truth overlay. **Intro tab.** | extend existing `/ab-context` | new tab with pitch-by-pitch scrubber |
| **4. Confusion matrix & per-class** | Interactive 7×7 confusion matrix. Click a cell, see example pitches that fall there. Per-class precision/recall/F1 bars. | `GET /diagnostics/confusion?model=v6` | new tab with table + drill-down |
| **5. Calibration display** | Reliability diagram per head (type, zone, result), ECE bar, before/after temperature scaling. **Your differentiator — Pi 2018 / Ahn 2026 don't show this.** | `GET /diagnostics/reliability?model=v6&head=type` | new tab with reliability curves |

**Order:** 1 → 2 → 3 → 5 → 4. Tab 1 is the easiest to start (just renders profile vector). Tab 3 is the most concrete intro for users. Tab 5 is the differentiator.

### Path B — Complete the causal layer

The skeleton is there; the work is integration and validation.

#### B.1 — Audit existing causal/ for v6 readiness (1-2 days)

Open each of the 6 files and check:

- [ ] **`nuisance.py`**: confirm `NuisanceModels.from_checkpoint(path)` works with the v6 checkpoint and profile_dim=218. The model loads via `ckpt["config"]` so it should auto-detect, but worth verifying.
- [ ] **`positivity.py`**: validate the τ=0.01 / 0.05 thresholds still make sense given v6's calibrated π̂. Per ADR 002 the bands are locked, but sanity-check on real ABs.
- [ ] **`crossfit.py`**: this likely needs k=5 v6 checkpoints (one per fold), not just fold-0. **You currently only have fold 0 trained.** Decision: train v6 on folds 1-4 too (expensive — ~5h each on Modal = 20h total) OR run single-fit AIPW first as a placeholder and bump to cross-fit later.
- [ ] **`g_computation.py`**: confirm the rollout integrates the v6 propensity head shape. Should be fine — same 8-class output as v5.
- [ ] **`aipw.py`**: validate doubly-robust scoring on a few real ABs. Print named per-class probabilities (per CLAUDE.md bug-prevention rule #2).
- [ ] **`sensitivity.py`**: E-value computation is model-agnostic, should just work.

#### B.2 — Train v6 on folds 1-4 (parallel to other work)

Required for K=5 cross-fitting. Each fold ~5h on Modal A100. Run with `--detach` (per the lesson learned in v6 training). Total wallclock: 20h, but can run multiple folds in parallel if Modal billing allows.

```bash
for fold in 1 2 3 4; do
    modal run --detach modal_app.py::train_remote \
        --size tiny --fold-id $fold --epochs 3 \
        --run-name tiny-fold${fold}-v6 &
done
```

Pull checkpoints to `checkpoints_modal/tiny-fold${fold}-v6/`.

#### B.3 — Wire causal endpoints into the API (1 week)

Add to `inference/api.py`:

- `POST /causal/g-computation` — rigorous rollout for one (AB, intervention)
- `POST /causal/aipw-per-unit` — single-pitch effect estimate with positivity gate
- `POST /causal/aipw-population` — average treatment effect across a cohort
- `GET /causal/positivity?game_pk=...&at_bat_number=...&pitch_index=...&treatment=...` — quick gate check before showing UI

Each route returns: estimate, CI, positivity band (GREEN/YELLOW/RED), ESS, E-value, refusal_reason if applicable.

#### B.4 — Build causal tabs 10-12 in frontend (1-2 weeks after B.3)

- **Tab 10 — True counterfactual.** Pick an AB, force a pitch type swap, show effect on run value / outcome distribution. Distinguish from Tab 7 ("alternative completion") via the trust gauge and refusal logic.
- **Tab 11 — Trust gauge.** Three-state visual (gauge/traffic-light). Reads π̂ at the swap action, maps to GREEN/YELLOW/RED. ESS readout for multi-step.
- **Tab 12 — Refusal UX.** When the positivity gate fires RED, show a clear "I cannot estimate this" UI explaining why. Include E-value sensitivity panel.

#### B.5 — Negative-control tests (per CLAUDE.md hard rule)

The causal layer's defensibility requires negative controls: pick treatments that shouldn't have a causal effect (e.g., umpire_id, ballpark fixed effects in non-park-dependent outcomes) and confirm the AIPW estimate centers on zero. Document in `docs/decisions/` as an ADR.

### Path C — Live game tracker (deferred but planned)

Independent of A and B. Builds on top once they're done.

Phases:
- **C.1** — Statcast polling: pull recent pitches via pybaseball or MLB Stats API; map to existing profile cache (need to handle pitchers/batters NOT in the cache → cold-start UX).
- **C.2** — Real-time prediction service: subscribe to the in-progress game stream, predict each pitch ~1s before it's thrown.
- **C.3** — Live UI tab: scoreboard with model overlay. Shows predicted P(FF/SL/...) and zone heatmap. Confidence interval rendered with the trust gauge.
- **C.4** — Daily forecast: Monte Carlo of today's games using v6 + lineups + the causal harness for individual matchups.

**Estimated effort:** 1 week C.1 + 1 week C.2 + 1 week C.3 + 1 week C.4.

Defer until A and B are done because:
1. Without the trust gauge (B.4), the live tracker is "yet another pitch predictor" — generic
2. With the trust gauge, it's "AI that knows when not to trust itself" — your story
3. Live data infra adds operational complexity (rate limits, latency, freshness checks) that distracts from the methods writeup

---

## 4. Sprint breakdown — what to do, in order

### Sprint 1 (week 1) — v6 integration audit + Tab 1
- B.1: audit `causal/` files for v6 readiness; fix any v5-isms
- A.1: build the pitcher profile inspector tab end-to-end (most concrete starting point)
- B.2: kick off Modal training for folds 1-4 (run in background)

### Sprint 2 (week 2) — Tabs 2-3 + AIPW spot-check
- A.2: batter profile inspector (parallel structure to A.1)
- A.3: single AB rollout viewer — this exercises the propensity head end-to-end
- B.1 cont: spot-check AIPW on 5-10 real ABs with named probabilities (per CLAUDE.md bug discipline)

### Sprint 3 (week 3) — Tabs 4-5 + cross-fit ready
- A.4: confusion matrix + per-class diagnostic tab
- A.5: calibration display (reliability diagrams) — your differentiator
- B.2 fin: all 5 folds trained; verify K=5 cross-fit works end-to-end on small sample

### Sprint 4 (week 4) — Causal API + negative controls
- B.3: wire `/causal/*` endpoints into `inference/api.py`
- B.5: run negative-control tests; document results in an ADR
- (optional) start integrating the calibration tab with the trust-gauge component

### Sprint 5 (week 5) — Causal tabs
- B.4 part 1: Tab 10 — true counterfactual
- B.4 part 2: Tab 11 — trust gauge component
- B.4 part 3: Tab 12 — refusal UX

### Sprint 6 (week 6) — Tipping detector
Separate skill (`tipping-analysis`). Build:
- batter-observable variant of the propensity head
- T_start metric (the first start where pitch type becomes predictable from batter-observables)
- 4 validation runs from the skill (subsequent-start performance, time-through-order, batter behavior, case studies)
- Tab 13 — T_start viz

### Sprint 7+ — Path C live tracker

Defer until everything above is stable.

---

## 5. Open decision points (resolve before starting)

1. **Cross-fit folds 1-4 of v6: train now or defer?** Training cost is 5h × 4 folds = 20h of Modal time + $$. Alternative: run single-fit AIPW initially as a stand-in, label it explicitly as "single-fit, not cross-fit, expect ~10% bias," and bump to K=5 once you've validated the architecture. Recommended: defer cross-fit until causal API endpoints are working end-to-end.

2. **Should the v6 result head be retrained with a dedicated swing head?** Current marginal P(swing) is 0.7245 OOD vs Ahn 2026's 0.78. **Don't chase now.** Revisit if/when the causal layer needs swing-decision factorization explicitly.

3. **Demo cache strategy.** The frontend will be slow if every tab requires live model inference. Pre-compute a "demo cache" for ~100 well-known pitchers/batters/ABs and serve those instantly. New ABs trigger live inference via the API. Decision: cache structure + invalidation policy.

4. **Tipping detector — separate model or batter-observable variant of the existing one?** Per the `tipping-analysis` skill (read it), it's a batter-observable variant. That means training another tiny on batter-observable features only. ~5h Modal per fold.

5. **Per CLAUDE.md hard rule #5: language discipline.** Tab 7 = "alternative completion" / "model rollout." Tab 10 = "counterfactual" only when AIPW+cross-fit+positivity all fire. Audit all UI strings before shipping.

---

## 6. Acceptance criteria for "the demo is done"

A reasonable bar for the v1 demo:

- [ ] Tabs 1-5 functional with v6 propensity model
- [ ] Tabs 10-12 functional with the causal layer (single-fit AIPW acceptable for v1; cross-fit by v2)
- [ ] Trust gauge correctly turns RED on out-of-support queries with refusal UX
- [ ] Calibration metrics displayed prominently somewhere in the UI
- [ ] At least 3 negative-control tests passing (E-value sensitivity)
- [ ] No causal language used outside Tabs 10-12 (CLAUDE.md hard rule #5)
- [ ] Demo loads in < 5s on a fresh visitor (use demo cache)
- [ ] Mobile-responsive (frontend-system skill: design system constraints)

---

## 7. Starting context for a fresh session

If picking this up in a new Claude Code session, read these first (in order):

1. **`CLAUDE.md`** at repo root — hard rules, especially #5 (no causal language without machinery), #6 (calibration is primary), #8 (no SF Pro, accessibility constraints).
2. **This document** — the roadmap.
3. **`docs/superpowers/specs/2026-05-17-v6-profile-features-design.md`** — v6 design (just shipped).
4. **The five most relevant skills** — load via `Skill` tool:
   - `causal-layer` for Path B
   - `frontend-system` for Path A
   - `eval-protocol` for measurement
   - `tipping-analysis` for Sprint 6
   - `pressure-testing-claims` for any quantitative claim
5. **ADRs 001-008** in `docs/decisions/` — locked design decisions.
6. **`inference/api.py`** lines 99-260 to see how `NuisanceModels` is loaded — this is where v6 plumbing starts.
7. **`causal/__init__.py`** docstring — current module layout.
8. **The v6 checkpoint:** `checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt` (and `checkpoint_calibrated.pt` with temperatures).

### One-paragraph starting prompt for a new session

> I'm continuing PitchGPT after v6 propensity-model training. V6 is calibrated (ECE 0.0058 in-dist, 0.0144 OOD), beats Pi 2018 RNN and Ahn et al. 2026 LLM on accuracy at the matched recall=0.79 operating point on postseason OOD, and the v5 baseline at `checkpoints_modal/tiny-fold0-1778792736/` is preserved for comparison. The plan is `docs/superpowers/plans/2026-05-18-post-v6-roadmap.md`. Causal-layer skeleton exists at `causal/*` (~2000 lines across 6 files) and needs v6 integration audit. Frontend scaffold exists at `frontend/src/` with React+Vite+TS+Tailwind+strike-zone components. FastAPI app at `inference/api.py` has /games, /at-bats, /ab-context, /query endpoints (662 lines). Twelve ADRs locked. Start with Sprint 1 from the roadmap: v6 integration audit on `causal/nuisance.py` + build Tab 1 (pitcher profile inspector). Kick off Modal training for folds 1-4 in background. Pressure-test claims before shipping (per memory).

### Where things live (quick path index)

| What | Where |
|---|---|
| v6 design spec | `docs/superpowers/specs/2026-05-17-v6-profile-features-design.md` |
| v6 implementation plan | `docs/superpowers/plans/2026-05-17-v6-profile-features.md` |
| This roadmap | `docs/superpowers/plans/2026-05-18-post-v6-roadmap.md` |
| ADRs | `docs/decisions/001-*.md` through `012-*.md` |
| v6 model checkpoint | `checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt` |
| v5 baseline | `checkpoints_modal/tiny-fold0-1778792736/checkpoint_best.pt` |
| v6 profile cache | `data/profiles/*.parquet` (schema=6, all 5 folds) |
| v5 backup cache | `data/profiles_v5_backup/*.parquet` |
| v6 standardizer | `data/preprocess_artifacts/v1/profile_standardization.npz` (D=218) |
| v5 standardizer backup | `data/preprocess_artifacts/v1/profile_standardization_v5_backup.npz` |
| Causal layer code | `causal/nuisance.py`, `g_computation.py`, `aipw.py`, `crossfit.py`, `positivity.py`, `sensitivity.py` |
| Inference API | `inference/api.py` (662 lines) |
| Frontend | `frontend/src/{App.tsx,Scoreboard.tsx,StrikeZone.tsx,api.ts,types.ts}` |
| Eval scripts | `scripts/{calibrate_pitchgpt.py,diagnose_type_perclass.py,eval_postseason_ood.py}` |
| Training script | `scripts/train_pitchgpt.py` |
| Modal app | `modal_app.py` |

### Things to remember (gotchas hard-learned in v6)

1. **Modal training without `--detach` dies if local client disconnects.** Always use `modal run --detach`.
2. **Slot-name collisions are easy.** v6's `arsenal_*_b*s*` slots collided with `startswith("arsenal_")` in `model/pitchgpt_dataset.py`. Lesson: enumerate slots by exact name, not prefix. Fixed at line ~52 in that file.
3. **Schema version check is strict.** `data/profile_cache_loader.py` refuses to load mismatched cache. v5 checkpoint evaluations need the `_LegacyProfileLookup` workaround from `scripts/eval_postseason_ood.py`.
4. **Modal volume needs explicit upload after data changes.** `python scripts/upload_to_modal.py --only profiles preprocess_artifacts --force`.
5. **The slow batter-cache build path exists.** `scripts/build_profile_cache.py:404` — make sure you're calling `build_batter_cache_for_fold_fast`, not the slow version.
6. **Calibration is the primary metric.** Per CLAUDE.md hard rule #6. Don't ship accuracy-only tables.
7. **Pressure-test before shipping.** Per `feedback_pressure_test_proactively.md` in memory — verify empirical/numerical claims with evidence before they land in code, docs, or user-facing text.

---

## 8. What we are NOT doing (out of scope for now)

To prevent scope creep:

- **Dedicated swing decision head** — marginal from result head is adequate (deferred).
- **Cross-AB / game-level context** — would help raw accuracy (Ahn 2026 has this implicitly) but adds complexity. Not on critical path.
- **LLM-style backbone scale-up** — your differentiator is calibration + causal, not scale.
- **Video / multimodal** — explicitly out per CLAUDE.md.
- **Cross-sport transfer** — also explicitly out.
- **Bayesian shrinkage on per-(type×count) cells** — post-MVP per the v6 design doc.
- **Retrieval-augmented player profiles** — interesting future work, not now.

---

## 9. Quick-start sanity check (run on a fresh session)

After cloning / picking up the repo, run these to confirm the v6 state is healthy:

```bash
# 1. Tests pass
uv run pytest tests/test_profile_cache.py tests/test_player_profiles.py -v 2>&1 | tail -5
# Expected: 85 passed

# 2. v6 profile cache is loadable + correct dim
uv run python -c "from data.profile_cache_loader import ProfileCache; from data.profile_cache import PITCHER_VECTOR_LEN, PROFILE_SCHEMA_VERSION; pc = ProfileCache(role='pitcher', fold_id=0); print(f'pitcher_dim={PITCHER_VECTOR_LEN} schema=v{PROFILE_SCHEMA_VERSION}')"
# Expected: pitcher_dim=218 schema=v6

# 3. v6 checkpoint loads + model builds
uv run python -c "import torch; from model.config import PitchGPTConfig; from model.pitchgpt import PitchGPT; ckpt = torch.load('checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt', map_location='cpu', weights_only=False); cfg = PitchGPTConfig(**{k:v for k,v in ckpt['config'].items()}); model = PitchGPT(cfg); model.load_state_dict(ckpt['model_state_dict']); print(f'OK params={model.num_parameters():,}')"
# Expected: OK params=4,860,986

# 4. Postseason OOD eval reproduces the numbers in this doc
uv run python -m scripts.eval_postseason_ood --ckpt checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt 2>&1 | tail -20
# Expected: v6 OOD numbers matching table in §1
```

If any of these fail, fix the environment before starting roadmap work.
