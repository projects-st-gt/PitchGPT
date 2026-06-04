# ContextSwitcher — Pick up where this session left off

**Last updated**: 2026-06-03 (late) — hitter-model build handed off.

## ⭐⭐⭐⭐ pitchGPT + CASCADE SIMULATOR WORKS (2026-06-04)

**`g_compute(outcome_model="hitter", hitter_step_fn=...)` is built + working.**
pitchGPT (small-v7) samples each pitch sequentially (full context + sequence
preserved); the CASCADE decides the batter response (not the transformer's weak
outcome head). Default `"head"` mode byte-unchanged (8 natural-mode tests pass).
E2E: elite hitter vs RHP, 300 sims → OPS 1.24 / K% 25.8% / HR 11.4%.

The pieces (`hitter/rollout.py`, all tested):
- `cascade_to_result_probs` — translator: cascade outputs → 7-class result vocab.
- `build_step_features` — sampled pitch → cascade features (zone→centroid via
  `checkpoints/hitter/zone_centroids.json`, velo/spin from pitcher per-type means,
  prev pitch as the lag → SEQUENCE preserved inside the rollout).
- `make_hitter_step_fn` — per-cell closure bundling cascade + translator.

**NEXT (the actual experiment the user wants):**
1. **Backtest pitchGPT+cascade vs lookup+cascade vs baseline** on real PAs (does
   pitchGPT's batter-aware, sequence-aware pitch selection beat the count-only
   lookup? lookup already = 1.405 < 1.452 baseline). The MC rollout is slow, so
   sample. NOTE: the pitchGPT-path OPS looked hot (1.24) — its in-play mix differs
   from the lookup, so it likely needs its own outcome-map calibration check
   (rebuild the map on predicted-xwoba over PITCHGPT-sampled pitches, or confirm
   the existing map holds). Don't trust the level until calibrated + backtested.
2. **matchup cards**: `mcsim/matchup_card.py` → call g_compute hitter mode; run
   June 4; publish to demo (:8000/:5173 running). Switch DEFAULT_CKPT → small-v7.
3. **App A**: chain hitter-mode PAs through a base-out-inning game state machine.
4. (Future) retrain a PITCH-ONLY transformer (drop outcome heads → free capacity
   for π̂) — test pitch top-1/calibration; only after pitchGPT beats the lookup.

---

## ⭐⭐⭐ HITTER MODEL — CALIBRATED + BEATS BASELINE (2026-06-04)

**Per-PA backtest (the real "does it work" measure) now PASSES.** Cascade +
empirical-lookup pitch source scores **1.405 log-loss vs 1.452 league-average
baseline** on real held-out 2024H2 at-bats (lower=better). It was *worse* (1.460)
until the calibration fix below.

**KEY FIX — the xwOBA→outcome map must bin on PREDICTED xwOBA, not real.**
contact_quality hedges to the mean (pred std ~0.08), so a real-binned map never
fires its high bins → HR came out 5× too low (0.7% vs real 3.3%), singles too
high. Predicted-binning fixes it (HR drift → 3.2% vs 3.3%). Reproducible via
`python -m scripts.hitter.build_outcome_map`. **Re-run this whenever the
contact_quality node is retrained.**

**small-v7 capacity test (DONE, 3 epochs):** type_top1 0.485 (tiny-v7 0.478).
Compression diagnostic (same panel/pitcher, controlled): tiny-v7 5.43×/r0.29,
**small-v7 3.41×/r0.67**, **hitter cascade 2.62×/r0.90** (true-talent 1.37×/r0.94).
Capacity helps spread magnitude but NOT ordering; cascade wins both. Plan:
small-v7 = better π̂, cascade = μ̂. Calibrated ckpt at
`checkpoints_modal/small-fold0-v7/checkpoint_calibrated.pt`.

**contact_outcome multiclass head:** built + saved, but it slightly HURT OPS
spread (2.62→2.81×); xwoba pinned as default `outcome_mode`. Multiclass stays an
option (may help power calibration — untested vs the fixed map).

**OPEN — wire the cards + the pitch-source test:**
- `mcsim/matchup_card.py` still uses g_compute (transformer μ̂). To ship
  cascade-based June 4 cards: add an analytic-compose path (`compose_pa` +
  `build_empirical_pitch_provider`) + run `run_matchup_cards.py` (switch
  DEFAULT_CKPT to small-v7) + publish to the running demo (:8000/:5173).
- **Transformer-vs-empirical pitch-source backtest** (does pitchGPT's batter-aware
  pitch selection beat the count-only lookup): empirical side done (1.405).
  Transformer side blocked: pitchGPT conditions on count via pitch HISTORY, not a
  settable count factor — clean per-count π̂ needs synthetic pitch histories or MC
  rollout extraction (multi-hour). Honest caveat: the analytic engine is count-only
  state, so it mutes sequence regardless of pitch source — count+last-pitch state
  is the upgrade to let sequence show.

---

## ⭐⭐ HITTER MODEL — STEPS 1-4 DONE, PASSES THE GATE (2026-06-03 late)

**The dedicated hitter cascade works.** Compression diagnostic on held-out 2024H2
(12-batter panel, avg over 5 reference pitchers):
- **spread ratio 2.58×** (was **6.8×** for the transformer; target ~1×)
- **Pearson r 0.90** (was 0.66) — near-perfect hitter ordering
- **model_mean 0.76 == real_mean 0.76** (transformer regressed everyone to ~.70)
- robust across ref pitchers (2.44–2.78×, r 0.875–0.925)

**Built + tested (42 hitter tests green), all on real Statcast, no placeholders:**
- `hitter/train.py` — 4 XGBoost nodes trained on FULL 4.74M pitches →
  `checkpoints/hitter/`. Full-data AUC: swing 0.869, called_strike 0.985,
  whiff 0.787; contact_quality (xwOBA reg) RMSE 0.369 / r 0.216. **Honest
  held-out ECE** (not the in-sample ~0): swing 0.009, called_strike 0.004,
  whiff 0.007 — all well-calibrated. **macOS XGBoost segfault root-caused**:
  feed numpy not pandas (columnar adapter crash) + `n_jobs=1`/`OMP_NUM_THREADS=1`
  (libomp race). cat cols via int codes + `feature_types`.
- `hitter/model.py` — `HitterModel.predict_cascade` (round-trip tested).
- `hitter/compose.py` — analytic count-tree solve (absorbing Markov, closed form)
  + real `xwoba_outcome_map` (275K in-play balls) at `checkpoints/hitter/`.
- `hitter/eval.py` — `run_compression_diagnostic` + CLI (`python -m hitter.eval`).
- New CLAUDE.md hard rule **1a**: no placeholder/stub/boilerplate/fabricated
  constants in production paths; synthetic data only in tests/.

**Remaining gap is at the LOW end** (weak hitters .40–.51 real → .61–.70 model,
under-punished). The lever to push 2.58× → ~1× is **contact_quality** (monotonic
xwOBA↑middle constraint, or the {1B/2B/3B/HR/out} multiclass head).

**NEXT (handoff steps 5-6):**
5. Wire into `causal/g_computation.py` behind `outcome_model="head"|"hitter"`
   (cascade = μ̂; keep π̂ from PitchGPT). Re-check AIPW≈g-comp + negative
   control≈0 before using causal language on its outputs.
6. `mcsim/matchup_card.py` `compose="analytic"` option → re-run
   `run_matchup_cards.py` for an in-range date (≤2026-05-08).

**small-v7:** original ephemeral run was CANCELLED at epoch 1/3 (session ended).
RELAUNCHED detached as app **`ap-9gGQmUoG4mbk4DzXcdSWuf`** (full 3 epochs). Pull
when done: `modal volume get pitchgpt-data checkpoints/small-fold0-v7
./checkpoints_modal/` → calibrate → re-run compression diagnostic on it (capacity
test). NOTE: the hitter model already recovers most of the spread, so small-v7 is
now a comparison point, not the fix.

---

## ⭐ START HERE (current state — supersedes older sections below)

- **Active branch: `hitter-swing-model`** (17 ahead of `main`). SUPERSET of
  `mcsim-runner-mp`: contains ALL go-live work (multiprocessing, frontend MCSim
  tab, **as-of profile fallback**, NaN→null fix, SQLite threading fix,
  OPS/AVG/BB%/K% cells, progress logging) **plus** the hitter foundation. Keep
  working here; nothing is merged to `main` yet.
- **Active task: build the hitter/swing model** → see **"▶ NEW-SESSION HANDOFF"**
  at the very BOTTOM of this file (authoritative step-by-step). Design:
  `hitter/MODEL_DESIGN.md`, **APPROVED** — proceed with the §9 leans (xwOBA-on-
  contact regression target; **analytic count-tree composition**, not
  Monte-Carlo; fouls = count-constant in v0; continuous plate_x/z; 5 separate
  XGBoost boosters).
- **Done in `hitter/`:** `labels.py` ✅ + `features.py` ✅ (both tested; features
  incl. the leakage-safe batter-profile join). Remaining: `train.py` →
  `model.py` → `compose.py` → `eval.py` → wire into `causal/g_computation.py`
  (`outcome_model` flag) → matchup-card path.
- **Parallel:** `small-v7` on Modal (app `ap-55acWEObPHd29pPYMZibNh`, L4, ~10h) —
  capacity test for the same compression problem. Pull/calibrate/re-diagnose
  steps in the handoff.
- **Acceptance test:** compression diagnostic — real vs model OPS spread.
  Transformer = **6.8×** (real std 0.261 / model 0.038, r=0.66). Target ≈ **1×**.
- **App A (full-game sim)** is unblocked: `compose.py`'s per-PA engine is its core.
- Sections below ("GO-LIVE SPRINT", Steps 4–8) are HISTORICAL — App B backend is
  done. Trust the handoff for what's live.

---

### (historical) GO-LIVE SPRINT status (2026-06-03) — read this first

On branch **`mcsim-runner-mp`** (NOT merged yet), bundling the "run it for real" work:
- ✅ **Game-level multiprocessing** added to the runner (`--n-workers`; each worker = 1 game at torch threads=1; flat thread-scaling makes this win). Smoke: 2 games/2 workers ~2× concurrency. Committed + unit-tested.
- ✅ **Live-progress logging added** (`compute_matchup_card(progress_every=)` + runner `--progress-every`, default 25). The first n_paths=250 run was opaque (no per-game logging) and ran slow under 9-way contention; killed it and re-ran WITH logging.
- ✅ **PREDICTION RUN DONE for 2026-06-04**: `--n-workers 9 --n-paths 120 --rng-seed 1` → **9/9 cards, 338 cells each, in `data/mcsim.sqlite`** (78 min wall, fully visible via streamed progress). NOTE: n_paths=**120** (not 250) — re-run at higher quality later if desired. Verified via the `/mcsim` API: OPS populated + discriminating (median RV flat at −0.100, as expected). Many 06-04 games had no probable announced → `is_starter=False` (cosmetic).
- ✅ **MCSim frontend tab MVP** built (`frontend/src/MCSimTab.tsx` + tab in `App.tsx` + `/mcsim` client in `api.ts` + types). `tsc -b` clean; **data path verified** through the API. **Browser visual verification STILL PENDING** — `make demo`, open "Matchup cards" tab on date 2026-06-04.

**Next after the run:**
1. Verify frontend renders real 06-04 cards (dev server + screenshot).
2. **2026-06-05:** `python -m scripts.mcsim.fetch_actuals --date 2026-06-04` → actuals overlay appears in the tab.
3. **Calibration eval** (not started): pool real PAs → reliability diagram/ECE; calibration-check OPS magnitude. Needs actuals (06-05) or backfilled past games. The card predicts in NEUTRAL context but real PAs vary — calibration must context-match (re-run model per real PA) or restrict to neutral-context PAs.
4. Merge `mcsim-runner-mp` → main + push.

---

(historical) Step 4 done 2026-06-01; everything below predates the go-live sprint.

This is the handoff doc for a new Claude session (or a VS Code restart). Read it cold; the project state below is everything you need to keep going.

---

## TL;DR — where to resume

**MCSim App B v1 BACKEND IS COMPLETE.** Steps 4–7 are merged+pushed to `main`. **Step 8 (read API endpoints) is DONE on branch `mcsim-app-b-read-api`** (off `main`; NOT yet merged as of 2026-06-03). All five backend pieces exist: storage, synthetic-state builder, matchup-card computer, live MLB-API runner, post-game actuals fetcher, and read endpoints. **Next big piece: the MCSim frontend tab** (date carousel + card grid + result overlay) — separate scope, uses `frontend-system`. Next concrete step:

> **MCSim frontend tab.** Date carousel → per-game matchup-card grid (pitchers × hitters) → result overlay once a game finishes. Wire to the Step 8 endpoints (`/mcsim/predictions?date=...`, `/mcsim/predictions/{game_pk}`). Use `frontend-system` skill (Inter, 3 colors, color+glyph for pitch types). Headline cells on **mean RV or OPS**, not median RV.

> **Step 8 (done) — read API.** `inference/mcsim_api.py`: model-free `APIRouter` mounted on the main app. `GET /mcsim/predictions?date=` → per-game summaries (carousel); `GET /mcsim/predictions/{game_pk}?date=` → full card with each (pitcher,batter) cell stamped with the real PA(s) (`actual.pa_count`+`events`; cells that didn't happen → `null`; PAs outside the grid → `unmatched_event_count`). DB via `get_conn` dependency (override in tests). Tests: `test_mcsim_api.py` (6).

**Step 7 (done) — actuals layer.** `mcsim/mlb_actuals.py::get_game_actuals` parses the MLB live feed (`/api/v1.1/game/{pk}/feed/live`) → final score + winner + per-PA events (pitcher, batter, **start context: bases reconstructed from `runners[].movement.originBase`, outs from first pitch — NOT `matchup.splits.menOnBase`, which is a stat-split label and reports RISP for a leadoff hitter**, verified). `scripts/mcsim/fetch_actuals.py` is the CLI (per date, writes Final games via `storage.write_actual`; non-final skipped). Validated on game 777079 (Giants @ Jays 6–8, 75 PAs). Tests: `test_mcsim_mlb_actuals.py` (4) + `test_mcsim_fetch_actuals.py` (2).

**Key validation insight (drives the future eval step):** a single matchup cell CANNOT be validated — a hitter faces a pitcher only 1–4 times per game. Validate by POOLING thousands of real PAs into a reliability diagram/ECE (calibration is the project's primary metric), and lean on the already-built per-pitch calibration. The 1000-path sim is Monte Carlo to smooth the model's distribution, not the thing being graded.

See sections below for details.

---

## The user's immediate priority for the next session

Continuing App B v1 backend (~3 more focused days):

1. ~~**Step 4 — `mcsim/matchup_card.py`**~~ ✅ done (commit `67f0629`)
2. ~~**Step 5 — CLI runner** (`scripts/mcsim/run_matchup_cards.py`)~~ ✅ done
3. ~~**Step 6 — MLB Stats API client**~~ ✅ done — MERGED into Step 5 as `mcsim/mlb_api.py` (schedule + active-roster; lineups dropped in favour of all-vs-all roster grid)
4. ~~**Step 7 — Post-game actuals fetcher**~~ ✅ done — `mcsim/mlb_actuals.py` + `scripts/mcsim/fetch_actuals.py` (branch `mcsim-app-b-actuals`)
5. ~~**Step 8 — Read API endpoints**~~ ✅ done — `inference/mcsim_api.py` (`GET /mcsim/predictions?date=...` summaries + `GET /mcsim/predictions/{game_pk}` full card with per-cell actuals overlay), mounted on the main app.
6. **App B v1 backend is COMPLETE.** Next big piece: **MCSim frontend tab** (date carousel + per-game card grid + result overlay) — uses `frontend-system` skill. Headline the card on **mean RV or OPS, not median RV** (median is a weak discriminator — see below).
7. (perf, optional) Runner multiprocessing — needed for a full 15-game nightly batch at n_paths=250.
8. (eval) Pooled per-PA calibration: reliability diagram/ECE over real PAs vs the model's per-matchup probabilities (context-matched). The real App-B validation — single cells can't be validated (1–4 real PAs each). Also calibration-check the projected OPS magnitude.

After App B v1 lands: App A (daily score prediction) needs a multi-AB state machine. MCSim brainstorm doc has design notes; that's a separate substantial project.

---

## Current branch: `mcsim-app-b-matchup-card`

Six commits on this branch since branching from main:

| # | Commit | What | Tests |
|---|---|---|:---:|
| 1 | `8a0ce1b` | `docs/mcsim_appB_brainstorm.md` — design decisions D1–D8 with explicit "my lean" + user sign-off recorded in chat | — |
| 2 | `e12438e` | **D4 natural mode** in `causal/g_computation.py` — `intervention_type=None` samples from π̂(type \| h) instead of clamping | 5 |
| 3 | `4da1a1c` | **Storage layer** — `mcsim/storage.py` + SQLite schema (predictions, actuals, model_versions) + 12 round-trip tests | 12 |
| 4 | `d54bd81` | **Option C** — relax `intervention_position >= 1` to `>= 0` (first-pitch rollouts work — propensity at last context-token position) | 3 |
| 5 | `a94210f` | **Synthetic-AB builder** — `mcsim/state.py` with `ReferenceContext` dataclass + `build_synthetic_ab()` | 11 |
| 6 | `67f0629` | **Matchup-card computer** — `mcsim/matchup_card.py` with `compute_matchup_card` + `PitcherSpec`/`BatterSpec`; exposes `RolloutResult.intervention_type_propensity` for the trust gate | 6 |

**Full test suite: 347 pass.** Branch pushed to origin.

The brainstorm doc (`docs/mcsim_appB_brainstorm.md`) is the source of truth for all design decisions. Read it before writing more code.

---

## Open PRs (not on this branch)

| PR # | Branch | Status | Notes |
|---|---|---|---|
| **#1** | `v7-type-conditioned-heads` | **Merged** to main | v7 Part 1 — type-conditioned execution heads (ADR-013 D1) |
| **#4** | `demo-polish` | Open, awaiting review | v7 checkpoint switch + `make demo` + 6 API smoke tests + warm startup hook |
| **#5** | `recommender` | Open, awaiting review | `rank_pitch_types` + `POST /recommend` |

Two other branches with parked work (don't merge):
- `v7-cross-ab-context` — Part 2 of ADR-013. Decided not to ship; flag-default-off, no harm leaving the code on the branch.
- (Arsenal mask experiment) — code committed on `v7-cross-ab-context`; A/B showed it broke NLL (true labels in trailing-window-zero-mass classes). Not shipped.

---

## Step 4 done — `mcsim/matchup_card.py` (commit `67f0629`)

`compute_matchup_card(nuisance, *, game_pk, game_date, home_team, away_team,
home_pitchers, away_pitchers, home_lineup, away_lineup, …, n_paths=1000,
rng_seed=None, context=None) -> dict` is built and tested (6 tests). It loops
every (pitcher, batter) cell across both half-grids, builds a synthetic
reference-state AB, rolls it out in natural mode
(`intervention_position=0, intervention_type=None`), and packs the per-game
payload. Returns the dict; does NOT persist. Key choices that landed:

- RV median/p05/p95 come from the **per-path `run_value` array** (truncated
  paths are NaN and excluded) — not derived from mean/SE.
- Per-cell seed = `rng_seed + cell_index` (reproducible, not identically
  correlated).
- Trust flag reads the exposed `RolloutResult.intervention_type_propensity`
  (the exact π̂(type|h) the rollout sampled from) — no second forward pass.
- Language discipline: cells are **predictive rollouts, not causal
  estimates.** `PositivityGate` is borrowed only for its trust state; its
  causal rationale string is deliberately not surfaced.

Named numerical check from a real cell: `[HomeSP vs AwayBat1]` median RV
= −0.1000 (p05 −0.15, p95 +1.11); top-1 = out; modal type = FS at π̂=0.327;
trust = green; 0/50 truncated.

---

## Next step — Step 5+6 (MERGED): live MLB-API runner (BRAINSTORM IN PROGRESS)

**Decision (user, 2026-06-01):** pull live from the MLB Stats API *now* — this
merges the planned Step 6 (MLB client) into Step 5. And use an **all-vs-all
roster grid**, not the posted 9-batter lineup.

### Why all-vs-all (the lineup problem)
Empirically probed against `statsapi.mlb.com` this session:

| Run timing | Probable pitchers | Real lineups | Actuals |
|---|:---:|:---:|:---:|
| Past date (all 2025) | ✅ | ✅ (9/side) | ✅ |
| Today, day-of (Pre-Game/Warmup) | ✅ | ✅ (~hrs before 1st pitch) | — |
| Tomorrow / future | ✅ (13–14/15) | ❌ **0 games** | — |

So D5's "night-before with probable lineups" is **false** — MLB posts confirmed
lineups only ~2–4h pre-first-pitch. User's fix: skip lineups entirely, grid the
**full active roster** (available night-before via the roster endpoint):
`/api/v1/teams/{id}/roster?rosterType=active&date=…` → 26 players splitting
cleanly into ~13 pitchers + ~13 position players by `position.type=='Pitcher'`.
This is a *better* dugout doc — it helps build a lineup, not just react to one.

### Verified this session
- API reachable; `requests 2.33.1` + `pybaseball 2.2.7` installed; no existing MLB client code.
- statsapi player IDs are MLBAM = same namespace as Statcast `pitcher`/`batter` (no crosswalk). Skenes=694973.
- Roster endpoint returns id/name/position the night before. ✅
- **Per-cell benchmark (v7, CPU, 8 threads): 39.87s @ n_paths=1000, LINEAR in n_paths (~33–40 ms/path).** The old "sublinear" hunch was wrong.

### The compute catch
All-vs-all ≈ **338 cells/game** (2 × ~13 × ~13) vs the old 63. At n_paths=1000
that's **57.7h for 15 games** — NOT nightly-feasible single-process. Linear
knob → n_paths=200 ≈ 12.7h/15. Levers: lower n_paths, multiprocessing across
cores, or run a subset. **v1 plan: build correct + sequential, validate on ONE
real game, measure true wall-clock, THEN tune (per D8 "no cron until manual
end-to-end works").**

### Open flags to verify in implementation
- **Handedness**: `build_synthetic_ab` needs `pitcher_throws`/`batter_stand`;
  roster `person` object may need `hydrate=person` to expose pitchHand/batSide. VERIFY.
- **Missing profiles**: just-called-up players may have no trailing-window
  profile → per-cell try/except + skip-with-log (no crash, no fabrication).
- `compute_matchup_card` itself needs **no change** — it already loops
  pitchers × batters over arbitrary-length lists; runner just feeds full rosters.
- Flag the probable starter via `is_starter=True` inside the all-staff grid.

### Settled mechanics
- Reuse `NuisanceModels(V7_CKPT, device="cpu")`; default ckpt
  `checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt`.
- `model_ckpt_hash`: **no helper exists** — compute it (e.g. truncated sha256 of
  ckpt file); `register_model_version` once per run.
- Persist via `write_prediction(..., app="matchup_card")` (validated key, `storage.py:125`).
- CLI surface (planned): `--date`, `--game-pk` (repeatable filter), `--n-paths`,
  `--rng-seed`, `--ckpt`, `--db-path`, `--max-games`, `--dry-run`.
- New unit: `mcsim/mlb_api.py` (schedule + roster client, its own tests) so the
  HTTP concern is isolated and testable; runner orchestrates.

**STATUS: design + spec + plan APPROVED (2026-06-02). Ready to implement.**
- Spec: `docs/superpowers/specs/2026-06-02-mcsim-appB-step5-live-runner-design.md`
- Plan: `docs/superpowers/plans/2026-06-02-mcsim-appB-step5-live-runner.md` (5 TDD tasks)
- Extra verified finding: **missing profiles do NOT crash** — `ProfileCache.lookup`
  falls back to league-mean then zeros (`data/profile_cache_loader.py:149`). So
  the runner needs only a per-GAME try/except, not per-cell. Debut players get a
  league-mean profile (principled degrade, not fabrication).
- Switch hitters (`batSide=='S'`) resolve to `'L'` in v1 (vs the more common RHP);
  per-pitcher resolution deferred (would need `compute_matchup_card` to vary stand).

---

## Step 5+6 results (DONE — validated on a real game 2026-06-03)

Implemented via subagent-driven TDD (plan: `docs/superpowers/plans/2026-06-02-mcsim-appB-step5-live-runner.md`). Commits on branch:
- `mcsim/mlb_api.py` — `get_schedule` + `get_active_roster` (+ malformed-item hardening on both). Tests: `tests/test_mcsim_mlb_api.py` (7).
- `scripts/mcsim/run_matchup_cards.py` — `compute_ckpt_hash`, `run_matchup_cards`, `main` CLI. Tests: `tests/test_mcsim_run_matchup_cards.py` (3: hash, real-model E2E, fast skip-path).
- **Full suite: 356 passed.**

**Live validation:** `python -m scripts.mcsim.run_matchup_cards --date 2026-06-03 --game-pk 822727 --n-paths 40` → Marlins @ Nationals, **338 cells written to `data/mcsim.sqlite`**, ckpt hash `e67bd67d65a988dd`. Named numbers (row 0, Andrew Alvarez LHP, is_starter=True): vs Christopher Morel median RV −0.1000 (p05 −0.15, p95 +1.40), top1=out, modal FF π̂=0.305, trust=green, 0/40 trunc. All 338 cells green.

**Wall-clock (measured 794.81s / 338 cells @ n_paths=40 = 2.35 s/cell, linear):**

| n_paths | min/game | 15 games single-proc | 15 games 8-core mp |
|---|---|---|---|
| 40 | 13.2 | 3.3h | 0.4h |
| 100 | 33.1 | 8.3h | 1.0h |
| 250 (default) | 82.8 | **20.7h (infeasible)** | 2.6h |

→ For a real 15-game nightly batch at n_paths=250, **multiprocessing is required** (or drop n_paths). Functionally the runner is complete; this is a perf follow-up, not a correctness gap.

**Honest finding — median RV is a weak cross-cell discriminator.** It pinned to −0.1000 for every cell (when >50% of paths end in "out", the median path IS an out → median = out's run value). Batter signal lives in the tails (p05/p95) and the outcome distribution, NOT the median. **Frontend (Step 8+) should headline mean RV or P(reaches base), not median.**

---

## What's built up to this point on `main`

Substantially everything from the v6/v7 era. The model + causal layer + demo backend + frontend are all in place. v7 is the deployed model (PR #1 merged).

### Causal layer (`causal/`)
- `nuisance.py` — loads a calibrated checkpoint, exposes π̂ + μ̂.
- `g_computation.py` — Monte Carlo rollout. **Now supports natural mode (intervention_type=None) and intervention_position=0** (the D4 + Option C changes on this branch).
- `aipw.py` — doubly-robust estimator + influence-function SE.
- `crossfit.py` — K=5 cross-fit dispatcher.
- `positivity.py` — `PositivityGate(tau_refuse=0.01, tau_green=0.05)` (ADR-002).
- `sensitivity.py` — `e_value_for_continuous_effect()`.

### Recommender (`recommender/`, on the open PR #5)
- `rank.py::rank_pitch_types` — wraps g_compute in a ranking loop with positivity gating, tossup flag on overlapping 95% CIs.
- Pinned by 7 integration tests + 3 API tests.

### Inference (`inference/`)
- FastAPI app, 7 endpoints: `/health`, `/games`, `/at-bats`, `/ab-context`, `/query`, `/recommend` (on PR #5), `/pitchers`, `/pitcher/{id}/profile`.
- `AppState` lazy-loads NuisanceModels + val parquets on first request. Demo polish PR #4 adds a startup hook to warm it.

### Frontend (`frontend/`)
- 3 real tabs (~3000 LOC): CounterfactualExplorer, RolloutViewerTab, PitcherProfileTab.
- Design system locked per `frontend-system` skill (Inter, 3 colors, 4 type scales, color + glyph for pitch types).
- Wired to backend via Vite proxy at `/api/*` → `:8000`.

### Frontend tabs NOT YET built
- Recommender Tab — separate scope after PR #5 merges.
- Tipping page — separate substantial project (`tipping-analysis` skill).
- **MCSim Tab — App B's frontend (date carousel + per-game card + result overlay).** Will be a separate scope once App B's backend ships.

---

## How App B fits — pre-game daily batch architecture

User clarified (the framing matters; got it wrong twice before):

- **Score prediction** = simulate the WHOLE game from 0-0 top of 1st, ~10K times per game. Day-before-game prediction. Drives a daily prediction site.
- **Matchup report card** = static document a coach takes into the dugout. ~10K paths per (P, B) cell, ~63 cells per game.

App B is the second one and is being built first because:
1. Reuses single-AB `g_compute` directly — no multi-AB state machine needed.
2. Per-cell cost is ~41s at n_paths=1000 on CPU (measured in this session). 63 cells × 41s ≈ ~43 min per game; ~3h nightly for 4 games. CPU-feasible.

**No GPU needed at user's stated scales** — confirmed empirically by the `g_compute` benchmark in this session.

User requirement: **all predictions stored and re-readable via a date carousel after the game**, with the real result overlaid once the game finishes. Storage layer (commit 3) implements this — SQLite, three tables, COALESCE-on-update for two-pass actuals ingestion.

---

## Design decisions locked for App B (from brainstorm doc)

| # | Decision | Resolved as | Why |
|---|---|---|---|
| D1 | Storage backend | **SQLite** single file | Stdlib, queryable, ~MB-scale; perfect for this size |
| D2 | Reference context per cell | **Marginal** (0-0, no runners, 0 outs, mid-game) for v1 | Apples-to-apples comparable across cells |
| D3 | n_paths per cell | **1000** | Measured ~3h nightly for 4 games — comfortable budget |
| D4 | Natural rollout mode | **Add `intervention_type=None` to g_compute** | Done — commit 2 |
| D5 | Game-day cadence | **Once nightly** (evening before, probable lineups) | Day-of re-run is v2 |
| D6 | Grid scope | **Starter + bullpen (~6-8 arms) × starting 9 batters** (~63 cells) | Spec assumption |
| D7 | Calibration KPIs | **Three**: game-CI hit-rate, winner correctness, per-cell empirical-vs-predicted | All three surface different failure modes |
| D8 | Scheduler | **On-demand for dev, Modal cron for prod** | Don't ship a cron until end-to-end works manually |

Plus the matchup-cell mechanics question (was option A/B/C):

> **Option C — relax `g_compute`'s `intervention_position >= 1` guard.** Pitch 0's propensity lives at the last context-token position (NC-1); the causal mask blocks attention from there to pitch positions, so it works cleanly. Verified empirically: 2000-path natural-mode rollout's empirical type distribution matches π̂ within 3% — and π̂ is non-degenerate (modal first-pitch type > 20% probability).

---

## Storage schema (live, on disk)

`data/mcsim.sqlite` — gitignored. Three tables, `PRAGMA user_version = SCHEMA_VERSION = 1`.

- `predictions(id PK, game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json)` UNIQUE (game_pk, prediction_date, app)
- `actuals(game_pk PK, fetched_at, final_score_home, final_score_away, winner, matchup_events_json)` — COALESCE on UPDATE so two-pass ingest (line score → matchup events) doesn't null earlier fields
- `model_versions(ckpt_hash PK, trained_at, label, notes)`

Public API in `mcsim/storage.py`:
```
init_db                       open + create tables, idempotent
write_prediction              upsert (game_pk, date, app)
read_prediction               fetch one
read_predictions_for_date     date-carousel listing
write_actual                  two-pass COALESCE upsert
read_actual                   fetch one
register_model_version        idempotent provenance
lookup_model_version          fetch by ckpt_hash
```

---

## Critical conventions (read before writing model-interfacing code)

Documented in CLAUDE.md "Bug-prevention discipline." Headlines:

### The PAD-at-0 type vocab gotcha
The propensity TYPE head emits 8 logits where index 0 is PAD and indices 1..7 are PITCH_TYPES (FF..FS). `[:N_PITCH_TYPES]` = `[:7]` slices the WRONG columns. Use named constants from `data/dataset.py`:

```python
MODEL_PITCH_TYPES_START_IDX = 1   # FF lives here
MODEL_PITCH_TYPES_END_IDX = 8     # exclusive end
MODEL_TYPE_ID["FF"] = 1
```

Asymmetry: the RESULT head emits 7 logits with NO PAD column. Only TYPE has the off-by-one trap.

### MPS bug
`compute_losses` AB-outcome gather miscompiles on Apple silicon (Issue #2). CPU + CUDA fine; everything in this session's test infra forces CPU via `torch.backends.mps.is_available = lambda: False`.

### Datetime unit gotcha
Existing helper `_composite_sort_key` in `scripts/build_profile_cache.py` assumes `datetime64[ns]`. Pandas 3.x sometimes produces `[us]`. Tests pin the unit-invariance.

### Print named numerical outputs
Per CLAUDE.md's bug-prevention rule: **before claiming "smoke passed" on any model-interfacing code, print at least one named numerical output (e.g., `π̂(FF) = 0.47` on AB X).** The user explicitly enforces this — "what's π̂(FF) on the first AB?" is the canonical pushback.

---

## What's running in the background

Nothing should be running. If the user's restart was unclean and processes are stranded:

```sh
lsof -i :8000 -i :5173  # uvicorn + Vite — kill if present
ps aux | grep -E "build_profile_cache|build_matchup_cache|train_pitchgpt|run_matchup_cards"
```

---

## Files reference (load-bearing for App B work)

| File | Purpose |
|---|---|
| `docs/MCSim_brainstorm.md` | High-level MCSim design (both apps). Pre-game framing locked. |
| `docs/mcsim_appB_brainstorm.md` | **App B v1 spec.** Design decisions D1–D8. Implementation order. |
| `docs/recommender_brainstorm.md` | Recommender design (PR #5). Reused in App B's matchup-card thinking. |
| `mcsim/__init__.py` | Package docstring. |
| `mcsim/storage.py` | SQLite layer + 8 public functions. |
| `mcsim/state.py` | `ReferenceContext` + `build_synthetic_ab()`. |
| `mcsim/matchup_card.py` | `compute_matchup_card()` + `PitcherSpec`/`BatterSpec`. Step 4. |
| `causal/g_computation.py` | `g_compute(intervention_type=None, intervention_position=0)` both supported now; `RolloutResult.intervention_type_propensity` exposed for the trust gate. |
| `tests/test_g_compute_natural_mode.py` | 8 tests pinning D4 + Option C. |
| `tests/test_mcsim_storage.py` | 12 tests pinning the storage layer. |
| `tests/test_mcsim_state.py` | 11 tests pinning the state builder. |
| `tests/test_mcsim_matchup_card.py` | 6 tests pinning the card computer. |
| `model/pitchgpt_dataset.py` | `REQUIRED_AUG_COLS`, `PITCH_FACTOR_COLS_INT`, `CATEGORICAL_CTX_COLS` — the schema App B's state builder targets. |

---

## How to resume after the VS Code restart

1. **Open terminal in `/Users/sidthakur/Projects/PitchGPT`.**
2. **Confirm branch and clean tree:**
   ```sh
   git branch --show-current   # should print: mcsim-app-b-matchup-card
   git status --short          # should be empty
   ```
3. **Re-read this doc + `docs/mcsim_appB_brainstorm.md` § Implementation order.**
4. **Quick sanity that everything still works:**
   ```sh
   uv run pytest tests/test_mcsim_storage.py tests/test_mcsim_state.py tests/test_mcsim_matchup_card.py -q
   ```
   Should print: `29 passed`.
5. **Start step 5** — write `scripts/mcsim/run_matchup_cards.py` per the contract in "Next concrete step" above. Resolve the open design questions there first.
6. **Discipline:** before claiming anything works, print named numerical output (e.g., a real cell's median RV + π̂(modal type) from an end-to-end run that landed a SQLite row).
7. **When done with step 5**, commit + push + report to user. Don't push past step 5 without checking in.

---

## Key learnings from this session (so the next session doesn't repeat)

- **My ratings on novel ideas are not trustworthy.** Earlier in this session I rated an arsenal-mask experiment 8.5/10; tried it; NLL exploded because the trailing-window `has_pitch=0` flag is "absent from window," not "impossible." User pulled the recommendation privileges. Don't pitch model improvements; only ship what's measured.
- **The user wants you to think actual usage, not demo.** Pre-compute caches for curated demos are theatre; real workflows run nightly batches. SQLite + cron + actual results comparison is the right shape.
- **CPU is enough at the user's stated scales.** Don't reach for GPU prematurely. Measure first.
- **The user reads numbers.** "Smoke-tested" without showing named numerical output is unacceptable. Always print at least one named number per check.
- **Pre-game framing for MCSim.** Both apps are pre-game — score prediction simulates the whole game from 0-0, matchup card is a static doc. NOT live in-game.
- **The brainstorm-then-decisions-then-code pattern works.** Each major piece (recommender, App B) opened with a brainstorm doc, user signed off on decisions, then implementation followed. Don't skip the brainstorm step on big-enough work.
- **Pressure-test before claiming.** When something feels too good (n_paths scales sublinearly!) or too easy (cache will be quick to build!), verify with a measurement. Several wrong estimates in this session got caught only because the user asked for evidence.
- **PR scope discipline.** Each PR is one clean concern: PR #1 v7 Part 1, PR #4 demo polish, PR #5 recommender. App B will be one more. Don't bundle.

---

## End

If anything in this doc contradicts something in the actual codebase, the codebase is right and this doc is stale. Update this doc at the end of every session that materially advances the project state.

---

## small-v7 training — PREPARED (2026-06-03), needs a GPU to run

**Why:** tiny-v7 compresses hitter OPS ~6.8× on real data (real std 0.261 vs
model 0.038, Pearson r=0.66) — it regresses every hitter toward ~.700. Step 1
fix = more capacity (small: 6L/8H/d512, ~25M params vs tiny ~6M).

**Validated command** (smoke-tested end-to-end on CPU, exit 0):
```
PYTHONPATH=. python -m scripts.train_pitchgpt \
  --size small --fold 0 --type-conditioned-heads \
  --epochs 3 --batch-size 256 --run-name small-fold0-v7
```
- `--type-conditioned-heads` = the v7 (ADR-013) change; arsenal_per_pitch +
  propensity_situational are ON by default. (Optional, to fully match v7's EMD
  loss: `--zone-spatial-weight <w>` — needs the zone-centroids file; verify
  whether tiny-v7 used it before relying on it.)
- **Device reality:** this Mac's only GPU path is MPS, which has the AB-outcome
  gather miscompile (Issue #2) → training here would be WRONG (MPS) or far too
  slow (CPU, ~25M params on 7M pitches). **Full run needs CUDA/Modal.** The
  in-repo infra is `scripts/upload_to_modal.py` (upload only) — there is no Modal
  *training* entrypoint checked in; launching the GPU run is a manual/Modal step.

**After training:** `scripts.calibrate_pitchgpt` → `checkpoint_calibrated.pt`,
then re-run the compression diagnostic (real-vs-model OPS spread) to see if
capacity recovered the spread. If not → the dedicated hitter model
(`docs/Hitter_Swing_Model.md`).

## Hitter/swing model brainstorm — `docs/Hitter_Swing_Model.md`
Structural fix for the hitter-compression problem. Key calls: decompose into
swing→whiff→contact-quality nodes; compose a **tabular/tree** hitter model with
the existing pitch transformer inside `g_compute` (behind an `outcome_model`
flag); **a Transformer for the hitter model is most likely overkill** (batter
response to one pitch is tabular, not a long-sequence problem) and a tree is far
more interpretable (serves the interpretability project). Phased: small-v7 →
tabular hitter model → wire into rollout + A/B → (only if needed) sequence-aware.

---

## Hitter/swing model + small-v7 — IN FLIGHT (2026-06-03)

**small-v7 training:** LAUNCHED on Modal, detached (app `ap-55acWEObPHd29pPYMZibNh`,
`modal_app.py` train_remote, L4 GPU, `--size small --type-conditioned-heads
--epochs 3 --run-name small-fold0-v7`, ~10h). Checkpoints land on the
`pitchgpt-data` volume at `/data/checkpoints/small-fold0-v7/`. Pull when done:
`modal volume get pitchgpt-data checkpoints/small-fold0-v7 ./checkpoints_modal/`,
then `scripts.calibrate_pitchgpt`, then re-run the compression diagnostic.
Motivation: tiny-v7 compresses hitter OPS **~6.8× on real data** (real std
0.261 vs model 0.038, r=0.66) — capacity test.

**Hitter/swing model:** branch `hitter-swing-model`, package `hitter/` (design:
`docs/Hitter_Swing_Model.md`). Decision locked: **XGBoost** (spreadsheet-style,
not a Transformer — a batter's response to one pitch is short-range; feed
recent-pitch lag features instead of attention), composed with the pitch
sequence model in `g_compute` behind an `outcome_model` flag.
- ✅ `hitter/labels.py` — per-pitch swing/whiff/fair-contact + in-play outcome
  from `description`/`events`. Validated (swing 0.473, whiff-on-swing 0.248).
- TODO: `features.py` (pitch + count + **recent-pitch lags** + **batter profile
  via ProfileCache** + pitcher) → `train.py` (3 nodes) → `model.py` predict →
  `eval.py` (compression diagnostic, target spread-ratio ≈ 1× vs current 6.8×)
  → wire into `g_compute` → UI tabs.
- Baseline reality (repo's own leak-clean numbers): PitchGPT-small type top-1
  **0.478** > leak-clean LSTM **0.449**; the "0.691 LSTM" was LEAKY. No saved
  XGBoost number — building this produces it. (Transformer wins *pitch
  prediction*; XGBoost's case for the *hitter* model is capacity-focus +
  interpretability, NOT proven accuracy.)

**Also still running:** demo servers (uvicorn :8000 `bizkpwz6m`, vite :5173
`bilyl1h9s`) — the matchup-card tab. NOTE those cards are the n_paths=120
2026-06-04 run that PREDATES the as-of profile fix, so they're player-blind;
re-run a recent in-range date (≤2026-05-08) for meaningful cards.

---

# ▶ NEW-SESSION HANDOFF: build the hitter model + matchup cards (2026-06-03)

Context ran low mid-build. Everything needed to finish is below. Branch:
**`hitter-swing-model`**. Design is locked — read these 3 first:
- `hitter/MODEL_DESIGN.md` — the multi-stage cascade, features, **analytic
  count-tree composition** (no Monte-Carlo), open decisions (all have a "lean").
- `docs/Hitter_Swing_Model.md` — rationale + the ~6.8× compression that motivates it.
- This handoff.

## The goal in one line
A 3-model XGBoost cascade (swing → whiff → contact-quality) for the BATTER side,
composed with PitchGPT (pitch side) over the count tree, to fix the ~6.8×
hitter-OPS compression and power matchup cards / counterfactual / recommender —
and (later) full-game simulation (App A).

## What's DONE
- `hitter/labels.py` ✅ — per-pitch swing/whiff/fair-contact + in-play outcome
  from `description`/`events`. Tested (swing 0.473, whiff-on-swing 0.248).
- `hitter/features.py` ✅ — `build_base_features` (pitch+count+platoon+in_zone+
  within-AB lags) and `attach_batter_profile` (per-(batter,game) ProfileCache
  lookup, `as_of_fallback=True`, leakage-safe). Tested incl. real-data check that
  different batters get different profile vectors. `BASE_FEATURE_COLS` exported.
- `hitter/__init__.py`, `hitter/README.md`, `hitter/MODEL_DESIGN.md` ✅.

## Build steps (in order)
1. ✅ **DONE — `hitter/features.py`** (see above). Profile-join convention used
   (from `model/pitchgpt_dataset.py:244`): per AB `asof_date = first pitch
   game_date`, `asof_game_num = first["game_num"] if present else 1`,
   `ProfileCache(role="batter", fold_id=...).lookup(..., as_of_fallback=True)`.
   TODO when wiring training: also attach the **pitcher "stuff" profile** the same
   way (a `attach_pitcher_profile` mirror), and consider the finer 25-zone
   encoding for interpretability plots.
2. **`hitter/train.py`** — train the nodes (XGBoost), temporal split (train ≤2023,
   val 2024H1). Each node on its conditional population (swing=all; whiff=swings;
   fair=contact; contact-quality=balls-in-play). Monotonic constraints where
   sensible (chase↑ out-of-zone; xwOBA↑ middle). Per-node isotonic calibration.
   **S3 target = xwOBA-on-contact** (`estimated_woba_using_speedangle`) — NOT in
   `data/augmented/`; join from `data/raw/{year}/{date}.parquet`. Fallback for v0:
   discrete {1B/2B/3B/HR/out} from `events` (already in augmented). Save models to
   `checkpoints/hitter/`.
3. **`hitter/model.py`** — `HitterModel.predict(pitch_type, location, count,
   batter_feats, pitcher_feats, ctx) -> {swing, whiff, fair, xwoba/outcome}`.
   Loads the saved nodes; vectorized.
4. **`hitter/compose.py`** — the **analytic count-tree solve**. State = (balls,
   strikes[, last_pitch_type]). For each count, marginalize PitchGPT's
   pitch(type+zone) distribution × the cascade → transition probs over
   {ball, called/swing strike, foul (2-strike=no-op), in-play→out/1B/2B/3B/HR}.
   Build the absorbing-Markov transition matrix; solve for terminal
   distribution (walk, K, out, 1B, 2B, 3B, HR) in closed form → per-PA outcome →
   OPS/AVG/OBP/SLG/K%/BB%. (Keep a Monte-Carlo path as a cross-check.)
5. **`hitter/eval.py`** — THE acceptance test: the compression diagnostic
   (real OPS spread vs model OPS spread across a batter panel; transformer = 6.8×;
   target ≈ 1×). Plus per-node AUC/logloss/ECE, held-out hitters.
6. **Wire into `causal/g_computation.py`** behind `outcome_model="head" | "hitter"`
   — the cascade IS μ̂(y|do(pitch),h); keep π̂ from PitchGPT. AIPW/positivity/
   E-values still apply.
7. **Matchup cards** — `mcsim/matchup_card.py` gets an `outcome_model` /
   `compose="analytic"|"mc"` option → per cell, run the analytic composition →
   the existing payload (OPS/AVG/BB%/K% already shown in `frontend/src/MCSimTab.tsx`).
   Re-run `scripts/mcsim/run_matchup_cards.py` (with multiprocessing) for a recent
   IN-RANGE date (≤2026-05-08, so real profiles + actuals exist).

## small-v7 (capacity test, running in PARALLEL)
- Modal app `ap-55acWEObPHd29pPYMZibNh` (dashboard:
  https://modal.com/apps/siddhartha-thakur/main — find the `pitchgpt` app), L4,
  `--size small --type-conditioned-heads --epochs 3 --run-name small-fold0-v7`,
  ~10h. Check: `modal app list`. When done:
  `modal volume get pitchgpt-data checkpoints/small-fold0-v7 ./checkpoints_modal/`
  → `python -m scripts.calibrate_pitchgpt --ckpt .../checkpoint.pt` →
  re-run the compression diagnostic on it. If small-v7 recovers the hitter spread
  on its own, the dedicated hitter model may be less urgent; if not, the hitter
  model is the fix. (Either way the hitter model is the better long-term answer.)

## App A (full future-game simulation) — yes, this unblocks it
The per-PA outcome engine (steps 4–5) IS the core App A needs. A full game =
chain PAs through a GAME state machine: lineup cycling (1–9), base-out state
(use the RE24 / base-out tables in `data/run_value/`), inning/outs, score, both
bullpens. Backtest on past games (predictions vs real finals — `mcsim` storage +
`fetch_actuals` already exist) before predicting future games. App A is a
separate `mcsim` module on top of the same PitchGPT+hitter engine.

## The compression diagnostic (reuse verbatim as the gate)
Real OPS per batter from held-out `events`; model OPS via the engine; compare
spread (std) + Pearson r. Transformer baseline: real std 0.261 vs model 0.038
(6.8×), r=0.66. Target for the hitter model: ratio ≈ 1×.

## Misc state
- Demo servers may still be running (uvicorn :8000, vite :5173). The 06-04 cards
  in `data/mcsim.sqlite` are PLAYER-BLIND (predate the as-of fix) — re-run a
  ≤2026-05-08 date for meaningful cards.
- Unmerged branches: `mcsim-runner-mp` (mp + frontend + as-of fix + NaN fix + OPS
  + logging — NOT merged to main), `hitter-swing-model` (this work).
