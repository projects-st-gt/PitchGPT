# ContextSwitcher — Pick up where this session left off

**Last updated**: 2026-06-01, mid-implementation of MCSim App B (pre-game matchup card). Step 4 (matchup-card computer) is done, tested, committed, and pushed. Step 5 (CLI runner) is next.

This is the handoff doc for a new Claude session (or a VS Code restart). Read it cold; the project state below is everything you need to keep going.

---

## TL;DR — where to resume

You are in the middle of building **MCSim App B (pre-game matchup card)** on branch `mcsim-app-b-matchup-card`. Six commits in. Step 4 (the matchup-card computer) just landed. Next concrete step:

> **Implement `scripts/mcsim/run_matchup_cards.py`** — the end-to-end CLI runner that takes a date + game_pk list, calls `compute_matchup_card`, and persists each card via `storage.write_prediction`. This is the first place the storage layer and the card computer meet.

See "Next concrete step" section below for the precise contract.

---

## The user's immediate priority for the next session

Continuing App B v1 backend (~3 more focused days):

1. ~~**Step 4 — `mcsim/matchup_card.py`**~~ ✅ done (commit `67f0629`)
2. **Step 5 — CLI runner** (`scripts/mcsim/run_matchup_cards.py`) ← next
3. **Step 6 — MLB Stats API client** (`scripts/mcsim/mlbstats.py` — schedule, probable pitchers, lineups, bullpen days-rest)
4. **Step 7 — Post-game actuals fetcher**
5. **Step 8 — Read API endpoints** (`GET /mcsim/predictions?date=...`, `GET /mcsim/predictions/{game_pk}`)

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

## Next concrete step — `scripts/mcsim/run_matchup_cards.py` (Step 5)

The CLI runner that takes a date + game_pk list, calls `compute_matchup_card`
for each game, and persists each card via `storage.write_prediction`. This is
the first place the storage layer and the card computer meet.

Open design questions to resolve at the top of Step 5 (Step 6's MLB Stats API
client doesn't exist yet, so lineups/pitchers must come from somewhere):

- **Where do `PitcherSpec`/`BatterSpec` lists come from for v1?** A hand-authored
  JSON/YAML game-spec file passed via `--spec`? Or stub IDs for a smoke run
  until Step 6 lands the real fetcher?
- **Model checkpoint loading** — reuse `NuisanceModels` loader; record
  `model_ckpt_hash` via `storage.register_model_version`.
- **Idempotency / re-run** — `write_prediction` upserts on (game_pk, date, app);
  CLI sets `app="matchup_card"` (the validated key; `storage.py:125` rejects
  anything but `"matchup_card"`/`"score_prediction"`).
- **CLI surface** — `--date`, `--game-pk` (repeatable) or `--spec`, `--n-paths`,
  `--rng-seed`, `--db-path`, `--dry-run` (compute + print, don't write).

Estimated time: ~half day. Tests should stub `compute_matchup_card` (or use
n_paths=50, 1 game, 2×2 grid) so CI stays fast.

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
