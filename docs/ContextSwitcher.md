# ContextSwitcher — Pick up where this session left off

**Last updated**: 2026-05-14, post 14-zone migration commit.

This is the handoff doc for a new Claude session. Read it cold; the project state
below is everything you need to keep going.

## The user's immediate priority for the next session

**14-zone migration code is LANDED (2026-05-14).** Backend schema changes,
preprocess re-run, frontend rework, and tests are all green. Remaining work
is the longer-running cache/standardizer/training pipeline:

1. Fold-0 profile cache build (in flight at session-end — check `/tmp/cache_14zone_fold0.log`).
2. Refit the profile standardizer once fold-0 cache lands (uses the new 118 / 57
   profile dims).
3. Smoke-print named numerical outputs from a real AB (e.g., `π̂(FF)` on AB X
   with the new schema) — the bug-prevention discipline says do this BEFORE
   claiming the retrain is sane.
4. Folds 1-4 cache rebuild (overnight, ~20h).
5. Modal training fold 0 as a smoke test (~$10, ~5h). Need user OK on Modal spend.
6. Folds 1-4 training (~$40, ~5h each).

## Current state of the project

### What's been built (Sprint 1 + 2 mostly complete)

- **Causal layer** (`causal/`): 6 modules — `nuisance.py`, `positivity.py`,
  `sensitivity.py`, `g_computation.py`, `aipw.py`, `crossfit.py`. All smoke-tested
  end-to-end on real ABs. The g-computation rollout uses the *rigorous* state machine
  (count progression, natural termination on 3 strikes / 4 balls / in-play).
- **Inference backend** (`inference/`): FastAPI app at `inference/api.py` with
  endpoints `/health`, `/games`, `/at-bats` (game-filtered), `/ab-context`, `/query`.
  Player names via Chadwick register cache. Game-team lookup cached at
  `data/preprocess_artifacts/game_teams_2024.parquet`.
- **Demo frontend** (`frontend/`): Vite + React + TypeScript + Tailwind, Apple-minimalist.
  Two-step picker (game → AB), scoreboard with ESPN-CDN team logos, strike-zone widget
  with batter silhouette + whiff heatmap, intervention controls with arsenal-aware
  disabled pitches, continuous trust gauge (no hard refusal — ADR 002 Option D), result
  panel with effect + CI + E-value + dotted-baseline outcome distribution.

### Key UX pivots from earlier in the session (load-bearing)

1. **No hard refusal** (ADR 002 Option D): the rollout always runs; the gauge shows
   `high / moderate / low` support. Refusal-as-block was the previous design; user
   pushed back. We now ALWAYS surface counterfactual numbers but label low-support
   queries as "not a causal claim."
2. **Separate type/zone thresholds**: `tau_refuse=0.01 / tau_green=0.05` for type;
   `tau_refuse=0.003 / tau_green=0.02` for zone (because 26 cells means per-cell
   marginals are ~3-7%). After 14-zone retrain these may want re-tuning.
3. **Plain-English everywhere**: no `π̂(type)` math notation in user-facing copy.
   "Throwing a slider: 33% — common" / "In that specific zone: 0.50% — essentially
   never goes there."

### What's running in the background (as of session end)

- `build_profile_cache --role batter --folds 1,2,3,4` (PID 21179, ~4h elapsed).
  **KILL THIS** before starting 14-zone work — it's building at the current v4/26-zone
  schema, which we're about to bump to v5/14-zone.
- FastAPI uvicorn on port 8000 (process probably still alive).
- Vite dev server on port 5173.

### What's stale / known issues

- **None of folds 1-4 are trained on Modal yet.** Cross-fit AIPW machinery exists
  but can't run end-to-end until those checkpoints exist. v5-A1 is the only trained
  checkpoint and is fold 0 — and it's pre-14-zone, so it's now invalid on the new
  schema.
- **14-zone migration code is in.** Augmented parquets regenerated under
  `pitchgpt_schema_version=2`, `feature_zone ∈ [0, 12]`. Fold-0 cache rebuild
  running at session-end.
- **All previously-trained checkpoints are invalid** under the new schema
  (n_zones embedding shrunk 26→13, profile dims shrunk 230→118 / 105→57).
  Loading will fail with a state-dict mismatch — that's the intentional guard rail.
- **Test set untouched** (2024-H2 + 2025 + 2026 partial). This is *correct discipline*
  — only touched once after writeup is locked.

## Critical conventions (read before writing model-interfacing code)

Already documented in CLAUDE.md's "Bug-prevention discipline" section. Repeating
the headline because it has bitten this project three times in one session:

**The PAD-at-0 type vocab gotcha**: the propensity TYPE head emits 8 logits where
index 0 is PAD and indices 1..7 are PITCH_TYPES (FF..FS). `[:N_PITCH_TYPES]` =
`[:7]` slices the WRONG 7 columns (PAD + first 6 of 7 types, missing FS). Use
the named constants from `data/dataset.py`:

```python
MODEL_PITCH_TYPES_START_IDX = 1   # FF lives here
MODEL_PITCH_TYPES_END_IDX = 8     # exclusive end
MODEL_TYPE_ID["FF"] = 1
```

**Asymmetry warning**: the RESULT head emits 7 logits with NO PAD column. So result
head reads use `[:N_RESULTS]` correctly. Only TYPE has the off-by-one trap.

**Convention is encoded as constants** in `data/dataset.py` lines 33-58. Always
import from there.

## 14-zone retrain — detailed plan (priority #1 for next session)

### Step 1: 14-zone scheme — VERIFIED + LANDED (2026-05-14)

**Scheme**: Statcast / SIS native 14-zone (the labels go 1-14 but zone 10 doesn't
exist, so there are **13 actual zones**). The Statcast `zone` column already
publishes this directly — we just remap to dense internal indices via
``data.zones.SIS_TO_INTERNAL``:

| SIS label | Internal idx | Location |
|---|---|---|
| 1, 2, 3 | 0, 1, 2 | In-zone top row (left, middle, right) |
| 4, 5, 6 | 3, 4, 5 | In-zone middle row |
| 7, 8, 9 | 6, 7, 8 | In-zone bottom row |
| 11 | 9 | Upper-left OOZ quadrant |
| 12 | 10 | Upper-right OOZ quadrant |
| 13 | 11 | Lower-left OOZ quadrant |
| 14 | 12 | Lower-right OOZ quadrant |

**Why direct mapping (not compute from plate_x/plate_z)**: Statcast's zone
classifier accounts for the per-batter strike zone (sz_top/sz_bot). It's more
umpire-accurate than rolling our own from a flat plate_x box.

**NaN handling**: Statcast `zone` is NaN for ~0.3% of "pitches" — these are
book-keeping rows (`automatic_ball` from intentional walks since 2017,
`automatic_strike` from pitch-clock violations since 2023). They have NaN
plate_x/plate_z/pitch_type too — not real pitches. Drop at preprocess.

### Step 2: Code changes — LANDED

All changes shipped in the 14-zone-migration commit:

| File | Change | Status |
|---|---|---|
| `data/zones.py` | Added `assign_feature_zone_14()` (Statcast `zone` column → dense 0..12). Added `SIS_TO_INTERNAL` / `INTERNAL_TO_SIS` maps, `N_FEATURE_ZONES_14=13`, `N_IN_ZONE_CELLS_14=9`. Kept legacy `assign_feature_zone()` for back-compat. | ✓ |
| `data/preprocess.py` | `harmonize_and_tag()` switched to `assign_feature_zone_14()`. Drops NaN-zone rows. | ✓ |
| `data/preprocess_pitchgpt.py` | `SCHEMA_VERSION` 1 → 2. | ✓ |
| `model/config.py` | `n_zones: 26 → 13`. `pitcher_profile_dim: 230 → 118`. `batter_profile_dim: 105 → 57`. | ✓ |
| `data/profile_cache.py` | `PROFILE_SCHEMA_VERSION: 4 → 5`. Heatmap auto-shrinks (7×9=63), grids auto-shrink (3×9=27) via `N_IN_ZONE_CELLS`. | ✓ |
| `data/player_profiles.py` | `N_IN_ZONE_CELLS: 25 → 9`, `OUT_OF_ZONE_CELL: 25 → 9`. Gating swapped from `== OOZ` to range checks (`< N_IN_ZONE_CELLS` / `>= N_IN_ZONE_CELLS`). | ✓ |
| `scripts/build_profile_cache.py` | Added `"zone"` to `NEEDED_COLUMNS`; switched to `assign_feature_zone_14`. | ✓ |
| `inference/api.py` + `schemas.py` | Comments / API contract updated to length-13 zone vector. | ✓ |
| `causal/g_computation.py` | One comment fix. | ✓ |
| `tests/test_zones.py` | Added 6 new tests for `assign_feature_zone_14` (SIS→internal map, NaN refusal, bad-label refusal, missing column, bijection). | ✓ |
| `tests/test_profile_cache.py`, `tests/test_player_profiles.py`, `tests/test_dataset*.py`, `tests/test_preprocess.py` | Updated fixtures using out-of-range zone values; updated `swing_z` / `heatmap_z` slot-index assertions. | ✓ |
| `frontend/src/types.ts` | Rewrote `gridCellToFeatureZone` / `featureZoneToGridCell` for 3×3 in-zone. Added `OOZQuadrant` type + `OOZ_QUADRANT_TO_FEATURE_ZONE` / `isOOZ` / `oozQuadrantOf` helpers. | ✓ |
| `frontend/src/StrikeZone.tsx` | Re-drew grid as 3×3 in-zone + 4 clickable OOZ quadrant rects (UL/UR/LL/LR). Heatmap render loops 5→3. Observed-pitch placement uses quadrant centers for OOZ. | ✓ |

### Step 3: Run order

1. **Kill the running batter-cache build** (`pkill -f "build_profile_cache.*batter"`).
2. **Modify `data/zones.py`** with the 14-zone scheme. Run tests.
3. **Re-run preprocess on all years**:
   `cd /Users/sidthakur/Projects/PitchGPT && make preprocess` (or equivalent). ~30-60 min.
4. **Update profile_cache.py + player_profiles.py + tests**. Run pytest.
5. **Rebuild full profile cache for fold 0** (pitcher + batter) as a smoke test.
   `uv run python -m scripts.build_profile_cache --role both --folds 0` (~3h).
   **Verify dims look right before scaling up.**
6. **Rebuild folds 1-4** (pitcher first ~2h, then batter ~8h). Use the
   `python -u` flag or set `PYTHONUNBUFFERED=1` so the log isn't buffered.
7. **Refit standardizer**: `uv run python -m scripts.fit_profile_standardizer`.
8. **Upload all caches + standardizer** to Modal volume (`modal volume put`).
9. **Train fold 0 on Modal** as a smoke test (~5-6h, ~$10). Verify accuracy
   isn't catastrophically lower. Expected: π̂ top-1 around 0.47, μ̂ around 0.52
   (vs 0.478/0.542 at 26-zone). Drop of ~0.5-2pp is expected and acceptable.
10. **If smoke OK, train folds 1-4** for cross-fit (~5h each × 4 = 20h sequential,
    parallel if Modal slots permit). ~$40.
11. **Sprint 6: production retrain through 2025** (TRAIN_END = 2025-12-31). Can
    bundle with the 14-zone work — one combined retrain. ~$15-20 extra.
12. **Update frontend `StrikeZone.tsx`** to render the new 14-zone layout.

### Step 4: What to verify

- π̂ top-1 on val should be ~0.47 (was 0.478 at 26-zone). Slight drop expected.
- ECE should be similar or slightly better (~0.005).
- Positivity: per-cell marginal goes from ~3.8% → ~7.1%. Trust gauge should
  refuse far less often.
- All 29 tests in `tests/test_profile_cache.py` should pass.

### Step 5: Total cost estimate

- Engineering: ~5 days.
- Compute: ~30h sequential / ~10h parallel.
- Dollars: ~$50-60 Modal + ~$15-20 for the bundled Sprint 6 retrain.
- Calendar: ~1 week.

## Other pending work (after 14-zone)

### Sprint 1 finish (causal layer end-to-end validation)

The causal layer is BUILT but hasn't been validated end-to-end on a real query yet
(per task #21). Once folds 1-4 land:

1. Run K=5 cross-fit AIPW on one slice (e.g., "all 0-2 counts to RHB, SL vs FF").
2. Verify: `AIPW ≈ g-computation` within 0.02 runs/PA.
3. Run a negative control: same intervention on "next batter's PA outcome" should
   give effect ≈ 0.
4. Run a positivity-violation test: pick a query that should refuse, confirm it does.
5. Run rollout stability: top-10 effects at N=100 vs N=1000 → rank corr > 0.9.

Pass criteria: all 5 → green light to demo + writeup. Failure: diagnose before
proceeding.

### Sprint 3 — Tipping detector (Tab 6)

Per ADR 005 + the tipping-analysis skill. Builds on `arm_angle` (Sprint 0b is done).

1. Train a *batter-observable* classifier on the visible-cue features only:
   `(arm_angle, release_pos_x/y/z, release_extension, prior pitches in AB,
   count, runners, outs)` → next pitch type.
2. Compute `T_start` per (pitcher, pitch type): earliest AB position where this
   classifier beats marginal pitch-type accuracy by ≥ 5pp.
3. Run the 4 validation checks in ADR 005.
4. Build the UI (Tab 6 in `docs/UX_Ideas.md`).

Effort: ~1.5 weeks.

### Sprint 2 finish — Demo polish + Tab 1

- Add Tab 1 (Play-by-play X-ray) per `docs/UX_Ideas.md`.
- Wrap current counterfactual in a tab structure.
- Polish: better silhouette? player photos? past-matchups footer?

Effort: ~1 week.

### Sprint 4 — Methods writeup

After Sprints 1-3 land, write the methods document. Note the constraint from
CLAUDE.md: don't write causal claims until validation table is green.

Effort: ~1 week of focused writing.

### Sprint 6 — Production retrain through 2025-12-31

`TRAIN_END = 2025-12-31`. Bundle with the 14-zone retrain to save compute.

### Sprint 7 (optional) — Live demo (Tab 3)

Per `docs/UX_Ideas.md` Tab 3. ~3 weeks.

**Prerequisite TODO: nightly profile-cache refresh job for live games.**
Right now the profile cache (`data/profiles/{role}_fold_{k}.parquet`) is a
static snapshot built once by `scripts/build_profile_cache.py`. For live
demo, today's games would have no cache entry and inference would fall
back to league-mean / zero-fill (collapsing the player-specific profile).

Path 1 (chosen, cheapest): a nightly cron job that:

1. Runs `make extract START=<yesterday> END=<yesterday>` to fetch the
   prior day's Statcast data.
2. Runs `make preprocess` to regenerate yesterday's augmented parquet
   (idempotent — only writes new files since `SCHEMA_VERSION` guard).
3. Runs `scripts.build_profile_cache --role both --folds 0` for fold 0
   only (the inference-serving fold) to refresh cache entries for any
   player who played yesterday. Important: only fold 0 — the cross-fit
   folds 1-4 should NOT be refreshed mid-evaluation period since that
   changes the held-out content of historical caches.

Acceptance: today's evening game can be queried in the demo and the
pitcher/batter profiles reflect prior-night content. Stale by ≤12h.

Open considerations:

- Fold 0 incremental build vs full rebuild: the current
  `build_profile_cache.py` rebuilds the entire fold-0 cache (~3h). For a
  nightly job we want incremental — only add new (player, asof_date)
  keys for yesterday's games and append. Needs a `--since DATE` flag and
  append-mode parquet write.
- Standardizer freezing: the standardizer (`profile_standardization.npz`)
  must NOT be refit nightly — that would shift the per-feature scale
  used by the trained checkpoint. Lock it post-training and version with
  the checkpoint.
- Failure mode: if the nightly extract fails, the demo silently degrades
  to league-mean fallback. Should log + alert.

Effort to land path 1: ~3 days (incremental cache + cron + dashboard alert).

### Stretch / opportunistic

- Tabs 4, 5, 7-12 per `docs/UX_Ideas.md`.
- Task #25 (autoregressive factor sampling, R3-style) — fixes the joint sampling
  approximation in g_compute.
- Task #13 (audit order-dependent consumers for parquet-order leak) — still pending.

## Files reference (only the load-bearing ones)

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project hard rules + Bug-prevention discipline. Read first. |
| `docs/UX_Ideas.md` | Tab 1-12 designs + cross-cutting principles. |
| `docs/iteration-results.md` | Running log of model iterations (v1-v5). |
| `docs/decisions/` | ADRs. Locked decisions. ADR 002 (positivity, with Option D pivot) + ADR 006 (cross-fit) are load-bearing. |
| `data/dataset.py` | `MODEL_TYPE_ID` + `PITCH_TYPES` + `RESULT_CLASSES` + temporal split constants. |
| `data/zones.py` | Strike-zone definitions. **Will be modified for 14-zone.** |
| `data/profile_cache.py` | Profile schema + `PROFILE_SCHEMA_VERSION`. **Will be bumped to 5.** |
| `data/player_profiles.py` | Per-player profile builders. |
| `data/preprocess_pitchgpt.py` | Raw → augmented parquet pipeline. |
| `model/config.py` | `PitchGPTConfig`. Has all the flag toggles. |
| `model/pitchgpt.py` | Main model class. |
| `model/pitchgpt_dataset.py` | Dataset + collate. |
| `causal/g_computation.py` | Rigorous rollout state machine. |
| `causal/aipw.py` | AIPW estimator + influence-function SE. |
| `causal/crossfit.py` | K=5 cross-fit dispatcher (needs folds 1-4). |
| `causal/positivity.py` | Trust gauge thresholds. |
| `causal/sensitivity.py` | E-values. |
| `inference/api.py` | FastAPI endpoints. |
| `inference/schemas.py` | Pydantic request/response models. |
| `inference/player_names.py` | Chadwick register name lookup. |
| `frontend/src/App.tsx` | Main React app. Two-step picker, scoreboard, result panel. |
| `frontend/src/StrikeZone.tsx` | The strike-zone + silhouette widget. **Will need 14-zone rework.** |
| `frontend/src/Scoreboard.tsx` | Scoreboard component with ESPN team logos. |
| `frontend/src/types.ts` | TypeScript types mirroring Pydantic. |

## How to resume

1. Read this doc fully.
2. Read `CLAUDE.md` (especially "Bug-prevention discipline").
3. Check what's still running: `ps aux | grep build_profile_cache`, `lsof -i :8000 -i :5173`.
4. If batter cache build still running at 26-zone: kill it.
5. Start the 14-zone work per the plan above.
6. **Discipline**: before any model-interfacing code, print named numerical outputs.
   The PAD-at-0 trap has bitten three times.

## Key learnings from this session (so next session doesn't repeat)

- **Python pipes to `tee` block-buffer stdout**. Use `PYTHONUNBUFFERED=1` or `python -u`
  for long-running scripts so logs are visible mid-run.
- **uvicorn needs to run from project root**. `cd /Users/sidthakur/Projects/PitchGPT &&
  PYTHONPATH=. uv run uvicorn inference.api:app --host 127.0.0.1 --port 8000`.
- **Pandas `pd.NA in bool()` raises**. For columns that can be NA, use
  `pd.notna(val) and val` (in that order — short-circuit).
- **Schema-version checks are guard rails, not enemies**. When they fire, that's
  the system catching you trying to load incompatible cache + checkpoint.
- **Always run `npm run build` from `frontend/`** — the `cd frontend` may fail in
  some bash session contexts if the CWD has reset.
- **The user's tolerance for retraining is HIGH** — "stop shying away from retrain
  if it improves things." Don't defer expensive-but-correct moves.
- **The user reads numbers**. "Smoke-tested" without showing actual numerical output
  is not acceptable. Always print at least one named number per check.

Done. New session: start with `pkill -f "build_profile_cache.*batter"` and read this
doc end-to-end before touching anything.
