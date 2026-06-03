# MCSim App B — Pre-Game Matchup Report Card (Brainstorm)

**Status:** brainstorm / not yet started
**Branch:** `mcsim-app-b-matchup-card`
**Depends on:** the recommender's `g_compute` reuse pattern (PR #5). Does **not** need PR #4 (demo polish) or PR #5 to be merged first — this is its own pipeline.

This is a planning doc. App B is the cheaper of the two MCSim apps because it reuses the existing single-AB `g_compute` directly — no multi-AB state machine, no pitcher-change model needed for v1.

---

## What App B is

A static **pre-game document** generated nightly for each of tomorrow's MLB games. For each game, a grid: every pitcher on the home team's staff (starter + bullpen) × every batter on the visiting team's lineup, and vice versa. Each cell is one expected run-value distribution under a reference context, with honest uncertainty.

> "Show me, before the game starts, how each of my available arms is projected against each of their hitters. For each cell: simulate 1,000 ABs under a reference context, return the run-value distribution + the top-1 outcome (K / BB / 1B / 2B / 3B / HR / out) + a trust flag."

**Stored. Re-readable via a date carousel after the game. Real result overlay once the game finishes.**

Decision the document supports: bullpen sequencing under leverage. A coach uses it to plan "P1 for high-leverage spots in the 7th, P2 for lefty-pocket in the 8th, closer for the 9th."

---

## End-to-end pipeline (one nightly run)

```
                    +----------------------+
                    | nightly scheduler    |  (Modal cron OR local crontab)
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | MLB Stats API client | --> pulls tomorrow's schedule:
                    |                      |     - probable starters
                    +----------+-----------+     - starting lineups (best-effort)
                               |                 - bullpen state (days_rest)
                               v
                    +----------------------+
                    | pre-game state builder|--> for each (pitcher, batter):
                    |                      |     - load pitcher/batter profiles
                    +----------+-----------+     - build synthetic AB at reference state
                               |
                               v
                    +----------------------+
                    | matchup card computer|--> for each (P, B) cell:
                    |                      |     - g_compute(n_paths=1000)
                    +----------+-----------+     - extract RV distribution, AB outcomes
                               |
                               v
                    +----------------------+
                    | storage writer       | --> SQLite predictions table
                    +----------------------+

                  ──── then, after each game ends ────

                    +----------------------+
                    | actuals fetcher      | --> MLB Stats API + Statcast
                    | (cron, ~3h post-game)|     - final score, winner
                    +----------+-----------+     - per-(P, B) ABs that occurred
                               |
                               v
                    +----------------------+
                    | storage writer       | --> SQLite actuals table
                    +----------------------+

                  ──── frontend reads on demand ────

                    +----------------------+
                    | GET /predictions     |  date-keyed; date carousel walks history
                    | GET /predictions/{pk}|  per-game card; overlay actuals when present
                    +----------------------+
```

Every box above is a small, isolated chunk. The compute box is the heaviest.

---

## What we already have (reuse map)

- `causal/g_computation.py::g_compute()` — the single-AB MC engine. Per the benchmark in this session: **~41 s at n_paths=1000 on CPU**.
- `causal/nuisance.py::NuisanceModels` — loaded checkpoint, exposes π̂ + μ̂.
- `data/profile_cache_loader.py::ProfileCache` — per-(role, fold) profile lookups.
- `inference/player_names.py` — name resolution for MLBAM IDs.
- `recommender/rank.py` — the ranking-loop pattern; **not used directly here** (App B doesn't rank candidates; it just produces a single per-cell distribution).
- `data.dataset.PITCH_TYPES`, run-value tables, etc. — model conventions.

---

## What needs to be built

### 1. MLB Stats API client (`scripts/mcsim/mlbstats.py` or similar)
- Pull tomorrow's schedule from `statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD`.
- For each game: probable pitchers, projected lineups (best-effort from `linescore` / `boxscore` endpoints), bullpen state.
- Post-game: pull final line score from `boxscore`; per-AB matchups from `live/feed` endpoint.
- ~200 LOC of HTTP + JSON parsing. ToS allows non-commercial use.

### 2. Pre-game state builder (`recommender/synthetic_ab.py` or `mcsim/state.py`)
- Given `(pitcher_id, batter_id, asof_date, reference_context)`, build a one-pitch synthetic AB the model can roll out from.
- The "reference context" is the count + base/outs state at which we evaluate (D2 below).
- Output: a DataFrame in the same schema `g_compute` consumes (`ab_pitches`).

### 3. Matchup card computer (`mcsim/matchup_card.py`)
- For each game and each (P, B) cell: call `g_compute(n_paths=1000)`.
- Per cell: median RV + 5/95% percentile band + top-1 outcome + trust flag (from π̂ at the natural choice).
- Total: ~63 cells × ~41 s = ~43 min per game.

### 4. Storage (`mcsim/storage.py` + a SQLite db)
- The three tables from the spec: `predictions`, `actuals`, `model_versions`.
- Single-file DB at `data/mcsim.sqlite` (gitignored).
- Plain Python `sqlite3` stdlib — no ORM.

### 5. Post-game actuals fetcher (`mcsim/actuals.py`)
- ~3 h after first pitch, pull line score + AB-by-AB matchups.
- Match each AB to the predicted cell by `(pitcher_id, batter_id)`.
- Write to `actuals` table.

### 6. Read API endpoints (extension of `inference/api.py`)
- `GET /mcsim/predictions?date=YYYY-MM-DD` — list of games + their matchup cards.
- `GET /mcsim/predictions/{game_pk}` — full card for one game (predictions + actuals if present).
- `GET /mcsim/calibration?window=30d` — aggregated calibration KPI for the date carousel header.

### 7. Frontend Tab (separate scope, later)
- Date carousel.
- Per-game card: score-prediction header + matchup grid.
- Real-result overlay where present (per-cell empirical vs predicted; per-game winner / final score vs predicted CI).
- The 30-day calibration KPI front and center — the project's epistemic-humility story turned into a live number.

---

## Design decisions

### D1 — storage backend
- **(a) SQLite**, single file. Stdlib, queryable, easy to back up, perfect for this scale (~3–4 games × N days × 63 matchups × ~1 KB each = a few MB per year).
- **(b) Plain JSON files** on disk, one per game per date. Easier to git-track in early dev; awkward for "list all predictions for last 30 days" queries.
- **(c) Postgres**. Real DB infrastructure. Overkill for v1.

**Lean: (a) SQLite.** Confirmed in the prior chat. Migrate to Postgres only if multi-process or external integration ever matters.

### D2 — reference context for cells
The cell shows "what would happen if THIS pitcher faces THIS batter at..." what?

- **(a) Marginal context** — count 0–0, no runners, 0 outs, mid-game inning (4 or 5), neutral score. Simplest, most-comparable across cells. Probably the right v1.
- **(b) Realistic-context distribution** — sample the context from the distribution the pairing is likely to actually face (starter usually faces leadoff with no runners; closer usually faces high-leverage). Better fidelity, but cells aren't directly comparable.
- **(c) Multiple contexts per cell** — ship both, or ship (a) as the main grid + (b) as a per-cell deep-dive.

**Lean: (a) for the v1 main grid; (c) as a v2 click-through.** Marginal context is the right starting place; comparing cells should be apples-to-apples.

### D3 — sample size per cell (`n_paths`)
The benchmark showed `g_compute` is roughly linear in `n_paths`:

| n_paths | per-cell | per game (63 cells) | 4 games nightly |
|---|---|---|---|
| 500 | 21 s | 22 min | ~1.5 h |
| 1000 | 41 s | 43 min | ~2.9 h |
| 2000 | 85 s | 89 min | ~5.9 h |

**Lean: 1000 paths per cell.** Stays within a comfortable overnight window (~3 h on CPU for 4 games), and the CI tightens enough to make cell rankings stable. Drop to 500 if compute budget bites; raise to 2000 only if the demo-quality of CI bands matters for the UI.

### D4 — how to roll out "natural" play in a cell
The matchup card cell wants the **natural** behavior of pitcher P vs batter B at the reference state — *not* an intervened pitch type. But `g_compute` requires an `intervention_type`.

Two clean options:

- **(a) Sample-then-rollout**: at the intervention position, sample the type from `π̂(type | h)`, then call `g_compute(intervention_type=sampled_type, n_paths=1)` per realization. Repeat 1000 times. Honest: the rollout reflects the pitcher's natural mix, weighted by his propensities. **Cost: 1000 × 1-path g_compute calls — ~10× slower than one 1000-path call, because each call pays the trunk-pass overhead.**
- **(b) Type-marginal rollout via 7 batched g_computes**: run `g_compute(intervention_type=t)` for each of the 7 types at `n_paths = 1000 × π̂(t)`, then concatenate. Honest mixture; matches (a) in expectation. **Cost: 7 × the per-cell time = ~5 min per cell, too slow.**
- **(c) Add a "no-intervention" mode to g_compute** that samples the type naturally at the intervention position before continuing the rollout. Minor refactor (`intervention_type=None` → sample from π̂). **Cost: identical to the existing per-cell time. Cleanest.**

**Lean: (c).** Modest refactor to `causal/g_computation.py`: when `intervention_type` is `None`, sample from `π̂(type|h)` per path. Same external surface for `g_compute`, no new function needed downstream. The recommender already calls `g_compute(intervention_type=specific)`; that path is untouched.

### D5 — game day cadence
Lineups firm up ~3 h before first pitch. Options:

- **(a) Once nightly**, evening before, using *probable* lineups. Final lineups missing for ~5% of cells.
- **(b) Twice**: evening preview + day-of morning re-run with confirmed lineups.

**Lean: (a) for v1.** Marginalize over plausible probable-lineup variations later if it's a real problem (probably not for v1 evaluation).

### D6 — scope per game
Who's in the grid?

- **Rows (pitchers):** starter + every reliever on the active 25-man with `days_rest >= 1`. Typically 6–8 arms.
- **Columns (batters):** the 9 starters in the opposing lineup. Key bench bats deferred to v2.

**Lean: 6–9 pitchers × 9 batters = ~54–81 cells per game.** Average ~63 (the spec assumption).

### D7 — calibration metric for the 30-day KPI
What does "the model was right" mean?

- **(a) Game-level:** % of games where the actual final score was inside the predicted 80% CI.
- **(b) Game-level winner:** % of games where the predicted winner won.
- **(c) Per-cell:** for each predicted matchup that actually occurred, % where the actual outcome was inside the per-cell predicted distribution.

**Lean: show all three on the calibration KPI panel.** They surface different failure modes.

### D8 — scheduler (Modal cron vs local crontab)
- **(a) Modal cron** — `modal.Schedule.cron("...")`. Runs in the same image as training. CPU-only function, ~3 h, ~few cents per night.
- **(b) Local crontab** — run on a always-on box (the user's Mac / a Pi / a small EC2).
- **(c) On-demand triggered** — manual nightly run for now, scheduler later.

**Lean: (c) for v1 dev; (a) for production.** Don't ship a cron until the pipeline works end-to-end manually.

---

## Storage schema (concrete SQL)

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk            INTEGER NOT NULL,
    prediction_date    TEXT    NOT NULL,      -- "YYYY-MM-DD" (the *game* date)
    made_at            TEXT    NOT NULL,      -- ISO timestamp
    model_ckpt_hash    TEXT    NOT NULL,
    app                TEXT    NOT NULL,      -- "matchup_card" | "score_prediction"
    payload_json       TEXT    NOT NULL,      -- the full grid / score-dist
    UNIQUE (game_pk, prediction_date, app)
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions (prediction_date);

CREATE TABLE IF NOT EXISTS actuals (
    game_pk             INTEGER PRIMARY KEY,
    fetched_at          TEXT NOT NULL,
    final_score_home    INTEGER,
    final_score_away    INTEGER,
    winner              TEXT,
    matchup_events_json TEXT                 -- per-(P, B) actual ABs that occurred
);

CREATE TABLE IF NOT EXISTS model_versions (
    ckpt_hash TEXT PRIMARY KEY,
    trained_at TEXT,
    label TEXT,                              -- "v7 tiny fold0", etc.
    notes TEXT
);
```

Payload JSON shape for a matchup card (per game):

```json
{
  "game_pk": 776543,
  "home_team": "NYY",
  "away_team": "BOS",
  "starter_home": { "pitcher_id": 543037, "name": "Gerrit Cole" },
  "starter_away": { "pitcher_id": 657277, "name": "Brayan Bello" },
  "rows": [
    { "pitcher_id": 543037, "name": "Gerrit Cole", "is_starter": true,
      "cells": [
        {
          "batter_id": 605141, "batter_name": "Mookie Betts",
          "predicted_rv_median": -0.012,
          "predicted_rv_p05": -0.087, "predicted_rv_p95": +0.061,
          "predicted_top1_outcome": "K",
          "predicted_outcome_dist": {"K": 0.28, "BB": 0.10, "1B": 0.16, ...},
          "p_hat_top_type": 0.41,
          "trust_state": "green",
          "n_paths": 1000
        },
        ...
      ]
    },
    ...
  ],
  "n_paths_per_cell": 1000,
  "reference_context": { "count": "0-0", "runners": "empty", "outs": 0, "inning": 5 }
}
```

---

## Open questions

- **Probable-starter accuracy:** the MLB Stats API `probablePitcher` is ~95% correct night-before. What's the right fallback when it's wrong? Skip the game? Use last-start pitcher? Worth measuring once we have data.
- **Held-out / debutant pitchers:** model handles them OK via the league-fallback path in the profile cache, but should the trust flag drop a tier? Probably yes.
- **Reference inning** for D2: inning 4 vs 5 — does it matter? Inning-bucket categorical is a context input; experimentally check whether late-inning effects show up at this level.
- **Per-cell ESS / multi-step positivity:** the rollouts are single-AB so this is straightforward, but worth surfacing the per-cell n_truncated and ESS in the payload.
- **Statcast enrichment timing for actuals:** line score is near-real-time, but pitch-by-pitch enrichment is 1–2 days. Two-pass actuals fetcher: quick line-score fetch on game-end, full Statcast match-by-AB the next morning.

---

## Honest constraints (to surface in any UI that ships this)

- Model trained ≤ 2023 Statcast. 2026 game predictions are 2+ seasons OOD.
- Reference-context cells abstract away mid-game leverage; the realistic-context per-cell deep-dive is a v2 add.
- Statcast pitch-type classification is noisy at the slider/cutter and sinker/four-seam boundaries — propensities for these pairings can swap.
- App B's matchup cells are **predictive, not causal**. The recommender's positivity-gating semantics provide trust flags, but these cells are model rollouts, not AIPW estimates. UI copy must use predictive language.
- Per the `causal-layer` skill: game-level predictions are never causal estimates. Same applies here.

---

## Implementation order

1. **D4 refactor** — add `intervention_type=None` natural-sampling to `g_compute` (the cleanest path per D4). Tests pinning that natural rollout reproduces the propensity distribution at position k. **~half day.**
2. **`mcsim/storage.py` + SQLite schema** — write/read primitives, model-version table, basic round-trip tests. **~half day.**
3. **`mcsim/state.py`** — pre-game synthetic-AB builder. Given `(pitcher_id, batter_id, reference_context)`, produces the DataFrame `g_compute` wants. **~half day.**
4. **`mcsim/matchup_card.py`** — for one game's `(pitchers, batters)` list, compute all cells, return + persist the card. **~half day.**
5. **`scripts/mcsim/run_matchup_cards.py`** — end-to-end CLI: takes a date + game_pk list, writes cards to SQLite. **~half day.**
6. **`scripts/mcsim/mlbstats.py`** — MLB Stats API client (schedule, probable pitchers, lineups, bullpen state). **~1 day.**
7. **`scripts/mcsim/fetch_actuals.py`** — post-game scraper. **~half day.**
8. **`GET /mcsim/predictions`** + `GET /mcsim/predictions/{pk}` API endpoints. **~half day.**
9. **Live test** — run nightly on real upcoming games, manually inspect cards. **~few nights.**
10. **Frontend tab** — separate scope.

**Approximate total: ~5–6 focused days for v1 backend + scheduler. Frontend is its own project.**

---

## Things explicitly out of scope (for v1)

- Full game simulation (that's App A, separate project — needs the multi-AB engine).
- Pitcher-change modeling (App A territory).
- Realistic-context per-cell deep-dive (D2 (c)) — v2.
- Bench-bat / pinch-hitter modeling — v2.
- A betting / handicapping framing — different ToS, different model-quality bar.
- GPU inference — measured CPU is sufficient at the user's scales.
