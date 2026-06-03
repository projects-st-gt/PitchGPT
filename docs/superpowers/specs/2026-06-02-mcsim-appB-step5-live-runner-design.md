# MCSim App B — Step 5+6 (merged): live MLB-API matchup-card runner

**Date:** 2026-06-02
**Branch:** `mcsim-app-b-matchup-card`
**Status:** design approved (user, 2026-06-02); spec under review before plan.

## Summary

Build the end-to-end runner that turns a calendar date into persisted pre-game
matchup cards, pulling real games and rosters **live from the MLB Stats API**.
This deliberately merges the originally-separate Step 6 (MLB client) into Step 5,
per the user's decision to pull from the API now rather than hand-author a spec
file.

The grid is **all rostered pitchers × all rostered position players** (both
halves of the game), not the posted 9-batter lineup — because MLB does not
publish confirmed lineups until ~2–4h before first pitch, whereas active rosters
are known the night before. An all-vs-all card is also a better dugout document:
it helps *build* a lineup rather than react to one.

## Motivation / findings (empirically verified 2026-06-01)

Probed against `https://statsapi.mlb.com`:

| Run timing | Probable pitchers | Real lineups | Actuals |
|---|:---:|:---:|:---:|
| Past date (all 2025) | yes | yes (9/side) | yes |
| Today, day-of (Pre-Game/Warmup) | yes | yes (~hrs pre-pitch) | — |
| Tomorrow / future | yes (13–14/15) | **no (0 games)** | — |

This falsifies brainstorm decision D5 ("night-before with probable lineups").
Resolution: skip lineups, grid the full active roster.

Other verified facts:
- `requests 2.33.1` + `pybaseball 2.2.7` installed; no existing MLB client code.
- statsapi player IDs are MLBAM = same namespace as Statcast `pitcher`/`batter`
  (no crosswalk needed). e.g. Skenes = 694973.
- Roster endpoint `/api/v1/teams/{id}/roster?rosterType=active&date=…` returns
  26 players splitting into ~13 pitchers + ~13 position players by
  `position.type == 'Pitcher'`, available the night before.
- **Per-cell benchmark (v7 ckpt, CPU, 8 threads): 39.87s @ n_paths=1000, LINEAR
  in n_paths (~33–40 ms/path).** The earlier "sublinear" hunch was wrong.

## Compute reality

All-vs-all ≈ **338 cells/game** (2 × ~13 × ~13) vs the old 63-cell plan. At
n_paths=1000 that is ~57.7h for 15 games — not nightly-feasible single-process.
Because cost is linear, n_paths is a proportional knob (n_paths=250 ≈ 14 min/game).

**Decision:** v1 is built correct and **sequential**, validated on one real game
end-to-end, then true all-vs-all wall-clock is measured before any nightly-batch
tuning (multiprocessing / subsetting). This honors D8 ("no cron until manual
end-to-end works"). Multiprocessing is explicitly out of scope for v1.

## Architecture — two units

### 1. `mcsim/mlb_api.py` — MLB Stats API client (isolated, testable)

Thin wrapper over `statsapi.mlb.com` (`requests`, no auth). Public surface:

- `get_schedule(date: str) -> list[GameInfo]`
  GameInfo: `game_pk`, `home_team_id`, `away_team_id`, `home_team`, `away_team`,
  `home_probable_pitcher_id | None`, `away_probable_pitcher_id | None`.
- `get_active_roster(team_id: int, date: str) -> tuple[list[PitcherSpec], list[BatterSpec]]`
  Splits the active roster by `position.type`; attaches handedness
  (`throws` / `stand`) from the person hydrate. Returns the exact
  `PitcherSpec` / `BatterSpec` dataclasses `compute_matchup_card` consumes.

Rationale for a separate module: the HTTP concern is the one flaky failure
source here and needs its own mocked tests; isolating it keeps
`compute_matchup_card` pure and the runner orchestration-only.

### 2. `scripts/mcsim/run_matchup_cards.py` — orchestrator

Per-run flow:
1. Load `NuisanceModels(--ckpt, device="cpu")` once. Compute `model_ckpt_hash`
   (truncated sha256 of the ckpt file — **no helper exists, add one**).
   `register_model_version(conn, ckpt_hash=…, label=…)`.
2. `get_schedule(--date)`; optionally filter to `--game-pk`; cap at `--max-games`.
3. Per game: `get_active_roster` for both teams → feed
   **home pitchers × away hitters** and **away pitchers × home hitters** into
   `compute_matchup_card` (which already loops arbitrary-length lists — **no
   change to `mcsim/matchup_card.py`**). Probable starter → `is_starter=True`.
4. `write_prediction(conn, game_pk=…, prediction_date=--date,
   app="matchup_card", payload=card, model_ckpt_hash=…)`.
5. Per-game log line: cell count, wall-clock, players skipped.

CLI surface:
- `--date` (required, `YYYY-MM-DD`)
- `--game-pk` (repeatable; filters schedule)
- `--n-paths` (default **250**)
- `--rng-seed`
- `--ckpt` (default `checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt`)
- `--db-path` (default `mcsim.storage.DEFAULT_DB_PATH`)
- `--max-games`
- `--dry-run` (compute + print named numbers; do NOT write)

## Data flow

```
--date ──► get_schedule ──► [GameInfo...]
                               │  (per game)
        get_active_roster(home), get_active_roster(away)
                               │
        (home_pitchers × away_hitters) + (away_pitchers × home_hitters)
                               │
                     compute_matchup_card  ──► payload dict
                               │
                     write_prediction(app="matchup_card")  ──► SQLite row
```

## Error handling (honest-by-default)

- Player with no trailing-window profile → per-cell `try/except`, **skip with a
  logged warning**, never fabricate. (Exact failure mode of `build_synthetic_ab`
  on an unknown id to be confirmed in implementation.)
- Game with no probable pitcher → still runs; `is_starter` simply not set.
- API failure on one game → log and continue to the next game; do not abort the
  whole batch.
- No fabricated pitches/lineups anywhere (hard rule #1). Roster/schedule metadata
  from the API is real; profiles remain real Statcast trailing windows.

## Testing

- `tests/test_mcsim_mlb_api.py` — unit tests with a **mocked HTTP layer** (canned
  schedule + roster JSON; no live calls in CI). Assert GameInfo parsing,
  pitcher/hitter split, handedness attachment, probable-pitcher extraction.
- `tests/test_mcsim_run_matchup_cards.py` — one small end-to-end test with the
  API client monkeypatched to return 2 pitchers × 2 hitters, n_paths=50, a temp
  SQLite db; assert a row lands and round-trips via `read_prediction`. Include
  one **named-numerical** assertion (a real cell's median RV / π̂(modal type))
  per CLAUDE.md bug-prevention discipline.
- Manual validation gate (not CI): run against ONE real game end-to-end, print
  named numbers, confirm a real SQLite row, report true wall-clock.

## Flags to verify during implementation (not assumed)

1. Whether the roster `person` hydrate actually exposes `pitchHand` / `batSide`
   (handedness is required by `build_synthetic_ab`).
2. The exact missing-profile failure mode (so the per-cell skip catches the right
   exception).

## Out of scope for v1

- Multiprocessing / nightly-batch performance tuning (revisited after measuring
  real all-vs-all wall-clock).
- Modal cron scheduler (D8 — only after manual end-to-end works).
- Projected lineups for future-date coverage (the all-vs-all roster grid removes
  the need; a `--project-lineups` mode could be a later opt-in).
- Post-game actuals fetch (Step 7) and read API endpoints (Step 8).
