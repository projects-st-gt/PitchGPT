# App A — Full-Game Simulation Engine (design)

**Date:** 2026-06-05
**Branch:** `hitter-swing-model` (do NOT merge to main yet)
**Status:** spec — pending user review, then `writing-plans`
**Brainstorm source:** `docs/ContextSwitcher.md` → "⭐⭐⭐⭐⭐ APP A — FULL-GAME SIM ENGINE"

## 1. Goal & scope

Simulate a whole MLB game from 0–0, top of the 1st, ~10K Monte Carlo times to
produce a **projected score, run-total distribution, and win probability** —
**pre-game** (not live in-game) — and **backtest** it against real finals.

App A is the **per-GAME layer**. The two layers beneath it are already built and
validated and are NOT rebuilt here:

- **Per-pitch** (pitchGPT π̂ + hitter cascade μ̂) — what pitch, what the batter
  does to it. ✅
- **Per-at-bat** (`g_compute(outcome_model="hitter")` → a matchup-card cell's
  `predicted_outcome_dist` over `['K','BB','1B','2B','3B','HR','out']`) — chains
  pitches into a PA *result*. ✅ **This is App A's per-PA fuel.**

App A adds the missing piece: **PA results → runs → score**, which requires a
base-running model the lower layers say nothing about.

### Out of scope (v0)
- Live / in-game updating.
- Daily-prediction path + demo tab (a later step; this spec ends at a working,
  backtested engine).
- The matchup-card "actual H-AB overlay" (a separate small App-B/demo display
  task — data already exists in `actuals.matchup_events_json`; not part of the
  engine).
- Player-level base-running (sprint speed), pitch-around / TTO context in the
  per-PA dist, and individual SB/CS/error events — all documented v0 limitations.

## 2. Key feasibility insight (already decided, restated)

Do **not** re-run the ~32s/PA rollout inside the game loop. The per-(pitcher,
batter) outcome distribution is **precomputed once** — that is exactly what a
matchup-card cell already holds. The 10K game sims just **sample** from those
precomputed dists through a fast pure-Python state machine. A game is ~76 PAs;
10K games × 76 ≈ 760K cheap iterations — runs locally in seconds, **no Modal
needed for the sim itself** (Modal is only for generating the cards).

## 3. Decisions locked this session

| Decision | Choice |
|---|---|
| Base-running model | **(A) Consecutive-PA empirical** base-out Markov transition matrix from real PBP (see §5). (B) text-parsing and (C) deterministic rules rejected. |
| Pitching changes | **Rule-based hook, behind a pluggable policy.** Pull starter when simulated batters-faced ≥ his recency-weighted personal average; cycle bullpen by order. A `ScriptedActuals` policy is left as a future backtest-diagnostic seam. |
| Pitcher workload stat | Not in `PitcherSpec`/profiles today → **derive** a real **recency-weighted avg batters-faced-per-start** from raw PBP, trailing window ending strictly before the game date, exponential decay favoring recent starts; league-average fallback for low-sample arms. (hard-rule-1a compliant: data-derived, no magic threshold.) |
| Extra innings | **Ghost runner on 2nd** to start each extra half-inning (2020+ rule). |
| Backtest set | **June 4–5 2026 only** (cards already exist). Reported as a **mechanics / sanity check**, NOT a calibration claim — ~24 games is too small N for credible win-prob calibration. A larger 2024H2/2025 slate is a future step. |
| Storage app key | Reuse the **existing `app="score_prediction"`** slot (`mcsim/storage.py:52,149`) — supersedes the ContextSwitcher's `"game_sim"` name. No schema change. |

## 4. Architecture & module boundaries

```
data builds (train-split tables + as-of-date features)
  data/run_value.py            (+ build_base_out_transition_matrix)  → data/run_value/base_out_transition.parquet
  data/pitcher_workload.py     (new) build_pitcher_workload(as_of)   → recency-weighted BF/start per pitcher

engine (new package: gamesim/)
  gamesim/outcomes.py          EVENTS→7-class map + 25-state base-out encoding constants
  gamesim/transition.py        BaseOutTransition: load + sample((from_state, outcome), rng) → (to_state, runs)
  gamesim/state.py             GameState.step(outcome) → runs; half-inning/inning/extras/ghost-runner logic
  gamesim/bullpen.py           PitchingPolicy interface; RuleBasedHook(workload_table)
  gamesim/montecarlo.py        simulate_game(card, lineups, starters, policy, n_sims, rng) → GameSimResult
  gamesim/backtest.py          run_backtest(date): cards+actuals → win-prob/run-total/score-dist metrics

CLI
  scripts/gamesim/run_game_sim.py    run sims for a date, persist via storage(app="score_prediction")
  scripts/gamesim/run_backtest.py    June 4–5 backtest harness

reuse (unchanged): mcsim/storage.py, scripts/mcsim/fetch_actuals.py, data/run_value/ tables, the matchup cards
```

Each unit has one job and a typed interface; each is independently testable. The
transition matrix and workload are pure data builds; the engine is pure Python
with an injected RNG (deterministic given a seed); the Monte Carlo and backtest
are thin orchestrators over them.

## 5. Base-out transition matrix (the correctness-critical build)

**Formulation — 25-state base-out Markov.** States = the 24 RE24 cells
(`base_state` 0–7 × `outs` 0–2, same encoding as `data/run_value.py`:
`on_1b*4 + on_2b*2 + on_3b*1`) **plus one absorbing `INNING_OVER` state**. This
makes GIDP, sac flies, and inning-ending PAs all fall out uniformly — a PA that
records the 3rd out simply transitions to `INNING_OVER`.

**Source data:** train split only, **2017–2023** raw PBP in `data/raw/` (a
league-structural table, same discipline as RE24 per the `statcast-pipeline`
skill and hard rule 2). Era drift in base-running is a documented caveat when
applied to 2026 backtest games.

**Construction (per real PA):**
1. Reduce per-pitch parquet to per-PA rows (`events` non-null), grouped within a
   half-inning, ordered by at-bat number.
2. Map `events` → 7-class outcome via `gamesim/outcomes.py:EVENT_TO_OUTCOME`:
   - `strikeout`, `strikeout_double_play` → `K`
   - `walk`, `intent_walk`, `hit_by_pitch`, `catcher_interf` → `BB`
     (base-forcing equivalent; the 7-class vocab has no HBP/CI)
   - `single`→`1B`, `double`→`2B`, `triple`→`3B`, `home_run`→`HR`
   - `field_out`, `grounded_into_double_play`, `double_play`, `force_out`,
     `sac_fly`, `sac_fly_double_play`, `sac_bunt`, `fielders_choice`,
     `fielders_choice_out` → `out`
   - `field_error`, `truncated_pa`, anything else → **dropped** (not
     representable in the card's outcome space; ~1% — a documented v0 limitation:
     the sim omits error-driven runs).
3. `from_state = (base_state_before, outs_before)`.
4. `runs = post_bat_score − bat_score` (captures runs that score even on an
   inning-ending PA).
5. `to_state`:
   - if a subsequent PA exists in the same half-inning →
     `(next.base_state, next.outs_when_up)`
   - else → `INNING_OVER`.
6. Aggregate counts over `(from_state, outcome) → (to_state, runs)`; normalize to
   a probability per `(from_state, outcome)`. Persist long-format parquet with
   `n` per cell for trust/coverage.

**Cells with no data** (rare illegal/never-observed combos) raise on sample —
never silently default — per hard rule 1a.

### TDD — hand-checkable cells (print NAMED numbers, per CLAUDE.md discipline)
- `HR` from `(bases_empty, 0 outs)` → `to_state` mass entirely on
  `(bases_empty, 0 outs)`, `P(runs=1)=1.0`. Assert and print.
- `HR` from `(bases_loaded, 1 out)` → `P(runs=4)=1.0`. Print.
- `1B` from `(runner_on_2nd, 0 outs)` → print `P(runner scores)` (= mass on
  `runs≥1`); assert it's in a sane empirical band (~0.5–0.65) — the analog of the
  `π̂(FF)` named-number check.
- `out` from `(runner_on_1st, 0 outs)` → print `P(GIDP-like: to_state has 2 outs
  recorded)` is non-trivial (> 0.05), proving the bucket carries double plays.
- `BB` from `(bases_loaded, 2 outs)` → `P(runs=1)=1.0` (forced run) — proves the
  matrix learns base-forcing without special-casing.

## 6. GameState (pure Python, TDD)

State: `inning, is_top, outs, base_state, score_home, score_away,
lineup_ptr[home/away] (0–8, cycles), current_pitcher[home/away]`.

`step(outcome) -> runs_scored_this_pa`:
1. Sample `(to_state, runs)` from the transition matrix for
   `((base_state, outs), outcome)`.
2. Add `runs` to the batting team.
3. Advance the **batting team's** lineup pointer (the batter completed his PA
   regardless of the result — next PA is the next hitter).
4. If `to_state == INNING_OVER` → flip half-inning: reset `outs=0`,
   `base_state=empty`, toggle `is_top` (and increment `inning` when flipping from
   bottom→top). On the first PA of an extra half-inning (`inning > 9`), seed the
   **ghost runner** on 2nd.
5. Else set `(base_state, outs)` from `to_state`.

Game loop: play 9 innings; if tied after the bottom of the 9th, play extra
innings (ghost runner) until a half-inning ends with unequal scores at its
boundary. Standard walk-off / no-bottom-9th-if-home-leads rules applied. A
`MAX_INNINGS` safety backstop (e.g. 30) guards against pathological inputs (an
all-out card can never resolve a tie) — on hitting it the loop stops and records
the current score; the engine logs that the backstop fired so it never silently
masks a real bug.

### TDD — hand-checked cases
- `1B` from bases-empty/0-out with a transition fixture forcing
  "batter to 1st, 0 runs" → `base_state == on_1st`, `outs == 0`, returns 0.
- A forced `INNING_OVER` transition → outs reset to 0, bases empty, half flips.
- Bottom-9th home team takes the lead → game ends immediately (walk-off).
- Tie after 9 → extra inning starts with runner on 2nd (assert `base_state`).
- Full deterministic mini-game with a fixed RNG + fixture matrix → exact final
  score is hand-recomputable.

## 7. Bullpen policy (pluggable)

`PitchingPolicy` interface: `current_pitcher(team, batters_faced, inning,
score_diff) -> pitcher_id`.

`RuleBasedHook(workload_table)`: keep the starter until simulated
`batters_faced ≥ round(workload[starter])` (clamped to a sane floor/cap, e.g.
[18, 30] BF), then advance through the team's bullpen list in order. `workload`
is the recency-weighted BF/start from `data/pitcher_workload.py`. The per-PA fuel
for relievers already exists in the card grid (`get_active_roster` returns the
full active staff).

Future seam (not built now): `ScriptedActuals` replays the real box-score
pitching sequence to isolate engine error in backtests.

### TDD
- A starter with `workload=22` is replaced exactly when the 23rd batter comes up.
- Bullpen exhaustion falls back to the last reliever (never crashes / never
  reuses the starter).
- Workload build: a synthetic pitcher with recent heavy starts + old light
  starts yields a BF/start closer to the recent value (recency weighting works);
  print the named number.

## 8. Monte Carlo & result

`simulate_game(card, home_lineup, away_lineup, home_starter, away_starter,
policy, n_sims, rng) -> GameSimResult`:
- For each sim, run the GameState loop; each PA looks up the
  `predicted_outcome_dist` for `(current_pitcher_id, current_batter_id)` from the
  card grid, samples an outcome, and steps.
- Aggregate over sims: `GameSimResult` with home/away final-score arrays,
  `win_prob_home`, projected score (mean/median), run-total distribution, and a
  per-score-margin histogram.

Persistence: `scripts/gamesim/run_game_sim.py` writes the result payload via
`mcsim.storage.write_prediction(app="score_prediction")`, keyed
`(game_pk, date)`, with the card's `model_ckpt_hash` for provenance.

### TDD / sanity (named numbers)
- Degenerate card where every cell is `P(out)=1` → every sim is a 0–0 tie that
  hits the `MAX_INNINGS` backstop; assert `mean_total_runs == 0.0` and that the
  backstop fired (this is exactly the pathological input the backstop exists
  for). Print both.
- Card with inflated HR mass → `mean_total_runs` rises monotonically; print it.
- A real June-4 card → print projected score + `win_prob_home` and eyeball
  against the actual final.

## 9. Backtest harness (June 4–5 2026)

`gamesim/backtest.py:run_backtest(date)`:
- For each game with both a card (`read_prediction`) and an actual
  (`read_actual`, via `fetch_actuals`), simulate and compare.
- Metrics: (1) **win-prob reliability** (bucketed predicted vs realized — flagged
  as low-N descriptive on ~24 games), (2) **run-total error** (predicted mean/
  median vs actual total; MAE + bias), (3) **score-distribution coverage** (did
  the actual final land within the simulated 5–95 band?).
- Output a small table + the per-game rows; persist nothing beyond the score
  predictions already written.

D8 discipline: backtest on these completed games before any future-date
prediction path is built.

## 10. Hard-rule compliance checklist
- **Real data only / 1a:** transition matrix + workload derived from real PBP;
  empty cells raise, never default; no magic thresholds (workload is
  data-derived, hook clamp bounds are explicit engineering guards, not stand-ins
  for data).
- **Temporal split:** transition matrix from 2017–2023; workload uses trailing
  data ending strictly before each game date (leakage-safe, like profiles).
- **Named-number discipline:** every smoke test prints a named probability/score,
  not "passes".
- **No causal language:** App A outputs are **predictive model rollouts**, not
  causal estimates — UI/copy uses "projected", "the model expects", never
  "effect of".

## 11. Build order (→ writing-plans will expand)
1. `gamesim/outcomes.py` + `build_base_out_transition_matrix` (+ tests, §5).
2. `gamesim/transition.py` loader/sampler (+ tests).
3. `data/pitcher_workload.py` recency-weighted build (+ tests, §7).
4. `gamesim/state.py` GameState.step + game loop (+ tests, §6).
5. `gamesim/bullpen.py` RuleBasedHook (+ tests).
6. `gamesim/montecarlo.py` simulate_game (+ sanity tests, §8).
7. `gamesim/backtest.py` + `scripts/gamesim/*` (+ June 4–5 run, §9).

Each step: TDD, print named numerical output, commit per step, keep
`docs/ContextSwitcher.md` updated live.
