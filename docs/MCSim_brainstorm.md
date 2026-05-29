# MCSim — Pre-Game Monte Carlo Simulator (Brainstorm)

**Status:** brainstorm / not yet started
**Sequencing:** after demo polish and recommender (queued behind them by user's priority order)
**Engine:** reuses `causal/g_computation.py::g_compute()` as the per-AB sampler

This is a planning doc, not a spec. The goal is to capture ideas before they're forgotten and surface open questions so the actual implementation plan can be written from a real starting point — not from scratch.

**Important framing decision (2026-05-29):** both applications are **pre-game**, not live. They run the day before the game from announced inputs (probable starter, lineup, weather forecast, bullpen availability) and produce *artifacts a person uses* — a daily prediction page and a manager's prep document. This removes the live-ingestion problem entirely but makes **multi-AB game state machinery and pitcher-change modeling non-optional**, not deferred.

---

## What we're trying to build

A Monte Carlo simulator that, given a pre-game starting configuration, samples N forward rollouts of the entire game using PitchGPT as the per-pitch propensity and outcome model. Two headline applications:

### A. Daily score prediction (pre-game, ~24 h before first pitch)
> "Tomorrow's Yankees vs Red Sox game at Fenway, 7:10 PM start, 64°F clear, Probable starters Cole vs Bello. Simulate the whole game **N = 10,000 times** from a **0–0 top of the 1st** (not from any mid-game state), return: score distribution, win probability, expected run totals per team, and the in-game leverage moments most likely to swing the result."

**Key: every sim starts at the start.** Not given a mid-game state. The model imagines tomorrow's game from first pitch through last out, 10K independent times.

Inputs are all things published the day before the game:
- Probable starters
- Probable lineups (tentative; firms up ~3 h before first pitch)
- Bullpen availability + days-rest for each arm
- Weather forecast (temp, wind, roof state)
- Ballpark, umpire crew

Drives a **daily prediction site** — one card per scheduled MLB game, published every evening for the next day's slate. The card surfaces the score distribution + win probability + a short "what to watch" callout (e.g., "Cole projects 6 IP / 2 ER with a 30% chance of giving up a HR to Judge"). Honest framing throughout: predictive, not causal; OOD-flagged where appropriate.

### B. Pre-game matchup document (a sheet a coach takes into the dugout)
> "Show me the full grid — **every one of my pitchers × every one of their hitters** — so I can decide ahead of time who to pitch against whom. For each `(my pitcher, their batter)` cell, simulate **~10,000 at-bats** and give me an expected run-value distribution with honest uncertainty."

This is a *static pre-game document*, not an in-game tool. The coach generates it the morning of the game and walks it into the manager's meeting. The decision the document supports is **bullpen sequencing under leverage** — given the opponent's lineup and my available arms, who comes in for whom and when.

**Grid shape (concrete v1):**
- Rows: every pitcher available to me — starter + every reliever in the bullpen (~6–8 arms).
- Columns: every batter in the opponent's likely lineup (~9 starters + key bench bats).
- One cell per `(pitcher, batter)` pair. **N = 10,000 single-AB sims per cell.**
- Cell contents (decision-relevant numbers, not just one mean):
  - Median expected run value + 5/95 percentile band
  - Probability of "any run scoring from this AB" (BB / 1B / 2B / 3B / HR outcomes)
  - Top-1 outcome prediction (K / BB / in-play / etc.)
  - Trust flag — green/yellow/red from the positivity gate based on π̂(this pitcher actually throws his typical mix to this batter in a realistic context).

Two reasonable framings for what each cell's *context prior* is:

- **Marginal-context cell** — count 0–0, no runners, 0 outs, neutral inning. Comparable across cells, easy to reason about. **Probably the right v1.**
- **Realistic-context cell** — average over the context distribution this pairing is likely to actually face (starter usually faces leadoff with no runners; closer usually faces high-leverage spots). Decision-relevant but cells aren't directly comparable.

For v1, ship the marginal grid as the main page. The realistic-context per-cell deep-dive can be a v2 click-through.

Critically: these are **single-AB sims**, not full-game sims. **No multi-AB state machine, no pitcher-change model needed.** That's what makes the matchup document much cheaper to build than the daily score prediction (Application A) — it's the right place to start.

---

## What we already have (engine pieces to reuse)

- **`causal/g_computation.py::g_compute()`** — single-AB MC engine. Returns per-path terminal step, terminal kind (K/BB/in-play), AB outcome, run value, log-weights for ESS. Already battle-tested in the demo. *This is the core.*
- **`causal/nuisance.py::NuisanceModels`** — loads a calibrated checkpoint, exposes π̂ + μ̂.
- **`causal/positivity.py`** — multi-step ESS tracking — already refuses rollouts that wander out of support.
- **Run-value table (`data/run_value/`)** — RE24-style mapping from AB outcomes to expected runs given (base state, outs).

---

## What needs to be built

### 1. Multi-AB / inning state transitions
After an AB terminates, we need to evolve the *game* state, not just the AB state:

- **Runner advancement** from the AB outcome (K → no advancement; BB/HBP → walk-on runners; 1B/2B/3B/HR → standard advancement; outs → context-dependent advancement; double plays → complex special case). RE24 already encodes some of this implicitly but as expected runs, not as state transitions.
- **Outs counter**.
- **Inning rollover** at 3 outs (top↔bottom flip, base state clears, batting team flips).
- **Game termination** (9 innings, walk-off rules, extras).
- **Next batter** — needs the lineup; rotate by 1 within batting team.

### 2. Pitcher-change modeling (the specific ask)
A pitcher coming out is a *decision*, not a sample from PitchGPT. Three approaches, all worth considering:

#### (a) Manager-rule heuristics
Hand-coded triggers approximating "average managerial behavior":
- **Pitch count thresholds** — pull starter at ~100 pitches; pull reliever at ~30.
- **Third-time-through-order** for starters (well-known TTO penalty — managers increasingly pull before TTO3).
- **High-leverage matchup rules** — late innings, base/out state determines closer entry, lefty-specialist swap, etc.
- **Bases loaded + high pitch count** — natural pull trigger (user's idea).
- **Score state** — blowouts (5+ run lead) → leave starter in to "eat innings"; close games → pull earlier and stretch the bullpen.

#### (b) Per-pitcher empirical hook
Compute, from historical Statcast:
- **Average innings pitched per appearance** for this pitcher (user's idea — easy from data).
- **Average pitch count at pull** under different score/baseout states.
- **Days-rest sensitivity** (already encoded via `days_rest_bucket` in the profile — could be reused).

Then sample pull-time as a stochastic function of the live state, calibrated to that pitcher's history.

#### (c) "Pre-script" the bullpen
Simplest for the recommender / matchup-permutation case: the user specifies the pitching plan ("P1 for 5 IP, then P2 for 2, then P3"). No model decision needed — sim runs the plan and reports outcomes. Useful for "what-if" planning even without a full pull-time model.

**My read:** start with (c) for the matchup-permutation case (it's the cheapest path to a usable tool), add (a) as a v1 game-sim, defer (b) to "if we can show it's worth the complexity."

### 3. Lineup / roster state
- Starting lineup (9 batters in order).
- Available bullpen with `days_rest` per pitcher (already computable from the existing pitcher profiles).
- Substitutions — pinch hitters, double-switches, pitcher batting (NL-style) — likely deferred.

### 4. Multi-step positivity / ESS handling for game-level sims
For a 30-AB game-level rollout, the inverse-propensity weights compound. The existing `causal/positivity.py` ESS tracker will report collapse, and we have two honest choices:
- **Surface refusals at the game level**: *"the rollout left the trust region after inning 6 — score distribution is shown only for the in-support prefix."* Most honest.
- **Drop AIPW-style causal-validity framing for the sim outputs**: frame as "expected behavior under the model" — predictive, not causal. The `causal-layer` skill is explicit that game-level predictions are not causal estimates.

For the matchup-permutation use case (single AB), this is a non-issue.

### 5. Pre-game data ingestion (the practical version)
Since both apps are pre-game, we don't need a live feed. The inputs we need are all things published the day before:

- **Probable pitchers** — published the day before (MLB Stats API endpoint, ~95% accuracy).
- **Lineups** — published ~3 h before first pitch. For the daily prediction site, this means a late-evening or morning re-run as lineups firm up. For the manager's prep document the coach generates it after their own lineup is set.
- **Bullpen availability + days-rest** — derivable from the prior 1–2 days of game logs (who threw, how many pitches, who's unavailable).
- **Weather forecast** — third-party (OpenWeather, Visual Crossing) or MLB's published ballpark conditions. Maps onto the `temp_bucket` and `roof_state` categoricals the model already uses.
- **Ballpark + umpire crew** — published the day before. Direct lookups for the `ballpark` / `umpire` categoricals.

A nightly job pulls these via `pybaseball` + a small MLB Stats API client, builds the pre-game state for each scheduled game, runs the sim, writes the prediction artifact + the manager prep doc. **No real-time polling required.**

Statcast's 1–2 day enrichment latency is fine here — by the time we predict tomorrow's game, today's game has fully enriched data, which is what feeds the day-of pitcher profiles.

---

## Open questions worth surfacing now

- **Score-distribution CIs that are honest about model uncertainty, not just sampling error.** With 10K paths and an ~30-AB game, sampling error on final score is ~0.01 runs — but model uncertainty (architecture, training data, OOD on 2026) is much larger. How do we surface this without a quantified posterior? Probably: explicit caveats on the UI + an OOD indicator per inning.
- **2026 = OOD beyond test split.** The model's test split is 2024 H2 + 2025; 2026 game predictions are an extrapolation. Held-out-pitcher generalization held up at single-AB; game-level multi-AB amplifies that error.
- **What's the unit the manager actually wants in the prep document?** Expected run value? Probability of any run scoring? Probability of multi-run inning? Whiff rate? The cells need to match the decision-relevant statistic — worth asking an actual coach before designing the page.
- **Marginal vs realistic context for the matchup cells.** The marginal version (0-0 count, no runners, 0 outs) is comparable across cells but ignores leverage. The realistic-context version is more decision-relevant but cells aren't directly comparable. Probably ship both — marginal as the main grid, realistic as a deep-dive per cell.
- **Cross-inning offensive variance vs pitcher-batter matchup variance** — which dominates the score-prediction CI? Worth a sensitivity analysis early on.
- **Lineup uncertainty.** Lineups are public ~3 h before first pitch. If we publish the daily prediction the evening before, we have *probable* lineups but not final. Either (a) publish twice — evening preview + morning update, or (b) marginalize over plausible lineups.
- **Pinch-runner / pinch-hitter modeling** — likely deferred. State the assumption explicitly.

---

## Honest constraints (worth stating in any UX that ships this)

- Model trained on ≤ 2023 Statcast. 2026 game predictions are 2+ seasons OOD.
- The model has no notion of weather changes mid-game, umpire fatigue, momentum, recent-result emotional state. Some of these matter to humans; the model is blind to them.
- The categorical context tokens (`inning_bucket`, `score_diff_bucket`, etc.) cap at certain values — extras-innings and blowouts are underrepresented in training and likely behave poorly OOD.
- Per the project's `causal-layer` skill: **game-level predictions are not causal estimates.** They're *"expected behavior under the model."* All UX copy on the sim outputs must use predictive language (not "if pitcher P had thrown X").

---

## Suggested implementation order (once we start)

Updated for the pre-game framing:

1. **Matchup-permutation document (Application B) first.** Single-AB sims under a fixed reference context. Reuses `g_compute` almost directly. **No multi-AB state machine, no pitcher-change model needed for v1.** Biggest immediate coach-facing value, smallest new code surface. Output: a PDF / web page per game with a `(pitcher × batter)` grid of run-value distributions. Probably 2–3 days of focused work for a usable v1.
2. **Multi-AB engine.** Runner advancement state machine, inning rollover, lineup rotation, ESS-aware termination. *Required for the daily score prediction.*
3. **Pitcher-change model.** Start with pre-scripted bullpen + manager-rule heuristics (options *c* + *a* above). Empirical hooks (option *b*) only if heuristics fall short.
4. **Daily score prediction (Application A).** Composes the above: full-game sim from 0–0 top-of-1, N=10K paths, output the score and run-total distributions, win probability, expected leverage moments.
5. **Pre-game ingestion job + scheduling.** Nightly cron that pulls probable pitchers + lineups + weather + bullpen state, runs both apps for each scheduled game, writes the artifacts.
6. **The daily prediction site.** Static-ish (regenerated each evening) — a per-game card with the score distribution + the matchup doc download.

---

## Things explicitly out of scope

- Tracking ball/strike calls beyond the model's view of the umpire (the umpire ID is already a categorical input).
- Anything that requires reading defensive alignment or shift (Statcast has it; not in PitchGPT's current input).
- Real-time *betting* applications — separate ToS, separate model-quality bar.
- Video / multimodal inputs (CLAUDE.md hard rule — explicitly out).
