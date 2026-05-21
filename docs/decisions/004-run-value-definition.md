# ADR 004 — Run Value Definition

**Status:** Accepted (locked 2026-05-09)
**Date:** 2026-05-08

## The question, in plain English

The demo and recommender both make claims like "this pitch was worth +0.13 runs" or "the slider is +0.04 runs better than the fastball here." For those numbers to mean anything consistent, we need one definition of run value used everywhere.

What is "the run value of a pitch"?

## Why this matters

- Inconsistency = every downstream comparison is meaningless. The recommender ranking, the AIPW estimates, the tipping validations — they all flow from this number.
- The wrong choice (e.g., raw outcome instead of expected outcome) introduces noise that swamps the actual signal we're trying to detect.

## Components

### 1. RE24 — base/out run expectancy

For each of the 24 (base state × outs) combinations, compute the expected runs scored in the rest of the inning. This is empirical: the average runs scored after that state is reached, computed from the training data.

- **Refresh per season** — run environments shift (juiced ball years, etc.). RE for 2018 ≠ RE for 2024.
- **Training data only** — never compute RE24 using val/test data, otherwise we leak.
- This is a 50-line script. Output: a 24-cell table per season.

### 2. State value at the per-pitch level

The state of an at-bat is (base state, outs, balls, strikes). For each state, define a value V(state). Combining RE24 with count-state run expectancy gives V — count-state run expectancy is recomputed from training data per season (no fixed external constant; the run environment shifts year to year).

### 3. Per-pitch run value

- **Non-terminal pitch** (count changes, AB continues): `Δrun_value = V(state_after) − V(state_before)`. A called strike at 0-0 → 0-1 has a small negative value for the offense (positive for defense).
- **Terminal pitch** (AB ends, in-play or strikeout or walk):
  - Strikeout: `V(0 PA, +1 out, base state unchanged) − V(state_before)`.
  - Walk: `V(state with batter on first, runners advanced if forced) − V(state_before)`.
  - In-play: use **xwOBA** (expected wOBA based on launch parameters) rather than actual outcome. Map xwOBA to runs via the league-average wOBA-to-runs conversion, recomputed from training data.

### 4. Why xwOBA, not actual outcome

xwOBA captures the *expected* value of a batted ball given launch angle, exit velocity, and spray angle. Using actual outcome introduces variance from defensive positioning, ballpark dimensions, and pure luck. We're trying to evaluate the *pitcher's decision*, which controls the batted ball's launch parameters — not whether the left fielder was shifted.

This is the standard public-baseball-analytics convention. Using it makes our run-value numbers comparable to other published work.

## Recommendation

Align with the brainstorm. Specifically:

1. Compute RE24 per season from training data only.
2. Compute count-state run expectancy per season from training data only.
3. Define `V(state) = RE24[base, outs] + count_value[balls, strikes]` (additive, league-average).
4. Per-pitch run value = `V_after − V_before`, with xwOBA-based mapping for in-play.
5. Compute the per-pitch in-play wOBA-to-runs slope empirically per season via `compute_in_play_woba_to_runs_slope`. This slope (~0.49 on 2023) is what `in_play_run_value(xwoba, slope)` uses to convert a hypothetical pitch's xwOBA-on-contact into a run-value contribution during causal rollouts. **Footnote on labeling:** an earlier draft of this ADR called this "Tango's wOBA-to-runs constant ≈ 0.7" and asked for verification against the published value. That was a confabulation — there is no canonical "0.7" wOBA-to-runs constant at this aggregation level. FanGraphs's published "wOBA scale" (~1.20 for 2023) is for a different calculation entirely (runs above average per PA × wOBA-difference). The fix: compute the per-pitch in-play slope empirically per season; do not compare against a non-existent published reference. See `.claude/skills/pressure-testing-claims/SKILL.md` for the verification trail.

Locked once: lives at `data/run_value.py`. Imported everywhere downstream.

## Divergence from the brainstorm

None. This ADR specifies the per-season recompute and the xwOBA-to-runs source, both of which the brainstorm gestured at without locking.

## Consequences

- Every result-head output of μ̂ maps deterministically to a run-value number via this layer. The mapping is not learned — it's fixed by this ADR.
- Recommender's "expected runs" claims are first-principles run-value; no separate value head needed unless we add one as a Phase 9 sanity check.
- Sensitivity analysis works in run-value units, which makes E-values interpretable as "how strong would unmeasured confounding need to be to flip the sign of this many runs."
- Tipping validations (subsequent-start xwOBA-against, etc.) use this same run-value scale, so cross-component comparisons are valid.
