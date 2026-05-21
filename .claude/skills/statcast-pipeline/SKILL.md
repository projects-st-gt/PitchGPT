---
name: statcast-pipeline
description: Use this skill whenever working on the PitchGPT data pipeline — pulling pitches via pybaseball, harmonizing pitch types, normalizing zones, binning velocity, building player profiles, computing the RE24 run-value table, or constructing the PyTorch Dataset. Trigger this skill any time the user mentions Statcast, pybaseball, extraction, preprocessing, harmonization, player profiles, trailing windows, run value, RE24, xwOBA, or the Dataset class. Also trigger when the user asks about confounders, leakage, temporal splits, or "is this feature safe to use." This skill enforces the no-fake-data rule and the no-leakage rule, both of which are easy to violate accidentally.
---

# Statcast Pipeline

The pipeline is the foundation: every downstream claim depends on it. This skill
covers extraction, harmonization, feature engineering, and the strict leakage
rules that the player-profile and run-value layers must obey.

## Extraction

Use `pybaseball.statcast(start_dt, end_dt)` day by day from 2017-03-15 through
the latest available date. Implementation details:

- **Checkpoint and resume.** The full pull is ~3,300+ daily requests across
  2017–present. Wallclock varies from a few hours (warm pybaseball cache) to
  several days (cold). Persist progress to `data/raw/_checkpoint.json` after
  every successful day so a crash never restarts from day 1.
- **Exponential backoff** on rate limits (429) and connection errors. Start
  at 5s, cap at 5 min. Do *not* substitute fake data on failure — wait it out.
- **Per-day parquet layout** at `data/raw/{year}/{date}.parquet`. Year-as-
  directory keeps each day's write atomic — no read-modify-write on a growing
  year-level file. A partitioned read of the year directory just works:
  `pd.read_parquet("data/raw/2024/")`.
- **Idempotency:** re-running `make extract` after partial completion picks up
  exactly where the checkpoint left off. The script has been verified to
  re-run as a no-op on completed dates.

See `data/extract_statcast.py` for the canonical implementation.

## Pitch type harmonization

Statcast emits ~15 pitch types. Collapse to 7 canonical:

| canonical | maps from                  |
|-----------|----------------------------|
| FF        | FF, FA                      |
| SI        | SI, FT                      |
| FC        | FC, CT                      |
| SL        | SL, ST, SV                  |
| CU        | CU, KC, CS, KN              |
| CH        | CH, FO                      |
| FS        | FS                          |

Anything else (eepus, intentional ball, etc.) → drop the at-bat. The mapping
table lives at `data/harmonization.py:PITCH_TYPE_MAP`. Update there, never
inline.

## Game metadata (umpire, weather) — secondary source

Two confounders ADR 003 requires are not present in the pybaseball pull:

- `umpire` — column exists in the dataframe but is **100% null** on 2024+
  data (verified empirically). Cannot be imputed from Statcast.
- `temperature`, `weather_condition`, `wind_*` — not present at all.

Source: MLB Stats API game-feed endpoint, one HTTP request per `game_pk`:

```
https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
```

Returns ~820KB JSON. Relevant fields:

- `gameData.weather.{condition, temp, wind}` — `temp` is a JSON string;
  parse with `_safe_int`. `wind` is `"speed mph, direction"`; speed `0`
  arrives with direction `"None"` (literal string), normalized to Python
  `None`. `condition: "Roof Closed"` exposed as `roof_closed: True`.
- `liveData.boxscore.officials[]` — list with `Home Plate`, `First Base`,
  etc. Each entry has the umpire's MLBAM `id` and `fullName`. Some games
  list fewer than 4 officials; missing positions become `None`.

Output: `data/game_metadata/games.ndjson` (append-only, crash-safe one-line-
per-game writes), with `data/game_metadata/_checkpoint.json` tracking
completed `game_pk`s. Downstream join on `game_pk`.

Implementation: `data/extract_game_metadata.py`. Run after Statcast extraction
completes via `make extract-meta`. ~30K games, polite at 1 req/sec.

## Zone normalization

`plate_z` is in feet. A 6'5" batter's high strike is not a 5'9" batter's high
strike. Normalize:

```
z_norm = (plate_z - sz_bot) / (sz_top - sz_bot)
```

Then bin into 5 equal bands. `plate_x` is already batter-symmetric in feet;
clip to [-1.5, 1.5] and bin into 5 equal columns. Total: 25 zones.

**Platoon mirroring augmentation:** for left-handed batters, flip `plate_x`
sign so the model sees a canonical "batter-side" frame. This doubles effective
data for L/R interaction learning. Apply at training time only, never to the
held-out test set.

### Feature zones vs. action zones

The 25-zone scheme above is the **feature encoding** — what the model
observes about location. The **action space** for causal counterfactuals is
a coarser 5-zone collapse per ADR 001 (`up`, `down`, `arm-side`, `glove-side`,
`out-of-zone`). Both coexist: μ̂ and π̂ see the 25-zone observation; the
causal layer's `do(A = a)` operates on the 5-zone action. Don't conflate
them — they are different abstractions for different jobs.

## Velocity binning — type-relative, not absolute

A 78 mph fastball is BP; a 78 mph curveball is normal. Absolute bins leak
"this is a slow fastball" into a meaningless bucket.

For each pitcher × pitch_type, compute the trailing-window mean and std of
release_speed. Bin the *deviation* from that mean into deciles. Falls back to
league-mean-by-type if the pitcher has fewer than 30 examples in the window.

Implementation: `data/preprocess.py:type_relative_velocity_bin()`.

## Spin axis and handedness — not optional

Both must be first-class features, not buried in profile vectors:

- `spin_axis` (0–360°) bucketed into 12 bins of 30°. Distinguishes a gyro
  slider from a sweeper at the same RPM.
- `p_throws` (P, R/L) and `stand` (B, R/L) as categorical context tokens
  prepended to every at-bat, alongside the pitcher and batter context.

## Confounders to capture

The causal layer fails silently if confounders are missing. ADR 003 is the
authoritative list; what follows is the operational extraction view.

**From the Statcast pull (`data/raw/`):**

- `fielder_2` — catcher MLBAM ID, fully populated. Catchers heavily influence
  sequencing and have framing effects.
- `pitcher_pitch_count_in_game` — fatigue. Derive from cumulative
  `pitch_number` per pitcher per game.
- `time_through_order` (1, 2, 3+) — known performance penalty. Derive from
  batter's lineup ordinal × times faced.
- `score_diff`, `leverage_index` — derived from base-out-score-inning state
  at pitch time.
- `runner_state` (8 states: 000, 100, …, 111) — from `on_1b/2b/3b` columns.
- `outs` (0, 1, 2) — from `outs_when_up`.
- `ballpark_id` — derive from `home_team` (one park per team modulo
  Toronto's 2020–21 split, which we handle case-by-case).
- `days_rest` for the pitcher — derive from per-pitcher `game_date` diffs.

**From the secondary game-metadata pull (`data/game_metadata/games.ndjson`):**

- `hp_umpire_id` — home-plate umpire. Embed as a categorical with hierarchical
  pooling (~100 unique umpires).
- `temp_f`, `weather_condition`, `roof_closed`, `wind_speed_mph` — weather.
  `roof_closed` is the dominant moderator: indoor games are climate-controlled
  and the `temp_f` reading is uninformative for ball-flight purposes there.

**Recent-form features (per ADR 003):**

- **Pitcher last-30-days xwOBA-against** (primary recent-form feature).
  Time-based window: all of this pitcher's pitches with
  `(game_date, game_num) < (current.game_date, current.game_num)` and
  within 30 calendar days. Works uniformly for starters and relievers.
  When the window is data-light (injury return, start of season), the
  function returns `n_pitches` so the model knows the mean is low-confidence;
  paired with the `days_since_last_appearance` feature below, the model can
  contextualize layoffs without us fabricating values.
- Pitcher last-3-starts xwOBA-against (starter-specific). Same window
  rule but counted in completed *starts* rather than days. Retained for
  tipping analysis per ADR 005 (where the start-by-start trajectory is
  the unit of interest); not used as the general recent-form feature
  because it produces NaN holes for relievers.
- Batter last-14-days wOBA. Window: all PAs with `(game_date, game_num) <
  (current.game_date, current.game_num)` within 14 days. Same-day prior
  games (doubleheader G1) ARE included if before current; same-game prior
  PAs are NOT.
- Pitcher's in-start pitch mix to-date — already implicit in the autoregressive
  feed for the sequence model; explicit feature for non-sequential baselines.
- **`days_since_last_appearance`** (proposed, to be added alongside the
  recent-form feature). For each AB, the number of days between the
  pitcher's most recent prior appearance and the current game date.
  Captures layoffs from injury, IL stints, or off-season. The model can
  use this to discount the recent-form mean when it's known to be stale.

**No-leakage discipline (mandatory).** Every windowed feature must end strictly
before the current AB's `(game_date, game_num)` ordinal. Same-game prior
content is never in any trailing window — the autoregressive sequence already
feeds in-game prior pitches; the windowed features are explicitly out-of-game.
Dataset-construction unit tests assert this invariant per row; CI fails if it
ever trips.

## Result encoding — keep two parallel encodings

The 7-class result head needs the discrete encoding:

```
ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, in_play_hr
```

But the run-value layer needs more granularity. So *also* store, per pitch:

- `estimated_woba_using_speedangle` (xwOBA-on-contact) when in play
- `delta_run_exp` (Statcast's per-pitch run-value field)

The run-value layer maps the result distribution from μ̂ through these fields.

## Player profiles — trailing-window discipline

Profiles are computed per (player_id, date). The trailing window must end
*strictly before* the at-bat being predicted. No within-game leakage either:
the window ends at the start of the current game, not the current AB.

For pitchers (last 1000 pitches before window-end-date):
- arsenal composition (% by canonical type)
- mean velo, mean spin, mean spin-axis per type
- 25-zone heatmap per type (flattened to 175-vector)
- platoon splits (vs LHB, vs RHB)
- conditional-pitch-type entropy by count state

For batters (last 1000 PA before window-end-date):
- 25-zone swing%, whiff%, xBA grids
- chase rate by pitch type
- exit velo mean and 90th percentile
- K%, BB%, hard-contact rate

**Rookies and low-PA players:** use the league-mean profile blended with
whatever data exists, with a `profile_confidence` scalar passed alongside.
Never fabricate values.

**Window ordering and doubleheaders.** The trailing window ends at
`(game_date, game_num) < (current.game_date, current.game_num)`. Doubleheader
G1 is *before* G2 by `game_num`, so G1's content is eligible for G2's window
but not vice versa. ADR 003 has the full ordering rule; profile windows and
recent-form windows both follow it.

**Recent-form features are separate from these big-window profiles.** Pitcher
last-3-starts xwOBA-against and batter last-14-days wOBA (ADR 003) live
alongside the 1000-pitch / 1000-PA profiles, computed by the same module
with their own window definitions. Big profiles capture stable identity;
recent-form captures short-horizon trajectory. The model gets both.

Implementation: `data/player_profiles.py`. See the unit tests for leakage
detection — there's a synthetic test that *should* fail loudly if any
windowed feature includes content from at or after the target AB's
`(game_date, game_num)`.

## Run-value table (per ADR 004)

Compute from training data only — never from val/test, never from external
sources. Three pieces:

1. **RE24 by season.** For each (base state, outs) cell — 24 total — mean
   runs scored in the rest of the half-inning. Recomputed per training
   season because run environments shift across years (juiced ball, etc.).
2. **Count-state value by season.** For each (balls, strikes) state, the
   mean expected wOBA contribution. Same per-season recompute discipline.
3. **State value layer.** `V(state) = RE24[base, outs] + count_value[balls,
   strikes]`. Additive, league-average, frozen per season once computed.

Per-pitch run value:

- **Non-terminal pitch** (count changes, AB continues):
  `Δrun_value = V(state_after) − V(state_before)`.
- **Terminal pitch** (AB ends):
  - Strikeout / walk / HBP: `V(after) − V(before)` via deterministic state
    transition.
  - In-play: use **xwOBA-on-contact** (`estimated_woba_using_speedangle`),
    *not* actual outcome, mapped to runs via the per-pitch in-play
    contact slope computed empirically per season by
    `compute_in_play_woba_to_runs_slope` (~0.49 on 2023). There is no
    canonical published constant at this aggregation level — earlier
    drafts referenced "Tango ~0.7", which is a confabulation; see ADR 004
    and the `pressure-testing-claims` skill for the verification trail.

Store as `data/run_value/re24_{year}.parquet` and
`data/run_value/count_value_{year}.parquet`, one per season in the training
window. Frozen for evaluation; recompute only when the training window
changes.

**Why xwOBA, not actual outcome:** we evaluate the *pitcher's decision*,
which controls the launch parameters of contact, not whether the left
fielder was shifted into the gap. xwOBA captures the controllable variance.

The recommender and counterfactual outputs map μ̂'s result distribution
through this layer — see the `causal-layer` skill for the integration.

## Temporal split — non-negotiable

```
train: 2017-01-01 ≤ game_date ≤ 2023-12-31
val:   2024-01-01 ≤ game_date ≤ 2024-07-15
test:  2024-07-16 ≤ game_date  (through latest)
```

The split function lives at `data/dataset.py:temporal_split()`. There is also
a `held_out_pitchers` cohort: any pitcher whose first MLB pitch is in 2024.
That cohort tests profile-based generalization and is reported separately in
the eval table.

## Dataset class

`data/dataset.py:AtBatDataset` returns per item:

```
{
  "pitcher_profile": Tensor[d_profile_p],
  "batter_profile":  Tensor[d_profile_b],
  "context_tokens":  LongTensor[n_context],     # handedness, park, umpire, etc.
  "pitch_factors":   dict[str, LongTensor[T]],  # type, zone, velo, spin, ...
  "result_factors":  LongTensor[T],
  "target_factors":  same shape as pitch_factors, shifted by 1
  "padding_mask":    BoolTensor[T]
}
```

At-bats vary in length (1–~12 pitches). Pad to the longest in the batch; the
attention mask blocks attention to padding. Cross-at-bat attention is also
blocked when packing — see the `pitchgpt-model` skill for the masking spec.

## Sanity checks before declaring extraction "done"

Run `make data-sanity` and confirm:

- Pitch count by year matches Baseball Savant's published totals within 1%
- Pitch-type histogram matches league-published distributions within 0.5pp
- No NaN in `release_speed`, `plate_x`, `plate_z` after the harmonization step
- Held-out-pitcher cohort is non-empty
- Player-profile leakage test passes (no windowed feature contains content
  from the target AB's `(game_date, game_num)` or after)
- Game-metadata coverage: every distinct `game_pk` in `data/raw/` has a
  matching row in `data/game_metadata/games.ndjson`. Missing rows mean the
  metadata scrape needs another pass.
- Recent-form windows are populated for ≥95% of ABs in the training window
  (small minority of early-career and post-injury ABs will have empty
  windows; flag, don't fail).

If any check fails, do not proceed to training. Diagnose and fix.
