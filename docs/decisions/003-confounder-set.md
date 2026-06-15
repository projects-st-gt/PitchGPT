# ADR 003 — Confounder Set

**Status:** Accepted (locked 2026-05-09); Amended 2026-05-10 (see Amendment 1 below)
**Date:** 2026-05-08

## The question, in plain English

For our causal estimates to be valid, we need to assume that "given everything we condition on, the pitch choice was effectively random." Anything we don't condition on becomes *unmeasured confounding* — and our causal estimates absorb its bias silently.

So: what do we condition on?

## Why this matters

This is the single biggest assumption in the entire causal layer. Sequential conditional ignorability is going to be heroic no matter what we do (we will never have catcher's pre-pitch read of the batter, or scouting reports, or pitcher's mechanical state). Every confounder we *do* observe shrinks the gap. Every one we miss becomes a knob the sensitivity analysis has to argue about.

## Always-included (the baseline state)

These are non-negotiable — the model has them by construction:

- Count (balls, strikes), outs, base state (which bases occupied)
- Inning, score differential
- Batter handedness, pitcher handedness, batter stance
- Prior pitches in the current AB (type + zone + result)
- Pitcher and batter trailing-window profiles (30 days, strictly before the AB — see `statcast-pipeline` skill for leakage rules)

## Brainstorm-added confounders (locked)

The brainstorm specifically called these out as missing from earlier drafts:

| Confounder | Why it's a confounder | Encoding |
|---|---|---|
| Catcher identity | Huge sequencing influence; framing affects called-strike rates | Embedding (~200 catchers) with hierarchical pooling |
| Pitch count in game | Fatigue affects both pitch choice and outcome | Continuous (count + innings pitched today) |
| Days rest | Affects stuff and selection | Continuous, capped at 7 |
| Leverage index | Affects pitch selection (high-leverage = primary stuff) and outcome distribution | Continuous (formal LI from base-out-score-inning) |
| Time through the order | Batter familiarity changes both selection and outcome | Categorical (1st, 2nd, 3rd, 4th+) |
| Umpire identity | Zone effects are real and pitch-type-dependent | Embedding (~100 umpires) with hierarchical pooling |
| Previous batter's outcome | Carryover into next AB's pitch selection | Categorical (HR, BB, K, etc.) |
| Ballpark | Park factors affect pitch effectiveness; some pitches play better in some parks | Embedding (30 parks) |

## Additional confounders worth including

Not in the brainstorm but I'd argue for them:

- **Game-time temperature.** Affects ball flight and grip on breaking balls. Cheap to pull from Statcast.
- **Pitcher's recent form.** xwOBA-against over the last 30 days, time-based so the metric works uniformly for starters and relievers (a starter's 30 days ≈ 5–6 starts; a reliever's 30 days ≈ 20–25 outings — same calendar window, different outing rhythms). Earlier draft used "last 3 starts," which only works for starters and produces NaN holes for relievers; that variant is retained as a *starter-specific* metric in `pitcher_last_n_starts_xwoba` and is used for tipping analysis per ADR 005, but the general recent-form profile feature is the time-based version `pitcher_last_n_days_xwoba`.
- **Cross-season staleness signals (`long_window_span_days`, `long_window_pct_current_season`).** `profile_confidence` measures *window fullness* (n_pitches / window_pitches); these two features measure *window freshness*. The canonical case: an April-1 at-bat for a starter has a full 1000-pitch window that's mostly previous-October data — `profile_confidence ≈ 1` makes it look fresh, but the data is months old and arsenals can shift over the offseason (new pitches, mechanical changes, recovery from injury). The two staleness features expose this directly: `span_days` is the day-gap from the oldest pitch in the window to the asof date, and `pct_current_season` is the fraction of window pitches whose year matches the asof year. Both are NaN for empty windows. The model can learn to discount the long profile vector when these flags indicate staleness. Companion to `pitcher_days_since_last_appearance` (which captures the layoff gap directly). Added in profile schema v2 (2026-05-09).
- **Batter's recent form.** Last 14 days wOBA. Same logic.
- **Pitcher's in-start pitch mix to-date.** Already implicit in autoregressive feed for sequence models, but explicit feature for non-sequential baselines.

### No-leakage discipline for recent-form features

This is easy to get wrong in code, so it gets spelled out here and enforced by the dataset construction tests.

For an at-bat occurring in game G with start time T(G):

- **"Pitcher last 3 starts"** = the pitcher's three most recent *completed* starts where `(game_date, game_num) < (G.game_date, G.game_num)`. The current game G itself is excluded, even if the at-bat is late in G. If the pitcher made a relief appearance earlier the same day (doubleheader G1), it does not count as a start; only games where the pitcher was the starter are eligible. Computed at AB-start boundaries; all pitches within the same AB share the same window.
- **"Batter last 14 days"** = all PAs the batter recorded with `(game_date, game_num) < (G.game_date, G.game_num)` and `G.game_date − PA.game_date ≤ 14` days. Same-day prior games (doubleheader G1) ARE included if they meet the strict-before condition; same-game prior PAs in G itself are NOT included. Computed at AB-start boundaries.
- **General rule** for any windowed feature: the window ends *strictly* before the current AB's game-and-game_num ordinal. Same-game prior content is never in any trailing window — the autoregressive sequence already feeds the model in-game prior pitches; the windowed features are explicitly out-of-game.
- **Verification:** dataset-construction unit tests assert that for every windowed feature, no row in the trailing window has `(game_date, game_num) ≥ (current.game_date, current.game_num)`. CI fails if this assertion ever trips.

This pattern matches the 30-day pitcher/batter profile windows already documented in the `statcast-pipeline` skill — same discipline, different window length.

I'd skip:

- **Wind speed/direction.** Inconsistent availability, marginal additional explanatory power once temperature is in.
- **Personal-catcher pairings.** Already captured by catcher × pitcher interaction in embeddings.
- **Dugout-side intel signals (sign-stealing).** Unobservable in principle. The sensitivity analysis is the right tool here, not a feature.

## Recommendation

Include the **baseline state + all eight brainstorm-added confounders + temperature + recent-form windows for both pitcher and batter.**

High-cardinality embeddings (catcher, umpire) get hierarchical pooling: shared-prior on the embedding mean so cells with few observations regress toward the league mean instead of overfitting to noise.

## Divergence from the brainstorm

**Adds four:** temperature, pitcher recent form (last 3 starts), batter recent form (last 14 days), and explicit pitcher in-start mix as a feature for non-sequential baselines.

Otherwise matches.

## Consequences

- The unmeasured-confounding gap that sensitivity analysis has to bound includes (at minimum): catcher's pre-pitch read of the batter, scouting reports we don't have access to, pitcher's mechanical state ("how does my slider feel today"), in-game tells that aren't in the prior-pitch sequence, sign-stealing.
- Embedding tables for catcher/umpire/ballpark/batter/pitcher need careful initialization and freeze schedules during cross-fitting (see ADR 006 and 007).
- The "about" page in the demo lists what we condition on and what we don't — the unmeasured confounders get a paragraph each, not a single bullet.

---

## Amendment 1 — 2026-05-10

Two changes to the operational encoding of confounders. Neither changes *what* we condition on; both change *how* it enters the model.

### Change A: Replace derived Leverage Index with raw state components

**Original encoding (locked 2026-05-09):** "Continuous (formal LI from base-out-score-inning)."

**New encoding:** Drop the derived LI feature. Provide the model with the raw state components that LI is built from:

| New feature | Vocab | Source |
|---|---|---|
| `inning_bucket` | 14 (innings 1–12, extras 13, PAD 0) | Statcast `inning` |
| `score_diff_bucket` | 11 (signed, clipped to [−5, +5]) | `bat_score − fld_score` |
| `inning_half` | 3 (top, bot, PAD) | Statcast `inning_topbot` |
| (already present) `runners` | 8 | per-pitch base-occupancy state |
| (already present) `outs` | 3 | per-pitch outs-when-up |

`inning_bucket` and `score_diff_bucket` join the **categorical context token** (pitcher/batter handedness, ballpark, umpire, catcher, days_rest, tto, temp, roof, *now also inning and score_diff*). `runners` and `outs` are already in the per-pitch factor stream.

#### Why

Tom Tango's Leverage Index is a deterministic function:

```
LI = f(inning, inning_half, outs, base_runners, score_diff)
```

By the **sufficient-statistic property** of confounding adjustment, conditioning on the inputs `(inning, inning_half, outs, runners, score_diff)` is strictly sufficient for adjusting for LI. It is additionally at least as good as conditioning on LI directly, and strictly better when there is residual heterogeneity that LI's many-to-one mapping discards.

Two concrete places where raw state beats a scalar LI:

1. **Signed score asymmetry.** LI is symmetric in `|score_diff|`. Pitcher behavior is not — defending a 3-run lead and chasing a 3-run deficit produce different pitch mixes (defensive nibbling vs. aggressive zone attack). Raw `score_diff_bucket` preserves the sign; LI loses it.
2. **Within-LI heterogeneity.** Two states with the same LI but different `(inning, half)` can produce different behavior in extras-rules era. Raw state distinguishes them.

This is not a methodological shortcut — it's the standard sufficient-statistic argument from causal-inference textbooks. We are providing more (not less) information to the propensity and outcome models.

#### What we lose

- The single named "leverage" embedding direction is gone; the model's internal LI-equivalent representation is now distributed across `inning_emb + score_diff_emb + inning_half_emb + runners_emb + outs_emb`. The trunk learns the relevant interaction during training.
- Demo / writeup language must say "we adjust for game-state pressure via raw (inning, half, score_diff, runners, outs)" rather than "we adjust for leverage." This is a more transparent statement, not a weaker one.

#### Trunk capacity check (pre-registered)

With Tiny (4 layers × 4 heads × d_model=256) and ~7M training pitches, the trunk has ample capacity to learn the joint state → behavior interaction. If post-Phase-B diagnostics show the model is failing to differentiate high-LI from low-LI states in its hidden representations (probe: cluster terminal-pitch hidden states by computed LI and check separation), we revisit by adding a precomputed `leverage_quintile` as a *supplementary* feature alongside the raw state — not a replacement.

### Change B: Convert pitcher in-game fatigue to bucketed categorical

**Original encoding:** "Continuous (count + innings pitched today)."

**New encoding:** `pitcher_fatigue_bucket` — per-pitch categorical, vocab 12: `0–9, 10–19, 20–29, ..., 100+, PAD`.

The feature is derived at preprocess time as the cumulative pitch count for `(game_pk, pitcher)` after sorting by `(at_bat_number, pitch_number)`. It joins the **per-pitch factor stream** (not the context tokens) because fatigue increases *within* an AB.

#### Why bucketed instead of continuous

The rest of the per-pitch factor stream is categorical embeddings; introducing a single continuous feature requires a separate projection path and complicates the residual stream. Buckets keep the architecture uniform and the regularization story consistent (each bucket has its own embedding, learned independently). The trade is a small loss of within-bucket resolution; with ~7M pitches across 12 buckets we average ~580K pitches/bucket — plenty to learn distinct representations.

### Updated confounder encoding table

Replaces rows in the original "Brainstorm-added confounders" table:

| Confounder | Encoding (post-amendment) |
|---|---|
| Catcher identity | Embedding (vocab ~384, includes PAD/UNK) |
| Pitcher in-game fatigue | `pitcher_fatigue_bucket` per-pitch, vocab 12 |
| Days rest | Bucketed categorical, vocab 9 (0–7+, PAD) |
| ~~Leverage index~~ | **Removed**; replaced by `inning_bucket` + `score_diff_bucket` + `inning_half` (see Change A) |
| Time through the order | Categorical (1st, 2nd, 3rd, 4th+, PAD), vocab 5 |
| Umpire identity | Embedding (vocab ~256) |
| Ballpark | Embedding (vocab ~64) |
| Previous batter's outcome | (Implicit in autoregressive sequence; not a separate feature) |

### What does not change

- The set of confounders adjusted for is unchanged.
- The temporal/no-leakage discipline is unchanged.
- Recent-form features (pitcher 30-day, batter 14-day) are unchanged.
- Hierarchical pooling for catcher/umpire embeddings is unchanged.
- The unmeasured-confounding gap and sensitivity-analysis story are unchanged.

### Files affected by this amendment

- `model/config.py` — add `n_inning_buckets=14`, `n_score_diff_buckets=11`, `n_inning_half=3`, `n_pitcher_fatigue_buckets=12`; remove `n_leverage_buckets`.
- `model/embeddings.py` — `ContextTokens` swaps `leverage_emb` for `inning_emb + score_diff_emb + inning_half_emb`; `FactorEmbeddings` adds `pitcher_fatigue_emb`.
- `data/preprocess_pitchgpt.py` (new) — derives all the above columns and writes augmented parquets.
- PitchGPT model skill — context-token list and factor-vocab table updated.
- `tests/test_pitchgpt_model.py` — fixture dicts updated for new keys.
