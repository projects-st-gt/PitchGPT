# ADR 008 — Profile Cache Fold-Awareness

**Status:** Accepted (locked 2026-05-09)
**Date:** 2026-05-09

## The question, in plain English

ADR 006 specified K=5 cross-fitting blocked by `game_pk`. The idea: when we
train the propensity / outcome models on folds {1,2,4,5} and evaluate on
fold 3, the trained models should not have seen any fold-3 data.

The profile cache is a *feature*, not training data, but it's derived
*from* the entire pitch corpus. If we precompute one cache shared across
all folds, then a profile feature handed to the model for an at-bat in fold 3
is partially derived from earlier-in-time pitches that *also* belong to fold 3
(folds are blocked by `game_pk`, not by date).

That's a real but indirect leakage path. The contamination is bounded
(profile features are aggregated over hundreds of pitches), but it
violates the strict separation cross-fitting was designed to enforce.

## Why this matters

The headline pitch of this project is *"sequential causal inference done
correctly."* Cross-fitting is one of the four pieces (alongside positivity,
sensitivity analysis, and calibration) that distinguishes this work from the
typical ML "counterfactual" paper. A reviewer who asks *"did your nuisance
models see fold-K data through any feature path?"* needs to get a clean
"no, by construction" answer. Hedging on this weakens the claim.

## Options considered

**A. Fold-aware cache (chosen).** Build K=5 separate caches. The cache for
fold X is computed using only pitches from folds ≠ X (still subject to the
`before_asof` no-leakage rule on top of that). At training time, the
profile lookup uses the AB's fold ID to read from the correct cache.

- Pros: methodologically clean. By construction, π̂ trained on folds
  {1,2,4,5} sees zero fold-3 content through profile features.
- Cons: 5× storage (~1GB instead of ~200MB), 5× build time (~5 hours
  instead of ~1), one extra integer in the cache key, marginal complexity
  in the lookup path.

**B. Single shared cache; document the contamination.** Use the entire
corpus to build one cache. Argue that profile features are highly
aggregated so any single fold's marginal contribution is small.

- Pros: simplest, smallest footprint.
- Cons: contamination is real, even if small. Reviewer can rightly call
  out that AIPW's asymptotic guarantees no longer apply strictly.

**C. Hybrid: shared cache for long-window features, fold-aware for
short-window.** Long-window features (1000-pitch arsenal, heatmaps) get
contributions of <20% from any one fold by construction; short-window
features (last-30-days xwOBA, last-3-starts) can be dominated by one fold.

- Pros: smaller footprint than A, addresses the worst contamination paths.
- Cons: two storage formats; more conceptual complexity; "small enough
  contamination" is still a hedge.

## Decision

**Option A.** Build K=5 fold-aware caches. The principled choice; aligns
with the project's framing.

The operational costs are bounded and one-time:
- Storage: ~1GB total, irrelevant on modern disk.
- Build time: ~5 hours wallclock, parallelizable across folds.
- Lookup complexity: cache key is `(player_id, role, asof_date, asof_game_num, fold_id)`
  instead of `(player_id, role, asof_date, asof_game_num)` — one extra integer.

## Consequences

- The profile cache builder takes a `fold_assignments` table (one row per
  `game_pk` → `fold_id`) and emits K caches. The fold assignments come
  from the same K-fold split used by the training/eval orchestration.
- The dataset class's `pitcher_profile_lookup` and `batter_profile_lookup`
  callables receive `fold_id` along with the asof key.
- The methods writeup includes a one-sentence statement: *"Profile features
  are computed under fold-aware cross-fitting: the profile for an at-bat
  in fold k uses only pitches from folds ≠ k, in addition to the
  strict-temporal `before_asof` filter."*
- For inference (demo / production), there is no fold concept — the
  inference path uses a single all-data cache (computed alongside the K
  fold caches; trivial extra cost). Inference results are not used for
  training or evaluation, so no contamination concern.
- ADR 006 is unchanged but cross-references this ADR.

## Things deliberately not addressed here

- *How fold assignments are stored or distributed.* That's an
  implementation detail of the cache builder.
- *Whether the league-mean fallback cache is also fold-aware.* It is —
  same construction, same key shape minus `player_id`.
- *Inference cache update cadence when new data lands.* Out of scope;
  decide when productionizing.
