# ADR 006 — Cross-Fit Blocks

**Status:** Accepted (locked 2026-05-09)
**Date:** 2026-05-08

## The question, in plain English

To get unbiased causal estimates, we can't train the propensity and outcome models on the same data we then use to evaluate the causal effect — we'd be double-dipping. The standard fix is **cross-fitting**: split the data into K folds, train on K−1, evaluate on the held-out 1, rotate.

Two questions follow: what's the *unit* of splitting (individual pitches? at-bats? games? pitchers?), and what's K?

## Why this matters

- **Wrong unit = leakage.** If two pitches from the same game land in different folds, the catcher's read of the batter (built up across the game) is leaking from training to evaluation. Same for fatigue accumulating within a start, or any in-game adjustment.
- **No cross-fitting at all** = AIPW estimates are asymptotically biased. Most "deep counterfactual" papers skip this. One of the things we explicitly committed to *not* doing in this project.

## Options for the splitting unit

**A. Random by pitch.** Worst — heavy within-game and within-AB leakage. Don't.

**B. By at-bat.** Better, but two ABs from the same game still leak via catcher state, fatigue, umpire's evolving zone, etc.

**C. By game (`game_pk`).** Brainstorm default. Every pitch in a given game lands in the same fold. Eliminates within-game leakage entirely.

**D. By pitcher-game.** Stronger — eliminates catcher carryover effects between starts (a catcher might be in the training fold for one of his pitcher's starts but the test fold for another). But shrinks effective N because games with multiple pitchers (e.g., relievers) get treated as multiple units.

## K choice

- K = 5: standard, manageable compute. Each fold contains ~20% of the data.
- K = 10: 2× compute, modestly lower variance. Worth it if compute permits, otherwise overkill.
- K = 20: usually only matters when N is small. We have 7M pitches. Skip.

## Stratification

Folds should be balanced across:

- **Season** — so each fold contains roughly the same year mix. Otherwise we accidentally learn season-specific run environments unevenly.
- **Pitcher tier** (loose) — so high-volume pitchers aren't all clustered in one fold and absent from another. Game-blocking already does most of this.

## Recommendation

- **Unit:** game (`game_pk`). Option C, the brainstorm default.
- **K:** 5. Standard, manageable.
- **Stratification:** by season, with explicit fold balance verification (each fold should be within ±2% of mean season representation).
- **Verification step:** after splitting, log per-fold pitch count by season; if any fold deviates by more than ±2%, regenerate the split with a different seed.

## Divergence from the brainstorm

None. This formalizes the brainstorm's choice and adds the explicit fold-balance verification.

## Consequences

- Training cost is **5× a single training run.** This is the dominant compute cost of the project, and we accept it because cross-fitted AIPW is the headline methodological feature.
- Each cross-fit produces 5 model checkpoints. Inference at evaluation time uses the held-out fold's model on each fold's pitches. The demo uses one of the 5 (any will do — they're functionally equivalent on out-of-fold data).
- The data pipeline must produce a per-pitch `fold_id` column that's stable across runs (deterministic from `game_pk` hash). Otherwise we can't reproduce a cross-fit.
- ADR 007 (nuisance decoupling) interacts with this: if we go with the "head-decoupled cross-fit" option, the partition of folds-to-heads happens here.
- Pitcher held-out generalization (`eval-protocol` skill) is a *separate* split (debut-in-2024 cohort) and uses different machinery — don't confuse them.
