# ADR 005 — Tipping Metric

**Status:** Accepted (locked 2026-05-09)
**Date:** 2026-05-08

## The question, in plain English

How do we detect "this pitcher is unusually predictable today, in a way the batter could exploit"?

## Why this matters

Tipping is operationally defined by what *the batter can use*. Most natural-feeling metrics fail this test:

- **Marginal entropy of pitch types** — misses conditional patterns. A pitcher who always throws fastball → slider has high marginal entropy and is maximally tipped.
- **Self-surprisal under full PitchGPT** — conflates "generally predictable" with "unusually predictable today." Mariano Rivera throwing 95% cutters isn't tipping; it's identity.
- **Generic-vs-specific gap** — measures pitcher-specific structure, but pitcher-specific structure isn't the same as exploitable structure.

The brainstorm correctly rejected all three. The chosen approach measures predictability *from the batter's information set*.

## The chosen approach

Train a **batter-observable variant of PitchGPT**. Same architecture as the main model, but feature inputs are masked to only what the batter sees pre-pitch.

Then for each game-start by pitcher *p*:

```
T_start(p) = mean over pitches of [ log P_batter-obs(actual pitch | observable history) ] - μ_p
```

where `μ_p` is the pitcher's rolling baseline (last 5 starts) of the same quantity. Flag starts where `T_start > 1.5σ` above zero.

A higher T_start means: today, with the batter's information set, the actual pitches were unusually predictable relative to this pitcher's own baseline.

## Feature mask (locked)

This is the most important piece — it defines what counts as "batter-observable." Lock it here so it's not silently shifted later.

**Batter sees (in batter-observable PitchGPT):**

- Count, score, outs, runners, inning
- Prior pitches in the current AB: pitch type, observed location (where it crossed the plate, not release coords), result
- Pitch results in earlier ABs against this pitcher *today*: pitch type + observed location + result, in sequence
- Pitcher's season-to-date arsenal (pitch type usage rates, year-to-date)
- Pitcher handedness, ballpark, umpire identity
- Standard scouting (publicly available year-to-date splits)

**Batter does NOT see:**

- Spin axis, spin rate
- Release point micro-data (3D release coords)
- Launch parameters of prior batted balls in detail (just the result class)
- Pitcher's intent signals or mechanical features
- Catcher's sign or pre-pitch positioning
- Anything from the pitcher's perspective the batter couldn't physically observe

## Validation hierarchy

The brainstorm's four validations, with honest framing for each:

1. **Subsequent-start performance.** Flagged starts predict elevated xwOBA-against and lower whiff rate in the pitcher's next 1–3 starts. *Caveat:* downstream xwOBA regresses for many reasons (fatigue, opponent quality, weather, regression-to-mean). This is a correlational signal, not a confirmation.
2. **Within-start TTO amplification.** Flagged starts show larger 3rd-time-through penalties than the pitcher's TTO baseline. *Caveat:* 3rd-time penalties are noisy with small samples.
3. **Batter behavior.** Flagged starts: more aggressive early-count swings, better contact on guessed pitch types. *Caveat:* requires careful confounder control — opponent quality, home/away, etc.
4. **Case studies.** Known tipping incidents (Darvish 2017 WS, etc.) should fall in flagged set. *Caveat:* small N, anecdotal.

**Honest framing:** if 2+ validations correlate as predicted, T_start tracks tipping signatures. We are not claiming T_start *measures tipping directly* — there is no ground truth for that. The writeup uses "correlates with downstream signatures of tipping," not "detects tipping."

If only 1 validation correlates, partial finding — write it up that way. If 0, T_start doesn't track tipping and we say so. (This failure mode is itself informative.)

## Recommendation

Align with the brainstorm. Lock the feature mask above. Keep the four validations with the corrected framing.

## Divergence from the brainstorm

No methodological divergence. Two operational tightenings:

1. **Feature mask is fully specified** — the brainstorm sketched "what the batter sees" but didn't enumerate. Without this lock, future-me silently expands or contracts the mask.
2. **Validation framing tightened** — "T_start correlates with downstream tipping signatures" replaces any "T_start measures tipping" language. The four validations are correlational, and we don't oversell.

## Consequences

- Two transformer training runs total: full PitchGPT (uses everything) and batter-observable PitchGPT (uses the mask above). Same architecture, same hyperparameters, different feature inputs.
- The batter-observable variant is *not* used as the propensity model for the causal layer — that uses the full model. The variant exists exclusively for tipping.
- Validations 1–3 require their own causal control (else "flagged starts have worse xwOBA later" could be confounded by mean reversion). The methods write-up acknowledges this.
- Threshold (1.5σ above pitcher's own baseline) is provisional; calibrate against the case-study set after Phase 5.
