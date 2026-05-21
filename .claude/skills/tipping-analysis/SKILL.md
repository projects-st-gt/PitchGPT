---
name: tipping-analysis
description: Use this skill whenever working on tipping detection, predictability decay, batter-observable PitchGPT variant, the T_start metric, or any of the four validation runs (subsequent-start performance, time-through-order amplification, batter behavior, case studies). Trigger on mentions of tipping, predictability, sequencing entropy, batter perspective, observable features, T_start, Darvish, or any "is this pitcher predictable" framing. The metric specification here is operationally aligned with what tipping actually means (information available to the opposing side), and it is not the same as marginal entropy — older drafts of the project used marginal entropy, which is wrong. Read this before changing the metric definition or validation protocol.
---

# Tipping Analysis

Tipping is operationally defined by what the *batter* can exploit. The metric
must therefore be batter-observable predictability, not marginal sequencing
entropy. This skill specifies the metric, the model variant that powers it,
and the four validation runs.

## The wrong metric (do not use)

Marginal sequencing entropy averaged over a start:

```
H(start) = -Σ_a P(a) log P(a)
```

A pitcher who throws fastball-then-slider every time they're behind in the
count has high marginal entropy and is maximally tipped. The metric misses
conditional patterns entirely. Earlier versions of the project used this
metric; the change is documented in `docs/decisions/005-tipping-metric.md`.

## The right metric — batter-observable T_start

Train a *second* PitchGPT variant with masked features: the model only sees
what a batter could plausibly observe. Then measure how well *this* model
predicts the actual pitches in a start, relative to the pitcher's own
rolling baseline.

Define for each pitcher-start:

```
ℓ_start(p) = mean over pitches in the start of
             log P_obs( actual pitch | observable history, pitcher p )

T_start(p) = ℓ_start(p) - μ_p,    μ_p = mean of ℓ over last 5 starts
```

T_start > 1.5σ above zero (where σ is per-pitcher across the season) ⇒
flag as predictability spike.

Implementation: `tipping/t_start.py`.

## Batter-observable feature mask

The batter-observable variant uses the same architecture as the main
PitchGPT but with these feature visibility rules:

**Visible to batter (and to this variant):**
- Count, score, runners, outs, inning, leverage
- Prior pitches in the current at-bat (type, observed velocity, observed
  zone, observed result)
- Pitch results in *earlier at-bats against this same pitcher today*
- Pitcher's season-to-date arsenal frequencies (public scouting)
- Pitcher and batter handedness, ballpark, umpire (publicly known)
- Catcher identity (visible)

**Hidden from batter (zeroed in this variant):**
- Pitcher profile vector beyond the public arsenal frequencies (no internal
  features like trailing-window spin axis the batter wouldn't track)
- spin_axis on prior pitches (batter sees movement, not exact axis)
- Spin rate (batter sees movement, not RPM)
- Time-through-order (visible) but pitcher_pitch_count_in_game (hidden;
  approximate via TTO and inning)
- Days rest (the pitcher's, not the team's; hidden)

The mask is applied at the data layer in `tipping/observable_dataset.py`.
A unit test confirms the mask does not leak — train a probe to predict a
hidden feature from the masked input; if it succeeds, the mask has a hole.

## Validation runs — all four required

A T_start spike is a hypothesis, not a finding. Validate four ways. At least
two must show the predicted direction for the project to claim the metric
tracks tipping.

### 1. Subsequent-start performance

For each pitcher-start flagged as a T_start spike, compare the pitcher's
performance in the *next 1–3 starts* to their season baseline:

- xwOBA against (expected: elevated)
- Whiff rate (expected: lower)
- CSW% (expected: lower)

Match flagged starts to non-flagged starts by pitcher × month × opponent
quality. Difference-in-differences with bootstrap CIs.

Implementation: `tipping/validation/subsequent_perf.py`.

### 2. Time-through-order amplification

The 3rd-time-through-the-order penalty is a known phenomenon. If a pitcher
is tipping, the penalty should be *amplified* in flagged starts: batters
have had multiple looks at the same predictable pattern.

For each flagged start, compute the gap between 1st-TTO xwOBA and 3rd-TTO
xwOBA, and compare to the pitcher's season-average TTO gap. Flagged starts
should show larger gaps.

Implementation: `tipping/validation/tto_amplification.py`.

### 3. Batter behavior

If batters can predict pitches, they should:
- Swing earlier in counts (lower 0–0 take rate)
- Make better contact (higher exit velocity, lower whiff on
  expected-pitch-types)
- Show pitch-type-specific advantages (worse contact on the pitch they
  *didn't* predict, better contact on the one they did)

The third sub-test is the most discriminating. Use the batter-observable
variant's *predictions* per pitch, then condition batter outcomes on
"prediction matched" vs "prediction missed."

Implementation: `tipping/validation/batter_behavior.py`.

### 4. Case studies

Known tipping cases that should fall in the flagged set:

- Yu Darvish, 2017 World Series Games 3 and 7 (publicly diagnosed in
  postgame analysis)
- Trevor Bauer, mid-2018 stretches (publicly discussed)
- Other documented incidents — pull the list from the curated reference
  at `tipping/validation/case_studies.json`

If the metric flags none of these, something is wrong. If it flags only
some, that is informative — note which and why in the writeup.

## The "no signal" outcome is informative

If T_start does not correlate with any of the four validations, that is a
real finding: batter-observable signals do not drift in detectable ways at
this granularity. Report it as such.

This is *why* the metric was chosen over self-surprisal or generic-vs-specific
gap: it has a clean operational meaning. A null result here is interpretable.
Resist the temptation to swap to a more permissive metric to find a
"finding."

## Pitcher-level reporting in the demo

The tipping page surfaces:

- A line chart of T_start over the season for the selected pitcher, with
  the rolling baseline overlaid and flagged starts marked
- For each flagged start, a card showing:
  - Pitch-type distribution that start vs season average
  - Most exploitable count states (where T_start contribution is highest)
  - The four validation outcomes for that start (subsequent xwOBA delta,
    TTO gap, batter behavior, where applicable)
- A summary header: "X starts flagged in 2024. Y of them showed
  subsequent-start xwOBA elevation."

Do not present a flagged start as "this pitcher was tipping." Use:
"the model detected a predictability spike consistent with potential
tipping. Subsequent performance was [elevated/typical], [TTO penalty
amplified/normal]." Let the validation outcomes speak.

## Things to avoid

- **Computing T_start without the batter-observable mask.** The metric
  becomes circular — you're measuring how predictable the pitcher is to
  a model that sees everything the pitcher sees.
- **Using an unmasked PitchGPT to flag tipping.** Same problem.
- **Claiming a flagged pitcher "was tipping" without subsequent-start
  validation.** Career consequences for pitchers; uncertainty quantification
  is non-optional.
- **Tuning the 1.5σ threshold post-hoc to make case studies fire.** Lock
  the threshold via `docs/decisions/005-tipping-metric.md` before running
  the validations.
