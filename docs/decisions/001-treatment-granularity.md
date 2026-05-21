# ADR 001 — Treatment Granularity

**Status:** Accepted (locked 2026-05-09)
**Date:** 2026-05-08

## The question, in plain English

When the demo asks "what if he'd thrown a different pitch?", how specific is *a different pitch*?

- Just the pitch type (slider vs. fastball vs. changeup)?
- Pitch type plus where it was thrown in the zone?
- Pitch type plus location plus how hard?

This is the choice of "action space" for our causal layer. Every counterfactual query operates on this action space.

## Why this matters

- **Too coarse** → "throw a slider" averages over very different sliders (low-and-away vs. heart-of-zone) and the demo loses interesting queries.
- **Too fine** → almost every counterfactual hits the positivity wall ("not enough data") and the demo refuses everything. Coverage collapses, ESS collapses on multi-step rollouts, training cells empty out.

The right granularity is the largest action space where most realistic counterfactuals still have support.

## Options

**A. Pitch type only.** ~6–8 actions (FF, SI, SL, CB, CH, FC, etc.). Maximum support — almost every action has high π̂ for almost every state. Loses "where" entirely.

**B. Pitch type × zone bin.** With 9 standard Statcast zones: ~55 actions. With a coarser 5-zone scheme (in-up, in-down, in-arm-side, in-glove-side, out-of-zone): ~30 actions. Captures the "where" the demo needs without exploding cardinality.

**C. Pitch type × zone × velo tertile.** ~150+ actions. Velo tertiles defined within (pitcher, type) since absolute velo is meaningless across pitchers. Captures intent ("backup slider" vs. "good slider") but velo is heavily correlated with pitch type and pitcher fatigue, so cells are sparse and the action space mostly overlaps with itself.

## Recommendation

**Option B with a 5-zone collapse.** Five zones: `up`, `down`, `arm-side`, `glove-side`, `out-of-zone`. Action space = pitch type × zone ≈ 30 cells.

Velocity is *not* part of the action; it's a feature of the resulting outcome and is conditioned in μ̂. So "throw a slider low" is the action; how hard the slider was is part of the outcome model's job.

### Zone definitions (precise)

Coordinates: Statcast `plate_x` (lateral, 0 = middle of plate, sign convention per Statcast), `plate_z` (height, ft), and the per-pitch dynamic zone `sz_top` / `sz_bot`. Plate half-width = 0.83 ft.

Define:

```
z_norm   = (plate_z - sz_bot) / (sz_top - sz_bot)        # 0 = bottom of zone, 1 = top
x_norm   = plate_x / 0.83                                 # -1 = left edge, +1 = right edge
in_zone  = (-1 <= x_norm <= 1) and (0 <= z_norm <= 1)
arm_sign = +1 for RHP, -1 for LHP                         # arm-side is positive plate_x · arm_sign
```

Cell assignment:

| Cell | Condition |
|---|---|
| `up` | `in_zone` AND `z_norm > 0.67` |
| `down` | `in_zone` AND `z_norm < 0.33` |
| `arm-side` | `in_zone` AND `0.33 ≤ z_norm ≤ 0.67` AND `(x_norm · arm_sign) > 0` |
| `glove-side` | `in_zone` AND `0.33 ≤ z_norm ≤ 0.67` AND `(x_norm · arm_sign) ≤ 0` |
| `out-of-zone` | NOT `in_zone` |

Boundary handling: `z_norm` thresholds use `<` and `>` strictly to avoid double-assignment; pitches exactly on a boundary go to whichever cell the inequality lands them in by floating-point comparison. Pitches with `sz_top - sz_bot < 1.0` ft (Statcast measurement errors) are dropped at the dataset stage, not classified.

Out-of-zone may be further partitioned into four OZ quadrants (high-OZ, low-OZ, arm-side-OZ, glove-side-OZ) if a specific query needs it; the default action space treats all OZ as one cell.

### Why this 5-zone scheme rather than alternatives

- **Why not a uniform 3×3 grid?** The 9-zone scheme inflates the action space to ~55 cells without sequencing-relevant gains. Most strategy decisions don't distinguish "middle-up-arm-side" from "middle-up-glove-side" — they distinguish "up" from "everything else."
- **Why full-width `up` and `down`?** The strategically meaningful high-pitch and low-pitch decisions are about height first; arm/glove-side becomes secondary near the top/bottom of the zone. A high fastball over the heart of the plate and a high fastball just off the outside corner are functionally similar setup pitches; treating them as separate actions burns positivity for no signal gain.
- **Why arm/glove-side only at mid-height?** The east-west axis is where slider/sinker semantics live. Arm-side mid is the running-fastball/two-seam location; glove-side mid is the cutter/slider corner. These are genuinely different intents and deserve separate cells.
- **Why pitcher-frame (arm-side / glove-side) rather than batter-frame (in / away)?** The pitcher chooses location relative to their own arm; the batter-frame label flips with batter handedness and produces two "different" actions for one pitcher decision. Pitcher-frame keeps the action label invariant under batter swap and is the right frame for π̂.
- **Why `out-of-zone` as one cell by default?** The dominant binary the demo cares about is "did he attack the zone or try to get a chase?" Refining OZ further multiplies cells without evidence the additional resolution improves either positivity or interpretability. Reserved as a per-query refinement, not a default.

This scheme keeps the action space at ~30 cells — small enough that most realistic counterfactuals have π̂ > τ, large enough to support the "low and away vs. heart" queries that make the demo interesting.

## Divergence from the brainstorm

The brainstorm listed treatment granularity as an open question (three options, no pick). **This picks the middle option (B) and explicitly rejects the velocity-tertile option (C)** because of positivity collapse — the multi-step IPW weights blow up when each cell is rare.

## Consequences

- The 5-zone scheme needs to be specified concretely once (zone boundaries, edge cases for borderline pitches). Lives in `data/zones.py`.
- The recommender's "supported alternatives" panel shows up-to-30 alternatives per state, ranked by E[run-value | do(action)]. Many will be filtered by positivity; that's expected.
- The trust-gauge UX is per-action: each of the ~30 actions in a state gets its own gauge color.
- Adding velocity as a *contextual* feature (the pitcher's velo tendency on this pitch type, today's average) is fine and complementary; it just isn't part of the action.
