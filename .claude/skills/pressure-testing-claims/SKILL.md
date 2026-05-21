---
name: pressure-testing-claims
description: Use BEFORE asserting a non-obvious quantitative or factual claim — when citing external constants ("Tango's value is ~0.7"), claiming library/API behavior ("requests respects timeouts"), describing data shape ("this column is populated"), or making methodological claims ("this is doubly-robust"). Verify with evidence before the claim lands in code, docstrings, ADRs, or user-facing text. Trigger on phrases like "I'm pretty sure", "I recall", "the canonical X is Y", or any time a number-with-precision is being cited from memory rather than computed. Not for routine edits, already-verified items in the same session, or pure subjective preferences.
---

# Pressure-Testing Claims

The instinct to assert from training-data memory is strong. The cost of an
unverified claim is sometimes silent and high — it propagates into code,
methods documents, downstream calculations, and the user's understanding.
This skill is the habit of *verifying before asserting*, not second-guessing
after.

## The rule

Before any non-obvious quantitative or factual claim lands in a durable
artifact (code, comment, docstring, ADR, user-facing summary, methods
write-up), pressure-test it. Find a way to check the claim against the actual
system or data, run that check, and then write the claim with the evidence
behind it.

If you can't find a way to verify the claim, either drop the claim or
explicitly mark it as unverified and note what would be needed to verify.

## When to pressure-test

Always test:

- **Cited constants.** "Tango's wOBA-to-runs is ~0.7." "The standard
  τ for positivity is 0.01." "MLB has ~750K pitches per season."
- **Library/API behavior.** "`requests` respects timeouts by default."
  "`pandas` handles NaN automatically here." "pybaseball returns these
  columns."
- **Data shape and content.** "This column exists." "Values are
  normalized." "The cohort has at least N members."
- **Performance claims.** "This will be fast enough." "Memory will fit."
  "The API will rate-limit us at X req/sec."
- **Methodological claims.** "This estimator is unbiased." "This is
  doubly-robust." "Cross-fitting handles this kind of leakage."
- **Cross-system integration claims.** "This scrape will work." "This
  format will deserialize." "These two columns will join correctly."

Don't test:

- Routine code edits where the diff itself is the verification.
- Decisions the user has already explicitly made.
- Pure subjective design preferences with no factual content.
- Items already verified earlier in the same session that aren't suspected
  of having changed.

## How to pressure-test

1. **Identify the load-bearing claim.** What's the specific factual
   assertion that, if wrong, would break the work or mislead the user?
   Strip away the framing; isolate the claim.
2. **Pick the cheapest reliable test.**
   - For data claims: load real data and check the column / value / shape.
   - For API claims: curl or call it once and inspect the response.
   - For library claims: read the source or docs, or write a 5-line script
     that exercises the claim.
   - For numerical claims: derive from first principles or run an empirical
     fit on real data.
3. **Run the test.** Do not skip this step. Reading the test code is not
   running it.
4. **Report with evidence visible.** Show the verification — the script
   output, the curl response, the cite. If the claim is confirmed, the
   evidence is what makes it credible. If it's disconfirmed, say so
   explicitly and revise. Don't paper over a wrong claim with hedging.

## When the user asks "are you sure?"

That's a direct invitation to pressure-test. Don't reaffirm; go verify.
Report back with the evidence. The user doing this is doing the right thing,
and the right response is to do the work, not to double down on an unverified
assertion.

## Anti-patterns

- **"I'm pretty sure"** / **"I recall that"** — red flags. Verify or omit.
- **Asserting a number with decimal-place precision** without a citation or
  computation. If the precision isn't earned, round to less or omit.
- **Citing a paper or person** without quoting/computing the actual value.
  If the citation isn't quoted, it isn't load-bearing — drop it.
- **Tests that encode the same confused belief** as the bug. If the test
  was written from memory of "what the function should do," verify that
  the test's expected value is itself empirically grounded.
- **Hedging-as-substitute.** "It should work" / "this is roughly right"
  is not a substitute for evidence. Either verify and assert, or say "I
  haven't verified this, here's what would be needed."

## Worked example: the wOBA-to-runs constant

The claim that landed in code, docstrings, and an ADR: *"Tango's published
wOBA-to-runs constant is ~0.7."*

Pressure-testing it:

1. **Load-bearing claim isolated.** "Tango's published constant for the
   wOBA → runs mapping equals 0.7."
2. **Tests run on real 2023 Statcast data:**
   - Per-pitch in-play slope: **0.487**
   - Per-PA actual-wOBA slope: **0.602**
   - Per-event ratios (mean Δrun_exp / mean wOBA value):
     single 0.499, double 0.583, triple 0.632, HR 0.766, walk 0.335, HBP 0.517
3. **Findings:** No aggregation produces "~0.7" as a canonical value.
   FanGraphs' published "wOBA scale" is ~1.20 for 2023 — but that's runs
   above average per (PA × wOBA-difference), a different calculation entirely.
4. **Conclusion:** the "0.7" was confabulated. Cost of *not* pressure-testing:
   the constant would have entered the run-value layer and silently mislabeled
   downstream causal estimates as "verified against published value."

What replaced it: per-year empirical slopes from
`compute_in_play_woba_to_runs_slope`, saved to
`data/run_value/woba_to_runs.json`. No magic-number constant; the slope is
recomputed per training season from the same data the rest of the run-value
layer uses.

The lesson generalizes: **the moment a number-with-precision shows up in your
output, you owe the reader either a citation or a computation. Memory alone
isn't sufficient evidence to write it down.**
