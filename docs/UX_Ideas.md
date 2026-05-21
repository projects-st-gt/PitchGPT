# PitchGPT — UX / Tab Ideas

Capturing the full menu of user-facing experiences we've brainstormed. Tabs
1-6 are the immediate roadmap (current + planned for Sprint 2/3). Tabs 7-12
are stretch goals and what-a-team-actually-needs ideas.

## Tab 1 — Play-by-play X-ray (educational, easy)

**Pitch**: Pick a real game + at-bat, see the model's pitch-by-pitch predictions
side-by-side with what actually happened.

**Mechanic**: For each pitch in the AB:
- Actual: type, zone, count progression, result.
- Model's *pre-pitch* top-3 prediction (already returned by `/ab-context`).
- "Score": was the top-1 right? Top-3? Was the actual a low-π̂ surprise?

**Layout**: Split-pane. Left = actual AB unfolding. Right = at each pitch, model's
prediction *before* it knew what came. A small "report card" at the bottom: per-AB
accuracy + ECE on this slice.

**Value**: Trust-builds the model. Shows that it's doing real prediction (not
canned answers). Great for non-technical audiences and methods-paper figures.

**Effort**: ~2 days frontend; no new backend (data already in `/ab-context`).

## Tab 2 — Counterfactual (current Sprint 2 build)

**Pitch**: "What if the pitcher had thrown X here?"

**Mechanic**: Already built. Pick AB → pick intervention position → pick type +
zone → rollout. Continuous trust gauge (no hard refusal) labels how supported
the causal claim is.

**Status**: Largely complete. Ongoing polish: 14-zone after retrain, two-step
picker (just landed), arm-slot wiring.

## Tab 3 — Live game

**Pitch**: Pick an in-progress MLB game, see real-time pitch predictions evolve.

**Mechanic**:
- Poll `https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live` every
  3-5 sec for pitch events.
- For each new pitch event:
  - Update displayed game state (scoreboard, count, runners).
  - Show: model's predicted next-pitch distribution + zone heatmap *before*
    the pitch is thrown.
  - When the actual pitch arrives, render comparison vs model.
- Win probability: separate Markov-style calculator that integrates per-AB
  outcome distributions over the rest of the game.

**Components**:
- Live ingestion service (poll → diff → SSE/WebSocket push to frontend).
- Win-prob calculator (Markov over inning × outs × base state × score).
- Real-time UI with SSE consumer.

**Effort**: ~3 weeks total. Live ingest + UI ~1.5 weeks; win-prob ~1 week; polish ~0.5 week.

**Caveat**: Model trained ≤ 2023. By mid-2026, ~30% of pitchers are post-train debutants
who fall back to league mean. After Sprint 6 (retrain through 2025), this drops to ~10%.

**Key value-add**: Visible, demonstrable real-time epistemic humility — the demo's
refusal under low-support live cases lands harder than any static example.

## Tab 4 — Daily slate / Season projections

**Pitch**: Predicted outcome for every MLB game today (and project the season).

**Mechanic**:
- Full-game simulator: chains AB-level g-computation into 9-inning games.
- Needs: lineup data, starting pitcher info, pitcher-change rules, pinch-hit /
  defensive-sub logic.
- For each scheduled game: run N=300-500 full simulations →
  `P(home wins)`, expected score, expected runs, expected pitcher pitch counts.

**Validation**: Backtest on 2024-H2 → compare `P(home win)` to Vegas closing odds.
Calibration plot is the headline figure.

**Effort**: ~3 weeks. Game-state simulator core: ~1.5 weeks. Manager-decision
rules (pitch count thresholds, leverage-based pulls, TTO rules): ~1 week.
Polish + validation: ~0.5 week.

**Important**: this is the *foundational* tab. Tabs 3, 7, 8, 11 all reuse the
full-game simulator built here.

## Tab 5 — Pitch Strategy Optimizer / Trust-Region Recommender

**Pitch**: "What should the pitcher throw here?"

**Mechanic**:
- Input: pitcher, batter, count, runners, outs.
- Enumerate every `(pitch_type, action_zone)` combination — 7 × 5 = 35 candidates
  per ADR 001 (or 7 × 14 after the 14-zone retrain).
- For each: run g-computation → expected run value.
- Apply the **trust region**: only show candidates with π̂(pitch|state) > τ.
- Display as a heatmap per pitch type, color-coded by expected run value
  (green = pitcher-favored, red = batter-favored).
- Highlight the recommended pitch.

**Value**: Directly answers the question pitchers / catchers / coaches ask
constantly. Trust region prevents nonsense recommendations.

**Effort**: ~1 week (UI work + an enumerate-and-rollout loop using existing g_compute).

## Tab 6 — Tipping Audit (Sprint 3)

**Pitch**: "Is your pitcher tipping pitches? Where?"

**Mechanic**:
- Train a *batter-observable* classifier: given the visible cues a batter sees
  (pitcher's arm slot at release, release-point variance, prior pitch types/results
  in this AB), predict the next pitch type.
- `T_start`: earliest pitch index within an AB where this classifier's accuracy
  beats the marginal pitch-type distribution by a meaningful margin.
- Per-pitch-type "tipping score" — continuous 0-1.
- Identify specific cues: "Arm angle drops 4° on average for sliders vs fastballs."

**Inputs needed**: arm slot data (from Statcast `arm_angle`, already wired in Sprint 0b).

**Value**: Directly actionable for pitching coaches. "Your slider releases 4° lower" is
a coachable insight.

**Effort**: ~1.5 weeks (separate batter-observable classifier + four validation runs
per ADR 005 + UI).

## Tab 7 — Bullpen Optimizer

**Pitch**: "Given the state, which reliever should we bring in?"

**Mechanic**:
- For each available reliever in the bullpen:
  - Apply their pitcher profile to the upcoming AB(s).
  - Run g-computation under that reliever.
  - Compute expected runs allowed in the next inning(s).
- Rank relievers; flag the optimal call.

**Effort**: ~1 week. Reuses existing g_compute; just a UI + enumeration loop.

## Tab 8 — Trade Evaluator / Roster What-Ifs

**Pitch**: "What does adding pitcher X do to our team's projected wins?"

**Mechanic**:
- Drop the candidate pitcher into a team's rotation/bullpen.
- Use the full-game simulator (from Tab 4) to project the team's next 30 / 60
  games under the new roster vs. current roster.
- Δ wins, Δ team ERA, Δ leverage performance.

**Effort**: ~2 weeks (reuses Tab 4 simulator + light UI).

## Tab 9 — Umpire-Aware Game Plan

**Pitch**: "Tonight's home plate umpire is [name]. Here's how to adjust."

**Mechanic**:
- Each umpire has a profile: their historical strike-zone tendencies
  (where they expand vs tighten).
- Recompute strategy recommendations (Tab 5) conditional on this umpire's profile.
- "This ump expands down-and-away vs RHB by ~3%. Throw more sliders down-away tonight."

**Inputs**: Per-umpire historical called-strike data (already a feature in the model;
we have `umpire_id` as a categorical context).

**Effort**: ~1 week — mostly UI; the model already conditions on umpire.

## Tab 10 — Hitter Coaching: "What should you look for?"

**Pitch**: Viewpoint-flip of Tab 5 — for the batter facing a specific pitcher.

**Mechanic**:
- For each likely pitch from the pitcher (sorted by π̂):
  - Probability + zone heatmap + expected result.
- Highlights the pitches the batter should be "looking for" — most probable + most
  punishable (high `(P(hit) × P(reachable swing))`).

**Effort**: ~0.5 week (essentially Tab 5 with the run-value reversed).

## Tab 11 — Player Development / Minor-League Projection

**Pitch**: "How will this minor-league pitcher perform in MLB?"

**Mechanic**:
- Need a *projection layer* — map minor-league stats / video tracking to MLB-equivalent
  profile. We don't have this today.
- Possible approach: profile-similarity matching — find MLB pitchers with similar
  arsenal/velocity/spin from training data, project the rookie's performance based on
  weighted average.

**Effort**: ~2-3 weeks (the projection layer is the hard part).

## Tab 12 — Auto Game-Recap

**Pitch**: After a game ends, auto-generate a written recap.

**Mechanic**:
- Chain an LLM on top of the model's per-pitch outputs.
- Highlight: biggest win-prob shifts, questionable manager decisions, surprising
  outperformances.

**Effort**: ~1 week (LLM + template + per-game data extraction).

---

## Cross-cutting design principles

(Locked from CLAUDE.md and our discussions:)

- **Apple-minimalist design**: Inter font only (SF Pro license violation),
  monochromatic + one teal accent + amber warn + red refusal-only.
  Generous whitespace (`py-16` to `py-24`). Hairline borders (gray-200).
  Three weights (400/500/600), three sizes (14/16/22/32). 200ms ease-out
  on state changes.
- **Pitch types use color + glyph + name** (accessibility: ~8% of male users
  can't distinguish by color alone).
- **No hard refusal — continuous trust gauge** (ADR 002 Option D). Always show
  the rollout; label it `high / moderate / low` support. Plain-English
  explanations ("essentially never goes there" rather than `π̂(zone) = 0.005`).
- **Real Statcast only** (no synthetic / mocked data anywhere).
- **Temporal split is sacred** (train ≤ 2023, val = 2024 H1, test = 2024 H2 +
  2025; test touched once at the end).

## Foundational dependency

Every advanced tab (3, 4, 7, 8, 11) depends on the **full-game simulator** built for
Tab 4. Sequence it first; the rest follow more cheaply.

## Team-pitched MVP

Tabs 1 + 2 + 5 + 6 — *play-by-play X-ray + counterfactual + strategy recommender +
tipping audit* — is a real product an analytics dept would actually use. ~6-8 weeks
from current state. Everything else is upside.
