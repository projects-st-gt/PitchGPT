# PitchGPT

A sequential causal model for baseball at-bats. One autoregressive transformer
trained on ~7M Statcast pitches (2017–2025) does double duty as both the
propensity model π̂(a | h) and the conditional outcome model μ̂(y | a, h) inside
a g-computation / AIPW pipeline. The deliverables are (1) a working causal
inference layer with cross-fitting and positivity gating, (2) a tipping
detector based on batter-observable predictability, and (3) an interactive demo
that *teaches epistemic humility* — showing a trust gauge, refusing
out-of-support queries, and surfacing sensitivity to unmeasured confounding.

The primary target is the demo and a methods writeup. A paper falls out only if
the methods work hold up. Do not optimise for publishability over correctness.

## What this project is not

- Not a flat-vocabulary GPT with a "novel architecture" claim. Factored
  embeddings are sound engineering, not a contribution.
- Not a "counterfactual" demo in any causal sense unless the causal machinery
  is actually running. Use "alternative completion" or "model rollout" otherwise.
- Not era-mixing speculation, cross-sport transfer, or video/multimodal work.
  These are scope-creep traps.
- Not chasing next-pitch top-1 accuracy as the headline metric. Calibration,
  trust-region behaviour, and the causal estimands are what matter.

## Repo map

- `data/`         Statcast extraction, harmonization, player profiles, dataset
- `model/`        factored embeddings, transformer, two-stage result head, training
- `causal/`       propensity wrapper, g-computation rollout, AIPW, cross-fitting,
                  positivity, sensitivity analysis
- `tipping/`      batter-observable variant, T_start computation, validations
- `recommender/`  trust-region-restricted causal recommendation
- `inference/`    FastAPI app, pre-computed demo cache
- `frontend/`     React + TS + Tailwind demo
- `eval/`         baselines, calibration, held-out-pitcher generalization
- `docs/decisions/`  ADRs (see decision log below)

## Commands

Canonical commands live in the `Makefile`. Common ones:

- `make extract`            run pybaseball extraction with checkpoint/resume
- `make preprocess`         harmonize, tokenize, build profiles, write parquet
- `make train MODEL=small`  train PitchGPT (sizes: tiny | small)
- `make crossfit K=5`       run K-fold cross-fitting for nuisance models
- `make eval`               full eval table: baselines + PitchGPT + calibration
- `make demo`               start FastAPI + Vite dev server

Reach for `make eval` after any model or baseline change before claiming an
improvement. Do not paste accuracy numbers into PRs without it.

## Hard rules — never violate

1. **Real Statcast data only.** Every pitch in training, evaluation, and the
   demo cache must come from real Statcast via pybaseball. No synthetic,
   mocked, generated, or interpolated pitches anywhere. The frontend may
   scaffold against a JSON fixture of *real* pitches before the API is up.
1a. **No placeholder / stub / boilerplate code in production paths.** Every
   function shipped in a non-test module must be a real, complete
   implementation — no `pass`/`return None` stubs, no "TODO: implement", no
   fabricated constants standing in for values that should be computed from
   data, no hardcoded fallback numbers presented as if real. If a value can be
   derived from the real data, derive it; if a degenerate-case guard is truly
   needed, make it raise or use an *explicitly data-derived* fallback (e.g. the
   empirical mean), never a magic literal. Synthetic data is allowed ONLY in
   `tests/` as controlled fixtures for unit-testing pure functions — never in
   training, eval, inference, or demo code. When in doubt, compute it for real
   or raise; do not paper over a gap.
2. **Temporal splits.** Train ≤ 2023, val = 2024H1, test = 2024H2 + 2025.
   Never shuffle at-bats across years. Player-profile trailing windows must
   end strictly before the at-bat in question — no within-game leakage either.
3. **Cross-fitting is mandatory** for any AIPW or population-level causal
   estimate. K=5, stratified by season, blocked by `game_pk`. Single-fit
   AIPW numbers do not get reported.
4. **Positivity gate is mandatory** for any individual counterfactual query.
   Below τ=0.01 propensity, the system *refuses* a point estimate and shows an
   extrapolation warning. The refusal is a feature, not a bug — do not silence
   it to make the demo look smoother.
5. **No causal language without the causal machinery.** "Counterfactual,"
   "effect of," "if he had thrown" are reserved for outputs from g-computation
   or AIPW with cross-fitting and positivity checks. Otherwise: "alternative
   completion," "model rollout," "what the model expects."
6. **Calibration is a primary metric** for π̂, μ̂, and every baseline. Reports
   include ECE and reliability diagrams alongside accuracy. Accuracy-only
   tables get rejected.
7. **Reserved words in writeups.** Do not call factored embeddings "novel."
   Do not claim PitchGPT "discovers" anything that wasn't measured against a
   pre-registered baseline. Do not promise effects without confidence intervals.
8. **No SF Pro on the web** (license violation). Use Inter. Pitch types must
   use color *plus* a glyph or shape — color alone fails ~8% of male users.

## Bug-prevention discipline — for model-interfacing code

The project's input/output factor conventions are not uniform: the type
factor stores ids 1..7 in the parquet with PAD=0, and the propensity TYPE
HEAD emits 8 logits (PAD at index 0, PITCH_TYPES at indices 1..7) — but the
RESULT HEAD emits 7 logits with no PAD column. This asymmetry has bitten
three code paths in one session, every one a slice that mixed up the
convention. Five disciplines to prevent it; these override generic instincts
about brevity or "smoke-tested = done."

1. **Print a sample before writing the slice.** When writing code that reads
   any model head output, first print one row's *named* values. Trace which
   index maps to which pitch type / result class. Only then write the slice.

2. **Show output, never just claim "smoke tested."** A test that says
   "passes" without printing `μ̂`, `π̂`, or a named per-class probability hides
   convention bugs. Always print at least one *named* number —
   `π̂(FF) = 0.35`, not `argmax matches: True`. Named numbers surface
   convention errors immediately.

3. **Test the convention first.** Before AIPW / cross-fit / g-comp code that
   reads `propensity_probs["type"]`, write `assert pi[MODEL_TYPE_ID["FF"]] >
   0.15` on a real AB. ~3 lines, catches the bug class deterministically.

4. **Name the convention as constants — don't slice with magic numbers.**
   `[:N_PITCH_TYPES]` is a convention claim; name it. Use
   `MODEL_PITCH_TYPES_START_IDX` / `MODEL_PITCH_TYPES_END_IDX` /
   `MODEL_TYPE_ID["FF"]` from `data/dataset.py`. Magic numbers are where
   convention bugs hide; named constants make the claim explicit at the
   call site.

5. **One specific numerical check per smoke test.** Replace "tests pass" with
   "the model predicts top-1 of SL on AB X" or "π̂(FF) on AB Y = 0.42." If
   the user can't recompute the answer mentally and gut-check it, the test
   isn't doing anything.

**User-enforceable rule:** I commit to printing at least one *named numerical
output* per smoke test, before any claim of "working." If I write a smoke
test without such an output, the user is expected to reject it with "what's
`π̂(FF)` on the first AB?" — I should not have shipped the test without that
check baked in.

## Skill index

Skills load on demand. Trigger by topic, not just by keyword:

- `statcast-pipeline`  — pybaseball extraction, harmonization, profile
                          leakage rules, run-value table
- `pitchgpt-model`     — factored embeddings, two-stage result head, training
                          conventions, attention masking, model sizes
- `causal-layer`       — g-computation, AIPW, cross-fitting, positivity gating,
                          E-values, negative controls
- `tipping-analysis`   — batter-observable variant, T_start metric, four
                          validation runs, failure-mode reporting
- `eval-protocol`      — baseline implementations, calibration metrics,
                          held-out-pitcher cohort, bootstrapped CIs
- `frontend-system`    — design tokens, components, motion, trust-gauge UI,
                          refusal UX

When working in any of these areas, read the skill before writing code. If a
convention in CLAUDE.md and a skill conflict, the skill is more specific —
follow the skill and flag the conflict in the PR.

## Workflow

- Plan before code for non-trivial work. For changes that touch a hard rule,
  the model architecture, or a causal-layer component, draft an approach in
  `docs/decisions/` first and link it in the PR.
- Reference files by `path:line` rather than pasting snippets. Snippets go
  stale; pointers do not.
- Update the corresponding `SKILL.md` when conventions change. Do not
  duplicate skill content into `CLAUDE.md`.
- Use `gh` (GitHub CLI) for PRs and CI checks. Do not paste CI logs into chat
  unless asked.

## Decision log

ADRs live in `docs/decisions/`. The set that must exist before training
starts:

- `001-treatment-granularity.md`  pitch type only? type+zone? type+zone+velo-tertile?
- `002-positivity-threshold.md`   τ value, calibration check, ESS rules for multi-step
- `003-confounder-set.md`         catcher, fatigue, TTO, umpire, leverage, ballpark, days rest
- `004-run-value-definition.md`   RE24 from data + xwOBA-on-contact mapping
- `005-tipping-metric.md`         batter-observable T_start vs alternatives considered
- `006-cross-fit-blocks.md`       K=5, season-stratified, `game_pk`-blocked

If a decision is not in the log and not in CLAUDE.md, treat it as undecided
and ask before assuming.

## State to check before starting work

- `make status`          shows extraction progress, latest checkpoint, last eval
- `docs/decisions/`      open ADRs that may affect the task
- `.claude/skills/`      relevant skill for the area being touched
- `eval/results/latest/` last full eval run, for regression comparison
