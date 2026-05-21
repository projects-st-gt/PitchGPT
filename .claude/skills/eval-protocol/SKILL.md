---
name: eval-protocol
description: Use this skill any time the project is evaluating a model — implementing baselines, computing calibration metrics, running held-out-pitcher tests, comparing AIPW to g-computation, or producing the eval table. Trigger on mentions of baseline, accuracy, calibration, ECE, reliability diagram, top-k, bootstrapped, confidence interval, held-out, generalization, n-gram, retrieval, XGBoost, or LSTM. The default expectation is that the XGBoost baseline lands within 2–3 points of PitchGPT on top-1 next-pitch accuracy; the transformer's edge is in the *applications* (calibration of rollouts, conditional surprisal, in-support recommendations), not raw prediction. Read this skill before claiming any improvement.
---

# Eval Protocol

Every numerical claim in the project — accuracy, calibration, effect sizes,
tipping detections — passes through this protocol. Skipping a step here
means the writeup either understates uncertainty or overstates the model's
contribution.

## Baselines — implement before training PitchGPT

Implement and evaluate all five before touching the transformer. They
anchor every later claim.

### 1. Marginal frequency

Predict the most common pitch type unconditionally. Floor for sanity.
Implementation: `eval/baselines/marginal.py`.

### 2. Count-conditional frequency

For each of the 12 count states, predict the most common pitch type.
Surprisingly hard to beat on top-1 accuracy. The "bag-of-words" equivalent.
Implementation: `eval/baselines/count_conditional.py`.

### 3. Per-pitcher n-gram with Bayesian smoothing

For each pitcher, build conditional pitch-type tables given (count, last_n
pitch types). Smooth toward the league-conditional prior with a Dirichlet
concentration parameter tuned on the validation set.

This is the strongest non-neural baseline because pitcher identity carries
enormous predictive weight. Use n ∈ {1, 2, 3} and report the best.
Implementation: `eval/baselines/pitcher_ngram.py`.

### 4. Retrieval k-NN

Embed each pitch state as (count, batter handedness, pitcher arsenal vector,
last pitch type). For a query state, find the k=200 nearest neighbours in
the training set and return the empirical pitch-type distribution.

Often beats parametric models on structured-prediction problems. Forces a
defense of "why a transformer?".
Implementation: `eval/baselines/retrieval_knn.py`.

### 5. XGBoost on engineered features

Features: count, batter handedness, pitcher arsenal frequencies, runners,
outs, leverage, score_diff, previous pitch type, previous pitch result,
TTO, days_rest, ballpark, umpire one-hot. Target: pitch type.

This is the bar PitchGPT must clear to justify its existence. Expect XGBoost
to land within 2–3 points of PitchGPT on top-1 accuracy. That is fine. The
transformer's edge is calibration, conditional rollouts, and the recommender
behaving sensibly under positivity gating — not top-1 accuracy. Frame
accordingly.
Implementation: `eval/baselines/xgboost_baseline.py`.

### Optional: LSTM

Same inputs as PitchGPT, LSTM instead of transformer. Tests whether
attention specifically matters for sequence modeling here. Run only if
budget allows. Implementation: `eval/baselines/lstm_baseline.py`.

## Metrics — calibration is primary

Every model in the eval table reports all of:

| metric                       | applies to              |
|------------------------------|-------------------------|
| top-1 accuracy (pitch type)  | all                     |
| top-3 accuracy (pitch type)  | all                     |
| top-1 accuracy (zone)        | PitchGPT, retrieval     |
| ECE (pitch type)             | all                     |
| ECE (zone)                   | PitchGPT, retrieval     |
| reliability diagram          | all (rendered as plot)  |
| Brier score (pitch type)     | all                     |
| log-loss (pitch type)        | all                     |
| AB-outcome AUC               | PitchGPT only           |

Calibration tooling: `eval/metrics/calibration.py`. Reliability diagrams use
15 quantile bins by default; ECE is the equal-mass version.

**Why calibration is primary.** Pitchers are deliberately stochastic. If
the true marginal entropy of pitch-type-given-count is ~1.7 bits, *no model
can score above ~55% top-1*. What matters for the causal layer is whether
"60% fastball" actually corresponds to 60% fastballs in that situation.

## Generalization — held-out pitcher cohort

Standard eval is on the temporal test split. Additionally report metrics on
the **held-out pitchers** cohort: pitchers whose first MLB pitch is in 2024.

If accuracy collapses (more than 5 points below the standard test) for
PitchGPT but not for XGBoost on this cohort, the player-profile encoder is
doing more memorization than generalization, and the writeup should say so
explicitly.

Implementation: `eval/generalization/held_out_pitchers.py`.

## Confidence intervals — bootstrap, always

Every metric reported in the table has a 95% bootstrapped CI. Resample
at-bats with replacement (not pitches — at-bats are the independence unit),
B=1000. Confidence intervals are reported in the eval table; tables without
them get rejected.

Bootstrap utilities: `eval/metrics/bootstrap.py`.

## Causal validation table (separate from prediction table)

This is the table that justifies the causal layer. Every row computed under
K=5 cross-fitting:

| check                                | metric                    | pass criterion              |
|--------------------------------------|---------------------------|-----------------------------|
| μ̂ calibration                       | ECE                       | < 0.03 per result class     |
| π̂ calibration                       | ECE                       | < 0.03 per pitch type       |
| AIPW vs g-computation agreement      | mean abs. diff. on slice  | < 0.02 runs per AB          |
| natural-experiment matched-pair      | Pearson r                 | > 0.5 with model effect     |
| negative control (next batter PA)    | effect estimate           | CI overlaps zero            |
| domain spot-check (0-2 down/away)    | model recommends correctly| sign agrees with literature |
| rollout stability (N=100 vs N=1000)  | rank correlation top-10   | > 0.9                        |

Failures here are blockers, not footnotes. A red row in this table prevents
shipping the demo's causal panel.

Implementation: `eval/causal/`.

## Reporting discipline

Each eval run produces an artifact directory under
`eval/results/{YYYY-MM-DD-HH-MM}-{git_sha}/`:

- `prediction_table.csv` and `.md` for the standard eval
- `causal_validation_table.csv` and `.md`
- `reliability_diagrams/` — one PNG per (model, head)
- `bootstrap_distributions/` — pickled distributions for re-aggregation
- `config.yaml` — exact hyperparameters and data splits
- `git_status.txt` — branch, commit, dirty-file list

`make eval` produces all of the above. PRs claiming improvement must link to
the artifact directory. Do not paste tables into PR descriptions without the
link — it removes the audit trail.

## When PitchGPT does not beat the baseline

This is OK *if and only if* one of the following is true:

- PitchGPT's calibration is meaningfully better at comparable accuracy
- PitchGPT enables a downstream task the baseline cannot (rollouts, AIPW
  with shared nuisance, recommender under positivity)
- PitchGPT generalizes to held-out pitchers where the baseline collapses

Frame the writeup around the strength that exists, not the strength that
doesn't. Inflated accuracy claims tank credibility on every other claim.

## Things to avoid

- **Reporting accuracy without calibration.**
- **Bootstrapping over pitches instead of at-bats.** Wrong independence unit.
- **Tuning hyperparameters on the test set.** Use validation. The test set
  is touched once, at the end, per model variant.
- **Cherry-picking a slice where PitchGPT wins.** Pre-register slices in
  `eval/slices.yaml`.
- **Skipping the causal validation table because the prediction numbers
  look good.** Prediction does not imply causal validity.
