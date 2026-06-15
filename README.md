# PitchGPT

A causal inference engine for baseball, built on an autoregressive transformer trained on ~7M MLB Statcast pitches (2017-2025). PitchGPT models the sequential structure of plate appearances pitch-by-pitch and powers two applications: a **counterfactual pitch-type analysis** (what happens if a pitcher throws a different pitch?) and a **full-game Monte Carlo simulator** that projects scores and win probabilities for real MLB matchups.

## What it does

**Pitch-level modeling.** A single transformer learns both the pitch-selection propensity (what pitch is likely?) and conditional outcome distribution (what happens after it?) from the full sequence of pitches in an at-bat. This dual role eliminates the need for separate propensity and outcome models.

**Causal inference.** The model feeds into a g-computation and AIPW (augmented inverse-propensity weighting) pipeline with 5-fold cross-fitting, positivity gating, and E-value sensitivity analysis. This lets us ask genuinely causal questions: "How much does throwing a slider here change the expected run value?" The system refuses to answer when the data doesn't support the counterfactual (positivity violations).

**Game simulation.** Pre-computed matchup cards (per pitcher-batter outcome distributions) feed a Monte Carlo engine that simulates full 9-inning games with realistic bullpen management, times-through-order penalties, park factors, and extra-inning rules. 10,000 simulations per game produce projected scores, win probabilities, and over/under distributions.

## Architecture

```
                    ┌──────────────────────────────┐
                    │       Statcast Pitches        │
                    │   ~7M pitches, 2017-2025      │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │         PitchGPT              │
                    │   4-layer transformer, 256d   │
                    │   Factored embeddings (11     │
                    │   factors per pitch token)    │
                    │   ~6M parameters              │
                    ├───────────────────────────────┤
                    │  Outputs:                     │
                    │  - π̂(type | history) propensity│
                    │  - μ̂(result | type, history)  │
                    └──────┬──────────────┬────────┘
                           │              │
              ┌────────────▼──┐    ┌──────▼──────────┐
              │  Causal Layer │    │ Hitter Cascade   │
              │  g-computation│    │ 5 XGBoost models │
              │  AIPW + xfit  │    │ swing → whiff →  │
              │  positivity   │    │ called_strike →  │
              │  E-values     │    │ contact_quality →│
              │               │    │ contact_outcome  │
              └───────────────┘    └────────┬────────┘
                                            │
                                 ┌──────────▼──────────┐
                                 │   Matchup Cards      │
                                 │ per (pitcher, batter) │
                                 │ outcome distributions │
                                 └──────────┬──────────┘
                                            │
                                 ┌──────────▼──────────┐
                                 │  Game Simulator      │
                                 │  10K MC sims/game    │
                                 │  bullpen, TTO, parks │
                                 │  → scores, win prob  │
                                 └─────────────────────┘
```

### PitchGPT Transformer

Each pitch in an at-bat is represented as a **factored embedding**: 11 factors (pitch type, zone, velocity bucket, spin rate, movement, etc.) are independently embedded and summed into a single token. Three context tokens are prepended to every sequence encoding the pitcher profile, batter profile, and game situation (count, runners, outs, inning, ballpark, umpire). The transformer processes the sequence autoregressively and produces:

- A **pitch-type propensity head** predicting the distribution over 7 canonical pitch types (FF, SI, FC, SL, CU, CH, FS)
- A **result head** predicting the pitch outcome (ball, called strike, swinging strike, foul, in-play)

### Hitter Cascade

Five gradient-boosted trees (XGBoost) form a decision cascade that converts a pitch into a plate-appearance outcome:

1. **Swing** — does the batter swing? (binary)
2. **Whiff** — if swinging, does he miss? (binary)
3. **Called strike** — if not swinging, is it a strike? (binary, umpire-aware)
4. **Contact quality** — if contact, what's the expected quality? (regression: launch angle × exit velo → xwOBA)
5. **Contact outcome** — if contact, what happens? (multiclass: out, 1B, 2B, 3B, HR)

The cascade consumes pitch characteristics from PitchGPT plus batter/pitcher profiles, ballpark, and game context.

### Causal Inference Layer

For counterfactual questions ("what if he threw a changeup instead of a fastball?"), the system uses:

- **g-computation**: rolls out alternative pitch sequences through the model
- **AIPW**: doubly-robust estimation combining propensity and outcome models
- **5-fold cross-fitting**: stratified by season, blocked by game, to prevent overfitting-induced bias
- **Positivity gating**: refuses estimates when a pitch type has < 1% propensity (the model hasn't seen enough data to answer)
- **E-values**: quantifies how strong unmeasured confounding would need to be to explain away an effect

### Game Simulator

The Monte Carlo engine simulates full baseball games:

- Reads pre-computed matchup cards (outcome distributions for every pitcher-batter pair in both lineups)
- Simulates plate appearances by sampling from those distributions
- Models **times-through-order** penalties (batters hit better the 2nd and 3rd time they face a starter)
- Applies **park factors** (Coors inflates hits, Petco suppresses them)
- Manages **bullpen transitions** with workload-aware pitcher changes
- Handles **extra innings** with ghost runners, walk-offs, and a 30-inning backstop
- Produces win probabilities, projected scores, and run distributions from 10,000 simulations per game

## Project Structure

```
data/           Statcast extraction, harmonization, player profiles, dataset
model/          Factored embeddings, transformer, two-stage result head, training
causal/         Propensity wrapper, g-computation, AIPW, cross-fitting,
                positivity gating, sensitivity analysis (E-values)
hitter/         XGBoost cascade (5 nodes), feature engineering, rollout bridge
gamesim/        Game state machine, Monte Carlo engine, bullpen policy, park factors
mcsim/          Matchup card computation, MLB API integration, storage
inference/      FastAPI app, demo API
frontend/       React + TypeScript + Tailwind demo UI
eval/           Baselines, calibration metrics, held-out-pitcher generalization
scripts/        Training, extraction, card generation, Modal GPU runners
docs/decisions/ Architecture Decision Records (ADRs)
```

## Key Design Decisions

The project maintains Architecture Decision Records in `docs/decisions/`. Key ones:

| ADR | Decision |
|-----|----------|
| 001 | Treatment is pitch type only (not type+zone) — positivity requires it |
| 002 | Positivity threshold τ = 0.01; below this, refuse the estimate |
| 003 | Confounder set: catcher, fatigue, TTO, umpire, leverage, ballpark, days rest |
| 004 | Run value = RE24 from data + xwOBA-on-contact mapping |
| 005 | Tipping metric uses batter-observable pitch predictability |
| 006 | Cross-fitting: K=5, season-stratified, game_pk-blocked |

## Results

### PitchGPT Transformer

Evaluated on the 2024 H1 validation set (440K pitches, 113K at-bats):

| Metric | Value |
|--------|-------|
| Pitch-type top-1 accuracy (π̂) | **47.7%** |
| Result top-1 accuracy (μ̂) | **54.0%** |
| Pitch-type ECE (calibration) | 0.019 |
| Result ECE | 0.004 |

The model generalizes well to pitchers it has never seen: held-out pitchers (2024 debutants, n=206) achieve 47.9% type accuracy vs 47.7% for known pitchers. Pitcher-specific rare pitches (splitters, changeups) show expected degradation.

### Hitter Cascade

Per-node metrics on held-out test data:

| Node | Metric | Value |
|------|--------|-------|
| Swing | AUC | 0.869 |
| Whiff | AUC | 0.787 |
| Called Strike | AUC | 0.985 |
| Contact Quality | RMSE | 0.369 |
| Contact Outcome | Accuracy | 0.681 |

### Game Simulator

Backtested against ~157 real MLB games (June 2026). Results are actively being improved — the current iteration is fixing home-field advantage modeling and ballpark effects:

| Metric | Baseline | Target |
|--------|----------|--------|
| Pick accuracy | 54.2% | > 55% |
| Brier skill score | -0.006 | > 0 (better than coin flip) |
| Home-field gap | -0.071 | ~0 (calibrated) |

## Usage Examples

### Counterfactual Query (API)

Ask "what if the pitcher had thrown a slider instead?" via the FastAPI endpoint:

```bash
# Start the API server
uvicorn inference.api:app --port 8000

# Query a counterfactual
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "game_pk": 745534,
    "at_bat_number": 42,
    "pitch_number": 3,
    "intervention_type": "SL",
    "n_paths": 200
  }'
```

The response includes the estimated causal effect on run value, confidence interval, E-value (sensitivity to unmeasured confounding), and a trust gauge ("green" / "yellow" / "red") based on positivity.

### Natural-Mode Rollout (Python)

Roll out what the model expects a pitcher to do (no intervention — predictive, not causal):

```python
from causal.nuisance_v2 import load_nuisance_auto
from causal.g_computation import g_compute
from mcsim.state import build_synthetic_ab

# Load the model
nz = load_nuisance_auto("checkpoints/tiny-v1c1.pt")

# Build a synthetic at-bat context
ab = build_synthetic_ab(
    pitcher_id=543037, batter_id=665489,
    game_date="2024-08-01",
    pitcher_throws="R", batter_stand="L",
    ballpark_id=15, umpire_id=0, catcher_id=0,
)

# Roll out 500 paths in natural mode (model picks pitches)
result = g_compute(nz, ab,
    intervention_position=0, intervention_type=None,
    n_paths=500, rng_seed=42)

# Examine the outcome distribution
for name, prob in zip(result.ab_outcome_names, result.ab_outcome_distribution):
    print(f"  {name}: {prob:.3f}")
```

### Game Simulation

Simulate a full game from pre-computed matchup cards:

```bash
# Generate matchup cards for a date (requires GPU via Modal)
python -m scripts.mcsim.run_cards_modal --date 2026-06-15 --n-paths 300

# Run 10,000 Monte Carlo simulations per game
python -m scripts.mcsim.run_gamesim --all --n-sims 10000

# Analyze backtest results vs actuals
python -m scripts.mcsim.analyze_backtest
```

### Interactive Demo

```bash
# Start both the FastAPI backend and Vite frontend
make demo
# Opens at http://localhost:5173
```

The demo includes:
- **Rollout Viewer** — explore pitch-by-pitch model predictions for real at-bats
- **Pitcher Profile** — view a pitcher's arsenal, zone heatmaps, and tendencies
- **Score Predictions** — projected scores and win probabilities for today's games
- **Matchup Cards** — per-pitcher-batter outcome distributions with trust indicators

## Data

All training, evaluation, and inference data comes from real MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball). No synthetic, generated, or interpolated data is used anywhere outside of test fixtures.

**Temporal split** (strict, no leakage):
- Train: 2017-2023
- Validation: 2024 H1
- Test: 2024 H2 + 2025

## Quick Start

```bash
# Extract Statcast data
make extract

# Preprocess and build profiles
make preprocess

# Train PitchGPT (tiny: ~6M params, 4 layers, 256d)
make train MODEL=tiny

# Run 5-fold cross-fitting
make crossfit K=5

# Full evaluation (baselines + PitchGPT + calibration)
make eval

# Start the demo (FastAPI + Vite)
make demo
```

## GPU Training (Modal)

Large-scale training and matchup card generation run on [Modal](https://modal.com) GPUs:

```bash
# Train on A100
modal run modal_app.py::train_remote --size tiny --epochs 3

# Generate matchup cards on T4 (parallelized per game)
python -m scripts.mcsim.run_cards_modal --date 2026-06-15 --n-paths 300
```

## License

This project is for research and educational purposes.
