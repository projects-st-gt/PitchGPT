# `hitter/` — dedicated hitter / swing-decision model

Separate, trackable home for the XGBoost hitter model (design:
`docs/Hitter_Swing_Model.md`). Predicts the batter's response to a pitch —
swing → whiff → contact-quality — to fix the ~7× hitter-OPS compression of the
single transformer. Composes with the pitch sequence model inside the causal
rollout (`causal/g_computation.py`) behind an `outcome_model` flag, so matchup
cards / counterfactual / recommender all benefit.

## Layout (building incrementally)
- `labels.py`  — per-pitch swing/whiff/contact + in-play outcome labels (done)
- `features.py` — feature builder (pitch + count + recent-pitch lags + batter
  profile + pitcher) — next
- `train.py`   — train the 3 XGBoost nodes (swing, whiff, contact-outcome)
- `model.py`   — load + `predict(pitch, batter, count, ctx) -> outcome dist`
- `eval.py`    — the compression diagnostic (real vs predicted OPS spread)

## Status
Phase 1 in progress on branch `hitter-swing-model`.
