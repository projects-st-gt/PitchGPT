"""Rigorous g-computation rollout for one (AB, intervention) pair.

The autoregressive transformer *is* a learned sequential propensity score, and
its outcome head *is* the conditional outcome model. Rolling out under an
intervention at pitch index k is g-computation — see the ``causal-layer``
skill for the spec.

This module implements the **rigorous** variant — count state evolves
per baseball rules after each sampled result, AB terminates naturally
(3 strikes / 4 balls / in-play / HBP), and the AB length is an *output* of
the simulation, not a fixed input. Three counterfactual deliverables per
query:

1. Effect on expected runs (the headline causal claim).
2. Effect on AB length (second-order, often interesting).
3. Effect on AB-outcome distribution ({K, BB, 1B, 2B, 3B, HR, out} shift).

The state machine is pure baseball rules — see :func:`update_count` and
:func:`is_terminal`. The model is queried per step for (π̂, μ̂); intervention
clamps ``type[k] = a*`` while letting zone/velo/spin sample naturally.

**MVP scope** (this implementation):
- Run-value mapping uses UNCONDITIONAL expected values per AB-outcome class
  (K: −0.15, BB: +0.30, 1B: +0.45, 2B: +0.75, 3B: +1.05, HR: +1.40, out: −0.10).
- The proper version uses ``data.run_value`` (RE24 + count_value, conditional
  on the AB's starting base/out state). Wire that up as a follow-up.
- ``runners``/``outs`` stay constant within an AB. Wild-pitch / passed-ball /
  pickoff cases are not simulated. Realistic for ≥98% of ABs.
- ``pitcher_fatigue`` stays at its observed bucket — fatigue changes by ~1
  pitch per step at the game scale, which doesn't shift the 10-pitch bucket.
- ``spin_axis`` after the intervention position uses the observed AB's
  spin axis at each step (a placeholder — the model's spin-axis head is
  continuous and a proper sampler isn't wired here yet). This loses a little
  realism but doesn't bias the type/result distribution materially.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from causal.nuisance import (
    NuisanceModels,
    PROPENSITY_HEADS,
    build_single_ab_batch,
)
from data.dataset import (
    MODEL_PITCH_TYPES_END_IDX,
    MODEL_PITCH_TYPES_START_IDX,
    MODEL_TYPE_ID,
    N_PITCH_TYPES,
    N_RESULTS,
    PITCH_TYPES,
    PITCH_TYPE_TO_ID,
    RESULT_CLASSES,
    RESULT_TO_ID,
)

# ============================================================
# Result-class constants (0-indexed, model-side)
# ============================================================
RESULT_BALL = RESULT_TO_ID["ball"]
RESULT_CALLED_STRIKE = RESULT_TO_ID["called_strike"]
RESULT_SWINGING_STRIKE = RESULT_TO_ID["swinging_strike"]
RESULT_FOUL = RESULT_TO_ID["foul"]
RESULT_IN_PLAY_OUT = RESULT_TO_ID["in_play_out"]
RESULT_IN_PLAY_HIT = RESULT_TO_ID["in_play_hit"]
RESULT_IN_PLAY_HR = RESULT_TO_ID["in_play_hr"]

# Dataset/model factor convention: type/result IDs in the input parquet are
# 1-indexed (PAD=0, PITCH_TYPES at 1..7), but the model's head outputs are
# 0-indexed over PITCH_TYPES (0..6) with PAD at index 7. The dataset shifts
# result targets by -1 (pitchgpt_dataset.py:300). So:
#   - Model output indices: 0..N_PITCH_TYPES-1 (=0..6) for type; 0..N_RESULTS-1 (=0..6) for result.
#   - Input parquet factor ids: 1..N_PITCH_TYPES (=1..7) for type_id, 1..N_RESULTS (=1..7) for result_id.
# We sample from model-output indices, then SHIFT BY +1 when writing back to
# the input pitch_factors tensor.
TYPE_ID_OFFSET = 1
RESULT_ID_OFFSET = 1

# AB-outcome class names (data.pitchgpt_dataset.AB_OUTCOME_*).
AB_OUTCOME_NAMES = ["K", "BB", "1B", "2B", "3B", "HR", "out"]
AB_OUTCOME_K = 0
AB_OUTCOME_BB = 1
AB_OUTCOME_1B = 2
AB_OUTCOME_2B = 3
AB_OUTCOME_3B = 4
AB_OUTCOME_HR = 5
AB_OUTCOME_OUT = 6

# Unconditional expected run values per AB outcome class (MVP — TODO: condition
# on the AB's starting (base, outs) state via data.run_value tables).
# Values are rough empirical means from MLB run-environment work, in the
# RE24-anchored frame where the AB's marginal effect is roughly:
#   HR ≈ +1.40, 3B ≈ +1.05, 2B ≈ +0.75, 1B ≈ +0.45, BB ≈ +0.30,
#   out ≈ −0.10, K ≈ −0.15.
DEFAULT_AB_RUN_VALUE = np.array([
    -0.15,  # K
    +0.30,  # BB (includes HBP via the AB-outcome head's BB class)
    +0.45,  # 1B
    +0.75,  # 2B
    +1.05,  # 3B
    +1.40,  # HR
    -0.10,  # out
], dtype=np.float64)


# ============================================================
# Count-state transitions (pure baseball rules — no model)
# ============================================================


def count_state_id(balls: np.ndarray | int, strikes: np.ndarray | int):
    """Encode (balls, strikes) → ``count_state`` factor id.

    Same encoding as ``data.preprocess_pitchgpt.compute_count_state``:
    ``count_state = balls * 3 + strikes``, range [0, 11].
    Works element-wise on numpy arrays.
    """
    return np.asarray(balls) * 3 + np.asarray(strikes)


def update_count(
    balls: np.ndarray,
    strikes: np.ndarray,
    result_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply baseball count-update rules per result. Vectorized over paths.

    Rules:
      - ball: balls += 1.
      - called_strike / swinging_strike: strikes += 1.
      - foul: strikes = min(strikes + 1, 2).  Foul with 2 strikes doesn't count.
      - in_play_*: count unchanged (the AB will be terminated by the caller).

    Args:
        balls: (N,) int array.
        strikes: (N,) int array.
        result_class: (N,) int array of 0-indexed result classes.

    Returns:
        (new_balls, new_strikes), each (N,) int array.
    """
    balls = np.asarray(balls)
    strikes = np.asarray(strikes)
    result_class = np.asarray(result_class)

    new_balls = balls.copy()
    new_strikes = strikes.copy()

    is_ball = result_class == RESULT_BALL
    is_strike = (result_class == RESULT_CALLED_STRIKE) | (result_class == RESULT_SWINGING_STRIKE)
    is_foul = result_class == RESULT_FOUL

    new_balls = np.where(is_ball, balls + 1, new_balls)
    new_strikes = np.where(is_strike, strikes + 1, new_strikes)
    # Foul with strikes < 2 → strikes + 1; foul with strikes >= 2 → unchanged.
    foul_increment = is_foul & (strikes < 2)
    new_strikes = np.where(foul_increment, new_strikes + 1, new_strikes)
    return new_balls, new_strikes


def is_terminal(
    balls: np.ndarray,
    strikes: np.ndarray,
    result_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized terminal-check. Returns (is_terminal, terminal_kind).

    ``terminal_kind`` is an int code:
      - 0 = not terminal
      - 1 = K (strikes ≥ 3)
      - 2 = BB (balls ≥ 4)
      - 3 = in_play (any in_play_* result class)
    """
    balls = np.asarray(balls)
    strikes = np.asarray(strikes)
    result_class = np.asarray(result_class)

    is_in_play = (
        (result_class == RESULT_IN_PLAY_OUT)
        | (result_class == RESULT_IN_PLAY_HIT)
        | (result_class == RESULT_IN_PLAY_HR)
    )
    is_k = strikes >= 3
    is_bb = balls >= 4

    terminal = is_in_play | is_k | is_bb
    kind = np.zeros_like(result_class)
    kind = np.where(is_k, 1, kind)
    kind = np.where(is_bb, 2, kind)
    kind = np.where(is_in_play, 3, kind)  # in-play wins if multiple flags fire
    return terminal, kind


TERMINAL_KIND_NOT_YET = 0
TERMINAL_KIND_K = 1
TERMINAL_KIND_BB = 2
TERMINAL_KIND_IN_PLAY = 3


# ============================================================
# Result dataclass
# ============================================================


@dataclass
class RolloutResult:
    """One (AB, intervention) rollout's outputs."""

    n_paths: int
    intervention_position: int
    intervention_type: int
    intervention_type_name: str
    intervention_zone: Optional[int]   # 0..12 feature-zone (v5 SIS internal), or None for type-only
    max_steps: int

    # Per-path outputs, shape (N,)
    terminal_step: np.ndarray            # 0-indexed pitch position where AB ended
    terminal_kind: np.ndarray            # 0 not-yet / 1 K / 2 BB / 3 in_play
    ab_outcome: np.ndarray               # 0..6 = {K, BB, 1B, 2B, 3B, HR, out}; −1 if truncated
    run_value: np.ndarray                # expected run value per path
    log_weights_per_step: np.ndarray     # (max_steps + 1 - k, N) for ESS tracking

    # Aggregates
    mean_run_value: float
    se_run_value: float
    mean_ab_length: float                # mean # of pitches the AB ended in
    ab_outcome_distribution: np.ndarray  # P(AB-outcome), shape (7,)
    n_truncated: int                     # paths that didn't terminate within max_steps

    @property
    def ab_length_distribution(self) -> dict[int, int]:
        """Histogram of terminal_step (1-indexed AB lengths) across paths."""
        from collections import Counter
        lens = (self.terminal_step + 1).astype(int)
        return dict(sorted(Counter(lens.tolist()).items()))


# ============================================================
# Main rollout
# ============================================================


def _resolve_intervention_type(intervention_type: int | str) -> tuple[int, str]:
    """Returns (type_id_model_0indexed, type_name)."""
    if isinstance(intervention_type, str):
        if intervention_type not in PITCH_TYPE_TO_ID:
            raise ValueError(
                f"unknown pitch type {intervention_type!r}; must be one of {PITCH_TYPES}"
            )
        return PITCH_TYPE_TO_ID[intervention_type], intervention_type
    type_id = int(intervention_type)
    if not 0 <= type_id < N_PITCH_TYPES:
        raise ValueError(
            f"intervention_type id must be in [0, {N_PITCH_TYPES}); got {type_id}"
        )
    return type_id, PITCH_TYPES[type_id]


def _sample_from_probs(
    probs_2d: torch.Tensor,
    rng: np.random.Generator,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sample one class from each row of a (B, K) probability tensor.

    Uses cumulative-distribution sampling for speed; respects ``active_mask``
    by returning 0 for inactive rows (the value is ignored downstream).
    """
    p = probs_2d.numpy().astype(np.float64)
    # Renormalize defensively (temperature scaling preserves sum, but float drift).
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    p = np.where(s > 0, p / np.maximum(s, 1e-12), p)
    cdf = p.cumsum(axis=1)
    u = rng.uniform(size=(p.shape[0],))[:, None]
    samples = (u < cdf).argmax(axis=1).astype(np.int64)
    if active_mask is not None:
        samples = np.where(active_mask, samples, 0)
    return samples


def g_compute(
    nuisance: NuisanceModels,
    ab_pitches: pd.DataFrame,
    *,
    intervention_position: int,
    intervention_type: int | str,
    intervention_zone: Optional[int] = None,
    n_paths: int = 1000,
    max_steps: int = 12,
    rng_seed: Optional[int] = None,
    run_value_table: np.ndarray = DEFAULT_AB_RUN_VALUE,
) -> RolloutResult:
    """Run the rigorous Monte Carlo g-computation rollout.

    See the module docstring for the simulation rules. Outputs a
    :class:`RolloutResult` with per-path terminal steps + outcomes + run values
    and the aggregates a demo (or AIPW) would consume.
    """
    if intervention_position < 1:
        raise ValueError(
            f"intervention_position must be ≥ 1 (the model doesn't autoregressively "
            f"predict pitch[0] from no history). Got {intervention_position}."
        )
    if intervention_position >= len(ab_pitches):
        raise ValueError(
            f"intervention_position {intervention_position} ≥ AB length "
            f"{len(ab_pitches)}; pick an earlier position."
        )
    if max_steps < intervention_position + 1:
        raise ValueError(
            f"max_steps {max_steps} must be > intervention_position {intervention_position}"
        )
    if run_value_table.shape != (7,):
        raise ValueError(
            f"run_value_table must be shape (7,) for {AB_OUTCOME_NAMES}; got {run_value_table.shape}"
        )

    intervention_type_id, intervention_type_name = _resolve_intervention_type(intervention_type)
    rng = np.random.default_rng(rng_seed)

    # --- Build batch + extract observed initial sequence -----------------------
    # build_single_ab_batch sets up the FULL observed AB replicated N times.
    # We use it to populate the constant per-AB context (pitcher/batter profiles,
    # arsenal, categorical context). The pitch_factors tensors will be REWRITTEN
    # below — only positions 0..intervention_position-1 stay as observed; from
    # ``intervention_position`` onward, we sample / clamp.
    batch = build_single_ab_batch(nuisance, ab_pitches, n_replicates=n_paths)
    N = n_paths
    T_obs = batch["pitch_factors"]["type"].shape[1]  # observed AB length
    assert T_obs == len(ab_pitches)

    # Allocate a fresh pitch_factors with width=max_steps. Copy positions 0..k-1.
    k = intervention_position
    factors_long_keys = ["type", "zone", "velo", "spin_rate", "result", "count",
                          "runners", "outs", "pos", "pitcher_fatigue"]
    full = {}
    for key in factors_long_keys:
        v = batch["pitch_factors"][key]  # (N, T_obs)
        wide = torch.zeros(N, max_steps, dtype=v.dtype)
        wide[:, :k] = v[:, :k]
        full[key] = wide
    # spin_axis is (N, T_obs, 2) float.
    sa = batch["pitch_factors"]["spin_axis"]  # (N, T_obs, 2)
    wide_sa = torch.zeros(N, max_steps, 2, dtype=sa.dtype)
    wide_sa[:, :k] = sa[:, :k]
    full["spin_axis"] = wide_sa
    intended = {
        "type": full["type"].clone(),
        "zone": full["zone"].clone(),
        "velo": full["velo"].clone(),
        "spin_axis": full["spin_axis"].clone(),
    }
    # Padding mask: we'll grow this each step.
    pad = torch.zeros(N, max_steps, dtype=torch.bool)
    pad[:, :k] = True  # positions 0..k-1 are "real"

    # --- Per-path state (mutated across steps) ---------------------------------
    # Initial count at position k = the observed AB's count at position k.
    # We read it from the OBSERVED row's balls/strikes (stored in count_state).
    obs_count_state = int(batch["pitch_factors"]["count"][0, k].item())
    balls = np.full(N, obs_count_state // 3, dtype=np.int64)
    strikes = np.full(N, obs_count_state % 3, dtype=np.int64)
    pos_step = np.full(N, k, dtype=np.int64)
    active = np.ones(N, dtype=bool)
    terminal_step = np.full(N, -1, dtype=np.int64)
    terminal_kind = np.full(N, TERMINAL_KIND_NOT_YET, dtype=np.int64)
    log_weights = []  # one (N,) per step

    # The observed AB's runners/outs/pitcher_fatigue/pos are AB-level for our
    # MVP: runners/outs stay constant, pitcher_fatigue stays at obs bucket, pos
    # increments by 1 per step.
    obs_runners = int(batch["pitch_factors"]["runners"][0, k].item())
    obs_outs = int(batch["pitch_factors"]["outs"][0, k].item())
    obs_fatigue = int(batch["pitch_factors"]["pitcher_fatigue"][0, k].item())
    # spin_axis placeholder for k..max_steps-1: copy the observed AB's last spin_axis
    # (rolls forward if intervention extends beyond observed length).
    if T_obs > 0:
        spin_axis_fill = sa[:, min(k, T_obs - 1), :]  # (N, 2)
    else:
        spin_axis_fill = torch.zeros(N, 2, dtype=sa.dtype)

    # --- Main loop --------------------------------------------------------------
    for step in range(k, max_steps):
        # Set up factors at position `step`:
        #   - pos = step (clipped to n_positions vocab implicitly by the model)
        #   - count = count_state_id(balls, strikes)
        #   - runners = obs_runners, outs = obs_outs, fatigue = obs_fatigue
        full["pos"][:, step] = step
        # Clip count state to legal embedding-vocab range [0, 11]. A terminated
        # path could carry balls=4 or strikes=3 from the prior step's result;
        # those values are correct for the state-machine logic but out of vocab
        # for the embedding layer. Inactive paths are masked from sampling, but
        # the embedding LOOKUP still happens at this position, so the index
        # must be legal. Clip to the (3, 2) corner for terminal-but-not-yet-
        # cleared paths — the value doesn't affect their outputs (masked).
        clipped_balls = np.clip(balls, 0, 3)
        clipped_strikes = np.clip(strikes, 0, 2)
        full["count"][:, step] = torch.from_numpy(
            count_state_id(clipped_balls, clipped_strikes).astype(np.int64)
        )
        full["runners"][:, step] = obs_runners
        full["outs"][:, step] = obs_outs
        full["pitcher_fatigue"][:, step] = obs_fatigue
        full["spin_axis"][:, step, :] = spin_axis_fill
        # Type/zone/velo/spin_rate/result will be filled below from sampling.
        # pad: only the active rows have a real pitch at this step.
        pad[:, step] = torch.from_numpy(active)

        # Run forward on the (B=N, T=step+1) prefix.
        batch_step = {
            "pitcher_profile": batch["pitcher_profile"],
            "batter_profile": batch["batter_profile"],
            "arsenal": batch.get("arsenal"),
            "categorical_context": batch["categorical_context"],
            "pitch_factors": {kk: vv[:, : step + 1] for kk, vv in full.items() if kk != "spin_axis"},
            "intended_actions": {  # will refresh after sampling
                "type": intended["type"][:, : step + 1],
                "zone": intended["zone"][:, : step + 1],
                "velo": intended["velo"][:, : step + 1],
                "spin_axis": intended["spin_axis"][:, : step + 1, :],
            },
            "padding_mask": pad[:, : step + 1],
        }
        batch_step["pitch_factors"]["spin_axis"] = full["spin_axis"][:, : step + 1, :]
        out = nuisance.forward(batch_step)

        # π̂ at the position that PREDICTS pitch[step]: that's sequence index
        # NC + (step - 1) [propensity at position step-1 predicts pitch step].
        # The model's propensity[NC + step - 1] = "what's pitch[step]?". For
        # step == k, this is "what's the intervention pitch?".
        # NOTE: the model's heads at sequence position NC + step compute the
        # NEXT pitch's distribution given history through pitch[step]. Inputs
        # at NC + step are the position-step pitch tokens; we read its outputs
        # AFTER appending. For sampling step's pitch, we use propensity at
        # NC + (step - 1) — i.e., the model's prediction given pitches 0..step-1.
        seq_idx_for_predicting_step = nuisance.model.N_CONTEXT_TOKENS + (step - 1)
        # The TYPE head outputs over the 8-class vocab. Slice via the named
        # constants — see ``data.dataset.MODEL_PITCH_TYPES_*`` and the
        # "Bug-prevention discipline" in CLAUDE.md. Sampled value ∈ [0, 7) maps
        # to PITCH_TYPES; write back to the dataset's type_id by adding
        # TYPE_ID_OFFSET (=MODEL_PITCH_TYPES_START_IDX=1).
        type_probs = out.propensity_probs["type"][
            :, seq_idx_for_predicting_step,
            MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX,
        ]
        zone_probs = out.propensity_probs["zone"][:, seq_idx_for_predicting_step, :]
        velo_probs = out.propensity_probs["velo"][:, seq_idx_for_predicting_step, :]
        spin_rate_probs = out.propensity_probs["spin_rate"][:, seq_idx_for_predicting_step, :]
        type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Sample type (with intervention clamp at step == k).
        if step == k:
            sampled_type = np.full(N, intervention_type_id, dtype=np.int64)
        else:
            sampled_type = _sample_from_probs(type_probs, rng, active)
        # Track log-weight 1 / π̂(sampled type) for ESS — but ONLY for the
        # SAMPLED steps (intervention step has weight 1 trivially because we
        # clamped, but in the ESS calculation we still want the propensity of
        # the intervention for the gate decision and multi-step diagnostic).
        sampled_type_prob = type_probs.numpy()[np.arange(N), sampled_type].astype(np.float64)
        sampled_type_prob = np.clip(sampled_type_prob, 1e-8, 1.0)
        log_weights.append(-np.log(sampled_type_prob))  # inverse-propensity weight in log space

        # Sample zone (with intervention clamp at step == k if intervention_zone is set).
        # When both type and zone are intervened, the rollout's intervention is a
        # joint (type, zone) action. The model's heads sample type and zone
        # conditionally independent given the hidden state (MVP — see task #25
        # for the autoregressive fix). The result head still conditions on the
        # joint (type, zone, velo, spin) we feed it via intended_actions.
        if step == k and intervention_zone is not None:
            sampled_zone = np.full(N, int(intervention_zone), dtype=np.int64)
        else:
            sampled_zone = _sample_from_probs(zone_probs, rng, active)
        sampled_velo = _sample_from_probs(velo_probs, rng, active)
        sampled_spin_rate = _sample_from_probs(spin_rate_probs, rng, active)

        # Write the sampled factors into the input tensors. Use 1-indexed
        # convention for the type factor (the dataset's type_id is 1..7 with
        # PAD=0). The other factors are already in their own vocab spaces
        # consistent with the dataset.
        full["type"][:, step] = torch.from_numpy(
            np.where(active, sampled_type + TYPE_ID_OFFSET, 0).astype(np.int64)
        )
        full["zone"][:, step] = torch.from_numpy(sampled_zone.astype(np.int64))
        full["velo"][:, step] = torch.from_numpy(sampled_velo.astype(np.int64))
        full["spin_rate"][:, step] = torch.from_numpy(sampled_spin_rate.astype(np.int64))
        # Mirror sampled action into intended_actions so the result head reads them.
        intended["type"][:, step] = full["type"][:, step]
        intended["zone"][:, step] = full["zone"][:, step]
        intended["velo"][:, step] = full["velo"][:, step]
        # spin_axis: use the placeholder; intended spin_axis matches.
        intended["spin_axis"][:, step, :] = full["spin_axis"][:, step, :]

        # --- Forward AGAIN to read result probs at the just-filled step --------
        # The previous forward used unfilled type/zone/velo at step → result
        # head's read was on placeholders. Re-run with the sampled action.
        batch_step["pitch_factors"]["type"] = full["type"][:, : step + 1]
        batch_step["pitch_factors"]["zone"] = full["zone"][:, : step + 1]
        batch_step["pitch_factors"]["velo"] = full["velo"][:, : step + 1]
        batch_step["pitch_factors"]["spin_rate"] = full["spin_rate"][:, : step + 1]
        batch_step["pitch_factors"]["spin_axis"] = full["spin_axis"][:, : step + 1, :]
        batch_step["intended_actions"]["type"] = intended["type"][:, : step + 1]
        batch_step["intended_actions"]["zone"] = intended["zone"][:, : step + 1]
        batch_step["intended_actions"]["velo"] = intended["velo"][:, : step + 1]
        batch_step["intended_actions"]["spin_axis"] = intended["spin_axis"][:, : step + 1, :]
        out = nuisance.forward(batch_step)
        result_probs_step = out.result_probs[:, step, :]  # (N, 7)
        sampled_result = _sample_from_probs(result_probs_step, rng, active)
        full["result"][:, step] = torch.from_numpy(
            np.where(active, sampled_result + RESULT_ID_OFFSET, 0).astype(np.int64)
        )

        # --- Update count + check termination ----------------------------------
        balls, strikes = update_count(balls, strikes, sampled_result)
        term_flag, term_kind = is_terminal(balls, strikes, sampled_result)
        newly_terminal = active & term_flag
        terminal_step = np.where(newly_terminal & (terminal_step == -1), step, terminal_step)
        terminal_kind = np.where(newly_terminal & (terminal_kind == 0), term_kind, terminal_kind)
        active = active & ~term_flag

        if not active.any():
            break

    # --- AB-outcome assignment + run-value lookup ------------------------------
    ab_outcome = np.full(N, -1, dtype=np.int64)

    # K: terminal_kind == 1 → AB_OUTCOME_K (0)
    is_k = terminal_kind == TERMINAL_KIND_K
    ab_outcome = np.where(is_k, AB_OUTCOME_K, ab_outcome)
    # BB: terminal_kind == 2 → AB_OUTCOME_BB (1)
    is_bb = terminal_kind == TERMINAL_KIND_BB
    ab_outcome = np.where(is_bb, AB_OUTCOME_BB, ab_outcome)

    # In-play: sample from the AB-outcome head at terminal position, conditional
    # on the {1B, 2B, 3B, HR, out} sub-distribution. Read ab_outcome_per_pos at
    # each terminal step. We do ONE final forward to get the up-to-date probs.
    is_in_play = terminal_kind == TERMINAL_KIND_IN_PLAY
    if is_in_play.any():
        # Forward on the final populated sequence to get ab_outcome at terminal step.
        # max_terminal_step bounds T; pad mask up to that.
        max_term = int(terminal_step[is_in_play].max() + 1)  # inclusive end
        final_pad = torch.zeros(N, max_term, dtype=torch.bool)
        for i in range(N):
            ts = int(terminal_step[i])
            if ts >= 0:
                final_pad[i, : min(ts + 1, max_term)] = True
            else:
                # not-yet-terminal: keep its full max_steps (or up to current step)
                final_pad[i, :max_term] = True

        batch_final = {
            "pitcher_profile": batch["pitcher_profile"],
            "batter_profile": batch["batter_profile"],
            "arsenal": batch.get("arsenal"),
            "categorical_context": batch["categorical_context"],
            "pitch_factors": {kk: vv[:, :max_term] for kk, vv in full.items() if kk != "spin_axis"},
            "intended_actions": {
                "type": intended["type"][:, :max_term],
                "zone": intended["zone"][:, :max_term],
                "velo": intended["velo"][:, :max_term],
                "spin_axis": intended["spin_axis"][:, :max_term, :],
            },
            "padding_mask": final_pad,
        }
        batch_final["pitch_factors"]["spin_axis"] = full["spin_axis"][:, :max_term, :]
        out_final = nuisance.forward(batch_final)
        # ab_outcome_probs[b, t, :] = prob over {K, BB, 1B, 2B, 3B, HR, out} at pitch t.
        ab_probs_full = out_final.ab_outcome_probs  # (N, max_term, 7)

        # For in_play paths, pick the probability slice at their terminal step,
        # zero out K/BB classes, renormalize, sample.
        in_play_idx = np.where(is_in_play)[0]
        for i in in_play_idx:
            ts = int(terminal_step[i])
            p = ab_probs_full[i, ts].numpy().astype(np.float64)
            # Zero out K, BB (terminal_kind told us it's in_play).
            p[AB_OUTCOME_K] = 0.0
            p[AB_OUTCOME_BB] = 0.0
            s = p.sum()
            if s <= 0:
                # Pathological — fall back to "out".
                ab_outcome[i] = AB_OUTCOME_OUT
                continue
            p = p / s
            ab_outcome[i] = int(rng.choice(7, p=p))

    # Run value lookup (NaN for truncated paths).
    run_value = np.full(N, np.nan, dtype=np.float64)
    valid = ab_outcome >= 0
    run_value[valid] = run_value_table[ab_outcome[valid]]

    # --- Aggregates ------------------------------------------------------------
    n_truncated = int((~valid).sum())
    if valid.sum() > 0:
        mean_rv = float(np.nanmean(run_value))
        se_rv = float(np.nanstd(run_value, ddof=1) / np.sqrt(valid.sum()))
        mean_len = float((terminal_step[valid] + 1).mean())
        outcome_dist = np.zeros(7, dtype=np.float64)
        for c in range(7):
            outcome_dist[c] = float((ab_outcome[valid] == c).mean())
    else:
        mean_rv, se_rv, mean_len = float("nan"), float("nan"), float("nan")
        outcome_dist = np.full(7, np.nan)

    log_weights_arr = np.stack(log_weights, axis=0) if log_weights else np.zeros((0, N))

    return RolloutResult(
        n_paths=N,
        intervention_position=k,
        intervention_type=intervention_type_id,
        intervention_type_name=intervention_type_name,
        intervention_zone=int(intervention_zone) if intervention_zone is not None else None,
        max_steps=max_steps,
        terminal_step=terminal_step,
        terminal_kind=terminal_kind,
        ab_outcome=ab_outcome,
        run_value=run_value,
        log_weights_per_step=log_weights_arr,
        mean_run_value=mean_rv,
        se_run_value=se_rv,
        mean_ab_length=mean_len,
        ab_outcome_distribution=outcome_dist,
        n_truncated=n_truncated,
    )
