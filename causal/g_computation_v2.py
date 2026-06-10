"""Monte Carlo g-computation rollout for PitchGPTV2.

Adapts the V1 rollout (causal.g_computation) for the V2 model architecture.
The main structural difference: V2 has no discrete zone/velo/spin bins. Instead
it samples continuous (velo, spin, plate_x, plate_z) from a mixture-of-Gaussians
conditioned on the sampled pitch type. The cascade receives these continuous
values directly.

V2 also has no context tokens — the model uses adaLN conditioning from
pitcher/batter profiles, so there is no N_CONTEXT_TOKENS offset. Position 0
is the "start" token (type=PAD, continuous=zeros, result=0, pre-AB game state),
and position t predicts pitch t+1.

**Autoregressive convention:**
  - The model's type_logits at position t predict the NEXT pitch's type
    (left-shifted targets). So to sample pitch at step s, read
    type_logits[:, s, :] — the model has seen positions 0..s and predicts
    what comes next.
  - After sampling the type, call predict_continuous(hidden[:, s:s+1, :],
    sampled_type) to get the GMM for the continuous properties of that pitch.

**Sequence layout during rollout:**
  - Positions 0..k are the observed prefix (start token + observed pitches).
  - From position k+1 onward, each step samples type, then continuous, then
    the cascade determines the result. The sampled values are written back
    into the sequence for the next forward pass.

Reuses from causal.g_computation:
  - update_count, is_terminal — pure baseball count rules
  - _sample_from_probs — categorical sampling from probability tensors
  - RolloutResult — output dataclass
  - DEFAULT_AB_RUN_VALUE — run-value lookup table
  - Terminal-kind constants and AB-outcome constants
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from causal.nuisance_v2 import (
    NuisanceModelsV2,
    build_single_ab_batch_v2,
    denormalize_continuous,
    normalize_continuous,
)
from causal.g_computation import (
    # Count-state machine
    count_state_id,
    update_count,
    is_terminal,
    # Sampling
    _sample_from_probs,
    # Result dataclass + constants
    RolloutResult,
    DEFAULT_AB_RUN_VALUE,
    # Terminal-kind constants
    TERMINAL_KIND_NOT_YET,
    TERMINAL_KIND_K,
    TERMINAL_KIND_BB,
    TERMINAL_KIND_IN_PLAY,
    # Result-class constants (0-indexed)
    RESULT_BALL,
    RESULT_CALLED_STRIKE,
    RESULT_SWINGING_STRIKE,
    RESULT_FOUL,
    RESULT_IN_PLAY_OUT,
    RESULT_IN_PLAY_HIT,
    RESULT_IN_PLAY_HR,
    # AB-outcome constants
    AB_OUTCOME_NAMES,
    AB_OUTCOME_K,
    AB_OUTCOME_BB,
    AB_OUTCOME_1B,
    AB_OUTCOME_2B,
    AB_OUTCOME_3B,
    AB_OUTCOME_HR,
    AB_OUTCOME_OUT,
    # Index conventions
    TYPE_ID_OFFSET,
    RESULT_ID_OFFSET,
)
from data.dataset import (
    MODEL_PITCH_TYPES_START_IDX,
    MODEL_PITCH_TYPES_END_IDX,
    N_PITCH_TYPES,
    PITCH_TYPES,
    PITCH_TYPE_TO_ID,
)


# ============================================================
# Zone computation for cascade
# ============================================================


def plate_to_zone(plate_x: np.ndarray, plate_z: np.ndarray) -> np.ndarray:
    """Map continuous (plate_x, plate_z) to feature zone ids 0..12.

    Vectorized over N paths. The zone grid:
      - Zones 0..8: inside the strike zone (3x3 grid).
        Columns by plate_x: left (-0.83..-0.28), center (-0.28..0.28), right (0.28..0.83).
        Rows by plate_z: top (2.83..3.5), middle (2.17..2.83), bottom (1.5..2.17).
        Zone = row * 3 + col.
      - Zones 9..12: outside the strike zone, 4 quadrants:
        9 = top-left, 10 = top-right, 11 = bottom-left, 12 = bottom-right.

    Args:
        plate_x: (N,) float array — horizontal location in feet.
        plate_z: (N,) float array — vertical location in feet.

    Returns:
        (N,) int array of zone ids 0..12.
    """
    px = np.asarray(plate_x, dtype=np.float64)
    pz = np.asarray(plate_z, dtype=np.float64)
    N = len(px)

    in_zone_x = np.abs(px) <= 0.83
    in_zone_z = (pz >= 1.5) & (pz <= 3.5)
    in_zone = in_zone_x & in_zone_z

    # Inside the zone: 3x3 grid
    col = np.where(px < -0.28, 0, np.where(px < 0.28, 1, 2))
    row = np.where(pz < 2.17, 2, np.where(pz < 2.83, 1, 0))  # top row = 0
    in_zone_id = row * 3 + col

    # Outside the zone: 4 quadrants
    top = pz >= 2.5
    right = px >= 0.0
    out_zone_id = np.where(
        top & ~right, 9,
        np.where(
            top & right, 10,
            np.where(~top & ~right, 11, 12)
        )
    )

    return np.where(in_zone, in_zone_id, out_zone_id).astype(np.int64)


# ============================================================
# Intervention resolution
# ============================================================


def _resolve_intervention_type(
    intervention_type: int | str | None,
) -> tuple[Optional[int], Optional[str]]:
    """Returns (type_id_model_0indexed, type_name).

    None signals natural mode — the rollout samples from the model's
    propensity instead of clamping a specific type.
    """
    if intervention_type is None:
        return None, None
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


# ============================================================
# Main rollout
# ============================================================


def g_compute_v2(
    nuisance: NuisanceModelsV2,
    ab_pitches: pd.DataFrame,
    *,
    intervention_position: int = 0,
    intervention_type: int | str | None = None,
    n_paths: int = 300,
    max_steps: int = 12,
    rng_seed: int | None = None,
    run_value_table: np.ndarray = DEFAULT_AB_RUN_VALUE,
    hitter_step_fn=None,
    step_capture_fn=None,
    fractional_inplay: bool = False,
) -> RolloutResult:
    """Run the Monte Carlo g-computation rollout for PitchGPTV2.

    The rollout simulates at-bats by autoregressively sampling from the model.
    At each step:
      1. Forward pass to get type_logits and hidden states.
      2. Sample (or clamp) the pitch type from type_logits.
      3. Sample continuous (velo, spin, plate_x, plate_z) from the GMM.
      4. Pass to the cascade (hitter_step_fn) for the pitch result.
      5. Update the count, check for terminal conditions.
      6. Write sampled values back into the sequence for the next step.

    Args:
        nuisance: NuisanceModelsV2 wrapping a trained checkpoint.
        ab_pitches: one AB's pitch rows. At minimum the first row is used
            for the starting game state. The AB length determines the
            observed prefix.
        intervention_position: 0-indexed pitch position at which to intervene.
            0 = intervene on the first pitch (the AB starts fresh).
        intervention_type: pitch type to clamp at the intervention position.
            None = natural mode (sample from the model).
        n_paths: number of Monte Carlo paths.
        max_steps: maximum pitches per AB before truncation.
        rng_seed: for reproducible sampling.
        run_value_table: (7,) array mapping AB-outcome classes to run values.
        hitter_step_fn: cascade step function. Required — V2 has no result head.
            Signature: (type_ids, zone_ids, balls, strikes, prev_type_ids,
                        prev_zone_ids, n_prev, **kwargs) -> (result_probs, outcome5).
        step_capture_fn: optional diagnostics hook, called once per rollout step
            with a dict of named per-path arrays INCLUDING the active mask —
            capture code must mask to active paths (stats over terminated
            paths inflate rates; see the 2026-06-05 measurement-bug note).
        fractional_inplay: when True, a path that ends in-play contributes its
            EXACT cascade 5-way outcome split instead of one sampled outcome
            (Rao-Blackwellization of the terminal step only — mid-AB sampling
            is unchanged; this is NOT a Markov-chain conversion). Lowers the
            Monte-Carlo variance of outcome_dist and run_value at identical
            n_paths; per-path run_value becomes the path's EXPECTED run value.

    Returns:
        RolloutResult with per-path outcomes and aggregates.
    """
    if hitter_step_fn is None:
        raise ValueError(
            "hitter_step_fn is required for V2 rollout — V2 has no result head; "
            "the cascade (hitter model) must determine pitch outcomes."
        )
    if intervention_position < 0:
        raise ValueError(f"intervention_position must be >= 0; got {intervention_position}")
    if max_steps < intervention_position + 1:
        raise ValueError(
            f"max_steps {max_steps} must be > intervention_position {intervention_position}"
        )
    if run_value_table.shape != (7,):
        raise ValueError(
            f"run_value_table must be shape (7,) for {AB_OUTCOME_NAMES}; "
            f"got {run_value_table.shape}"
        )

    intervention_type_id, intervention_type_name = _resolve_intervention_type(
        intervention_type
    )
    rng = np.random.default_rng(rng_seed)

    N = n_paths
    k = intervention_position  # 0-indexed intervention pitch position

    # --- Build the initial batch from observed pitches -------------------------
    batch = build_single_ab_batch_v2(nuisance, ab_pitches, n_replicates=N)
    T_obs = batch["_n_observed"]       # number of real pitches in the observed AB
    seq_start = batch["_seq_len"]      # start token + observed pitches = T_obs + 1

    # The V2 sequence is: [start_token, pitch_1, pitch_2, ...].
    # The intervention position k refers to pitch k (0-indexed among real pitches),
    # which sits at sequence position k+1 (since position 0 is the start token).
    # The model at sequence position s predicts pitch s+1 (left-shifted targets),
    # BUT during training the target at position s is the NEXT pitch's type.
    # For autoregressive generation: to predict the pitch at sequence position p,
    # we run the model on positions 0..p-1 and read type_logits[:, p-1, :].
    #
    # In practice for the rollout, we build the full sequence up to the current
    # step and read the LAST position's output.

    # Allocate extended tensors for the full rollout (start + max_steps pitches).
    n_cont = int(nuisance.cfg.n_continuous)   # 4 (v1c) or 6 (v1c.1 +spin axis)
    max_seq = 1 + max_steps  # position 0 = start, positions 1..max_steps = pitches
    type_ids = torch.zeros(N, max_seq, dtype=torch.long)
    continuous = torch.zeros(N, max_seq, n_cont, dtype=torch.float32)
    result_ids = torch.zeros(N, max_seq, dtype=torch.long)
    count_state_t = torch.zeros(N, max_seq, dtype=torch.long)
    outs_t = torch.zeros(N, max_seq, dtype=torch.long)
    runners_t = torch.zeros(N, max_seq, dtype=torch.long)
    pitch_number_t = torch.zeros(N, max_seq, dtype=torch.long)
    padding_mask = torch.zeros(N, max_seq, dtype=torch.bool)

    # Copy the observed prefix (start token + pitches 0..k-1).
    # The intervention happens at pitch k, which is sequence position k+1.
    # So we copy the observed sequence up to and including position k (= k+1 positions).
    copy_len = min(k + 1, seq_start)  # copy start token + pitches 0..k-1
    type_ids[:, :copy_len] = batch["type_ids"][:, :copy_len]
    continuous[:, :copy_len] = batch["continuous"][:, :copy_len]
    result_ids[:, :copy_len] = batch["result_ids"][:, :copy_len]
    count_state_t[:, :copy_len] = batch["count_state"][:, :copy_len]
    outs_t[:, :copy_len] = batch["outs"][:, :copy_len]
    runners_t[:, :copy_len] = batch["runners"][:, :copy_len]
    pitch_number_t[:, :copy_len] = batch["pitch_number"][:, :copy_len]
    padding_mask[:, :copy_len] = True

    # --- Per-path state -------------------------------------------------------
    # Initial count at the intervention pitch = the count at the observed pitch k.
    # In the V2 convention, the count at pitch k is at sequence position k+1.
    # If k < T_obs, read from the observed data; otherwise start at 0-0.
    if k < T_obs:
        obs_count = int(batch["count_state"][0, k + 1].item())
    else:
        obs_count = 0  # 0-0 count if intervening beyond the observed AB
    balls = np.full(N, obs_count // 3, dtype=np.int64)
    strikes = np.full(N, obs_count % 3, dtype=np.int64)
    active = np.ones(N, dtype=bool)
    terminal_step = np.full(N, -1, dtype=np.int64)
    terminal_kind = np.full(N, TERMINAL_KIND_NOT_YET, dtype=np.int64)
    log_weights = []
    hitter_inplay = np.full(N, -1, dtype=np.int64)
    # Fractional terminal credit: per-path 5-way in-play split, recorded at
    # the step the path went in-play (used only when fractional_inplay).
    inplay_frac = np.zeros((N, 5), dtype=np.float64)

    # The observed AB's runners/outs stay constant within the AB (MVP simplification).
    obs_runners = int(batch["runners"][0, 0].item())
    obs_outs = int(batch["outs"][0, 0].item())

    # Captured at intervention step.
    intervention_velo_bin_mean: float = float("nan")
    intervention_type_propensity: np.ndarray = np.full(N_PITCH_TYPES, np.nan)

    # Track previous pitch type/zone for the cascade lag features.
    # Before the first rollout step, if k > 0, the previous pitch is observed.
    if k > 0 and k <= T_obs:
        prev_type = batch["type_ids"][:, k].numpy().copy()  # seq pos k = pitch k-1
        # Compute previous zone from observed continuous values. The batch
        # stores them z-score normalized (model input space) — map back to
        # raw feet before the zone grid.
        prev_cont = denormalize_continuous(
            batch["continuous"][:, k].numpy(), nuisance.cfg
        )  # (N, 4) raw
        prev_zone = plate_to_zone(prev_cont[:, 2], prev_cont[:, 3])
    else:
        prev_type = np.zeros(N, dtype=np.int64)
        prev_zone = np.full(N, -1, dtype=np.int64)

    # --- Main rollout loop ----------------------------------------------------
    # We iterate over pitch indices 0..max_steps-1. Each "step" produces pitch
    # at index `step`, which lives at sequence position `step + 1`.
    for step in range(k, max_steps):
        seq_pos = step + 1  # sequence position for this pitch

        # Fill game state at this sequence position.
        clipped_balls = np.clip(balls, 0, 3)
        clipped_strikes = np.clip(strikes, 0, 2)
        count_state_t[:, seq_pos] = torch.from_numpy(
            count_state_id(clipped_balls, clipped_strikes).astype(np.int64)
        )
        runners_t[:, seq_pos] = obs_runners
        outs_t[:, seq_pos] = obs_outs
        pitch_number_t[:, seq_pos] = min(step + 1, 14)  # 1-indexed pitch number, capped
        padding_mask[:, seq_pos] = torch.from_numpy(active)

        # Run forward on the prefix: positions 0..seq_pos.
        # The model's output at position seq_pos-1 predicts the pitch at seq_pos.
        # But we need to include position seq_pos in the input (with its game state
        # filled but type/continuous/result as zeros) so the model processes it.
        # Actually, the model's autoregressive convention: output at position t
        # predicts position t+1. So to predict the pitch at seq_pos, we feed
        # positions 0..seq_pos-1 and read the output at position seq_pos-1.
        #
        # However, for the V2 model specifically, the type_ids at the current
        # position are NOT yet known (we're about to sample them). The model
        # should only see positions 0..seq_pos-1 for the type prediction.
        fwd_len = seq_pos  # feed positions 0..seq_pos-1
        batch_step = {
            "pitcher_profile": batch["pitcher_profile"],
            "batter_profile": batch["batter_profile"],
            "type_ids": type_ids[:, :fwd_len],
            "continuous": continuous[:, :fwd_len],
            "result_ids": result_ids[:, :fwd_len],
            "count_state": count_state_t[:, :fwd_len],
            "outs": outs_t[:, :fwd_len],
            "runners": runners_t[:, :fwd_len],
            "pitch_number": pitch_number_t[:, :fwd_len],
            "padding_mask": padding_mask[:, :fwd_len],
        }
        out = nuisance.forward(batch_step)

        # Read type prediction at the LAST position of the prefix.
        # type_logits[:, fwd_len-1, :] predicts the next pitch (at seq_pos).
        # Temperature: count-conditional when calibrated — keyed by the count
        # the PREDICTED pitch is thrown at (count_state_t[:, seq_pos]).
        pred_pos = fwd_len - 1
        type_logits = nuisance.scale_type_logits(
            out["type_logits"][:, pred_pos, :],            # (N, 8) raw
            count_ids=count_state_t[:, seq_pos],
        )

        # Mask out PAD (index 0) and renormalize over the 7 real types.
        type_logits_masked = type_logits.clone()
        type_logits_masked[:, 0] = -1e9  # mask PAD
        type_probs = torch.softmax(type_logits_masked, dim=-1)[
            :, MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX
        ]  # (N, 7)
        type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Capture the natural type propensity at the intervention step.
        if step == k:
            if active.any():
                intervention_type_propensity = (
                    type_probs.numpy()[active].mean(axis=0).astype(np.float64)
                )
            else:
                intervention_type_propensity = np.full(N_PITCH_TYPES, np.nan)

        # Sample or clamp the pitch type.
        if step == k and intervention_type_id is not None:
            sampled_type = np.full(N, intervention_type_id, dtype=np.int64)
        else:
            sampled_type = _sample_from_probs(type_probs, rng, active)

        # Log-weight for ESS tracking (inverse propensity weight).
        sampled_type_prob = type_probs.numpy()[
            np.arange(N), sampled_type
        ].astype(np.float64)
        sampled_type_prob = np.clip(sampled_type_prob, 1e-8, 1.0)
        log_weights.append(-np.log(sampled_type_prob))

        # Convert to 1-indexed type_id for the model input.
        sampled_type_1idx = np.where(active, sampled_type + TYPE_ID_OFFSET, 0).astype(
            np.int64
        )
        type_ids[:, seq_pos] = torch.from_numpy(sampled_type_1idx)

        # --- Sample continuous from GMM conditioned on the sampled type -------
        hidden_at_pred = out["hidden"][:, pred_pos:pred_pos + 1, :]  # (N, 1, d)
        type_for_gmm = torch.from_numpy(sampled_type_1idx).unsqueeze(1)  # (N, 1)
        log_w, mu, log_std = nuisance.predict_continuous(hidden_at_pred, type_for_gmm)
        # Sample from the GMM.
        cont_sample = nuisance.model.gmm_head.sample(
            log_w.to(nuisance.device),
            mu.to(nuisance.device),
            log_std.to(nuisance.device),
        ).cpu()  # (N, 1, 4) — in z-score space (the GMM was trained on
                 # normalized targets)

        # Denormalize to raw units for clipping + the cascade.
        cont_raw = denormalize_continuous(cont_sample[:, 0, :].numpy(), nuisance.cfg)
        velo = cont_raw[:, 0].astype(np.float64)     # mph
        spin = cont_raw[:, 1].astype(np.float64)     # rpm
        plate_x = cont_raw[:, 2].astype(np.float64)  # feet
        plate_z = cont_raw[:, 3].astype(np.float64)  # feet

        # Clamp to reasonable physical bounds.
        velo = np.clip(velo, 60.0, 110.0)
        spin = np.clip(spin, 800.0, 3800.0)
        plate_x = np.clip(plate_x, -2.5, 2.5)
        plate_z = np.clip(plate_z, 0.0, 5.0)

        cols = [velo, spin, plate_x, plate_z]
        sax_sin = sax_cos = None
        if n_cont >= 6:
            # v1c.1: spin axis (sin, cos) — clamp each to [-1, 1]; the pair
            # is passed to the cascade which was trained on real sin/cos.
            sax_sin = np.clip(cont_raw[:, 4].astype(np.float64), -1.0, 1.0)
            sax_cos = np.clip(cont_raw[:, 5].astype(np.float64), -1.0, 1.0)
            cols += [sax_sin, sax_cos]

        # Write the clipped values back into the sequence in NORMALIZED space
        # — the model's next forward pass expects its training input scale.
        cont_clipped_raw = np.stack(cols, axis=1)
        continuous[:, seq_pos] = torch.from_numpy(
            normalize_continuous(cont_clipped_raw, nuisance.cfg)
        )

        # Capture mean velo at the intervention step.
        if step == k:
            if active.any():
                intervention_velo_bin_mean = float(velo[active].mean())
            else:
                intervention_velo_bin_mean = float("nan")

        # --- Compute zone from plate_x/plate_z for the cascade ---------------
        zone_ids = plate_to_zone(plate_x, plate_z)

        # --- Pass to cascade for the pitch result -----------------------------
        tids = sampled_type_1idx
        zids = zone_ids
        nprev = np.full(N, step, dtype=np.int64)

        step_kwargs = dict(plate_x=plate_x, plate_z=plate_z,
                           velo_native=velo, spin_native=spin)
        if sax_sin is not None:
            # The cascade trained on real spin_axis_sin/cos; v1c checkpoints
            # (4-dim) fed zeros here — v1c.1 supplies the sampled axis.
            step_kwargs["spin_axis_sin"] = sax_sin
            step_kwargs["spin_axis_cos"] = sax_cos
        rp_np, oc5 = hitter_step_fn(
            tids, zids, balls, strikes, prev_type, prev_zone, nprev,
            **step_kwargs,
        )
        result_probs_step = torch.from_numpy(rp_np.astype(np.float32))
        sampled_result = _sample_from_probs(result_probs_step, rng, active)

        # Record detailed in-play outcome for paths that just went in-play.
        inplay_now = active & np.isin(
            sampled_result,
            [RESULT_IN_PLAY_OUT, RESULT_IN_PLAY_HIT, RESULT_IN_PLAY_HR],
        )
        for i in np.where(inplay_now)[0]:
            p = oc5[i].astype(np.float64)
            ssum = p.sum()
            hitter_inplay[i] = int(rng.choice(5, p=p / ssum)) if ssum > 0 else 0
            inplay_frac[i] = (p / ssum) if ssum > 0 else np.eye(5)[0]

        # Write result into the sequence (1-indexed).
        result_ids[:, seq_pos] = torch.from_numpy(
            np.where(active, sampled_result + RESULT_ID_OFFSET, 0).astype(np.int64)
        )

        if step_capture_fn is not None:
            step_capture_fn({
                "step": step,
                "active": active.copy(),
                "type_1idx": sampled_type_1idx.copy(),
                "balls": clipped_balls.copy(),      # PRE-pitch count
                "strikes": clipped_strikes.copy(),
                "velo": velo.copy(), "spin": spin.copy(),
                "plate_x": plate_x.copy(), "plate_z": plate_z.copy(),
                "zone_ids": zone_ids.copy(),
                "result_probs": rp_np.copy(),
                "outcome5": oc5.copy(),
                "sampled_result": sampled_result.copy(),
            })

        # --- Update count + check termination --------------------------------
        balls, strikes = update_count(balls, strikes, sampled_result)
        term_flag, term_kind = is_terminal(balls, strikes, sampled_result)
        newly_terminal = active & term_flag
        terminal_step = np.where(
            newly_terminal & (terminal_step == -1), step, terminal_step
        )
        terminal_kind = np.where(
            newly_terminal & (terminal_kind == 0), term_kind, terminal_kind
        )
        active = active & ~term_flag

        # Update lag features for the next step.
        prev_type = sampled_type_1idx.copy()
        prev_zone = zone_ids.copy()

        if not active.any():
            break

    # --- AB-outcome assignment + run-value lookup -----------------------------
    ab_outcome = np.full(N, -1, dtype=np.int64)

    # K
    is_k = terminal_kind == TERMINAL_KIND_K
    ab_outcome = np.where(is_k, AB_OUTCOME_K, ab_outcome)
    # BB
    is_bb = terminal_kind == TERMINAL_KIND_BB
    ab_outcome = np.where(is_bb, AB_OUTCOME_BB, ab_outcome)

    # In-play: use the cascade's detailed outcome.
    is_in_play = terminal_kind == TERMINAL_KIND_IN_PLAY
    _OC5_TO_AB = np.array(
        [AB_OUTCOME_OUT,
         AB_OUTCOME_NAMES.index("1B"),
         AB_OUTCOME_NAMES.index("2B"),
         AB_OUTCOME_NAMES.index("3B"),
         AB_OUTCOME_NAMES.index("HR")],
        dtype=np.int64,
    )
    ip = np.where(is_in_play & (hitter_inplay >= 0))[0]
    ab_outcome[ip] = _OC5_TO_AB[hitter_inplay[ip]]

    valid = ab_outcome >= 0

    # Per-path 7-class outcome weights. Sampled mode: one-hot of the sampled
    # outcome. Fractional mode: K/BB stay one-hot (they are deterministic
    # given the path), in-play paths carry the cascade's exact 5-way split.
    outcome_w = np.zeros((N, 7), dtype=np.float64)
    if fractional_inplay:
        is_ip_valid = valid & (terminal_kind == TERMINAL_KIND_IN_PLAY)
        not_ip = valid & ~is_ip_valid
        outcome_w[not_ip, ab_outcome[not_ip]] = 1.0
        for j in range(5):
            outcome_w[is_ip_valid, _OC5_TO_AB[j]] += inplay_frac[is_ip_valid, j]
    else:
        outcome_w[valid, ab_outcome[valid]] = 1.0

    # Run value (NaN for truncated paths). Fractional mode: expected RV.
    # Elementwise multiply+sum instead of matmul: macOS Accelerate BLAS emits
    # spurious divide-by-zero warnings on small masked gemv calls.
    run_value = np.full(N, np.nan, dtype=np.float64)
    run_value[valid] = (outcome_w[valid] * run_value_table).sum(axis=1)

    # --- Aggregates -----------------------------------------------------------
    n_truncated = int((~valid).sum())
    if valid.sum() > 0:
        mean_rv = float(np.nanmean(run_value))
        se_rv = float(np.nanstd(run_value, ddof=1) / np.sqrt(valid.sum()))
        mean_len = float((terminal_step[valid] + 1).mean())
        outcome_dist = outcome_w[valid].mean(axis=0)
    else:
        mean_rv, se_rv, mean_len = float("nan"), float("nan"), float("nan")
        outcome_dist = np.full(7, np.nan)

    log_weights_arr = (
        np.stack(log_weights, axis=0) if log_weights else np.zeros((0, N))
    )

    return RolloutResult(
        n_paths=N,
        intervention_position=k,
        intervention_type=intervention_type_id,
        intervention_type_name=intervention_type_name,
        intervention_zone=None,  # V2 has no discrete zone intervention
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
        intervention_velo_bin_mean=intervention_velo_bin_mean,
        intervention_type_propensity=intervention_type_propensity,
    )
