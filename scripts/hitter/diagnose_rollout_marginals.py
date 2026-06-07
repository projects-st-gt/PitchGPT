"""Per-count pitch-type distribution diagnostic: real data vs teacher-forced model.

Compares PitchGPT's teacher-forced per-count type marginals against real
Statcast data to diagnose whether the rollout walk-deficit (6.4% predicted vs
9.3% real) stems from wrong per-count type distributions or from deeper
sequential-conditioning drift.

USAGE
-----
    python -m scripts.hitter.diagnose_rollout_marginals \\
        --ckpt checkpoints_modal/small-fold0-v8/checkpoint_calibrated.pt \\
        --n-pas 200

The script loads:
  1. Real per-count type marginals from augmented training data (≤2023).
  2. Model teacher-forced type marginals from sampled held-out validation PAs
     (augmented 2024-01-01..2024-07-15).

Convention notes (read CLAUDE.md "Bug-prevention discipline"):
  - type_id parquet: 1..7 (FF=1 ... FS=7), PAD=0.
  - Model type head: 8 logits, PAD at index 0, FF at index 1 ... FS at index 7.
  - Slicing model type head: propensity_probs["type"][..., 1:8] for real types.
  - count_state = balls * 3 + strikes, range [0, 11].
  - Autoregressive convention: propensity_probs["type"][b, NC + t, :] predicts
    pitch t+1 given pitches 0..t. For pitch t itself (t >= 1), prediction is at
    sequence position NC + t - 1. For the first pitch (t=0), prediction is at
    NC - 1 (the last context-token position).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# -- vocabulary / convention constants (Bug-prevention: named, not magic) --
from data.dataset import (
    PITCH_TYPES,  # ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]  7 types
    MODEL_PITCH_TYPES_START_IDX,  # 1  (PAD is at 0)
    MODEL_PITCH_TYPES_END_IDX,    # 8  (exclusive)
    MODEL_TYPE_ID,                # {"FF": 1, ..., "FS": 7}
    N_PITCH_TYPES,                # 7
)
from causal.nuisance import NuisanceModels, build_single_ab_batch
from model.pitchgpt import PitchGPT  # for N_CONTEXT_TOKENS

N_CONTEXT_TOKENS = PitchGPT.N_CONTEXT_TOKENS  # 3

# 12 count states: (balls=0..3, strikes=0..2)
_COUNT_LABELS: list[tuple[int, int]] = [
    (b, s) for b in range(4) for s in range(3)
]  # order: (0,0),(0,1),(0,2),(1,0),...,(3,2)

# Mapping: count_state integer -> (balls, strikes) label
_CS_TO_LABEL: dict[int, tuple[int, int]] = {b * 3 + s: (b, s) for b, s in _COUNT_LABELS}

# Augmented data directories
_AUG_BASE = Path("data/augmented")
_TRAIN_END_YEAR = 2023
_VAL_START = "2024-01-01"
_VAL_END = "2024-07-15"


# ============================================================
# 1. Real per-count type marginals from training data
# ============================================================


def _augmented_files(year_start: int, year_end: int,
                     date_start: str | None = None,
                     date_end: str | None = None) -> list[str]:
    """Sorted augmented parquet paths for the given year range.

    If date_start/date_end are provided they filter by the filename date
    (YYYY-MM-DD.parquet).
    """
    files: list[str] = []
    for yr in range(year_start, year_end + 1):
        files += sorted(glob.glob(str(_AUG_BASE / str(yr) / "*.parquet")))
    if date_start is not None:
        files = [f for f in files if os.path.basename(f)[:10] >= date_start]
    if date_end is not None:
        files = [f for f in files if os.path.basename(f)[:10] <= date_end]
    return files


def compute_real_marginals() -> dict[int, np.ndarray]:
    """Real per-count type marginals from augmented training data (≤2023).

    Returns:
        dict mapping count_state (0..11) -> float32 array of shape (7,)
        giving the probability of each pitch type (FF, SI, FC, SL, CU, CH, FS).
        Missing count states get uniform distribution with a warning.
    """
    files = _augmented_files(2017, _TRAIN_END_YEAR)
    if not files:
        raise FileNotFoundError(
            f"No augmented training data found under {_AUG_BASE}/2017..{_TRAIN_END_YEAR}"
        )

    # Accumulate counts: shape (12, 7) — [count_state, pitch_type_idx]
    counts = np.zeros((12, N_PITCH_TYPES), dtype=np.float64)
    n_files = len(files)
    print(f"[real] loading {n_files} training-year files...", flush=True)

    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["count_state", "pitch_type_canonical"])
        # drop rows with unrecognized/null pitch types
        valid_mask = df["pitch_type_canonical"].isin(PITCH_TYPES) & df["count_state"].between(0, 11)
        df = df[valid_mask]
        if df.empty:
            continue
        for cs, grp in df.groupby("count_state"):
            cs = int(cs)
            vc = grp["pitch_type_canonical"].value_counts()
            for j, pt in enumerate(PITCH_TYPES):
                counts[cs, j] += int(vc.get(pt, 0))
        if (i + 1) % 100 == 0:
            print(f"[real]   {i+1}/{n_files} files done", flush=True)

    print(f"[real] {n_files} files processed; total pitches: {int(counts.sum()):,}", flush=True)

    # Normalize to probabilities
    marginals: dict[int, np.ndarray] = {}
    for cs in range(12):
        row_sum = counts[cs].sum()
        if row_sum == 0:
            b, s = _CS_TO_LABEL[cs]
            print(f"[real] WARNING: no pitches for count ({b},{s}); using uniform",
                  flush=True)
            marginals[cs] = np.ones(N_PITCH_TYPES, dtype=np.float32) / N_PITCH_TYPES
        else:
            marginals[cs] = (counts[cs] / row_sum).astype(np.float32)

    return marginals


# ============================================================
# 2. Model teacher-forced type marginals from validation PAs
# ============================================================


def _load_val_ab_groups(n_abs: int, seed: int) -> list[pd.DataFrame]:
    """Sample ``n_abs`` at-bats from the validation split augmented data.

    Returns a list of DataFrames, each being one AB's rows in pitch order.
    Uses val split (2024-01-01..2024-07-15) per CLAUDE.md temporal split rules.
    """
    files = _augmented_files(2024, 2024, date_start=_VAL_START, date_end=_VAL_END)
    if not files:
        raise FileNotFoundError(
            f"No validation augmented data found for {_VAL_START}..{_VAL_END}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(files)  # shuffle file order so we get diversity without scanning all

    # Collect AB groups until we have at least n_abs or run out of files
    ab_groups: list[pd.DataFrame] = []
    seen: set[tuple] = set()
    file_budget = min(len(files), max(10, n_abs // 20 + 5))  # rough cap on I/O

    for f in files[:file_budget]:
        df = pd.read_parquet(f)
        # Keep only pitches with recognized type and valid count_state
        df = df[
            df["pitch_type_canonical"].isin(PITCH_TYPES)
            & df["count_state"].between(0, 11)
        ]
        if df.empty:
            continue
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        for (gk, ab), grp in df.groupby(["game_pk", "at_bat_number"], sort=False):
            key = (int(gk), int(ab))
            if key in seen:
                continue
            seen.add(key)
            ab_groups.append(grp.reset_index(drop=True))
            if len(ab_groups) >= n_abs * 3:  # over-sample; we'll subsample below
                break
        if len(ab_groups) >= n_abs * 3:
            break

    if not ab_groups:
        raise FileNotFoundError(
            "No valid ABs found in validation split. "
            "Check that data/augmented/2024/ exists with pitch_type_canonical column."
        )

    # Subsample deterministically
    idxs = rng.choice(len(ab_groups), size=min(n_abs, len(ab_groups)), replace=False)
    selected = [ab_groups[i] for i in idxs]
    print(f"[model] selected {len(selected)} ABs from {len(ab_groups)} candidates",
          flush=True)
    return selected


def compute_model_marginals(
    nuisance: NuisanceModels,
    ab_dfs: list[pd.DataFrame],
) -> dict[int, np.ndarray]:
    """Teacher-forced per-count type marginals from the model.

    For each AB, runs one forward pass (teacher-forced — the real pitch history
    is fed as context, not sampled). For each pitch t in the AB, reads the model's
    type probability distribution at the PREDICTION position for pitch t, which
    is at sequence position NC + t - 1 (for t >= 1) or NC - 1 (for t = 0).

    Convention check (Bug-prevention discipline):
      propensity_probs["type"] shape is (1, NC+T, 8).
      Real types live at indices 1:8 (MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX).
      PAD is at index 0 — excluded.
    """
    # Accumulate weighted type probs per count state: (12, 7)
    acc_probs = np.zeros((12, N_PITCH_TYPES), dtype=np.float64)
    acc_n = np.zeros(12, dtype=np.float64)

    n_total = len(ab_dfs)
    n_errors = 0

    for i, ab_df in enumerate(ab_dfs):
        try:
            batch = build_single_ab_batch(nuisance, ab_df, n_replicates=1)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"[model] AB {i} build_single_ab_batch error: {e}", flush=True)
            continue

        with torch.no_grad():
            fwd = nuisance.forward(batch)

        # type probs: (1, NC+T, 8) — take batch dim 0
        type_probs = fwd.propensity_probs["type"][0]  # (NC+T, 8)

        # CONVENTION VERIFICATION (Bug-prevention discipline step 3):
        # Real pitch type probs are at indices 1:8. The PAD mass at index 0
        # should be close to zero for non-degenerate inputs. We print one named
        # number from the first AB so the user can gut-check.
        if i == 0:
            nc_minus1_probs = type_probs[N_CONTEXT_TOKENS - 1]  # prediction for pitch 0
            real_type_probs_0 = nc_minus1_probs[MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX]
            top_idx = int(real_type_probs_0.argmax().item())
            top_type = PITCH_TYPES[top_idx]
            top_prob = float(real_type_probs_0[top_idx].item())
            ff_prob = float(real_type_probs_0[MODEL_TYPE_ID["FF"] - MODEL_PITCH_TYPES_START_IDX].item())
            pad_mass = float(nc_minus1_probs[0].item())
            print(f"[model] Convention check (first AB, pitch 0 prediction):")
            print(f"        π̂(top-1={top_type}) = {top_prob:.4f}  "
                  f"π̂(FF) = {ff_prob:.4f}  PAD_mass = {pad_mass:.4f}")
            print(f"        (PAD_mass should be ~0; top-1 should be a real pitch type)")

        T = len(ab_df)
        for t in range(T):
            # Sequence position of the PREDICTION for pitch t
            if t == 0:
                seq_pos = N_CONTEXT_TOKENS - 1
            else:
                seq_pos = N_CONTEXT_TOKENS + t - 1

            if seq_pos >= type_probs.shape[0]:
                continue  # sequence too short (shouldn't happen for well-formed AB)

            raw_type_dist = type_probs[seq_pos]  # (8,)
            # Slice real pitch types only (indices 1:8), renormalize
            real_dist = raw_type_dist[MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX].numpy()
            real_dist_sum = real_dist.sum()
            if real_dist_sum < 1e-9:
                continue
            real_dist = real_dist / real_dist_sum

            # Count state for pitch t
            cs = int(ab_df.iloc[t]["count_state"])
            if not (0 <= cs <= 11):
                continue

            acc_probs[cs] += real_dist
            acc_n[cs] += 1.0

        if (i + 1) % 20 == 0:
            print(f"[model]   {i+1}/{n_total} ABs done", flush=True)

    print(f"[model] {n_total} ABs processed ({n_errors} errors); "
          f"total pitch predictions: {int(acc_n.sum()):,}", flush=True)

    # Average per count state
    marginals: dict[int, np.ndarray] = {}
    for cs in range(12):
        b, s = _CS_TO_LABEL[cs]
        if acc_n[cs] == 0:
            print(f"[model] WARNING: no predictions for count ({b},{s}); using uniform",
                  flush=True)
            marginals[cs] = np.ones(N_PITCH_TYPES, dtype=np.float32) / N_PITCH_TYPES
        else:
            marginals[cs] = (acc_probs[cs] / acc_n[cs]).astype(np.float32)

    return marginals


# ============================================================
# 3. Shannon entropy helper
# ============================================================


def shannon_entropy(p: np.ndarray) -> float:
    """H = -Σ p · log(p), base-e nats. Safe against p=0."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


# ============================================================
# 4. Print comparison table
# ============================================================


def print_comparison(
    real_marginals: dict[int, np.ndarray],
    model_marginals: dict[int, np.ndarray],
) -> None:
    """Print the summary table and per-count per-type breakdown."""
    # ---- summary table ----
    header = f"{'Count':>7}  {'Real H':>10}  {'Model H':>10}  {'ΔH':>9}  {'Real top-1':>12}  {'Model top-1':>12}"
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    delta_hs: list[float] = []
    real_hs: list[float] = []
    model_hs: list[float] = []

    for cs in range(12):
        b, s = _CS_TO_LABEL[cs]
        label = f"({b},{s})"
        rp = real_marginals[cs]
        mp = model_marginals[cs]

        rH = shannon_entropy(rp)
        mH = shannon_entropy(mp)
        dH = mH - rH

        real_top = PITCH_TYPES[int(np.argmax(rp))]
        model_top = PITCH_TYPES[int(np.argmax(mp))]

        sign = "+" if dH >= 0 else ""
        print(f"{label:>7}  {rH:>10.4f}  {mH:>10.4f}  {sign}{dH:>8.4f}  "
              f"{real_top:>12}  {model_top:>12}")

        delta_hs.append(dH)
        real_hs.append(rH)
        model_hs.append(mH)

    print("=" * len(header))
    print(f"{'mean':>7}  {np.mean(real_hs):>10.4f}  {np.mean(model_hs):>10.4f}  "
          f"{'+' if np.mean(delta_hs) >= 0 else ''}{np.mean(delta_hs):>8.4f}")
    print()

    # ---- per-count per-type breakdown ----
    print("Per-count per-type breakdown:")
    print()
    for cs in range(12):
        b, s = _CS_TO_LABEL[cs]
        rp = real_marginals[cs]
        mp = model_marginals[cs]
        print(f"Count ({b},{s}):")
        for j, pt in enumerate(PITCH_TYPES):
            delta = float(mp[j]) - float(rp[j])
            sign = "+" if delta >= 0 else ""
            print(f"  {pt:>3}: real={float(rp[j]):.4f}  model={float(mp[j]):.4f}  "
                  f"Δ={sign}{delta:.4f}")
        print()

    # ---- summary stats ----
    print(f"Summary stats:")
    print(f"  Mean real  H(type): {np.mean(real_hs):.4f} nats")
    print(f"  Mean model H(type): {np.mean(model_hs):.4f} nats")
    print(f"  Mean ΔH (model − real): {np.mean(delta_hs):+.4f} nats")
    print()
    if np.mean(delta_hs) < -0.05:
        print("  Interpretation: model is UNDER-dispersed vs real data (lower entropy).")
        print("  The model concentrates mass on fewer types → potential walk-rate distortion.")
        print("  Stage 1 temperature tuning may help if ΔH is consistent across counts.")
    elif np.mean(delta_hs) > 0.05:
        print("  Interpretation: model is OVER-dispersed vs real data (higher entropy).")
    else:
        print("  Interpretation: per-count entropy is approximately matched.")
        print("  Walk deficit likely stems from sequential conditioning, not type marginals.")
        print("  Proceed to Stage 2 (noise injection retraining).")


# ============================================================
# 5. CLI entry point
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-count pitch-type distribution: real data vs teacher-forced model."
    )
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Path to calibrated checkpoint (.pt).")
    parser.add_argument("--n-pas", type=int, default=200,
                        help="Number of held-out validation PAs to sample (default: 200).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for PA sampling (default: 42).")
    args = parser.parse_args()

    # ---- 1. Real marginals ----
    print("=" * 60)
    print("Stage 1/2: computing real per-count type marginals (≤2023)")
    print("=" * 60)
    real_marginals = compute_real_marginals()

    # Sanity-print one named number (Bug-prevention step 2)
    ff_at_00 = float(real_marginals[0][PITCH_TYPES.index("FF")])
    sl_at_00 = float(real_marginals[0][PITCH_TYPES.index("SL")])
    print(f"[real] Sanity check (0,0): real P(FF)={ff_at_00:.4f}  P(SL)={sl_at_00:.4f}")

    # ---- 2. Model marginals ----
    print()
    print("=" * 60)
    print("Stage 2/2: computing model teacher-forced marginals")
    print("=" * 60)
    print(f"[model] loading checkpoint: {args.ckpt}", flush=True)
    nuisance = NuisanceModels(args.ckpt)
    print(f"[model] {nuisance}", flush=True)

    ab_dfs = _load_val_ab_groups(args.n_pas, seed=args.seed)
    model_marginals = compute_model_marginals(nuisance, ab_dfs)

    # Sanity-print one named number from model side (Bug-prevention step 2)
    m_ff_at_00 = float(model_marginals[0][PITCH_TYPES.index("FF")])
    m_sl_at_00 = float(model_marginals[0][PITCH_TYPES.index("SL")])
    print(f"[model] Sanity check (0,0): model P(FF)={m_ff_at_00:.4f}  P(SL)={m_sl_at_00:.4f}")

    # ---- 3. Print comparison ----
    print()
    print("=" * 60)
    print("Comparison: real vs teacher-forced model")
    print("=" * 60)
    print_comparison(real_marginals, model_marginals)


if __name__ == "__main__":
    main()
