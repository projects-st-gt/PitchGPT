"""Phase B underperformance investigation — clean full-val eval + diagnostics.

Loads the Phase B checkpoint, runs it over the *entire* val split (not the
64-batch in-training subsample), and reports:

  1. type top-1 on the propensity head's natural domain (pitches 1..T-1)
  2. per-pitch-position accuracy breakdown
  3. ECE on the type head (the metric the eval-protocol skill says actually matters)
  4. comparison vs per-pitcher-mode floor on the SAME pitch subset
  5. dataset-adapter sanity spot-checks (factor ids vs raw augmented parquet)

Run: python -m scripts.investigate_phase_b
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import PITCH_TYPE_TO_ID
from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset,
    collate_pitchgpt_at_bats,
)

CKPT = Path("checkpoints_modal/phase-b-tiny-fold0-v2/checkpoint.pt")
VAL_END = "2024-07-15"


def load_val_pitches() -> pd.DataFrame:
    files = sorted(glob.glob("data/augmented/2024/2024-*.parquet"))
    files = [f for f in files if f.split("/")[-1].replace(".parquet", "") <= VAL_END]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass ECE on the top-1 confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    n = len(conf)
    if n == 0:
        return 0.0
    bins = np.array_split(np.arange(n), n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        ece += len(b) / n * abs(correct[b].mean() - conf[b].mean())
    return float(ece)


def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
    model = PitchGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: step={ckpt['step']}, size={ckpt['size']}, n_params={model.num_parameters():,}")

    val_pitches = load_val_pitches()
    print(f"val pitches: {len(val_pitches):,}")

    pc_p = ProfileCache(role="pitcher", fold_id=0)
    pc_b = ProfileCache(role="batter", fold_id=0)
    ds = PitchGPTAtBatDataset(
        pitches=val_pitches, pitcher_profile_lookup=pc_p.lookup, batter_profile_lookup=pc_b.lookup
    )
    print(f"val at-bats: {len(ds):,}")

    # --- Dataset adapter sanity spot-check ---
    print("\n--- dataset adapter spot-check (first 3 ABs) ---")
    val_grouped = val_pitches.groupby(["game_pk", "at_bat_number"], sort=False)
    keys = list(val_grouped.groups.keys())
    ok = True
    for i in range(3):
        item = ds[i]
        gk, ab = keys[i]
        raw = val_pitches[(val_pitches.game_pk == gk) & (val_pitches.at_bat_number == ab)].sort_values("pitch_number")
        raw_type = raw["type_id"].to_numpy()
        ds_type = item["pitch_factors"]["type"].numpy()
        match = np.array_equal(raw_type, ds_type)
        ok &= match
        print(f"  AB {i}: raw type_ids={raw_type.tolist()}  ds={ds_type.tolist()}  match={match}")
        # spot-check categorical context
        raw_ballpark = int(raw["ballpark_id"].iloc[0])
        ds_ballpark = int(item["categorical_context"]["ballpark"])
        print(f"         ballpark raw={raw_ballpark} ds={ds_ballpark} match={raw_ballpark == ds_ballpark}")
    print(f"  spot-check {'PASSED' if ok else 'FAILED'}")

    # --- Full val eval ---
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_pitchgpt_at_bats)
    all_type_probs = []
    all_type_targets = []
    all_positions = []  # which pitch index in the AB does each target correspond to
    n_done = 0
    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    with torch.no_grad():
        for batch in loader:
            B, T = batch["padding_mask"].shape
            batch_d = to_dev(batch)
            out = model(
                pitcher_profile=batch_d["pitcher_profile"],
                batter_profile=batch_d["batter_profile"],
                categorical_context=batch_d["categorical_context"],
                pitch_factors=batch_d["pitch_factors"],
                intended_actions=batch_d["intended_actions"],
                padding_mask=batch_d["padding_mask"],
            )
            # propensity type at pitch positions (drop context), shape (B, T, 8)
            type_logits = out["propensity"]["type"][:, PitchGPT.N_CONTEXT_TOKENS:, :]
            type_probs = F.softmax(type_logits, dim=-1).cpu().numpy()  # (B, T, 8)
            type_target = batch["targets"]["propensity"]["type"].numpy()  # (B, T), -100 padded
            # target at position t corresponds to pitch index t+1 in the AB
            for b in range(B):
                for t in range(T):
                    tgt = type_target[b, t]
                    if tgt == -100:
                        continue
                    all_type_probs.append(type_probs[b, t])
                    all_type_targets.append(tgt)
                    all_positions.append(t + 1)  # this prediction is for pitch index t+1
            n_done += B
    type_probs = np.stack(all_type_probs)  # (N, 8)
    type_targets = np.array(all_type_targets)  # (N,) values in 1..7
    positions = np.array(all_positions)
    preds = type_probs.argmax(axis=1)

    overall_acc = (preds == type_targets).mean()
    print(f"\n--- full-val type top-1 (pitches 1..T-1) ---")
    print(f"  N predictions: {len(type_targets):,}")
    print(f"  top-1 accuracy: {overall_acc:.4f}")
    # top-3
    top3 = np.argsort(type_probs, axis=1)[:, -3:]
    top3_acc = np.mean([type_targets[i] in top3[i] for i in range(len(type_targets))])
    print(f"  top-3 accuracy: {top3_acc:.4f}")
    # ECE
    ece = expected_calibration_error(type_probs, type_targets, n_bins=15)
    print(f"  ECE (15 bins): {ece:.4f}")
    # log loss
    eps = 1e-9
    logloss = -np.mean(np.log(type_probs[np.arange(len(type_targets)), type_targets] + eps))
    print(f"  log-loss: {logloss:.4f}  (uniform-over-7 = {np.log(7):.4f})")

    # --- Per-position breakdown ---
    print(f"\n--- accuracy by pitch position ---")
    for p in sorted(set(positions.tolist()))[:8]:
        mask = positions == p
        print(f"  pitch #{p+1} (idx {p}): n={mask.sum():>7,}  acc={(preds[mask] == type_targets[mask]).mean():.4f}")

    # --- Per-pitcher-mode floor on the SAME subset ---
    # Need pitcher id aligned to each prediction. Re-derive: iterate the dataset again
    # but this time track the pitcher per prediction. Cheaper: recompute via the val df.
    val_pitches["pidx"] = val_pitches.groupby(["game_pk", "at_bat_number"]).cumcount()
    sub = val_pitches[val_pitches["pidx"] >= 1].copy()  # pitches predicted by the propensity head
    pmode = val_pitches.groupby("pitcher")["type_id"].agg(lambda s: s.mode().iloc[0])
    sub = sub.merge(pmode.rename("pmode"), left_on="pitcher", right_index=True)
    pm_acc = (sub["type_id"] == sub["pmode"]).mean()
    print(f"\n--- baselines on the same subset (pitches 1..T-1, val) ---")
    print(f"  per-pitcher-mode (uses val data — generous floor): {pm_acc:.4f}")
    global_mode = val_pitches["type_id"].mode().iloc[0]
    gm_acc = (sub["type_id"] == global_mode).mean()
    print(f"  global-mode ('always FF'): {gm_acc:.4f}")
    print(f"  PitchGPT-Tiny: {overall_acc:.4f}  (Δ vs per-pitcher-mode: {overall_acc - pm_acc:+.4f})")


if __name__ == "__main__":
    main()
