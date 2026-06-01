"""Inference-time A/B for the arsenal mask on an existing checkpoint.

Loads a calibrated PitchGPT checkpoint, runs the validation split once with
``arsenal_mask_type_logits`` OFF (what the checkpoint was trained and
calibrated with) and once ON (a pure post-process on the type logits).
Reports the calibration / accuracy / NLL delta on the type head, plus the
count of predictions that the mask actually re-routed (the model's argmax
WOULD have been an impossible type but the mask zeroed it out).

This is the free-calibration A/B for the proposal: no retraining, no
calibration re-fit (we reuse the saved temperatures, which are still valid
because the mask sets entries to ``-1e9`` and the temperature divides — so
``-1e9/T`` is still ``-1e9`` in float32 and ``softmax`` still treats it as
zero mass).

Run::

    PYTHONPATH=. python scripts/eval_arsenal_mask_delta.py \\
        --ckpt checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt

CPU-forced because the AB-outcome gather miscompiles on MPS (a pre-existing
issue unrelated to v7; see Issue #2). The mask itself is fine on MPS, but
the calibration loader runs through the same forward.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.backends.mps.is_available = lambda: False  # MPS ab_outcome gather miscompile

from data.dataset import (
    MODEL_PITCH_TYPES_START_IDX,
    MODEL_PITCH_TYPES_END_IDX,
    PITCH_TYPES,
)
from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
)
from torch.utils.data import DataLoader

VAL_END = "2024-07-15"


def ece_equal_mass(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    n = len(conf)
    if n == 0:
        return 0.0
    out = 0.0
    for b in np.array_split(np.arange(n), n_bins):
        if len(b):
            out += len(b) / n * abs(correct[b].mean() - conf[b].mean())
    return float(out)


def load_val_pitches(augmented_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(augmented_dir / "2024" / "2024-*.parquet")))
    files = [f for f in files if Path(f).stem <= VAL_END]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def to_dev(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    return x


def collect_type_logits_and_targets(model, loader, device, n_context_tokens):
    """One pass through val; return concatenated (logits, targets, arsenal)."""
    all_logits, all_targets, all_arsenal = [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = to_dev(batch, device)
            out = model(
                pitcher_profile=bd["pitcher_profile"],
                batter_profile=bd["batter_profile"],
                categorical_context=bd["categorical_context"],
                pitch_factors=bd["pitch_factors"],
                intended_actions=bd["intended_actions"],
                padding_mask=bd["padding_mask"],
                arsenal=bd.get("arsenal"),
            )
            type_logits = out["propensity"]["type"][:, n_context_tokens:, :].cpu()
            tg = batch["targets"]["propensity"]["type"]
            mask = tg != -100
            all_logits.append(type_logits[mask])
            all_targets.append(tg[mask])
            # Arsenal is (B, 14), one per AB; replicate so we keep alignment
            # with per-pitch positions (not strictly needed for metrics here
            # but useful for reasoning if we want per-row diagnostics later).
            all_arsenal.append(batch.get("arsenal").repeat_interleave(mask.sum(dim=1), dim=0)
                               if batch.get("arsenal") is not None else None)
    logits = torch.cat(all_logits).float()
    targets = torch.cat(all_targets).long()
    arsenal = torch.cat([a for a in all_arsenal if a is not None]) if all_arsenal[0] is not None else None
    return logits, targets, arsenal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    ap.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
    cfg.arsenal_mask_type_logits = False  # start with mask OFF (training-time state)
    model = PitchGPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    temps = ckpt.get("temperatures", {})
    T_type = float(temps.get("type", 1.0))
    fold_id = int(ckpt.get("fold_id", 0))
    print(f"loaded {args.ckpt}  step={ckpt.get('step')}  T_type={T_type:.4f}  fold={fold_id}")

    val_pitches = load_val_pitches(args.augmented_dir)
    pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=args.profiles_dir)
    pc_b = ProfileCache(role="batter", fold_id=fold_id, profiles_dir=args.profiles_dir)
    std = ProfileStandardizer() if args.standardize else None
    ds = PitchGPTAtBatDataset(
        pitches=val_pitches, pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup, profile_standardizer=std,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_pitchgpt_at_bats)
    print(f"val: {len(val_pitches):,} pitches, {len(ds):,} at-bats")

    NC = PitchGPT.N_CONTEXT_TOKENS

    print("\n--- pass 1: mask OFF (training-time / calibrated state) ---")
    cfg.arsenal_mask_type_logits = False
    logits_off, targets_off, _ = collect_type_logits_and_targets(model, loader, device, NC)
    print(f"  collected {len(targets_off):,} valid type predictions")

    print("\n--- pass 2: mask ON (inference-time post-process, same weights) ---")
    cfg.arsenal_mask_type_logits = True
    logits_on, targets_on, _ = collect_type_logits_and_targets(model, loader, device, NC)
    assert torch.equal(targets_off, targets_on), "target alignment broke between passes"

    # Apply the saved type-temperature to both (calibrate_pitchgpt fit T on the
    # raw logits; we keep that scaling for the A/B so the only delta is the mask).
    p_off = F.softmax(logits_off / T_type, dim=-1).numpy()
    p_on = F.softmax(logits_on / T_type, dim=-1).numpy()
    tnp = targets_off.numpy()

    nll_off = float(F.cross_entropy(logits_off / T_type, targets_off).item())
    nll_on = float(F.cross_entropy(logits_on / T_type, targets_on).item())
    ece_off = ece_equal_mass(p_off, tnp)
    ece_on = ece_equal_mass(p_on, tnp)
    acc_off = float((p_off.argmax(1) == tnp).mean())
    acc_on = float((p_on.argmax(1) == tnp).mean())

    # How often did the mask actually re-route a top-1 prediction? — i.e. the
    # unmasked argmax was a type the pitcher has never thrown (an impossible
    # event that the mask zeroed out).
    pred_off = p_off.argmax(1)
    pred_on = p_on.argmax(1)
    n_rerouted = int((pred_off != pred_on).sum())
    pct_rerouted = 100.0 * n_rerouted / len(tnp)

    # Also: how much probability mass was leaking to impossible types on average?
    # (Equivalent to mass on the masked classes pre-mask, summed per-row, averaged.)
    impossible_mass_per_row = (p_off.sum(1) - p_on.sum(1) * (p_off.sum(1) / p_on.sum(1) + 1e-12))
    # Simpler: how much mass moves between masked classes? — actually compute
    # the L1 distance between p_off and p_on per row, averaged.
    l1_shift = float(np.abs(p_off - p_on).sum(axis=1).mean())

    print("\n=== arsenal-mask delta (val 2024 H1, type head only) ===")
    print(f"  n predictions     : {len(tnp):,}")
    print(f"  T_type (saved)    : {T_type:.4f}")
    print()
    print(f"  {'metric':<14} {'mask OFF':>12} {'mask ON':>12} {'delta':>12}")
    print(f"  {'NLL':<14} {nll_off:>12.4f} {nll_on:>12.4f} {nll_on - nll_off:>+12.4f}")
    print(f"  {'ECE':<14} {ece_off:>12.4f} {ece_on:>12.4f} {ece_on - ece_off:>+12.4f}")
    print(f"  {'top-1 acc':<14} {acc_off:>12.4f} {acc_on:>12.4f} {acc_on - acc_off:>+12.4f}")
    print()
    print(f"  predictions re-routed by mask : {n_rerouted:,} of {len(tnp):,}  ({pct_rerouted:.2f}%)")
    print(f"  mean L1 shift in p̂(type | h)  : {l1_shift:.4f}")

    out_path = args.ckpt.with_name("arsenal_mask_delta.json")
    summary = {
        "ckpt": str(args.ckpt),
        "T_type": T_type,
        "n_predictions": int(len(tnp)),
        "nll_off": nll_off,
        "nll_on": nll_on,
        "nll_delta": nll_on - nll_off,
        "ece_off": ece_off,
        "ece_on": ece_on,
        "ece_delta": ece_on - ece_off,
        "acc_off": acc_off,
        "acc_on": acc_on,
        "acc_delta": acc_on - acc_off,
        "n_rerouted": n_rerouted,
        "pct_rerouted": pct_rerouted,
        "mean_l1_shift": l1_shift,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
