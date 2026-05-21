"""Per-class diagnostic for the propensity TYPE head.

For one or more checkpoints, reports — on the val split — the true vs
predicted FF rate, per-class precision/recall/support, and mean predicted
probability per class. The point is to surface whether the model is
over-predicting FF and how the focal/weights variants change that.

Conventions: model TYPE head emits 8 logits (PAD at 0, PITCH_TYPES[0]=FF
at index 1, ..., PITCH_TYPES[6]=FS at index 7). We slice [:, :, 1:] and
report under PITCH_TYPES names — no magic indices in user-facing output.

Run: python -m scripts.diagnose_type_perclass \
        --ckpt checkpoints_modal/tiny-fold0-1778792736/checkpoint_best.pt \
        --ckpt checkpoints_modal/tiny-fold0-1778979210/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import PITCH_TYPES, MODEL_PITCH_TYPES_START_IDX
from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
)

VAL_END = "2024-07-15"


def to_dev(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    return x


def load_val_pitches(augmented_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(augmented_dir / "2024" / "2024-*.parquet")))
    files = [f for f in files if Path(f).stem <= VAL_END]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def collect_type_logits(ckpt_path: Path, val_pitches: pd.DataFrame, args) -> tuple[np.ndarray, np.ndarray, dict]:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
    model = PitchGPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    fold_id = int(ckpt.get("fold_id", 0))
    pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=args.profiles_dir)
    pc_b = ProfileCache(role="batter", fold_id=fold_id, profiles_dir=args.profiles_dir)
    std = ProfileStandardizer() if args.standardize else None
    ds = PitchGPTAtBatDataset(
        pitches=val_pitches, pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup, profile_standardizer=std,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_pitchgpt_at_bats)
    NC = PitchGPT.N_CONTEXT_TOKENS

    logits_list, target_list = [], []
    with torch.no_grad():
        for batch in loader:
            bd = to_dev(batch, device)
            out = model(
                pitcher_profile=bd["pitcher_profile"], batter_profile=bd["batter_profile"],
                categorical_context=bd["categorical_context"], pitch_factors=bd["pitch_factors"],
                intended_actions=bd["intended_actions"], padding_mask=bd["padding_mask"],
                arsenal=bd.get("arsenal"),
            )
            lg = out["propensity"]["type"][:, NC:, :].cpu()  # (B, T, 8) -- model index 0..7
            tg = batch["targets"]["propensity"]["type"]      # (B, T), -100 padded; targets use model ids 1..7
            mask = tg != -100
            logits_list.append(lg[mask])
            target_list.append(tg[mask])
    logits = torch.cat(logits_list).numpy()      # (N, 8)
    targets = torch.cat(target_list).numpy()     # (N,) in {1..7}

    # Slice to the named-pitch logits only (drop PAD column 0). Targets stay 1..7,
    # so we shift to 0..6 to align with PITCH_TYPES index.
    type_logits = logits[:, MODEL_PITCH_TYPES_START_IDX:]   # (N, 7)
    type_targets = targets - MODEL_PITCH_TYPES_START_IDX     # (N,) in 0..6
    return type_logits, type_targets, {"size": ckpt.get("size"), "step": ckpt.get("step"),
                                       "focal_gamma": cfg.type_focal_gamma,
                                       "class_weight_alpha": cfg.type_class_weight_alpha}


def report(name: str, logits: np.ndarray, targets: np.ndarray) -> None:
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()  # (N, 7)
    preds = probs.argmax(axis=1)  # 0..6
    n = len(targets)
    print(f"\n=== {name}  N={n:,} ===")
    header = f"  {'pitch':<6} {'true_rate':>10} {'pred_rate':>10} {'mean_p':>8} {'precision':>10} {'recall':>8} {'F1':>6} {'support':>9}"
    print(header)
    for i, pt in enumerate(PITCH_TYPES):
        sup = int((targets == i).sum())
        true_rate = sup / n
        pred_rate = float((preds == i).mean())
        mean_p = float(probs[:, i].mean())
        # precision / recall for this class
        tp = int(((preds == i) & (targets == i)).sum())
        fp = int(((preds == i) & (targets != i)).sum())
        fn = int(((preds != i) & (targets == i)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {pt:<6} {true_rate:>10.4f} {pred_rate:>10.4f} {mean_p:>8.4f} {prec:>10.4f} {rec:>8.4f} {f1:>6.3f} {sup:>9,}")
    # Overall top-1
    acc = float((preds == targets).mean())
    print(f"  overall top-1 acc = {acc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, type=Path)
    ap.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    ap.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    args = ap.parse_args()

    val = load_val_pitches(args.augmented_dir)
    print(f"val: {len(val):,} pitches")

    for ck in args.ckpt:
        lg, tg, meta = collect_type_logits(ck, val, args)
        label = f"{ck.parent.name}  (gamma={meta['focal_gamma']}, alpha={meta['class_weight_alpha']})"
        report(label, lg, tg)


if __name__ == "__main__":
    main()
