"""Temperature scaling for a trained PitchGPT checkpoint.

Per the ``pitchgpt-model`` skill: freeze the model, fit one temperature
scalar per head on the validation split by minimising NLL, report ECE
before/after, and save the temperatures back into a *new* checkpoint
(``checkpoint_calibrated.pt``) — the original is left untouched.

Heads calibrated: the four propensity factor heads (type, zone, velo,
spin_rate) = pi-hat, plus the result head = mu-hat, plus the AB-outcome
head. Spin-axis is a von-Mises regression head, not categorical — skipped.

Run: python -m scripts.calibrate_pitchgpt \
        --ckpt checkpoints_modal/phase-b-tiny-fold0-v2/checkpoint.pt
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

from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
)

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


def nll(logits: torch.Tensor, targets: torch.Tensor, T: float = 1.0) -> float:
    return float(F.cross_entropy(logits / T, targets).item())


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """LBFGS on a single positive scalar (Guo et al. 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints_modal/phase-b-tiny-fold0-v2/checkpoint.pt"))
    ap.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    ap.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    ap.add_argument("--standardize", action="store_true", default=True,
                    help="apply ProfileStandardizer (matches the 'std' training recipe)")
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    ap.add_argument("--mdn-check", action="store_true",
                    help="run the MDN distributional check (v8 checkpoint)")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
    model = PitchGPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    fold_id = int(ckpt.get("fold_id", 0))
    print(f"loaded {args.ckpt}  step={ckpt.get('step')}  size={ckpt.get('size')}  fold={fold_id}  "
          f"params={model.num_parameters():,}  standardize={args.standardize}")

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
    PROP_HEADS = ["type", "zone", "velo", "spin_rate"]
    has_ab_head = getattr(cfg, "ab_outcome_head", True)
    all_heads = PROP_HEADS + ["result"] + (["ab_outcome"] if has_ab_head else [])
    bucket: dict[str, list] = {h: [[], []] for h in all_heads}

    # MDN distributional check accumulators
    has_mdn = getattr(cfg, "location_mdn", False)
    mdn_sampled_x: list[np.ndarray] = []
    mdn_real_x: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            bd = to_dev(batch, device)
            out = model(
                pitcher_profile=bd["pitcher_profile"], batter_profile=bd["batter_profile"],
                categorical_context=bd["categorical_context"], pitch_factors=bd["pitch_factors"],
                intended_actions=bd["intended_actions"], padding_mask=bd["padding_mask"],
                arsenal=bd.get("arsenal"),
            )
            pad = batch["padding_mask"]  # (B, T) bool
            for h in PROP_HEADS:
                lg = out["propensity"][h][:, NC:, :].cpu()
                tg = batch["targets"]["propensity"][h]
                mask = tg != -100
                bucket[h][0].append(lg[mask]); bucket[h][1].append(tg[mask])
            lg = out["result"].cpu()
            tg = batch["targets"]["result"]
            mask = tg != -100
            bucket["result"][0].append(lg[mask]); bucket["result"][1].append(tg[mask])
            if has_ab_head and "ab_outcome_per_pos" in out:
                lengths = pad.sum(dim=1)
                term_idx = (lengths - 1).clamp(min=0)
                lg_all = out["ab_outcome_per_pos"].cpu()
                term_lg = lg_all[torch.arange(lg_all.size(0)), term_idx]
                tg = batch["targets"]["ab_outcome"]
                mask = tg != -100
                bucket["ab_outcome"][0].append(term_lg[mask]); bucket["ab_outcome"][1].append(tg[mask])

            if has_mdn and args.mdn_check and "location_mdn" in out:
                lm = out["location_mdn"]
                loc_tgt = batch["targets"]["location"]  # (B, T, 2)
                valid = torch.isfinite(loc_tgt).all(dim=-1)
                if valid.any():
                    s = model.location_mdn.sample(
                        lm["log_w"][valid], lm["mu"][valid], lm["log_std"][valid])
                    mdn_sampled_x.append(s[:, 0].cpu().numpy())
                    mdn_real_x.append(loc_tgt[valid][:, 0].numpy())

    print(f"\n{'head':<12} {'n':>9}  {'NLL_before':>10} {'NLL_after':>10}  {'ECE_before':>10} {'ECE_after':>10}  {'acc':>7}  {'T':>7}")
    temps: dict[str, float] = {}
    for h in all_heads:
        logits = torch.cat(bucket[h][0]).float()
        targets = torch.cat(bucket[h][1]).long()
        if len(targets) == 0:
            print(f"{h:<12} {'(no samples)':>9}")
            continue
        T = fit_temperature(logits, targets)
        temps[h] = T
        p0 = F.softmax(logits, dim=-1).numpy()
        p1 = F.softmax(logits / T, dim=-1).numpy()
        tnp = targets.numpy()
        acc = float((p0.argmax(1) == tnp).mean())
        print(f"{h:<12} {len(targets):>9,}  {nll(logits, targets):>10.4f} {nll(logits, targets, T):>10.4f}  "
              f"{ece_equal_mass(p0, tnp):>10.4f} {ece_equal_mass(p1, tnp):>10.4f}  {acc:>7.4f}  {T:>7.4f}")

    if has_mdn and args.mdn_check and mdn_sampled_x:
        sx = np.concatenate(mdn_sampled_x)
        rx = np.concatenate(mdn_real_x)
        print(f"\n--- MDN distributional check ({len(sx):,} pitches) ---")
        print(f"  sampled mean |plate_x| = {np.abs(sx).mean():.3f}  (real {np.abs(rx).mean():.3f})")
        print(f"  sampled frac |plate_x|>1.1 = {(np.abs(sx) > 1.1).mean():.3f}  (real {(np.abs(rx) > 1.1).mean():.3f})")
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(sx, rx)
        print(f"  KS plate_x: stat={ks_stat:.4f}  p={ks_p:.4g}")

    out_path = args.ckpt.with_name("checkpoint_calibrated.pt")
    ckpt["temperatures"] = temps
    ckpt["calibration_val_end"] = VAL_END
    torch.save(ckpt, out_path)
    print(f"\ntemperatures: {temps}")
    print(f"saved calibrated checkpoint -> {out_path}  (original {args.ckpt} unchanged)")


if __name__ == "__main__":
    main()
