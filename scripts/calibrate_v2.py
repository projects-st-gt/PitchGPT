"""Temperature scaling for a trained PitchGPTV2 (base-v1c) checkpoint.

Mirrors ``scripts.calibrate_pitchgpt`` for the V2 architecture: freeze the
model, fit one temperature scalar for the TYPE head on the validation split
(2024 through 2024-07-15) by minimising NLL, report ECE before/after, and save
the temperature into a *new* checkpoint (``checkpoint_calibrated.pt``) — the
original is left untouched.

V2 has only two heads: the 8-way type softmax (PAD at index 0, types 1..7)
and the continuous GMM. Per the handoff plan, only the type head gets a
temperature. The GMM is *checked* (teacher-forced NLL + a plate_x
distributional check against real values, the v8 MDN discipline) but not
re-scaled.

Continuous inputs/targets are z-score normalised exactly as in
``scripts.train_v2`` (normalise AFTER the dataset's nan->0 fill — that is
what the model saw in training). The GMM check denormalises samples back to
raw units before comparing to real plate_x.

Run: python -m scripts.calibrate_v2 \
        --ckpt checkpoints_modal/tiny-v1c-base/checkpoint.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import MODEL_TYPE_ID
from data.profile_cache_loader import ProfileCache
from model.v2.config import V2Config
from model.v2.model import PitchGPTV2
from model.v2.dataset import V2AtBatDataset, collate_v2_at_bats
from model.pitchgpt_dataset import ProfileStandardizer
from scripts.calibrate_pitchgpt import (
    VAL_END,
    ece_equal_mass,
    fit_temperature,
    nll,
)
from scripts.train_v2 import move_to_device


def load_val_pitches(augmented_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(augmented_dir / "2024" / "2024-*.parquet")))
    files = [f for f in files if Path(f).stem <= VAL_END]
    if not files:
        raise FileNotFoundError(f"no 2024 validation parquets under {augmented_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints_modal/tiny-v1c-base/checkpoint.pt"))
    ap.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    ap.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    ap.add_argument("--standardize", action="store_true", default=True,
                    help="apply ProfileStandardizer (matches the V2 training recipe)")
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    ap.add_argument("--gmm-check", action="store_true", default=True,
                    help="teacher-forced GMM NLL + plate_x distributional check")
    ap.add_argument("--no-gmm-check", dest="gmm_check", action="store_false")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap val batches (smoke mode only)")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    valid_fields = {f.name for f in dataclasses.fields(V2Config)}
    cfg = V2Config(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
    model = PitchGPTV2(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    fold_id = int(ckpt.get("fold_id", 0))
    print(f"loaded {args.ckpt}  step={ckpt.get('step')}  size={ckpt.get('size')}  "
          f"fold={fold_id}  params={model.num_parameters():,}  "
          f"standardize={args.standardize}  device={device}")

    val_pitches = load_val_pitches(args.augmented_dir)
    pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=args.profiles_dir)
    pc_b = ProfileCache(role="batter", fold_id=fold_id, profiles_dir=args.profiles_dir)
    std = ProfileStandardizer() if args.standardize else None
    ds = V2AtBatDataset(
        pitches=val_pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        profile_standardizer=std,
        n_continuous=int(cfg.n_continuous),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                        collate_fn=collate_v2_at_bats)
    print(f"val: {len(val_pitches):,} pitches, {len(ds):,} at-bats")

    c_mean = torch.tensor(cfg.continuous_means[: cfg.n_continuous], device=device)
    c_std = torch.tensor(cfg.continuous_stds[: cfg.n_continuous], device=device)
    c_mean_np = np.asarray(cfg.continuous_means, dtype=np.float64)[: cfg.n_continuous]
    c_std_np = np.asarray(cfg.continuous_stds, dtype=np.float64)[: cfg.n_continuous]

    logits_acc: list[torch.Tensor] = []
    targets_acc: list[torch.Tensor] = []
    tcount_acc: list[torch.Tensor] = []   # count state of the TARGET pitch
    pos0_pad_mass: list[np.ndarray] = []

    gmm_nll_sum, gmm_nll_n = 0.0, 0
    gmm_sampled_x: list[np.ndarray] = []
    gmm_real_x: list[np.ndarray] = []
    gmm_sampled_velo: list[np.ndarray] = []
    gmm_real_velo: list[np.ndarray] = []

    import time
    t0 = time.time()
    n_batches_total = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            if i % 50 == 0 and i > 0:
                el = time.time() - t0
                eta = el / i * (n_batches_total - i)
                print(f"  [{i}/{n_batches_total} batches] {el:.0f}s elapsed, ETA {eta:.0f}s",
                      flush=True)
            batch = move_to_device(batch, device)
            # Z-score normalise inputs + targets exactly as in training
            # (dataset already replaced NaN inputs with raw 0.0).
            batch["continuous"] = (batch["continuous"] - c_mean) / c_std
            tc = batch["targets"]["continuous"]
            fm = torch.isfinite(tc)
            batch["targets"]["continuous"] = torch.where(fm, (tc - c_mean) / c_std, tc)

            out = model(
                pitcher_profile=batch["pitcher_profile"],
                batter_profile=batch["batter_profile"],
                type_ids=batch["type_ids"],
                continuous=batch["continuous"],
                result_ids=batch["result_ids"],
                count_state=batch["count_state"],
                outs=batch["outs"],
                runners=batch["runners"],
                pitch_number=batch["pitch_number"],
                padding_mask=batch["padding_mask"],
            )
            type_logits = out["type_logits"].cpu().float()           # (B, T, 8)
            type_targets = batch["targets"]["type"].cpu()             # (B, T)
            mask = type_targets != -100
            logits_acc.append(type_logits[mask])
            targets_acc.append(type_targets[mask])
            # The target at position t is pitch t+1, thrown at the count
            # stored at input position t+1 — shift count_state left by one.
            cs = batch["count_state"].cpu()
            target_count = torch.zeros_like(cs)
            target_count[:, :-1] = cs[:, 1:]
            tcount_acc.append(target_count[mask])

            # Named diagnostic: PAD probability mass at position 0 (the trained
            # first-pitch position — v9's untrained NC-1 had 79% here).
            p0 = F.softmax(type_logits[:, 0, :], dim=-1)[:, 0].numpy()
            pos0_pad_mass.append(p0)

            if args.gmm_check:
                # Teacher-forced GMM NLL + plate_x/velo distributional check.
                type_t = type_targets.clone()
                type_t[type_t == -100] = 0
                hidden = out["hidden"]
                log_w, mu, log_std = model.predict_continuous(
                    hidden, type_t.to(device).clamp(0, cfg.n_pitch_types - 1))
                cont_t = batch["targets"]["continuous"]
                valid = (type_targets.to(device) != -100) & torch.isfinite(cont_t).all(dim=-1)
                if valid.any():
                    nll_v = model.gmm_head.nll(
                        log_w[valid], mu[valid], log_std[valid], cont_t[valid])
                    n_v = int(valid.sum())
                    gmm_nll_sum += float(nll_v) * n_v
                    gmm_nll_n += n_v
                    s = model.gmm_head.sample(
                        log_w[valid], mu[valid], log_std[valid]).cpu().numpy()
                    real = cont_t[valid].cpu().numpy()
                    # Denormalise both back to raw units for the comparison.
                    s_raw = s * c_std_np + c_mean_np
                    r_raw = real * c_std_np + c_mean_np
                    gmm_sampled_velo.append(s_raw[:, 0])
                    gmm_real_velo.append(r_raw[:, 0])
                    gmm_sampled_x.append(s_raw[:, 2])
                    gmm_real_x.append(r_raw[:, 2])

    logits = torch.cat(logits_acc).float()
    targets = torch.cat(targets_acc).long()

    T = fit_temperature(logits, targets)
    p0 = F.softmax(logits, dim=-1).numpy()
    p1 = F.softmax(logits / T, dim=-1).numpy()
    tnp = targets.numpy()
    acc = float((p0.argmax(1) == tnp).mean())

    print(f"\n{'head':<12} {'n':>9}  {'NLL_before':>10} {'NLL_after':>10}  "
          f"{'ECE_before':>10} {'ECE_after':>10}  {'acc':>7}  {'T':>7}")
    print(f"{'type':<12} {len(targets):>9,}  {nll(logits, targets):>10.4f} "
          f"{nll(logits, targets, T):>10.4f}  {ece_equal_mass(p0, tnp):>10.4f} "
          f"{ece_equal_mass(p1, tnp):>10.4f}  {acc:>7.4f}  {T:>7.4f}")

    # --- Per-count temperatures -----------------------------------------------
    # The flat-T model over-commits to FF at hitter counts in rollout (2-0
    # +13.4pp, 2026-06-09 marginals). One temperature per count state of the
    # predicted pitch (group-conditional temperature scaling — ATS/contextual-
    # temperature family; arXiv:2409.19817, arXiv:2012.13575). Temperature is
    # monotonic: it shrinks the modal type's EXCESS at overconfident counts
    # but cannot reorder preferences.
    tcounts = torch.cat(tcount_acc).long()
    count_temps: dict[str, float] = {}
    print(f"\n--- per-count temperatures (count of the predicted pitch) ---")
    print(f"{'count':>5} {'n':>8}  {'T_cs':>7}  {'NLL_flat':>9} {'NLL_cs':>9}  "
          f"{'FF_flat':>8} {'FF_cs':>7} {'FF_real':>8}")
    nll_flat_sum, nll_cs_sum = 0.0, 0.0
    for cs_id in range(12):
        m = tcounts == cs_id
        n_cs = int(m.sum())
        if n_cs < 1000:
            continue
        lg, tg = logits[m], targets[m]
        T_cs = fit_temperature(lg, tg)
        count_temps[str(cs_id)] = T_cs
        nll_f, nll_c = nll(lg, tg, T), nll(lg, tg, T_cs)
        nll_flat_sum += nll_f * n_cs
        nll_cs_sum += nll_c * n_cs
        ff_flat = float(F.softmax(lg / T, dim=-1)[:, MODEL_TYPE_ID["FF"]].mean())
        ff_cs = float(F.softmax(lg / T_cs, dim=-1)[:, MODEL_TYPE_ID["FF"]].mean())
        ff_real = float((tg == MODEL_TYPE_ID["FF"]).float().mean())
        b, s = cs_id // 3, cs_id % 3
        print(f"  {b}-{s} {n_cs:>8,}  {T_cs:>7.4f}  {nll_f:>9.4f} {nll_c:>9.4f}  "
              f"{ff_flat:>8.4f} {ff_cs:>7.4f} {ff_real:>8.4f}")
    n_fit = int(sum((tcounts == int(k)).sum() for k in count_temps))
    print(f"  overall NLL: flat {nll_flat_sum / n_fit:.4f} -> per-count "
          f"{nll_cs_sum / n_fit:.4f}  ({len(count_temps)}/12 counts fitted)")

    # --- Named numerical checks (bug-prevention discipline) -------------------
    pad_mass = float(np.concatenate(pos0_pad_mass).mean())
    pi_post = F.softmax(logits / T, dim=-1)
    pi_ff = float(pi_post[:, MODEL_TYPE_ID["FF"]].mean())
    real_ff = float((tnp == MODEL_TYPE_ID["FF"]).mean())
    pad_pred_mass = float(pi_post[:, 0].mean())
    print(f"\n--- named checks ---")
    print(f"  mean π̂(FF) over val = {pi_ff:.4f}  (real FF share {real_ff:.4f})")
    print(f"  mean PAD mass at position 0 (first pitch) = {pad_mass:.4f}  "
          f"(v9's untrained NC-1 was 0.79 — should be ≈0 here)")
    print(f"  mean PAD mass over all predictions = {pad_pred_mass:.4f}")
    assert pi_ff > 0.15, f"π̂(FF)={pi_ff:.4f} — convention bug? FF should dominate"
    assert pad_mass < 0.05, f"PAD mass at position 0 = {pad_mass:.4f} — first pitch untrained?"

    if args.gmm_check and gmm_sampled_x:
        sx = np.concatenate(gmm_sampled_x)
        rx = np.concatenate(gmm_real_x)
        sv = np.concatenate(gmm_sampled_velo)
        rv = np.concatenate(gmm_real_velo)
        print(f"\n--- GMM distributional check ({len(sx):,} pitches, teacher-forced) ---")
        print(f"  GMM NLL (normalised space) = {gmm_nll_sum / max(gmm_nll_n, 1):.4f}")
        print(f"  sampled mean velo = {sv.mean():.2f} mph  (real {rv.mean():.2f})")
        print(f"  sampled mean |plate_x| = {np.abs(sx).mean():.3f} ft  (real {np.abs(rx).mean():.3f})")
        print(f"  sampled frac |plate_x|>1.1 = {(np.abs(sx) > 1.1).mean():.3f}  "
              f"(real {(np.abs(rx) > 1.1).mean():.3f})")
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_p = ks_2samp(sx, rx)
            print(f"  KS plate_x: stat={ks_stat:.4f}  p={ks_p:.4g}")
        except ImportError:
            print("  (scipy not available — skipping KS test)")

    out_path = args.ckpt.with_name("checkpoint_calibrated.pt")
    ckpt["temperatures"] = {"type": T}
    ckpt["count_temperatures"] = {"type": count_temps}
    ckpt["calibration_val_end"] = VAL_END
    torch.save(ckpt, out_path)
    print(f"\ntemperatures: {{'type': {T:.4f}}}  + per-count for {len(count_temps)} counts")
    print(f"saved calibrated checkpoint -> {out_path}  (original {args.ckpt} unchanged)")


if __name__ == "__main__":
    main()
