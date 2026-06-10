"""Closed-loop fine-tuning for PitchGPTV2 — tethered top-K self-input (v1c-cl).

Spec: docs/superpowers/specs/2026-06-10-v1c-cl-finetune-design.md
Evidence: CAT-K (Zhang et al., CVPR 2025, arXiv:2412.05334) adapted; detached
substitution per the pushforward trick (Brandstetter et al., ICLR 2022).

The model's INPUTS become its own sampled pitches (tethered to the real
sequence) while the TARGETS stay the real next pitches and the count/result
scaffold stays real. Two passes per batch:

  Pass 1 (no_grad, sequential over positions): at each pitch position t,
  read the model's prediction at position t-1 over the substituted-so-far
  prefix. If the REAL type at t is in the model's top-K -> keep the real
  type (tether hit). Else -> substitute the model's top-1 candidate. The
  continuous input at t is sampled from the GMM conditioned on the
  substituted type (detached, physically clamped, kept in z-score space —
  exactly matching rollout conditions).

  Pass 2 (grad): one standard forward on the substituted batch + the SAME
  losses as base training (type CE + GMM NLL vs the REAL targets), via
  compute_v2_losses.

No gradients flow through sampling or substitution (both threads of the
ml-research run converge on detaching as the principled choice).

Run locally (smoke)::

    python -m scripts.finetune_v2_cl --ckpt checkpoints_modal/tiny-v1c-base/checkpoint.pt \
        --max-pitches 200000 --max-steps 30

Modal: ``modal_app.py::finetune_v2_cl_remote``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.profile_cache_loader import ProfileCache
from model.v2.config import V2Config
from model.v2.model import PitchGPTV2
from model.v2.dataset import V2AtBatDataset, collate_v2_at_bats
from model.pitchgpt_dataset import (
    DEFAULT_PROFILE_STD_PATH,
    ProfileStandardizer,
    load_augmented_pitches,
    split_augmented,
)
from scripts.train_v2 import (
    compute_v2_losses,
    evaluate,
    move_to_device,
    select_device,
)

# Physical clamp bounds in RAW units [velo mph, spin rpm, plate_x ft, plate_z ft]
# — identical to causal.g_computation_v2's rollout clamps.
_CLAMP_LO = (60.0, 800.0, -2.5, 0.0)
_CLAMP_HI = (110.0, 3800.0, 2.5, 5.0)


@torch.no_grad()
def build_substituted_batch(
    model: PitchGPTV2,
    batch: dict,
    cfg: V2Config,
    *,
    k: int = 3,
    p_sub: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[dict, dict]:
    """Pass 1: build self-generated (tethered) inputs for one batch.

    Args:
        model: PitchGPTV2 in eval/train mode (called under no_grad).
        batch: collate_v2_at_bats output ALREADY z-score normalized and on
            the model's device. Modified copy is returned; original untouched.
        cfg: V2Config (for means/stds and vocab sizes).
        k: tether width — real type kept if within the model's top-K.
        p_sub: per-position substitution probability (curriculum ramp).
        generator: torch.Generator on the batch device for reproducibility.

    Returns:
        (sub_batch, stats) where sub_batch has substituted type_ids +
        continuous (targets/scaffold untouched) and stats has named numbers:
        tether_hit_rate, sub_rate, n_positions.
    """
    type_ids = batch["type_ids"].clone()          # (B, T)
    continuous = batch["continuous"].clone()      # (B, T, 4) normalized
    pad = batch["padding_mask"]                   # (B, T) bool
    B, T = type_ids.shape
    device = type_ids.device

    c_mean = torch.tensor(cfg.continuous_means, device=device)
    c_std = torch.tensor(cfg.continuous_stds, device=device)
    lo = torch.tensor(_CLAMP_LO, device=device)
    hi = torch.tensor(_CLAMP_HI, device=device)

    coin = torch.rand(B, T, device=device, generator=generator)

    n_sub = 0
    n_tether_hit = 0
    n_positions = 0

    # Position 0 is the start token (never substituted). Pitch positions are
    # 1..T-1. The prediction for position t lives at position t-1's output.
    for t in range(1, T):
        rows = pad[:, t]                          # real pitch at position t
        if not rows.any():
            break
        n_positions += int(rows.sum())
        do_sub = rows & (coin[:, t] < p_sub)
        if not do_sub.any():
            continue

        out = model(
            pitcher_profile=batch["pitcher_profile"],
            batter_profile=batch["batter_profile"],
            type_ids=type_ids,
            continuous=continuous,
            result_ids=batch["result_ids"],
            count_state=batch["count_state"],
            outs=batch["outs"],
            runners=batch["runners"],
            pitch_number=batch["pitch_number"],
            padding_mask=pad,
        )
        logits = out["type_logits"][:, t - 1, :]  # (B, 8) — predicts pitch t
        logits[:, 0] = -1e9                       # mask PAD
        kk = min(k, cfg.n_pitch_types - 1)
        topk = logits.topk(kk, dim=-1).indices    # (B, kk) values in 1..7

        real_t = type_ids[:, t]                   # current (still real) types
        in_topk = (topk == real_t.unsqueeze(1)).any(dim=1)
        # Tether: real type if in top-K, else the model's top-1 candidate.
        sub_type = torch.where(in_topk, real_t, topk[:, 0])
        new_type = torch.where(do_sub, sub_type, real_t)
        type_ids[:, t] = new_type

        n_sub += int(do_sub.sum())
        n_tether_hit += int((do_sub & in_topk).sum())

        # Continuous input for substituted positions: GMM sample conditioned
        # on the substituted type — matching rollout conditions. z-space
        # sample -> raw clamp -> back to z-space.
        hidden_t = out["hidden"][:, t - 1: t, :]              # (B, 1, d)
        log_w, mu, log_std = model.predict_continuous(
            hidden_t, new_type.unsqueeze(1))
        samp = model.gmm_head.sample(log_w, mu, log_std)[:, 0, :]  # (B, 4) z
        raw = samp * c_std + c_mean
        raw = torch.clamp(raw, lo, hi)
        samp = (raw - c_mean) / c_std
        continuous[:, t] = torch.where(
            do_sub.unsqueeze(1), samp, continuous[:, t])

    sub_batch = dict(batch)
    sub_batch["type_ids"] = type_ids
    sub_batch["continuous"] = continuous

    stats = {
        "tether_hit_rate": n_tether_hit / max(n_sub, 1),
        "sub_rate": n_sub / max(n_positions, 1),
        "n_positions": n_positions,
    }
    return sub_batch, stats


def gmm_mixweight_entropy(model: PitchGPTV2, out: dict, batch: dict) -> float:
    """Monitor for GMM component collapse (GraphCast blurring-risk analog).

    Mean entropy (nats) of the mixture weights at valid target positions.
    Uniform over K=5 components = ln(5) = 1.609; collapse drives this to 0.
    """
    with torch.no_grad():
        tt = batch["targets"]["type"].clone()
        tt[tt == -100] = 0
        log_w, _, _ = model.predict_continuous(
            out["hidden"], tt.clamp(0, model.cfg.n_pitch_types - 1))
        valid = batch["targets"]["type"] != -100
        if not valid.any():
            return float("nan")
        w = log_w[valid].exp()
        ent = -(w * w.clamp_min(1e-12).log()).sum(-1)
        return float(ent.mean())


def finetune(
    *,
    ckpt_path: Path,
    augmented_dir: Path = Path("data/augmented"),
    profiles_dir: Path = Path("data/profiles"),
    ckpt_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
    max_steps: int = 2500,
    max_pitches: int | None = None,
    batch_size: int = 256,
    lr: float = 3e-5,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    k_tether: int = 3,
    p_sub_ramp_steps: int = 1000,
    log_every: int = 25,
    eval_every: int = 250,
    seed: int = 42,
    top1_guard: float = 0.46,
    device: str | None = None,
) -> dict:
    """Fine-tune a trained V2 checkpoint with tethered self-inputs."""
    torch.manual_seed(seed)
    device = torch.device(device) if device else select_device()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid_fields = {f.name for f in dataclasses.fields(V2Config)}
    cfg = V2Config(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
    model = PitchGPTV2(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    fold_id = int(ckpt.get("fold_id", 0))
    size = ckpt.get("size", "tiny")

    run_name = run_name or f"{size}-v1c-cl"
    run_dir = ckpt_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    def log_event(event: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    log_event({"event": "init", "run_name": run_name, "device": str(device),
               "base_ckpt": str(ckpt_path), "k_tether": k_tether,
               "max_steps": max_steps, "lr": lr,
               "p_sub_ramp_steps": p_sub_ramp_steps})
    print(f"v1c-cl fine-tune: {ckpt_path} ({model.num_parameters():,} params) "
          f"on {device}, K={k_tether}, {max_steps} steps")

    # --- Data (identical recipe to scripts.train_v2) ---
    pitches_all = load_augmented_pitches(augmented_dir)
    splits = split_augmented(pitches_all)
    train_pitches, val_pitches = splits["train"], splits["val"]
    if max_pitches is not None:
        train_pitches = train_pitches.head(max_pitches)
        val_pitches = val_pitches.head(min(max_pitches // 4, len(val_pitches)))

    pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=profiles_dir)
    pc_b = ProfileCache(role="batter", fold_id=fold_id, profiles_dir=profiles_dir)
    standardizer = None
    candidate = DEFAULT_PROFILE_STD_PATH
    if not candidate.exists():
        candidate = augmented_dir.parent / "preprocess_artifacts" / "v1" / "profile_standardization.npz"
    if candidate.exists():
        standardizer = ProfileStandardizer(candidate)

    train_ds = V2AtBatDataset(
        pitches=train_pitches, pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup, profile_standardizer=standardizer)
    val_ds = V2AtBatDataset(
        pitches=val_pitches, pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup, profile_standardizer=standardizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_v2_at_bats,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_v2_at_bats)
    log_event({"event": "datasets_built", "n_train_ab": len(train_ds),
               "n_val_ab": len(val_ds)})

    # --- Baseline eval (the top-1 guard reference) ---
    base_eval = evaluate(model, val_loader, cfg, device=device)
    print(f"baseline teacher-forced: top1={base_eval['type_top1']:.4f} "
          f"loss={base_eval['loss_total']:.4f}")
    log_event({"event": "baseline_eval", **base_eval})

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                              weight_decay=weight_decay)
    gen = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    gen.manual_seed(seed + 1)
    if device.type == "mps":
        gen = None  # MPS generator quirks; fall back to global seeding

    c_mean = torch.tensor(cfg.continuous_means, device=device)
    c_std = torch.tensor(cfg.continuous_stds, device=device)

    step = 0
    best_val = base_eval["loss_total"]
    t0 = time.time()
    done = False
    while not done:
        for batch in train_loader:
            if step >= max_steps:
                done = True
                break
            batch = move_to_device(batch, device)
            # Normalize continuous inputs + targets exactly as base training.
            batch["continuous"] = (batch["continuous"] - c_mean) / c_std
            tc = batch["targets"]["continuous"]
            fm = torch.isfinite(tc)
            batch["targets"]["continuous"] = torch.where(fm, (tc - c_mean) / c_std, tc)

            p_sub = min(1.0, step / max(p_sub_ramp_steps, 1))

            model.eval()
            sub_batch, sub_stats = build_substituted_batch(
                model, batch, cfg, k=k_tether, p_sub=p_sub, generator=gen)

            model.train()
            out = model(
                pitcher_profile=sub_batch["pitcher_profile"],
                batter_profile=sub_batch["batter_profile"],
                type_ids=sub_batch["type_ids"],
                continuous=sub_batch["continuous"],
                result_ids=sub_batch["result_ids"],
                count_state=sub_batch["count_state"],
                outs=sub_batch["outs"],
                runners=sub_batch["runners"],
                pitch_number=sub_batch["pitch_number"],
                padding_mask=sub_batch["padding_mask"],
            )
            loss, per_head = compute_v2_losses(model, out, sub_batch, cfg)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            if step % log_every == 0:
                ent = gmm_mixweight_entropy(model, out, sub_batch)
                el = time.time() - t0
                sps = (step + 1) / max(el, 1e-6)
                eta = (max_steps - step) / max(sps, 1e-6)
                log_event({"event": "ft_step", "step": step, "p_sub": p_sub,
                           **per_head, **sub_stats, "gmm_w_entropy": ent})
                print(f"step {step:5d}/{max_steps} loss={per_head['total']:.3f} "
                      f"type={per_head['type']:.3f} cont={per_head['continuous']:.3f} "
                      f"p_sub={p_sub:.2f} tether={sub_stats['tether_hit_rate']:.3f} "
                      f"gmmH={ent:.3f} ({sps:.2f}/s, ETA {eta/60:.0f}m)", flush=True)

            if step > 0 and step % eval_every == 0:
                ev = evaluate(model, val_loader, cfg, device=device)
                log_event({"event": "eval", "step": step, **ev})
                print(f"  [eval @{step}] top1={ev['type_top1']:.4f} "
                      f"(baseline {base_eval['type_top1']:.4f}, guard {top1_guard}) "
                      f"loss={ev['loss_total']:.4f}", flush=True)
                if ev["type_top1"] < top1_guard:
                    log_event({"event": "abort_top1_guard", "step": step,
                               "type_top1": ev["type_top1"]})
                    print(f"ABORT: top-1 {ev['type_top1']:.4f} < guard {top1_guard}")
                    done = True
                    break
                if ev["loss_total"] < best_val:
                    best_val = ev["loss_total"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": asdict(cfg), "size": size, "fold_id": fold_id,
                        "step": step, "val_loss": ev["loss_total"],
                        "schema_version": 2, "finetune": "v1c-cl",
                        "base_ckpt": str(ckpt_path),
                    }, run_dir / "checkpoint_best.pt")
                    log_event({"event": "best_checkpoint", "step": step,
                               "val_loss": ev["loss_total"]})
            step += 1

    final_eval = evaluate(model, val_loader, cfg, device=device, max_batches=256)
    log_event({"event": "final_eval", "step": step, **final_eval})
    ckpt_out = run_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg), "size": size, "fold_id": fold_id,
        "step": step, "final_eval": final_eval, "schema_version": 2,
        "finetune": "v1c-cl", "base_ckpt": str(ckpt_path),
    }, ckpt_out)
    log_event({"event": "checkpoint_saved", "path": str(ckpt_out)})

    summary = {
        "run_name": run_name, "steps": step,
        "baseline_top1": base_eval["type_top1"],
        "final_eval": final_eval,
        "wallclock_s": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_out),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="V2 closed-loop fine-tune (v1c-cl)")
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints_modal/tiny-v1c-base/checkpoint.pt"))
    p.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    p.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--max-pitches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--k-tether", type=int, default=3)
    p.add_argument("--p-sub-ramp-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="override device (e.g. cpu — MPS train-mode forward "
                        "produces NaN; real runs go through Modal CUDA)")
    args = p.parse_args()
    finetune(
        ckpt_path=args.ckpt, augmented_dir=args.augmented_dir,
        profiles_dir=args.profiles_dir, ckpt_dir=args.ckpt_dir,
        run_name=args.run_name, max_steps=args.max_steps,
        max_pitches=args.max_pitches, batch_size=args.batch_size,
        lr=args.lr, k_tether=args.k_tether,
        p_sub_ramp_steps=args.p_sub_ramp_steps,
        eval_every=args.eval_every, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
