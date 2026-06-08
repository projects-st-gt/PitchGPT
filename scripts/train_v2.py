"""PitchGPTV2 training entry point.

Training loop for the v2 model (base-v1c). Two loss terms:
1. Type cross-entropy: 8-class logits (PAD at 0, types 1..7) with
   ignore_index=-100 for padding positions.
2. Continuous GMM NLL: mixture-of-Gaussians NLL on the 4 continuous
   features (velo, spin, plate_x, plate_z), conditioned on the REAL
   next pitch type (teacher forcing).

No zone/velo/spin/result/AB-outcome losses — the GMM replaces the
factored propensity heads from V1.

adaLN conditioning replaces context tokens, so there is no N_CONTEXT_TOKENS
offset to manage.

Run locally::

    python -m scripts.train_v2 --size tiny --fold 0 --max-pitches 250000 --max-steps 100

Modal dispatch can wrap ``train()`` the same way ``modal_app.py`` does for V1.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import N_PITCH_TYPES
from data.profile_cache_loader import ProfileCache
from model.v2.config import V2Config, tiny_v2_config, small_v2_config
from model.v2.model import PitchGPTV2
from model.v2.dataset import V2AtBatDataset, collate_v2_at_bats
from model.pitchgpt_dataset import (
    DEFAULT_PROFILE_STD_PATH,
    ProfileStandardizer,
    load_augmented_pitches,
    split_augmented,
)

SIZE_FACTORIES = {
    "tiny": tiny_v2_config,
    "small": small_v2_config,
}

CKPT_ROOT = Path("checkpoints")


# ============================================================
# Device selection
# ============================================================


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # MPS autocast (fp16/bf16) currently triggers dtype-mismatch errors during
    # backward on torch<=2.5; stick to fp32 locally and rely on Modal A100 for
    # the bf16 speedup.
    return torch.float32


# ============================================================
# LR schedule
# ============================================================


def cosine_with_warmup(
    step: int, *, warmup: int, max_steps: int, lr_max: float, lr_min: float
) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


# ============================================================
# Noise injection (input perturbation for rollout robustness)
# ============================================================


def build_empirical_type_dist(pitches: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per-count-state empirical type distribution from training data.

    Returns {count_state_id -> float32[n_pitch_types]} normalized probabilities.
    Used by noise injection to sample plausible replacement types at each count.
    """
    counts = np.zeros((12, N_PITCH_TYPES), dtype=np.float64)
    for cs, grp in pitches.groupby("count_state"):
        cs = int(cs)
        if not (0 <= cs <= 11):
            continue
        vc = grp["type_id"].value_counts()
        for tid, cnt in vc.items():
            tid = int(tid)
            if 1 <= tid <= N_PITCH_TYPES:
                counts[cs, tid - 1] += int(cnt)
    result = {}
    for cs in range(12):
        s = counts[cs].sum()
        if s > 0:
            result[cs] = (counts[cs] / s).astype(np.float32)
        else:
            result[cs] = np.ones(N_PITCH_TYPES, dtype=np.float32) / N_PITCH_TYPES
    return result


def perturb_v2_input_types(
    type_ids: torch.Tensor,
    count_state: torch.Tensor,
    padding_mask: torch.Tensor,
    p_corrupt: float,
    empirical_type_dist: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Replace a fraction of input type tokens with count-conditional empirical samples.

    Modifies type_ids in-place and returns it. For each real (non-padded)
    position, with probability p_corrupt, replaces the type token with a
    sample drawn from the empirical type distribution at that position's
    count state.
    """
    B, T = padding_mask.shape
    coin = torch.from_numpy(rng.random((B, T)).astype(np.float32))
    corrupt_mask = (coin < p_corrupt) & padding_mask.cpu()

    if not corrupt_mask.any():
        return type_ids

    for b in range(B):
        for t in range(T):
            if not corrupt_mask[b, t]:
                continue
            cs = int(count_state[b, t].item())
            if cs not in empirical_type_dist:
                continue
            td = empirical_type_dist[cs]
            sampled_type = int(rng.choice(len(td), p=td)) + 1  # +1 for 1-indexed
            type_ids[b, t] = sampled_type

    return type_ids


# ============================================================
# Loss computation
# ============================================================


def compute_v2_losses(
    model: PitchGPTV2,
    out: dict,
    batch: dict,
    cfg: V2Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Two losses: type cross-entropy + continuous GMM NLL.

    Type CE:
        type_logits shape: (B, T, 8) -- PAD at index 0, types at 1..7
        targets.type: (B, T) -- values 1..7 for real types, -100 for PAD
        Use F.cross_entropy with ignore_index=-100.

    Continuous GMM NLL:
        Call model.predict_continuous(hidden, teacher_forced_type) where
        teacher_forced_type = targets.type (the REAL next pitch type).
        Only compute NLL where targets.type != -100 AND targets.continuous
        is finite.

    total = cfg.w_type * type_loss + cfg.w_continuous * continuous_loss
    """
    type_logits = out["type_logits"]  # (B, T, 8)
    hidden = out["hidden"]            # (B, T, d_model)

    type_targets = batch["targets"]["type"]          # (B, T) -- 1..7 or -100
    cont_targets = batch["targets"]["continuous"]    # (B, T, 4) -- NaN where masked

    # --- Type CE ---
    type_loss = F.cross_entropy(
        type_logits.reshape(-1, cfg.n_pitch_types),  # n_pitch_types = 8 (includes PAD)
        type_targets.reshape(-1),
        ignore_index=-100,
        label_smoothing=cfg.label_smoothing,
    )

    # --- Continuous GMM NLL ---
    # Teacher forcing: condition the GMM on the REAL next pitch type.
    # Build a version of type_targets where -100 is replaced with 0 (PAD)
    # so the embedding lookup doesn't crash. We mask out those positions
    # in the NLL anyway.
    teacher_type = type_targets.clone()
    teacher_type[teacher_type == -100] = 0  # PAD idx for invalid positions
    log_w, mu, log_std = model.predict_continuous(hidden, teacher_type)

    # Valid mask: type target is real AND all 4 continuous targets are finite.
    valid = (type_targets != -100) & torch.isfinite(cont_targets).all(dim=-1)  # (B, T)

    if valid.any():
        log_w_v = log_w[valid]        # (N, K)
        mu_v = mu[valid]              # (N, K, D)
        log_std_v = log_std[valid]    # (N, K, D)
        cont_v = cont_targets[valid]  # (N, D)

        continuous_loss = model.gmm_head.nll(log_w_v, mu_v, log_std_v, cont_v)
    else:
        continuous_loss = torch.tensor(0.0, device=type_logits.device)

    total = cfg.w_type * type_loss + cfg.w_continuous * continuous_loss

    return total, {
        "type": float(type_loss.detach()),
        "continuous": float(continuous_loss.detach()),
        "total": float(total.detach()),
    }


# ============================================================
# Eval
# ============================================================


@torch.no_grad()
def evaluate(
    model: PitchGPTV2,
    loader: DataLoader,
    cfg: V2Config,
    *,
    device: torch.device,
    max_batches: int = 64,
) -> dict[str, float]:
    """Cheap eval pass: average losses and top-1 type accuracy."""
    model.eval()
    loss_sums = {"type": 0.0, "continuous": 0.0, "total": 0.0}
    n_batches = 0
    type_correct = 0
    type_total = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_to_device(batch, device)
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
        _, log = compute_v2_losses(model, out, batch, cfg)
        for k in loss_sums:
            loss_sums[k] += log[k]
        n_batches += 1

        # Type top-1 accuracy on non-PAD positions.
        # type_logits has 8 outputs (PAD at 0, types at 1..7).
        # targets are 1..7 for real, -100 for PAD.
        type_logits = out["type_logits"]  # (B, T, 8)
        type_target = batch["targets"]["type"]  # (B, T)
        valid = type_target != -100
        if valid.any():
            preds = type_logits.argmax(dim=-1)  # (B, T)
            type_correct += int((preds[valid] == type_target[valid]).sum())
            type_total += int(valid.sum())

    out_metrics = {f"loss_{k}": v / max(n_batches, 1) for k, v in loss_sums.items()}
    out_metrics["type_top1"] = type_correct / max(type_total, 1)
    return out_metrics


# ============================================================
# Device-moving helper
# ============================================================


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move tensors in the batch dict to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


# ============================================================
# Training function — reusable by local + Modal
# ============================================================


def train(
    *,
    augmented_dir: Path = Path("data/augmented"),
    profiles_dir: Path = Path("data/profiles"),
    fold_id: int = 0,
    size: str = "tiny",
    epochs: int = 3,
    max_steps: int | None = None,
    max_pitches: int | None = None,
    batch_size: int = 256,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    grad_clip: float = 1.0,
    log_every: int = 50,
    eval_every: int = 1000,
    ckpt_dir: Path = CKPT_ROOT,
    run_name: str | None = None,
    seed: int = 42,
    noise_p: float = 0.0,
    noise_ramp_steps: int = 3000,
) -> dict:
    """Run a single training pass; return summary dict.

    All paths are explicit so the same function can be called from Modal with
    paths pointing at a mounted Volume.
    """
    torch.manual_seed(seed)
    device = select_device()
    cfg = SIZE_FACTORIES[size]()

    run_name = run_name or f"v2-{size}-fold{fold_id}-{int(time.time())}"
    run_dir = ckpt_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    def log_event(event: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    log_event({
        "event": "init", "run_name": run_name, "device": str(device),
        "size": size, "fold_id": fold_id,
        "config": {k: v for k, v in asdict(cfg).items()},
    })

    # --- Load data ---
    t0 = time.time()
    pitches_all = load_augmented_pitches(augmented_dir)
    splits = split_augmented(pitches_all)
    train_pitches = splits["train"]
    val_pitches = splits["val"]
    if max_pitches is not None:
        train_pitches = train_pitches.head(max_pitches)
        val_pitches = val_pitches.head(min(max_pitches // 4, len(val_pitches)))
    log_event({
        "event": "data_loaded",
        "load_seconds": round(time.time() - t0, 2),
        "n_train_pitches": int(len(train_pitches)),
        "n_val_pitches": int(len(val_pitches)),
    })

    pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=profiles_dir)
    pc_b = ProfileCache(role="batter", fold_id=fold_id, profiles_dir=profiles_dir)

    # Probe actual profile dimensions from the data and override config
    # defaults. The config ships with placeholder dims (223/91) that may not
    # match the current profile cache schema.
    _probe_row = train_pitches.iloc[0]
    _probe_date = pd.Timestamp(_probe_row["game_date"])
    _probe_gn = int(_probe_row["game_num"]) if "game_num" in _probe_row.index else 1
    _pitcher_dim = len(pc_p.lookup(int(_probe_row["pitcher"]), _probe_date, _probe_gn)["vector"])
    _batter_dim = len(pc_b.lookup(int(_probe_row["batter"]), _probe_date, _probe_gn)["vector"])
    if _pitcher_dim != cfg.pitcher_profile_dim or _batter_dim != cfg.batter_profile_dim:
        print(f"Profile dims from data: pitcher={_pitcher_dim}, batter={_batter_dim} "
              f"(config had {cfg.pitcher_profile_dim}/{cfg.batter_profile_dim}) — overriding config")
        cfg.pitcher_profile_dim = _pitcher_dim
        cfg.batter_profile_dim = _batter_dim

    # Profile standardization — reuse V1's standardizer infrastructure.
    standardizer = None
    candidate = DEFAULT_PROFILE_STD_PATH
    if not candidate.exists():
        candidate = augmented_dir.parent / "preprocess_artifacts" / "v1" / "profile_standardization.npz"
    if candidate.exists():
        standardizer = ProfileStandardizer(candidate)
        log_event({"event": "profile_standardizer_loaded", "path": str(candidate)})

    train_ds = V2AtBatDataset(
        pitches=train_pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        profile_standardizer=standardizer,
    )
    val_ds = V2AtBatDataset(
        pitches=val_pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        profile_standardizer=standardizer,
    ) if len(val_pitches) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_v2_at_bats,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_v2_at_bats, drop_last=False,
        ) if val_ds is not None else None
    )

    log_event({
        "event": "datasets_built",
        "n_train_ab": len(train_ds),
        "n_val_ab": len(val_ds) if val_ds is not None else 0,
        "steps_per_epoch": len(train_loader),
    })

    # --- Build model + optimizer ---
    model = PitchGPTV2(cfg).to(device)
    n_params = model.num_parameters()
    log_event({"event": "model_built", "n_params": n_params})
    print(f"PitchGPTV2 ({size}) — {n_params:,} parameters on {device}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=weight_decay,
    )

    total_steps = max_steps if max_steps is not None else epochs * len(train_loader)
    amp_dtype = autocast_dtype(device)
    use_amp = amp_dtype != torch.float32 and device.type in ("cuda", "mps")

    # --- Noise injection setup ---
    noise_rng: Optional[np.random.Generator] = None
    empirical_type_dist: Optional[dict] = None
    if noise_p > 0:
        noise_rng = np.random.default_rng(seed + 7)
        empirical_type_dist = build_empirical_type_dist(train_pitches)
        log_event({
            "event": "noise_injection_enabled",
            "noise_p": noise_p,
            "noise_ramp_steps": noise_ramp_steps,
        })
        print(f"Noise injection ON: p={noise_p}, ramp={noise_ramp_steps} steps")

    # --- Training loop ---
    step = 0
    t_train_start = time.time()
    best_val_loss = float("inf")
    best_ckpt_path: Optional[Path] = None

    for epoch in range(epochs):
        for batch in train_loader:
            if max_steps is not None and step >= max_steps:
                break
            batch = move_to_device(batch, device)

            # Noise injection: perturb input type tokens.
            if noise_p > 0 and noise_rng is not None and empirical_type_dist is not None:
                current_p = min(noise_p, noise_p * step / max(noise_ramp_steps, 1))
                if current_p > 0:
                    perturb_v2_input_types(
                        batch["type_ids"],
                        batch["count_state"],
                        batch["padding_mask"],
                        p_corrupt=current_p,
                        empirical_type_dist=empirical_type_dist,
                        rng=noise_rng,
                    )

            lr = cosine_with_warmup(
                step, warmup=warmup_steps, max_steps=total_steps,
                lr_max=lr_max, lr_min=lr_min,
            )
            set_lr(optim, lr)

            model.train()
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp,
            ):
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
                loss, per_head = compute_v2_losses(model, out, batch, cfg)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            if step % log_every == 0:
                elapsed = time.time() - t_train_start
                steps_per_s = (step + 1) / max(elapsed, 1e-6)
                log_event({
                    "event": "train_step", "step": step, "epoch": epoch, "lr": lr,
                    **per_head, "steps_per_s": round(steps_per_s, 2),
                })
                print(
                    f"step {step:6d} epoch {epoch} "
                    f"loss={per_head['total']:.3f} "
                    f"type={per_head['type']:.3f} "
                    f"cont={per_head['continuous']:.3f} "
                    f"lr={lr:.2e} speed={steps_per_s:.1f}/s"
                )

            if val_loader is not None and step > 0 and step % eval_every == 0:
                eval_metrics = evaluate(model, val_loader, cfg, device=device)
                log_event({"event": "eval", "step": step, **eval_metrics})
                print(
                    f"  [eval @ {step}] type_top1={eval_metrics['type_top1']:.3f} "
                    f"val_loss={eval_metrics['loss_total']:.3f} "
                    f"type_loss={eval_metrics['loss_type']:.3f} "
                    f"cont_loss={eval_metrics['loss_continuous']:.3f}"
                )
                val_loss = eval_metrics["loss_total"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = run_dir / "checkpoint_best.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "config": asdict(cfg),
                            "size": size,
                            "fold_id": fold_id,
                            "step": step,
                            "val_loss": val_loss,
                            "schema_version": 2,
                        },
                        best_ckpt_path,
                    )
                    log_event({"event": "best_checkpoint", "step": step, "val_loss": val_loss})

            step += 1

        if max_steps is not None and step >= max_steps:
            break

    # --- Final eval + checkpoint ---
    final_eval = (
        evaluate(model, val_loader, cfg, device=device, max_batches=256)
        if val_loader is not None
        else {}
    )
    log_event({"event": "final_eval", "step": step, **final_eval})

    ckpt_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "config": asdict(cfg),
            "size": size,
            "fold_id": fold_id,
            "step": step,
            "final_eval": final_eval,
            "schema_version": 2,
        },
        ckpt_path,
    )
    log_event({"event": "checkpoint_saved", "path": str(ckpt_path)})

    summary = {
        "run_name": run_name,
        "steps_completed": step,
        "wallclock_s": round(time.time() - t_train_start, 2),
        "final_eval": final_eval,
        "checkpoint": str(ckpt_path),
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path else None,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
    }
    print(json.dumps(summary, indent=2))
    return summary


# ============================================================
# CLI
# ============================================================


def main() -> None:
    p = argparse.ArgumentParser(description="PitchGPTV2 training (local)")
    p.add_argument("--size", choices=list(SIZE_FACTORIES.keys()), default="tiny")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-pitches", type=int, default=None, help="cap training pitches (smoke mode)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    p.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    p.add_argument("--ckpt-dir", type=Path, default=CKPT_ROOT)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-p", type=float, default=0.0)
    p.add_argument("--noise-ramp-steps", type=int, default=3000)
    args = p.parse_args()

    train(
        augmented_dir=args.augmented_dir,
        profiles_dir=args.profiles_dir,
        fold_id=args.fold,
        size=args.size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        max_pitches=args.max_pitches,
        batch_size=args.batch_size,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ckpt_dir=args.ckpt_dir,
        run_name=args.run_name,
        seed=args.seed,
        noise_p=args.noise_p,
        noise_ramp_steps=args.noise_ramp_steps,
    )


if __name__ == "__main__":
    main()
