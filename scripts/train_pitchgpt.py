"""PitchGPT training entry point.

Single training run for a given model size and fold. The ``train()`` function
is the actual training loop and is callable from both the local CLI here and
from ``modal_app.py`` (which wraps it for remote A100 execution).

Defaults match the ``pitchgpt-model`` skill:
- AdamW, β=(0.9, 0.95), weight_decay=0.1
- lr=3e-4, cosine decay to 3e-5, linear warmup 2000 steps
- batch_size=256 at-bats, grad clip 1.0, dropout 0.1
- per-head loss weights from :attr:`PitchGPTConfig.head_weights`
- bf16 mixed precision (where supported) with fp32 master weights via AdamW

Run locally with::

    python -m scripts.train_pitchgpt \
        --size tiny --fold 0 --max-pitches 250000 --max-steps 100

Phase B is::

    python -m scripts.train_pitchgpt --size tiny --fold 0 --epochs 3

Modal dispatch lives in ``modal_app.py``; this script is local-only.
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

from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig, sanity_config, small_config, tiny_config
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    DEFAULT_PROFILE_STD_PATH,
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
    load_augmented_pitches,
    split_augmented,
)

SIZE_FACTORIES = {
    "sanity": sanity_config,
    "tiny": tiny_config,
    "small": small_config,
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
    # backward on torch≤2.5; stick to fp32 locally and rely on Modal A100 for
    # the bf16 speedup.
    return torch.float32


# ============================================================
# LR schedule
# ============================================================


def cosine_with_warmup(step: int, *, warmup: int, max_steps: int, lr_max: float, lr_min: float) -> float:
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
# Loss computation
# ============================================================


def compute_losses(
    out: dict,
    batch: dict,
    cfg: PitchGPTConfig,
    *,
    n_context_tokens: int,
    zone_centers: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of per-head cross-entropies, weighted by cfg.head_weights.

    Returns the scalar joint loss plus a dict of detached per-head losses for
    logging. The propensity logits are sliced to drop context-token positions
    before matching to targets.

    When ``cfg.zone_spatial_weight > 0`` and ``zone_centers`` is provided,
    an auxiliary EMD-style loss is added on the zone head: the squared
    distance between ``E_p[centroid]`` and the true zone's centroid, in feet.
    """
    w = cfg.head_weights

    # Propensity heads — drop context positions to align with per-pitch targets.
    type_logits = out["propensity"]["type"][:, n_context_tokens:, :]
    zone_logits = out["propensity"]["zone"][:, n_context_tokens:, :]
    velo_logits = out["propensity"]["velo"][:, n_context_tokens:, :]
    spin_rate_logits = out["propensity"]["spin_rate"][:, n_context_tokens:, :]

    if cfg.type_focal_gamma > 0.0:
        # Focal loss on type head: (1 - p_true)^gamma * CE, downweights examples
        # the model is already confident about. Helps when class is imbalanced
        # toward one dominant pitch type (FF). Per Lin et al 2017 "Focal Loss".
        # Class weights (1/freq^class_weight_alpha) added if class_weight_alpha > 0.
        type_logits_flat = type_logits.reshape(-1, cfg.n_pitch_types)
        type_target_flat = batch["targets"]["propensity"]["type"].reshape(-1)
        valid_mask = type_target_flat != -100
        if valid_mask.any():
            logits_v = type_logits_flat[valid_mask]
            target_v = type_target_flat[valid_mask]
            log_probs = F.log_softmax(logits_v, dim=-1)
            log_p_true = log_probs.gather(1, target_v.unsqueeze(-1)).squeeze(-1)
            p_true = log_p_true.exp()
            focal_w = (1.0 - p_true).pow(cfg.type_focal_gamma)
            # Optional class weighting (1/freq^alpha)
            if cfg.type_class_weight_alpha > 0.0 and cfg.type_class_freq is not None:
                freq = torch.as_tensor(cfg.type_class_freq, device=logits_v.device, dtype=logits_v.dtype)
                cls_w = (1.0 / (freq + 1e-6)).pow(cfg.type_class_weight_alpha)
                # Normalize so mean weight ≈ 1 (preserves overall loss scale)
                cls_w = cls_w * (freq.sum() / (cls_w * freq).sum())
                sample_w = cls_w[target_v]
            else:
                sample_w = torch.ones_like(p_true)
            type_loss = -(focal_w * sample_w * log_p_true).mean()
        else:
            type_loss = torch.tensor(0.0, device=type_logits.device)
    else:
        type_loss = F.cross_entropy(
            type_logits.reshape(-1, cfg.n_pitch_types),
            batch["targets"]["propensity"]["type"].reshape(-1),
            ignore_index=-100,
            label_smoothing=cfg.label_smoothing_type,
        )
    zone_loss = F.cross_entropy(
        zone_logits.reshape(-1, cfg.n_zones),
        batch["targets"]["propensity"]["zone"].reshape(-1),
        ignore_index=-100,
    )

    # Auxiliary EMD-style spatial loss on the zone head (see config docstring).
    # Skipped if disabled (weight=0) OR if centroids weren't provided.
    if cfg.zone_spatial_weight > 0.0 and zone_centers is not None:
        zone_target_flat = batch["targets"]["propensity"]["zone"].reshape(-1)
        zone_logits_flat = zone_logits.reshape(-1, cfg.n_zones)
        valid_z = zone_target_flat != -100
        if valid_z.any():
            zl = zone_logits_flat[valid_z]                 # (N, 13)
            zt = zone_target_flat[valid_z]                 # (N,)
            probs = torch.softmax(zl, dim=-1)              # (N, 13)
            pred_c = probs @ zone_centers                  # (N, 2)
            true_c = zone_centers[zt]                      # (N, 2)
            zone_spatial_loss = ((pred_c - true_c) ** 2).sum(dim=-1).mean()
        else:
            zone_spatial_loss = torch.tensor(0.0, device=type_loss.device)
    else:
        zone_spatial_loss = torch.tensor(0.0, device=type_loss.device)
    velo_loss = F.cross_entropy(
        velo_logits.reshape(-1, cfg.n_velo_bins),
        batch["targets"]["propensity"]["velo"].reshape(-1),
        ignore_index=-100,
    )
    spin_rate_loss = F.cross_entropy(
        spin_rate_logits.reshape(-1, cfg.n_spin_rate_bins),
        batch["targets"]["propensity"]["spin_rate"].reshape(-1),
        ignore_index=-100,
    )

    # Spin axis: 3 outputs from the head (mean_sin, mean_cos, log_kappa) for
    # the circular variant. For Phase B we skip the von-Mises NLL and just
    # treat it as a 2D regression on (sin, cos) of the NEXT pitch's axis,
    # using mean-squared error. Promote to von-Mises NLL in Phase E.
    spin_axis_out = out["propensity"]["spin_axis"][:, n_context_tokens:, :2]  # (B, T, 2)
    next_spin_axis = torch.zeros_like(spin_axis_out)
    next_spin_axis[:, :-1, :] = batch["pitch_factors"]["spin_axis"][:, 1:, :]
    spin_axis_valid = batch["padding_mask"].clone()
    spin_axis_valid[:, -1] = False  # last position has no successor
    spin_axis_loss = (
        ((spin_axis_out - next_spin_axis) ** 2).sum(dim=-1)[spin_axis_valid].mean()
        if spin_axis_valid.any()
        else torch.tensor(0.0, device=type_loss.device)
    )

    # Result head
    result_loss = F.cross_entropy(
        out["result"].reshape(-1, cfg.n_result_logits),
        batch["targets"]["result"].reshape(-1),
        ignore_index=-100,
    )

    # AB outcome — select terminal-pitch prediction per row.
    valid_lengths = batch["padding_mask"].sum(dim=1)  # (B,)
    has_pitches = valid_lengths > 0
    if has_pitches.any():
        terminal_pos = (valid_lengths - 1).clamp(min=0)
        idx_batch = torch.arange(out["ab_outcome_per_pos"].shape[0], device=type_loss.device)
        ab_logits = out["ab_outcome_per_pos"][idx_batch, terminal_pos, :]
        ab_loss = F.cross_entropy(
            ab_logits[has_pitches],
            batch["targets"]["ab_outcome"][has_pitches],
            ignore_index=-100,
        )
    else:
        ab_loss = torch.tensor(0.0, device=type_loss.device)

    total = (
        w["type"] * type_loss
        + w["zone"] * zone_loss
        + w["velo"] * velo_loss
        + w["spin_rate"] * spin_rate_loss
        + w["spin_axis"] * spin_axis_loss
        + w["result"] * result_loss
        + w["ab_outcome"] * ab_loss
        + cfg.zone_spatial_weight * zone_spatial_loss
    )

    return total, {
        "type": float(type_loss.detach()),
        "zone": float(zone_loss.detach()),
        "zone_spatial": float(zone_spatial_loss.detach()),
        "velo": float(velo_loss.detach()),
        "spin_rate": float(spin_rate_loss.detach()),
        "spin_axis": float(spin_axis_loss.detach()),
        "result": float(result_loss.detach()),
        "ab_outcome": float(ab_loss.detach()),
        "total": float(total.detach()),
    }


# ============================================================
# Eval helpers — top-1 accuracy per head on a slice
# ============================================================


@torch.no_grad()
def evaluate(
    model: PitchGPT,
    loader: DataLoader,
    cfg: PitchGPTConfig,
    *,
    device: torch.device,
    max_batches: int = 64,
    zone_centers: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Cheap eval pass: per-head average loss and top-1 accuracy on N batches."""
    model.eval()
    head_losses_sum = {k: 0.0 for k in (
        "type", "zone", "zone_spatial", "velo", "spin_rate", "spin_axis",
        "result", "ab_outcome", "total",
    )}
    n_batches = 0
    type_correct = 0
    type_total = 0
    result_correct = 0
    result_total = 0
    zone_correct = 0
    zone_top3_correct = 0
    zone_total = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model(
            pitcher_profile=batch["pitcher_profile"],
            batter_profile=batch["batter_profile"],
            categorical_context=batch["categorical_context"],
            pitch_factors=batch["pitch_factors"],
            intended_actions=batch["intended_actions"],
            padding_mask=batch["padding_mask"],
            arsenal=batch.get("arsenal"),
        )
        _, log = compute_losses(out, batch, cfg, n_context_tokens=PitchGPT.N_CONTEXT_TOKENS, zone_centers=zone_centers)
        for k in head_losses_sum:
            head_losses_sum[k] += log[k]
        n_batches += 1

        # Type accuracy at pitch positions
        type_logits = out["propensity"]["type"][:, PitchGPT.N_CONTEXT_TOKENS:, :]
        type_target = batch["targets"]["propensity"]["type"]
        valid = type_target != -100
        if valid.any():
            preds = type_logits.argmax(dim=-1)
            type_correct += int((preds[valid] == type_target[valid]).sum())
            type_total += int(valid.sum())

        # Zone accuracy (top-1 and top-3). The zone head has cfg.n_zones outputs
        # with no PAD offset, so the slice is straightforward — target indices
        # are dense 0..n_zones-1 (under v5 SIS 14-zone, 0..12).
        zone_logits = out["propensity"]["zone"][:, PitchGPT.N_CONTEXT_TOKENS:, :]
        zone_target = batch["targets"]["propensity"]["zone"]
        valid_z = zone_target != -100
        if valid_z.any():
            zlogits_v = zone_logits[valid_z]
            ztarget_v = zone_target[valid_z]
            preds = zlogits_v.argmax(dim=-1)
            zone_correct += int((preds == ztarget_v).sum())
            k = min(3, zlogits_v.shape[-1])
            top3 = zlogits_v.topk(k, dim=-1).indices
            zone_top3_correct += int((top3 == ztarget_v.unsqueeze(-1)).any(dim=-1).sum())
            zone_total += int(valid_z.sum())

        # Result accuracy
        result_target = batch["targets"]["result"]
        valid_r = result_target != -100
        if valid_r.any():
            preds = out["result"].argmax(dim=-1)
            result_correct += int((preds[valid_r] == result_target[valid_r]).sum())
            result_total += int(valid_r.sum())

    out_metrics = {f"loss_{k}": v / max(n_batches, 1) for k, v in head_losses_sum.items()}
    out_metrics["type_top1"] = type_correct / max(type_total, 1)
    out_metrics["zone_top1"] = zone_correct / max(zone_total, 1)
    out_metrics["zone_top3"] = zone_top3_correct / max(zone_total, 1)
    out_metrics["result_top1"] = result_correct / max(result_total, 1)
    return out_metrics


# ============================================================
# Device-moving helper
# ============================================================


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move tensors in the batch dict to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch


# ============================================================
# The training function — reusable by local + Modal
# ============================================================


def train(
    *,
    augmented_dir: Path,
    profiles_dir: Path,
    fold_id: int,
    size: str = "tiny",
    epochs: int = 3,
    max_steps: Optional[int] = None,
    max_pitches: Optional[int] = None,
    batch_size: int = 256,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    grad_clip: float = 1.0,
    log_every: int = 50,
    eval_every: int = 1000,
    ckpt_dir: Path = CKPT_ROOT,
    run_name: Optional[str] = None,
    seed: int = 42,
    early_stop_patience: int = 0,  # number of evals without improvement before stopping; 0 disables
    early_stop_min_delta: float = 1e-3,
    standardize_profiles: bool = True,  # z-score profile vectors before the context-token MLP
    profile_std_path: Path = DEFAULT_PROFILE_STD_PATH,
    arsenal_per_pitch: bool = True,  # ADR 009 — add the projected arsenal vector to every pitch token
    propensity_situational: bool = True,  # ADR 010 — fuse (count,runners,outs)[t+1] before the propensity heads
    concat_then_project: bool = False,  # ADR 011 ("fix #1") — concat+project the 11 factor embeddings instead of summing
    profile_film: bool = False,  # ADR 012 ("fix #2") — FiLM-condition the trunk on the player profile
    zone_spatial_weight: float = 0.0,  # v5 14-zone — EMD aux loss coeff on zone head (0 = off)
    type_focal_gamma: float = 0.0,     # focal loss on type head (0 = CE)
    type_class_weight_alpha: float = 0.0,  # inverse-freq class weighting (0 = uniform)
    type_conditioned_heads: bool = False,  # ADR 013 — type-condition the execution/result heads
) -> dict:
    """Run a single training pass; return summary dict.

    All paths are explicit so the same function can be called from Modal with
    paths pointing at a mounted Volume.
    """
    torch.manual_seed(seed)
    device = select_device()
    cfg = SIZE_FACTORIES[size]()
    cfg.arsenal_per_pitch = arsenal_per_pitch  # ADR 009 — overrides the (backward-compat) config default
    cfg.propensity_situational = propensity_situational  # ADR 010
    cfg.concat_then_project = concat_then_project  # ADR 011
    cfg.profile_film = profile_film  # ADR 012
    cfg.zone_spatial_weight = zone_spatial_weight  # v5 14-zone spatial aux loss
    cfg.type_focal_gamma = type_focal_gamma
    cfg.type_class_weight_alpha = type_class_weight_alpha
    cfg.type_conditioned_heads = type_conditioned_heads  # ADR 013
    run_name = run_name or f"{size}-fold{fold_id}-{int(time.time())}"
    run_dir = ckpt_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    def log_event(event: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    log_event({
        "event": "init", "run_name": run_name, "device": str(device),
        "size": size, "fold_id": fold_id,
        "config": {k: v for k, v in asdict(cfg).items() if not isinstance(v, dict)},
    })

    # --- Load data ---
    t0 = time.time()
    pitches_all = load_augmented_pitches(augmented_dir)
    splits = split_augmented(pitches_all)
    train_pitches = splits["train"]
    val_pitches = splits["val"]
    if max_pitches is not None:
        # Sanity / smoke mode — take the first N pitches by date.
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

    standardizer = None
    if standardize_profiles:
        # The profile path may be relative to a different cwd on Modal; resolve
        # against the data dir's grandparent if the default doesn't exist.
        candidate = profile_std_path
        if not candidate.exists():
            candidate = augmented_dir.parent / "preprocess_artifacts" / "v1" / "profile_standardization.npz"
        standardizer = ProfileStandardizer(candidate)
        log_event({"event": "profile_standardizer_loaded", "path": str(candidate)})

    train_ds = PitchGPTAtBatDataset(
        pitches=train_pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        profile_standardizer=standardizer,
    )
    val_ds = PitchGPTAtBatDataset(
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
        collate_fn=collate_pitchgpt_at_bats,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_pitchgpt_at_bats, drop_last=False,
        ) if val_ds is not None else None
    )

    log_event({
        "event": "datasets_built",
        "n_train_ab": len(train_ds),
        "n_val_ab": len(val_ds) if val_ds is not None else 0,
        "steps_per_epoch": len(train_loader),
    })

    # --- Type class frequencies for focal/class-balanced loss (optional) ---
    if cfg.type_focal_gamma > 0.0 or cfg.type_class_weight_alpha > 0.0:
        # Compute empirical class frequencies from the training corpus.
        # Vector aligned to model type-id indices (0 = PAD, 1..7 = canonical types).
        type_counts = train_pitches["type_id"].value_counts().sort_index()
        freq = np.zeros(cfg.n_pitch_types, dtype=np.float32)
        for idx, count in type_counts.items():
            if 0 <= int(idx) < cfg.n_pitch_types:
                freq[int(idx)] = float(count)
        freq = freq / freq.sum()
        cfg.type_class_freq = freq.tolist()
        log_event({
            "event": "type_class_freq_computed",
            "focal_gamma": cfg.type_focal_gamma,
            "class_weight_alpha": cfg.type_class_weight_alpha,
            "freq": [round(f, 4) for f in freq.tolist()],
        })

    # --- Zone centroids for the spatial aux loss (v5 14-zone, optional) ---
    zone_centers: Optional[torch.Tensor] = None
    if cfg.zone_spatial_weight > 0.0:
        # Centroids live alongside the v2 preprocess artifacts. Path is
        # `data/preprocess_artifacts/v2/zone_centroids.npy` of shape (13, 2):
        # column 0 = mean plate_x (ft), column 1 = mean plate_z (ft).
        candidate = augmented_dir.parent / "preprocess_artifacts" / "v2" / "zone_centroids.npy"
        if not candidate.exists():
            raise FileNotFoundError(
                f"zone_spatial_weight={cfg.zone_spatial_weight} but {candidate} not found; "
                f"run scripts to compute centroids first or set zone_spatial_weight=0.0"
            )
        zone_centers = torch.from_numpy(np.load(candidate)).to(device).float()
        assert zone_centers.shape == (cfg.n_zones, 2), (
            f"zone_centroids shape {tuple(zone_centers.shape)} != ({cfg.n_zones}, 2)"
        )
        log_event({
            "event": "zone_centers_loaded",
            "path": str(candidate),
            "shape": list(zone_centers.shape),
            "zone_spatial_weight": cfg.zone_spatial_weight,
        })

    # --- Build model + optimizer ---
    model = PitchGPT(cfg).to(device)
    log_event({"event": "model_built", "n_params": int(model.num_parameters())})

    optim = torch.optim.AdamW(
        model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=weight_decay
    )

    total_steps = max_steps if max_steps is not None else epochs * len(train_loader)
    amp_dtype = autocast_dtype(device)
    use_amp = amp_dtype != torch.float32 and device.type in ("cuda", "mps")

    # --- Training loop ---
    step = 0
    t_train_start = time.time()
    best_val_loss = float("inf")
    evals_since_best = 0
    best_ckpt_path: Optional[Path] = None
    stop_signal = False
    for epoch in range(epochs):
        if stop_signal:
            break
        for batch in train_loader:
            if max_steps is not None and step >= max_steps:
                break
            if stop_signal:
                break
            batch = move_batch_to_device(batch, device)

            lr = cosine_with_warmup(
                step, warmup=warmup_steps, max_steps=total_steps,
                lr_max=lr_max, lr_min=lr_min,
            )
            set_lr(optim, lr)

            model.train()
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(
                    pitcher_profile=batch["pitcher_profile"],
                    batter_profile=batch["batter_profile"],
                    categorical_context=batch["categorical_context"],
                    pitch_factors=batch["pitch_factors"],
                    intended_actions=batch["intended_actions"],
                    padding_mask=batch["padding_mask"],
                    arsenal=batch.get("arsenal"),
                )
                loss, per_head = compute_losses(
                    out, batch, cfg, n_context_tokens=PitchGPT.N_CONTEXT_TOKENS,
                    zone_centers=zone_centers,
                )
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
                    f"loss={per_head['total']:.3f} type={per_head['type']:.3f} "
                    f"zone={per_head['zone']:.3f} result={per_head['result']:.3f} "
                    f"lr={lr:.2e} speed={steps_per_s:.1f}/s"
                )

            if val_loader is not None and step > 0 and step % eval_every == 0:
                eval_metrics = evaluate(model, val_loader, cfg, device=device, zone_centers=zone_centers)
                log_event({"event": "eval", "step": step, **eval_metrics})
                print(
                    f"  [eval @ {step}] type_top1={eval_metrics['type_top1']:.3f} "
                    f"result_top1={eval_metrics['result_top1']:.3f} "
                    f"val_loss={eval_metrics['loss_total']:.3f}"
                )
                val_loss = eval_metrics["loss_total"]
                if val_loss + early_stop_min_delta < best_val_loss:
                    best_val_loss = val_loss
                    evals_since_best = 0
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
                            "schema_version": 1,
                        },
                        best_ckpt_path,
                    )
                    log_event({"event": "best_checkpoint", "step": step, "val_loss": val_loss})
                else:
                    evals_since_best += 1
                    if early_stop_patience > 0 and evals_since_best >= early_stop_patience:
                        log_event({
                            "event": "early_stop", "step": step,
                            "best_val_loss": best_val_loss,
                            "evals_since_best": evals_since_best,
                        })
                        print(f"  early-stop triggered at step {step} "
                              f"(best val_loss={best_val_loss:.3f}, "
                              f"no improvement for {evals_since_best} evals)")
                        stop_signal = True
                        break

            step += 1

        if max_steps is not None and step >= max_steps:
            break

    # --- Final eval + checkpoint ---
    final_eval = (
        evaluate(model, val_loader, cfg, device=device, max_batches=256, zone_centers=zone_centers)
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
            "schema_version": 1,
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
        "early_stopped": stop_signal,
    }
    print(json.dumps(summary, indent=2))
    return summary


# ============================================================
# CLI
# ============================================================


def main() -> None:
    p = argparse.ArgumentParser(description="PitchGPT training (local)")
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
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="early-stop after N evals without val-loss improvement; 0 disables")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    p.add_argument("--no-standardize-profiles", action="store_true",
                   help="disable per-feature z-score standardization of profile vectors")
    p.add_argument("--no-arsenal-per-pitch", action="store_true",
                   help="disable the per-pitch arsenal feature (ADR 009)")
    p.add_argument("--no-propensity-situational", action="store_true",
                   help="disable the situational two-stage propensity head (ADR 010)")
    p.add_argument("--concat-then-project", action="store_true",
                   help="concat+project the 11 per-pitch factor embeddings instead of summing (ADR 011)")
    p.add_argument("--profile-film", action="store_true",
                   help="FiLM-condition the trunk on the player profile (ADR 012)")
    p.add_argument("--zone-spatial-weight", type=float, default=0.0,
                   help="v5 14-zone EMD aux loss coeff on the zone head (0 = off)")
    p.add_argument("--type-focal-gamma", type=float, default=0.0,
                   help="focal loss gamma on the type head (0 = CE; 2 = canonical focal)")
    p.add_argument("--type-class-weight-alpha", type=float, default=0.0,
                   help="inverse-freq class weighting on type head (0 = uniform; 0.5 = mild; 1.0 = full balance)")
    p.add_argument("--type-conditioned-heads", action="store_true",
                   help="type-condition the execution/result heads via the type_fusion MLP (ADR 013)")
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
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ckpt_dir=args.ckpt_dir,
        run_name=args.run_name,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        standardize_profiles=not args.no_standardize_profiles,
        arsenal_per_pitch=not args.no_arsenal_per_pitch,
        propensity_situational=not args.no_propensity_situational,
        concat_then_project=args.concat_then_project,
        profile_film=args.profile_film,
        zone_spatial_weight=args.zone_spatial_weight,
        type_focal_gamma=args.type_focal_gamma,
        type_class_weight_alpha=args.type_class_weight_alpha,
        type_conditioned_heads=args.type_conditioned_heads,
    )


if __name__ == "__main__":
    main()
