"""Controlled experiments to isolate why PitchGPT-Tiny's pitch-type top-1
(~0.46) is far below the profile-aware LSTM (~0.69) on the same val split.

Runs short trainings on a fixed slice with one variable changed at a time:

  base       : as Phase B (no std, all 7 heads, profile only via context token)
  std        : + per-feature z-score standardization of profile vectors
  type_only  : head weights = {type:1, others:0} (isolate multi-task interference)
  std_type   : standardization + type-only
  arsenal    : + pitcher arsenal (7 dims) concatenated to the propensity head input
  arsenal_std: arsenal injection + standardization

Each reports val type top-1 after N steps so the levers' effects are comparable.
Not a final eval — just enough signal to find the dominant factor.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.profile_cache_loader import ProfileCache
from model.config import tiny_config
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
)

# Arsenal feature indices in the 223-dim pitcher profile vector (arsenal_FF..arsenal_FS).
ARSENAL_IDX = list(range(7))


def load_slice(n_train_pitches: int, n_val_pitches: int):
    import glob
    train_files = sorted(glob.glob("data/augmented/2023/2023-*.parquet"))
    train_p = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    train_p = train_p.sort_values(["game_pk", "at_bat_number", "pitch_number"]).head(n_train_pitches).reset_index(drop=True)
    val_files = sorted(glob.glob("data/augmented/2024/2024-*.parquet"))
    val_files = [f for f in val_files if f.split("/")[-1].replace(".parquet", "") <= "2024-07-15"]
    val_p = pd.concat([pd.read_parquet(f) for f in val_files], ignore_index=True)
    val_p = val_p.sort_values(["game_pk", "at_bat_number", "pitch_number"]).head(n_val_pitches).reset_index(drop=True)
    return train_p, val_p


class ArsenalPropensityWrapper(nn.Module):
    """Wrap a PitchGPT so the propensity TYPE head also sees the pitcher's
    arsenal (7 dims), projected and added to each pitch-position hidden state.

    This mimics the LSTM baseline, which gets the arsenal target-encoding as a
    direct per-pitch feature. We only patch the type head's input here.
    """

    def __init__(self, model: PitchGPT, d_model: int):
        super().__init__()
        self.model = model
        self.arsenal_proj = nn.Linear(7, d_model)
        nn.init.normal_(self.arsenal_proj.weight, std=0.02)
        nn.init.zeros_(self.arsenal_proj.bias)

    def forward(self, *, pitcher_arsenal, **kwargs):
        # Run the model normally, then re-derive the type logits with the
        # arsenal injected. We need the trunk hidden — ask for intermediates.
        out = self.model(**kwargs, return_intermediates=True)
        # final hidden after ln_final: recompute? The model already applied
        # ln_final inside forward. We approximate by re-projecting from the
        # propensity logits' pre-image is not available; instead, add the
        # arsenal contribution directly to the type logits via the embedding-
        # tied projection. Simpler: add arsenal_proj(arsenal) into the hidden
        # used for the type head. Since we can't reach that hidden cleanly from
        # outside, we instead append an arsenal-derived bias to the type logits.
        # arsenal (B,7) -> (B, n_pitch_types) via a small linear.
        # Re-use arsenal_proj output projected to vocab through the embedding.
        arsenal_h = self.arsenal_proj(pitcher_arsenal)  # (B, d_model)
        # tie to type embedding like the propensity head does
        type_emb_w = self.model.embed.type_emb.weight  # (n_pitch_types, d_model)
        arsenal_logits = arsenal_h @ type_emb_w.T  # (B, n_pitch_types)
        # broadcast over sequence positions
        out["propensity"]["type"] = out["propensity"]["type"] + arsenal_logits.unsqueeze(1)
        return out


def to_dev(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def eval_type_top1(model, loader, device, use_arsenal=False, arsenal_lookup=None) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        bd = to_dev(batch, device)
        kwargs = dict(
            pitcher_profile=bd["pitcher_profile"],
            batter_profile=bd["batter_profile"],
            categorical_context=bd["categorical_context"],
            pitch_factors=bd["pitch_factors"],
            intended_actions=bd["intended_actions"],
            padding_mask=bd["padding_mask"],
        )
        if use_arsenal:
            ars = arsenal_lookup(batch)  # (B,7) on cpu
            out = model(pitcher_arsenal=ars.to(device), **kwargs)
        else:
            out = model(**kwargs)
        logits = out["propensity"]["type"][:, PitchGPT.N_CONTEXT_TOKENS:, :]
        tgt = batch["targets"]["propensity"]["type"]
        valid = tgt != -100
        if valid.any():
            preds = logits.argmax(-1).cpu()
            correct += int((preds[valid] == tgt[valid]).sum())
            total += int(valid.sum())
    return correct / max(total, 1)


def run_experiment(name, *, standardize, type_only, arsenal, inject=False, head_inject=False, two_stage=False, low_reg=False, lr=3e-4, train_p, val_p, steps, batch_size, device):
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.inject_profiles_per_pitch = inject
    cfg.inject_profiles_to_head = head_inject
    cfg.propensity_type_two_stage = two_stage
    if low_reg:
        cfg.dropout = 0.0
        cfg.label_smoothing_type = 0.0
    if type_only:
        cfg.head_weights = {"type": 1.0, "zone": 0.0, "velo": 0.0, "spin_rate": 0.0,
                            "spin_axis": 0.0, "result": 0.0, "ab_outcome": 0.0}
    standardizer = ProfileStandardizer() if standardize else None
    pc_p = ProfileCache(role="pitcher", fold_id=0)
    pc_b = ProfileCache(role="batter", fold_id=0)
    train_ds = PitchGPTAtBatDataset(train_p, pitcher_profile_lookup=pc_p.lookup,
                                    batter_profile_lookup=pc_b.lookup, profile_standardizer=standardizer)
    val_ds = PitchGPTAtBatDataset(val_p, pitcher_profile_lookup=pc_p.lookup,
                                  batter_profile_lookup=pc_b.lookup, profile_standardizer=standardizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_pitchgpt_at_bats, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=collate_pitchgpt_at_bats)

    base_model = PitchGPT(cfg).to(device)

    # For the arsenal experiment we need the RAW (unstandardized) arsenal per AB.
    arsenal_lookup = None
    if arsenal:
        # Build a per-AB raw-arsenal lookup keyed by position in the dataset.
        # Simplest: read arsenal directly from the *unstandardized* pitcher cache.
        raw_pc = ProfileCache(role="pitcher", fold_id=0)

        def make_lookup(ds):
            keys = ds._ab_keys
            df = ds._df

            def lookup(batch_dict):
                # We don't have AB ids in the collated batch; instead, re-derive
                # arsenal from pitcher_profile by *un-standardizing* if needed.
                # But pitcher_profile in the batch may be standardized. So instead
                # we recompute arsenal from the raw cache using the pitcher id —
                # which isn't in the batch either. Punt: use the (possibly
                # standardized) first 7 dims of pitcher_profile as a proxy. If
                # standardized, that's z-scored arsenal, still informative.
                return batch_dict["pitcher_profile"][:, :7].clone()

            return lookup

        arsenal_lookup = make_lookup(train_ds)
        model = ArsenalPropensityWrapper(base_model, cfg.d_model).to(device)
    else:
        model = base_model

    wd = 0.0 if low_reg else 0.1
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    warmup = max(steps // 10, 50)
    step = 0
    t0 = time.time()
    model.train()
    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            cur_lr = lr * min((step + 1) / warmup, 1.0)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            bd = to_dev(batch, device)
            kwargs = dict(
                pitcher_profile=bd["pitcher_profile"],
                batter_profile=bd["batter_profile"],
                categorical_context=bd["categorical_context"],
                pitch_factors=bd["pitch_factors"],
                intended_actions=bd["intended_actions"],
                padding_mask=bd["padding_mask"],
            )
            if arsenal:
                ars = arsenal_lookup(batch).to(device)
                out = model(pitcher_arsenal=ars, **kwargs)
            else:
                out = model(**kwargs)
            type_logits = out["propensity"]["type"][:, PitchGPT.N_CONTEXT_TOKENS:, :]
            loss = F.cross_entropy(type_logits.reshape(-1, cfg.n_pitch_types),
                                   bd["targets"]["propensity"]["type"].reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    acc = eval_type_top1(model, val_loader, device, use_arsenal=arsenal,
                         arsenal_lookup=(lambda b: b["pitcher_profile"][:, :7]) if arsenal else None)
    print(f"  {name:14s}: type_top1={acc:.4f}  ({step} steps, {time.time()-t0:.0f}s)")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-pitches", type=int, default=800000)
    ap.add_argument("--val-pitches", type=int, default=200000)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device={device}, steps={args.steps}, batch={args.batch_size}")
    train_p, val_p = load_slice(args.train_pitches, args.val_pitches)
    print(f"train slice: {len(train_p):,} pitches; val slice: {len(val_p):,} pitches")

    experiments = [
        ("base",        dict(standardize=False, type_only=False, arsenal=False, inject=False, head_inject=False, two_stage=False)),
        ("std",         dict(standardize=True,  type_only=False, arsenal=False, inject=False, head_inject=False, two_stage=False)),
        ("type_only",   dict(standardize=False, type_only=True,  arsenal=False, inject=False, head_inject=False, two_stage=False)),
        ("std_type",    dict(standardize=True,  type_only=True,  arsenal=False, inject=False, head_inject=False, two_stage=False)),
        ("arsenal",     dict(standardize=False, type_only=True,  arsenal=True,  inject=False, head_inject=False, two_stage=False)),
        ("inject",      dict(standardize=True,  type_only=False, arsenal=False, inject=True,  head_inject=False, two_stage=False)),
        ("head_inject", dict(standardize=True,  type_only=False, arsenal=False, inject=False, head_inject=True,  two_stage=False)),
        ("two_stage",   dict(standardize=True,  type_only=False, arsenal=False, inject=False, head_inject=False, two_stage=True)),
        ("lowreg_hilr", dict(standardize=True,  type_only=False, arsenal=False, inject=False, head_inject=False, two_stage=False, low_reg=True, lr=1e-3)),
    ]
    if args.only:
        experiments = [(n, k) for n, k in experiments if n in set(args.only)]

    print("\n--- results ---")
    for name, kw in experiments:
        run_experiment(name, train_p=train_p, val_p=val_p, steps=args.steps,
                       batch_size=args.batch_size, device=device, **kw)


if __name__ == "__main__":
    main()
