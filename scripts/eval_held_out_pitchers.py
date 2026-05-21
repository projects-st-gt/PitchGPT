"""Held-out-pitcher cohort eval (per the ``eval-protocol`` skill).

Loads a *calibrated* checkpoint (with saved per-head temperatures from
``scripts.calibrate_pitchgpt``), runs inference on val (2024 H1), and
reports per-head accuracy + ECE + per-class type breakdown split into:

  - ``main`` cohort: pitchers whose first MLB pitch is before ``--debut-year``.
  - ``held_out`` cohort: pitchers whose first MLB pitch is in or after ``--debut-year``.
  - ``all``: the full val split (sanity reference).

The point of this eval: tests whether the player-profile encoder is doing
generalization or memorization. If held-out collapses much worse than main,
the model is leaning on per-pitcher identity (via the profile lookup), and
the writeup needs to say so.

Run::

    python -m scripts.eval_held_out_pitchers \
        --ckpt checkpoints/small-v1-arsenal-std/checkpoint_calibrated.pt
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

from data.dataset import PITCH_TYPES
from data.profile_cache_loader import ProfileCache
from eval.generalization.held_out_pitchers import held_out_pitcher_ids
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset,
    ProfileStandardizer,
    collate_pitchgpt_at_bats,
)

VAL_END = "2024-07-15"


def ece_equal_mass(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    if len(probs) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    n = len(conf)
    out = 0.0
    for b in np.array_split(np.arange(n), n_bins):
        if len(b):
            out += len(b) / n * abs(correct[b].mean() - conf[b].mean())
    return float(out)


def to_dev(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    return x


def load_val_pitches(augmented_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(augmented_dir / "2024" / "2024-*.parquet")))
    files = [f for f in files if Path(f).stem <= VAL_END]
    if not files:
        raise FileNotFoundError(f"no val parquets under {augmented_dir}/2024 ≤ {VAL_END}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


PROP_HEADS = ["type", "zone", "velo", "spin_rate"]


def run_inference(model, ds, device) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Accumulate logits + targets per head over the whole dataset."""
    loader = DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=0,
        collate_fn=collate_pitchgpt_at_bats,
    )
    NC = PitchGPT.N_CONTEXT_TOKENS
    bucket: dict[str, list] = {h: [[], []] for h in PROP_HEADS + ["result", "ab_outcome"]}
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
            pad = batch["padding_mask"]
            for h in PROP_HEADS:
                lg = out["propensity"][h][:, NC:, :].cpu()
                tg = batch["targets"]["propensity"][h]
                m = tg != -100
                bucket[h][0].append(lg[m]); bucket[h][1].append(tg[m])
            lg = out["result"].cpu()
            tg = batch["targets"]["result"]
            m = tg != -100
            bucket["result"][0].append(lg[m]); bucket["result"][1].append(tg[m])
            lengths = pad.sum(dim=1)
            term_idx = (lengths - 1).clamp(min=0)
            lg_all = out["ab_outcome_per_pos"].cpu()
            term_lg = lg_all[torch.arange(lg_all.size(0)), term_idx]
            tg = batch["targets"]["ab_outcome"]
            m = tg != -100
            bucket["ab_outcome"][0].append(term_lg[m]); bucket["ab_outcome"][1].append(tg[m])
    out_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for h in PROP_HEADS + ["result", "ab_outcome"]:
        if bucket[h][0]:
            out_dict[h] = (
                torch.cat(bucket[h][0]).float().numpy(),
                torch.cat(bucket[h][1]).long().numpy(),
            )
        else:
            out_dict[h] = (np.zeros((0, 0)), np.zeros((0,), dtype=int))
    return out_dict


def per_class_breakdown(
    logits: np.ndarray, targets: np.ndarray, class_names: list[str], T: float = 1.0
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1/support for the type head.

    CONVENTION (critical, has bitten us three times now): the model's type
    vocab is 8 classes — PAD at MODEL index 0, PITCH_TYPES (FF..FS) at MODEL
    indices 1..7. So when ``class_names[i]`` is "FF" (i=0 in PITCH_TYPES),
    the matching model index is i+1=1, NOT i. The dataset's targets and
    logits-argmax are in the model's index space (0..7), so we have to add 1
    to ``i`` when matching.
    """
    if len(logits) == 0:
        return {n: {"precision": float("nan"), "recall": float("nan"),
                    "f1": float("nan"), "support": 0} for n in class_names}
    probs = F.softmax(torch.from_numpy(logits) / T, dim=-1).numpy()
    pred = probs.argmax(axis=1)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        model_idx = i + 1  # PITCH_TYPES[i] lives at model index i+1 (PAD=0)
        tp = int(((pred == model_idx) & (targets == model_idx)).sum())
        fp = int(((pred == model_idx) & (targets != model_idx)).sum())
        support = int((targets == model_idx).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / support if support > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (
            precision and recall and precision + recall > 0
        ) else float("nan")
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/small-v1-arsenal-std/checkpoint_calibrated.pt"))
    ap.add_argument("--debut-year", type=int, default=2024)
    ap.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    ap.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
    model = PitchGPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    fold_id = int(ckpt.get("fold_id", 0))
    temps = ckpt.get("temperatures", {})
    print(f"loaded {args.ckpt}  step={ckpt.get('step')}  size={ckpt.get('size')}  "
          f"fold={fold_id}  params={model.num_parameters():,}  "
          f"temps={ {k: round(v, 4) for k, v in temps.items()} }")

    val_pitches = load_val_pitches(args.augmented_dir)
    # Compute "debut year" on the FULL corpus (train + val), not val alone.
    # If we used val alone, every pitcher's earliest game_date in the dataframe
    # would land in 2024 → all val pitchers would be misclassified as debutants.
    # The set of pitchers who appear in training (≤ 2023) is the "veteran"
    # complement; everyone in val but not in that set is a true held-out debutant.
    print(f"identifying held-out cohort by scanning training-period pitcher IDs (≤ 2023)...")
    train_pitchers: set[int] = set()
    for year_dir in sorted(args.augmented_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            yr = int(year_dir.name)
        except ValueError:
            continue
        if yr > args.debut_year - 1:
            continue
        for parquet in sorted(year_dir.glob("*.parquet")):
            df_p = pd.read_parquet(parquet, columns=["pitcher"])
            train_pitchers.update(df_p["pitcher"].astype(int).unique())
    val_pitchers = set(val_pitches["pitcher"].astype(int).unique())
    held_out_set = val_pitchers - train_pitchers
    main_set = val_pitchers & train_pitchers
    all_pitchers = val_pitchers
    print(f"val: {len(val_pitches):,} pitches | val pitchers: {len(val_pitchers)} | "
          f"in train (≤ {args.debut_year - 1}): {len(main_set)} | "
          f"held-out (debut ≥ {args.debut_year}): {len(held_out_set)}")

    pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=args.profiles_dir)
    pc_b = ProfileCache(role="batter",  fold_id=fold_id, profiles_dir=args.profiles_dir)
    std = ProfileStandardizer() if args.standardize else None

    cohorts: list[tuple[str, set[int]]] = [
        ("main",     main_set),
        ("held_out", held_out_set),
        ("all",      all_pitchers),
    ]

    for name, ids in cohorts:
        df = val_pitches[val_pitches["pitcher"].astype(int).isin(ids)].reset_index(drop=True)
        if len(df) == 0:
            print(f"\n=== cohort {name}: 0 pitches (skipping) ===")
            continue
        ds = PitchGPTAtBatDataset(
            pitches=df, pitcher_profile_lookup=pc_p.lookup,
            batter_profile_lookup=pc_b.lookup, profile_standardizer=std,
        )
        print(f"\n=== cohort {name}: {len(df):,} pitches, {len(ds):,} ABs, "
              f"{len(ids)} pitchers ===")
        results = run_inference(model, ds, device)

        print(f"{'head':<12} {'n':>9}  {'acc':>8} {'ECE':>8}  {'T':>7}")
        for h in PROP_HEADS + ["result", "ab_outcome"]:
            logits, targets = results[h]
            if len(targets) == 0:
                continue
            T = temps.get(h, 1.0)
            probs = F.softmax(torch.from_numpy(logits) / T, dim=-1).numpy()
            acc = float((probs.argmax(axis=1) == targets).mean())
            ece = ece_equal_mass(probs, targets)
            print(f"{h:<12} {len(targets):>9,}  {acc:>8.4f} {ece:>8.4f}  {T:>7.4f}")

        T_type = temps.get("type", 1.0)
        bd = per_class_breakdown(*results["type"], list(PITCH_TYPES), T=T_type)
        print(f"\n  per-class TYPE breakdown (cohort: {name}, T={T_type:.4f}):")
        print(f"    {'class':<6} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
        for c, m in bd.items():
            p = f"{m['precision']:.4f}" if m['precision'] == m['precision'] else "  nan"
            r = f"{m['recall']:.4f}"    if m['recall']    == m['recall']    else "  nan"
            f = f"{m['f1']:.4f}"        if m['f1']        == m['f1']        else "  nan"
            print(f"    {c:<6} {p:>10} {r:>10} {f:>10} {m['support']:>10}")


if __name__ == "__main__":
    main()
