"""Out-of-distribution evaluation on MLB postseason pitches.

Mirrors the evaluation protocol of Ahn et al. 2026 ("Neural Sabermetrics
with World Model") — train on regular season, evaluate on postseason — to
produce apples-to-apples comparisons vs that paper's reported numbers
(binary FF: accuracy 0.637, recall 0.792, F1 0.722; on Llama-3.2-3B).

Postseason games in Statcast carry ``game_type`` in {F, D, L, W}
(Wild Card / Division / League Championship / World Series); regular-season
games are tagged ``R``. The augmented parquets don't carry game_type, so
we join through the raw parquets to identify postseason game_pks.

Outputs side-by-side comparison:
- in-distribution baseline (val 2024H1)         — from existing calibrate script
- in-distribution v6 (val 2024H1)               — from existing calibrate script
- postseason OOD (2024 postseason + 2025 postseason)

Run:

    # Eval v5 baseline only
    python -m scripts.eval_postseason_ood \\
        --ckpt checkpoints_modal/tiny-fold0-1778792736/checkpoint_best.pt

    # Eval v5 + v6 side-by-side
    python -m scripts.eval_postseason_ood \\
        --ckpt checkpoints_modal/tiny-fold0-1778792736/checkpoint_best.pt \\
        --ckpt checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt

    # Eval just v6
    python -m scripts.eval_postseason_ood \\
        --ckpt checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt
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


class _LegacyProfileLookup:
    """Lookup over a previous-schema profile cache, bypassing the schema check.

    Used to evaluate v5-trained checkpoints against the v5 backup cache when
    PROFILE_SCHEMA_VERSION has moved on to v6. Same lookup contract as
    ``ProfileCache``: ``(player_id, asof_date, asof_game_num) -> {"vector": ndarray}``.
    Per-player slot → league-mean fallback → zero, matching the standard chain.
    """

    def __init__(self, role: str, fold_id: int, profiles_dir: Path):
        player_path = profiles_dir / f"{role}_fold_{fold_id}.parquet"
        league_path = profiles_dir / f"league_{role}_fold_{fold_id}.parquet"
        pdf = pd.read_parquet(player_path)
        ldf = pd.read_parquet(league_path)
        # In-memory dicts keyed by (player_id, asof_date, asof_game_num) and
        # (asof_date, asof_game_num).
        self._player: dict[tuple[int, pd.Timestamp, int], np.ndarray] = {}
        for _, row in pdf.iterrows():
            key = (int(row["player_id"]),
                   pd.Timestamp(row["asof_date"]).date(),
                   int(row["asof_game_num"]))
            self._player[key] = np.asarray(row["vector"], dtype=np.float32)
        self._league: dict[tuple[pd.Timestamp, int], np.ndarray] = {}
        for _, row in ldf.iterrows():
            key = (pd.Timestamp(row["asof_date"]).date(),
                   int(row["asof_game_num"]))
            self._league[key] = np.asarray(row["vector"], dtype=np.float32)
        self._vector_len = len(next(iter(self._player.values())))

    def lookup(self, player_id: int, asof_date, asof_game_num: int) -> dict:
        asof_d = pd.Timestamp(asof_date).date() if not hasattr(asof_date, "year") else asof_date
        player_key = (int(player_id), asof_d, int(asof_game_num))
        league_key = (asof_d, int(asof_game_num))
        player_vec = self._player.get(player_key)
        league_vec = self._league.get(league_key)
        if player_vec is not None and league_vec is not None:
            vec = np.where(np.isnan(player_vec), league_vec, player_vec).astype(np.float32)
        elif player_vec is not None:
            vec = np.nan_to_num(player_vec.copy(), nan=0.0).astype(np.float32)
        elif league_vec is not None:
            vec = np.nan_to_num(league_vec.copy(), nan=0.0).astype(np.float32)
        else:
            vec = np.zeros(self._vector_len, dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0)
        return {"vector": vec}


# Postseason runs Oct–Nov in modern MLB; expand if needed.
POSTSEASON_MONTHS = [
    "2024-10", "2024-11",
    "2025-10", "2025-11",
]


def identify_postseason_game_pks(raw_dir: Path) -> set[int]:
    """Read raw parquets for postseason months, return game_pks with game_type != 'R'."""
    postseason_pks: set[int] = set()
    type_breakdown: dict[str, int] = {}
    for month_prefix in POSTSEASON_MONTHS:
        year = month_prefix[:4]
        files = sorted(glob.glob(str(raw_dir / year / f"{month_prefix}-*.parquet")))
        if not files:
            print(f"  warn: no raw parquets for {month_prefix}")
            continue
        for f in files:
            df = pd.read_parquet(f, columns=["game_pk", "game_type"])
            non_reg = df[df["game_type"] != "R"]
            postseason_pks.update(non_reg["game_pk"].unique())
            for gt, n in non_reg["game_type"].value_counts().items():
                type_breakdown[gt] = type_breakdown.get(gt, 0) + int(n)
    return postseason_pks, type_breakdown


def load_postseason_pitches(augmented_dir: Path, postseason_pks: set[int]) -> pd.DataFrame:
    """Load augmented pitches restricted to postseason game_pks."""
    parts: list[pd.DataFrame] = []
    for month_prefix in POSTSEASON_MONTHS:
        year = month_prefix[:4]
        files = sorted(glob.glob(str(augmented_dir / year / f"{month_prefix}-*.parquet")))
        for f in files:
            df = pd.read_parquet(f)
            df = df[df["game_pk"].isin(postseason_pks)]
            if len(df):
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)


def binary_metrics_at_threshold(probs_pos: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    """Binary precision/recall/F1/accuracy at a fixed P(positive) threshold."""
    pred = (probs_pos >= threshold)
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(n, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        "predicted_positive_rate": (tp + fp) / max(n, 1),
    }


def threshold_for_target_recall(probs_pos: np.ndarray, y_true: np.ndarray, target_recall: float) -> float:
    """Find the threshold that achieves >= target_recall on the positive class."""
    if y_true.sum() == 0:
        return 0.5
    pos_probs_sorted = np.sort(probs_pos[y_true])[::-1]  # descending
    # To get target_recall, we need to capture target_recall * n_pos positives.
    # Threshold = the prob at index target_recall * n_pos (or the n_pos-th if exact).
    target_idx = int(np.ceil(target_recall * len(pos_probs_sorted))) - 1
    target_idx = max(0, min(target_idx, len(pos_probs_sorted) - 1))
    return float(pos_probs_sorted[target_idx])


def roc_auc(probs_pos: np.ndarray, y_true: np.ndarray) -> float:
    """Threshold-independent AUC. Returns NaN if all labels are one class."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    # Rank-based AUC = (rank_sum_of_positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(probs_pos)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs_pos) + 1)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    rank_sum_pos = float(ranks[y_true].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


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


def to_dev(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    return x


def eval_checkpoint(ckpt_path: Path, pitches: pd.DataFrame, args) -> dict:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PitchGPTConfig(**{k: v for k, v in ckpt["config"].items()})
    model = PitchGPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    fold_id = int(ckpt.get("fold_id", 0))
    is_v5 = cfg.pitcher_profile_dim == 118
    print(f"  loaded {ckpt_path}  step={ckpt.get('step')}  size={ckpt.get('size')}  "
          f"fold={fold_id}  pitcher_profile_dim={cfg.pitcher_profile_dim}"
          f"{' (v5 — using v5 backup cache)' if is_v5 else ' (v6)'}")

    if is_v5:
        # v5 checkpoint: must route to v5 backup cache. Bypass the current
        # loader's schema check (which expects PROFILE_SCHEMA_VERSION=6) via a
        # legacy lookup that mirrors the standard NaN→league→zero fallback.
        v5_dir = args.v5_profiles_dir
        if not (v5_dir / f"pitcher_fold_{fold_id}.parquet").exists():
            raise RuntimeError(
                f"v5 checkpoint requires v5 profile cache at {v5_dir}, but "
                f"pitcher_fold_{fold_id}.parquet is missing there."
            )
        pc_p = _LegacyProfileLookup(role="pitcher", fold_id=fold_id, profiles_dir=v5_dir)
        pc_b = _LegacyProfileLookup(role="batter", fold_id=fold_id, profiles_dir=v5_dir)
        # The v5 baseline was trained with the v5-fitted standardizer; the
        # current file on disk is the v6 refit. Use the backup if present.
        v5_std_path = Path("data/preprocess_artifacts/v1/profile_standardization_v5_backup.npz")
        if v5_std_path.exists():
            std = ProfileStandardizer(v5_std_path)
        else:
            print(f"  warn: v5 standardizer backup not at {v5_std_path}; using current standardizer")
            std = ProfileStandardizer() if args.standardize else None
    else:
        pc_p = ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=args.profiles_dir)
        pc_b = ProfileCache(role="batter", fold_id=fold_id, profiles_dir=args.profiles_dir)
        std = ProfileStandardizer() if args.standardize else None
    ds = PitchGPTAtBatDataset(
        pitches=pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        profile_standardizer=std,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                        collate_fn=collate_pitchgpt_at_bats)
    NC = PitchGPT.N_CONTEXT_TOKENS

    logits_list, target_list = [], []
    result_logits_list, result_target_list = [], []
    with torch.no_grad():
        for batch in loader:
            bd = to_dev(batch, device)
            out = model(
                pitcher_profile=bd["pitcher_profile"], batter_profile=bd["batter_profile"],
                categorical_context=bd["categorical_context"], pitch_factors=bd["pitch_factors"],
                intended_actions=bd["intended_actions"], padding_mask=bd["padding_mask"],
                arsenal=bd.get("arsenal"),
            )
            # Type head
            lg = out["propensity"]["type"][:, NC:, :].cpu()
            tg = batch["targets"]["propensity"]["type"]
            mask = tg != -100
            logits_list.append(lg[mask])
            target_list.append(tg[mask])
            # Result head
            rlg = out["result"].cpu()
            rtg = batch["targets"]["result"]
            rmask = rtg != -100
            result_logits_list.append(rlg[rmask])
            result_target_list.append(rtg[rmask])
    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(target_list).numpy()
    result_logits = torch.cat(result_logits_list).numpy()
    result_targets = torch.cat(result_target_list).numpy()

    # Type head: 8 logits (PAD at 0, pitch types 1..7). Slice to named-pitch slots.
    type_logits = logits[:, MODEL_PITCH_TYPES_START_IDX:]
    type_targets = targets - MODEL_PITCH_TYPES_START_IDX

    probs = torch.softmax(torch.from_numpy(type_logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    n = len(type_targets)

    # 7-class metrics
    acc_7 = float((preds == type_targets).mean())
    nll_7 = float(F.cross_entropy(torch.from_numpy(type_logits).float(),
                                  torch.from_numpy(type_targets).long()).item())
    ece_7 = ece_equal_mass(probs, type_targets)

    # Binary FF vs not-FF derived metrics
    FF_IDX = 0  # PITCH_TYPES[0] = "FF"
    is_ff_true = (type_targets == FF_IDX)
    probs_ff = probs[:, FF_IDX]
    mean_p_ff = float(probs_ff.mean())
    true_ff_rate = float(is_ff_true.mean())
    auc_ff = roc_auc(probs_ff, is_ff_true)

    # (a) Argmax-of-7 binary: predict FF iff FF is the 7-class argmax.
    argmax_pred_ff = (preds == FF_IDX)
    tp = int((argmax_pred_ff & is_ff_true).sum())
    fp = int((argmax_pred_ff & ~is_ff_true).sum())
    fn = int((~argmax_pred_ff & is_ff_true).sum())
    tn = int((~argmax_pred_ff & ~is_ff_true).sum())
    binary_argmax = {
        "threshold": float("nan"),  # threshold not applicable for argmax-of-7
        "accuracy": (tp + tn) / max(n, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        "predicted_positive_rate": (tp + fp) / max(n, 1),
    }
    # (b) Threshold P(FF) >= 0.5
    binary_t05 = binary_metrics_at_threshold(probs_ff, is_ff_true, 0.5)
    # (c) Threshold tuned to match Pi 2018 / Ahn 2026 operating point (recall ≈ 0.79).
    thr_recall = threshold_for_target_recall(probs_ff, is_ff_true, 0.79)
    binary_tunedR = binary_metrics_at_threshold(probs_ff, is_ff_true, thr_recall)
    # (d) Threshold tuned to maximize F1 (sweep)
    f1_sweep_thresholds = np.linspace(0.05, 0.95, 19)
    f1_sweep = [binary_metrics_at_threshold(probs_ff, is_ff_true, t) for t in f1_sweep_thresholds]
    best_f1_idx = int(np.argmax([m["f1"] for m in f1_sweep]))
    binary_bestF1 = f1_sweep[best_f1_idx]

    # Result head: 7 classes, swing = indices 2..6, no-swing = 0..1
    # (See data/dataset.py RESULT_CLASSES: ball, called_strike, swinging_strike, foul,
    #  in_play_out, in_play_hit, in_play_hr)
    result_probs = torch.softmax(torch.from_numpy(result_logits), dim=-1).numpy()
    swing_probs = result_probs[:, 2:].sum(axis=1)
    swing_targets = (result_targets >= 2).astype(int)  # 1 if swing, 0 if take
    swing_pred = (swing_probs >= 0.5).astype(int)
    swing_acc = float((swing_pred == swing_targets).mean()) if len(swing_targets) else float("nan")

    return {
        "n_pitches": n,
        "n_result": len(result_targets),
        "type_top1_7class": acc_7,
        "type_nll_7class": nll_7,
        "type_ece_7class": ece_7,
        "mean_p_ff": mean_p_ff,
        "true_ff_rate": true_ff_rate,
        "auc_ff": auc_ff,
        "binary_argmax": binary_argmax,
        "binary_t05": binary_t05,
        "binary_tunedR": binary_tunedR,
        "binary_bestF1": binary_bestF1,
        "swing_acc_marginal": swing_acc,
        "ckpt": str(ckpt_path),
        "step": int(ckpt.get("step", 0)),
        "size": ckpt.get("size", "?"),
    }


def _fmt_op_point(label: str, m: dict) -> str:
    thr = f"thr={m['threshold']:.3f}" if not np.isnan(m['threshold']) else "thr=argmax-of-7"
    return (f"    {label:<28} {thr}  "
            f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  F1={m['f1']:.4f}  "
            f"pred_pos_rate={m['predicted_positive_rate']:.4f}")


def fmt_report(name: str, m: dict) -> str:
    return (
        f"=== {name} ===\n"
        f"  ckpt: {m['ckpt']}  step={m['step']}  size={m['size']}\n"
        f"  n_pitches={m['n_pitches']:,}  (result rows: {m['n_result']:,})\n"
        f"  7-class:    type_top1={m['type_top1_7class']:.4f}  "
        f"NLL={m['type_nll_7class']:.4f}  ECE={m['type_ece_7class']:.4f}\n"
        f"  FF calib:   mean_p(FF)={m['mean_p_ff']:.4f}  true_rate(FF)={m['true_ff_rate']:.4f}\n"
        f"  binary FF (threshold-independent): AUC={m['auc_ff']:.4f}\n"
        f"  binary FF (different operating points):\n"
        f"{_fmt_op_point('(a) argmax-of-7', m['binary_argmax'])}\n"
        f"{_fmt_op_point('(b) P(FF) >= 0.5', m['binary_t05'])}\n"
        f"{_fmt_op_point('(c) tuned to recall ≈ 0.79', m['binary_tunedR'])}\n"
        f"{_fmt_op_point('(d) tuned to best F1', m['binary_bestF1'])}\n"
        f"  swing (marginal P(swing) from result head): acc={m['swing_acc_marginal']:.4f}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Postseason OOD eval (vs Ahn et al. 2026 protocol)")
    ap.add_argument("--ckpt", action="append", required=True, type=Path,
                    help="Repeat to compare multiple checkpoints.")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--augmented-dir", type=Path, default=Path("data/augmented"))
    ap.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"),
                    help="Current (v6) profile cache directory.")
    ap.add_argument("--v5-profiles-dir", type=Path, default=Path("data/profiles_v5_backup"),
                    help="v5 backup cache directory (used when a v5 checkpoint is evaluated).")
    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    args = ap.parse_args()

    print("Identifying postseason games via raw parquets (game_type != 'R')...")
    postseason_pks, type_breakdown = identify_postseason_game_pks(args.raw_dir)
    print(f"  found {len(postseason_pks):,} postseason game_pks")
    print(f"  type breakdown (pitches in raw): {type_breakdown}")

    print(f"\nLoading augmented postseason pitches...")
    pitches = load_postseason_pitches(args.augmented_dir, postseason_pks)
    print(f"  loaded {len(pitches):,} postseason pitches across "
          f"{pitches['game_pk'].nunique() if len(pitches) else 0} games")
    if len(pitches) == 0:
        print("\nNO POSTSEASON PITCHES FOUND. Are augmented parquets present for Oct/Nov?")
        return

    results: list[tuple[str, dict]] = []
    for ck in args.ckpt:
        print(f"\nEvaluating {ck} on postseason OOD...")
        m = eval_checkpoint(ck, pitches, args)
        results.append((ck.parent.name, m))

    print("\n" + "=" * 70)
    print(f"POSTSEASON OOD RESULTS — {len(pitches):,} pitches from "
          f"{pitches['game_pk'].nunique()} postseason games")
    print("=" * 70)
    for name, m in results:
        print(fmt_report(name, m))

    if len(results) > 1:
        print("\n--- side-by-side deltas (later - earlier), threshold-fair ---")
        base = results[0][1]
        for name, m in results[1:]:
            print(f"\n  {name} vs {results[0][0]}:")
            print(f"    7-class top-1:   {m['type_top1_7class'] - base['type_top1_7class']:+.4f}")
            print(f"    7-class ECE:     {m['type_ece_7class'] - base['type_ece_7class']:+.4f}")
            print(f"    binary FF AUC:   {m['auc_ff'] - base['auc_ff']:+.4f}")
            for op_label, key in [("argmax-of-7", "binary_argmax"),
                                   ("P(FF)>=0.5", "binary_t05"),
                                   ("tuned recall≈0.79", "binary_tunedR"),
                                   ("best F1", "binary_bestF1")]:
                d_acc = m[key]['accuracy'] - base[key]['accuracy']
                d_f1 = m[key]['f1'] - base[key]['f1']
                d_rec = m[key]['recall'] - base[key]['recall']
                print(f"    binary FF @{op_label:<20} acc {d_acc:+.4f}  recall {d_rec:+.4f}  F1 {d_f1:+.4f}")
            print(f"    swing acc:       {m['swing_acc_marginal'] - base['swing_acc_marginal']:+.4f}")

    print("\nReference numbers from the literature (all on postseason / OOD-style eval):")
    print("  Pi 2018 RNN (cited in Ahn et al.):    binary FF acc=0.633  recall=0.792  F1=0.720")
    print("  Ahn et al. 2026 (Llama-3.2-3B):       binary FF acc=0.637  recall=0.792  F1=0.722")
    print("  Ahn et al. 2026 swing decision:       in-zone 0.766  out-of-zone 0.792")
    print("  Note: Pi 2018 and Ahn 2026 report recall=0.792 — same operating point.")
    print("  Closest apples-to-apples row in our table: '(c) tuned to recall ≈ 0.79'.")


if __name__ == "__main__":
    main()
