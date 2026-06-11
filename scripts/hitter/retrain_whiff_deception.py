"""Retrain the swing + whiff nodes with deception features (step 3).

Targets the measured cascade-side K leak (whiff isolation, 2026-06-11):
the whiff node under-predicts swing-and-miss at 2 strikes by ~0.7pp/pitch.
New features (DECEPTION_FEATURE_COLS): prev_velo_diff, prev_loc_dist,
same_type_prev — the pitch-to-pitch contrast that drives whiffs, which the
sequence-aware pitch model can supply at rollout.

Scope: ONLY swing + whiff are retrained (called_strike/contact_quality keep
their artifacts, so the xwOBA map stays valid). meta.json node entries are
MERGED, not overwritten. Old artifacts are backed up as .pre_deception.bak.

Same temporal split as the original training (train ≤2023, val 2024H1).

Run: python -m scripts.hitter.retrain_whiff_deception
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib

from data.profile_cache_loader import ProfileCache
from hitter.train import (
    NODE_CATEGORICAL,
    NODE_FEATURES,
    NODE_MONOTONE,
    NODE_OBJECTIVE,
    build_training_frame,
    load_pitch_frame,
    node_population,
    train_node,
)

NODES = ("swing", "whiff")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="2017-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--val-start", default="2024-01-01")
    ap.add_argument("--val-end", default="2024-07-15")
    ap.add_argument("--fold-id", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints/hitter")
    args = ap.parse_args()

    out = Path(args.out_dir)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text())
    print("BEFORE (from meta.json):")
    for n in NODES:
        m = meta["nodes"].get(n, {}).get("metrics", {})
        print(f"  {n:>6}: AUC={m.get('auc'):.3f} logloss={m.get('logloss'):.3f} "
              f"ECE={m.get('ece'):.3f}")

    bcache = ProfileCache(role="batter", fold_id=args.fold_id)
    pcache = ProfileCache(role="pitcher", fold_id=args.fold_id)

    print(f"loading train [{args.train_start}..{args.train_end}] "
          f"+ val [{args.val_start}..{args.val_end}]")
    tr_aug, tr_raw = load_pitch_frame(args.train_start, args.train_end)
    va_aug, va_raw = load_pitch_frame(args.val_start, args.val_end)
    print(f"  train pitches={len(tr_aug):,}  val pitches={len(va_aug):,}")
    train_df = build_training_frame(tr_aug, tr_raw, bcache, pcache)
    val_df = build_training_frame(va_aug, va_raw, bcache, pcache)
    for col in ("prev_velo_diff", "prev_loc_dist", "same_type_prev"):
        assert col in train_df.columns, f"deception col {col} missing"
    print(f"  deception sample: prev_velo_diff mean={train_df['prev_velo_diff'].mean():.3f} "
          f"std={train_df['prev_velo_diff'].std():.3f}; "
          f"prev_loc_dist median={train_df['prev_loc_dist'].median():.3f}")

    for node in NODES:
        Xtr, ytr = node_population(train_df, node)
        Xva, yva = node_population(val_df, node)
        print(f"training {node}: n_train={len(Xtr):,} n_val={len(Xva):,} "
              f"({len(NODE_FEATURES[node])} features incl. deception)")
        res = train_node(
            Xtr, ytr, Xva, yva,
            objective=NODE_OBJECTIVE[node],
            feature_names=NODE_FEATURES[node],
            categorical=NODE_CATEGORICAL[node],
            monotone=NODE_MONOTONE[node],
        )
        m = res["metrics"]
        old = meta["nodes"].get(node, {}).get("metrics", {})
        print(f"  [{node}] AUC {old.get('auc'):.3f} -> {m['auc']:.3f}   "
              f"logloss {old.get('logloss'):.3f} -> {m['logloss']:.3f}   "
              f"ECE {old.get('ece'):.3f} -> {m['ece']:.3f}")

        art_path = out / f"{node}.joblib"
        shutil.copy(art_path, out / f"{node}.pre_deception.bak.joblib")
        joblib.dump(
            {k: res[k] for k in ("booster", "calibrator", "feature_names",
                                 "categorical", "cat_dtypes", "objective")},
            art_path,
        )
        meta["nodes"][node] = {"objective": res["objective"], "metrics": m}

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved {', '.join(NODES)} artifacts + merged meta -> {out}")


if __name__ == "__main__":
    main()
