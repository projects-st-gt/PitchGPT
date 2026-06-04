"""Build the xwOBA->outcome map used by hitter/compose.

CRITICAL CALIBRATION POINT (learned the hard way via the per-PA backtest):
bin by the MODEL'S PREDICTED xwOBA, not the real xwOBA. The contact_quality
regression hedges to the mean (predicted std ~0.08 vs a much wider real spread),
so a map binned on REAL xwOBA quantiles almost never gets hit in its high bins at
inference -> home runs come out ~5x too low and singles too high. Binning on
PREDICTED xwOBA makes the map a proper calibration of "when the model says X,
here's what actually happened", which fixes the per-PA drift (HR 0.7%->3.2% vs
real 3.3%) and flips the backtest from worse-than-baseline to clearly better
(1.460 -> 1.405 vs 1.452 league-average).

Usage:
    python -m scripts.hitter.build_outcome_map \
        --train-start 2023-01-01 --train-end 2023-12-31 --n-bins 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.profile_cache_loader import ProfileCache
from hitter.compose import build_xwoba_outcome_map
from hitter.model import HitterModel
from hitter.train import build_training_frame, load_pitch_frame, node_population


def main() -> None:
    ap = argparse.ArgumentParser(description="Build xwOBA->outcome map (predicted-binned)")
    ap.add_argument("--model-dir", default="checkpoints/hitter")
    ap.add_argument("--train-start", default="2023-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--fold-id", type=int, default=0)
    args = ap.parse_args()

    hm = HitterModel(args.model_dir)
    bc = ProfileCache(role="batter", fold_id=args.fold_id)
    pc = ProfileCache(role="pitcher", fold_id=args.fold_id)

    print(f"building feature frame [{args.train_start}..{args.train_end}]...")
    df = build_training_frame(
        *load_pitch_frame(args.train_start, args.train_end), bc, pc)
    X, _ = node_population(df, "contact_quality")        # balls in play
    pred = hm.predict_node("contact_quality", X)          # PREDICTED xwOBA
    events = X["events"].to_numpy()
    print(f"in-play balls={len(X):,}  predicted xwOBA mean={pred.mean():.3f} "
          f"std={pred.std():.3f}")

    m = build_xwoba_outcome_map(pred, events, n_bins=args.n_bins)
    out = Path(args.model_dir) / "xwoba_outcome_map.json"
    out.write_text(json.dumps(m))
    print(f"saved {out}  (binned on PREDICTED xwOBA — see module docstring)")


if __name__ == "__main__":
    main()
