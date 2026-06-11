"""Per-count recalibration of the cascade's binary nodes (swing / whiff /
called_strike).

Motivation (2026-06-11 whiff isolation): the cascade is well calibrated
globally but shows small per-count gaps that matter exactly where strikeouts
are decided — at 2 strikes it under-predicts whiff|swing by ~0.7pp and
over-predicts swing by ~0.9pp, costing ~0.9pp of K% in composition.

Method: for each node and each count (balls, strikes), fit an intercept-only
logit shift delta on a CALIBRATION window (2024H1 — strictly before the
2024H2+ eval window): p' = sigmoid(logit(p) + delta). The MLE for an
intercept-only adjustment matches the group mean, solved by bisection — no
tuned constants, everything data-derived.

Output: checkpoints/hitter/count_calibration.json, applied automatically by
HitterModel.predict_cascade when X carries balls/strikes columns.

Run: python -m scripts.hitter.calibrate_cascade_counts
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hitter.labels import is_swing, is_whiff_given_swing
from hitter.rollout import load_hitter_ctx
from hitter.train import build_inference_features

EPS = 1e-6
MIN_CELL = 500   # minimum pitches per (node, count) cell to fit a shift


def fit_logit_shift(p: np.ndarray, y: np.ndarray) -> float:
    """Intercept-only logit shift: solve mean(sigmoid(logit(p)+d)) = mean(y)."""
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    target = float(y.mean())
    if target <= 0.0 or target >= 1.0:
        return 0.0
    z = np.log(p / (1 - p))

    def mean_at(d):
        return float((1 / (1 + np.exp(-(z + d)))).mean())

    lo, hi = -4.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-07-15",
                    help="calibration window end — MUST stay before the eval window")
    ap.add_argument("--aug-dir", default="data/augmented")
    ap.add_argument("--hitter-dir", default="checkpoints/hitter")
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(f"{args.aug_dir}/2024/2024-*.parquet"))
             if args.start[:7] <= Path(f).stem[:7] and Path(f).stem <= args.end]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"calibration window {args.start}..{args.end}: {len(df):,} pitches")

    ctx = load_hitter_ctx(args.hitter_dir)
    X = build_inference_features(df, ctx["bc"], ctx["pc"])
    casc = ctx["hm"].predict_cascade(X)

    real_swing = is_swing(X["description"]).to_numpy().astype(bool)
    real_whiff = is_whiff_given_swing(X["description"]).to_numpy().astype(bool)
    real_cs = (X["description"] == "called_strike").to_numpy()
    balls = X["balls"].to_numpy().astype(int)
    strikes = X["strikes"].to_numpy().astype(int)

    # node -> (predictions, labels, population mask)
    pops = {
        "swing": (casc["swing"], real_swing.astype(float), np.ones(len(X), bool)),
        "whiff": (casc["whiff"], real_whiff.astype(float), real_swing),
        "called_strike": (casc["called_strike"], real_cs.astype(float), ~real_swing),
    }

    cal: dict[str, dict[str, float]] = {}
    print(f"\n{'node':>14} {'count':>5} {'n':>7}  {'pred':>7} {'real':>7} "
          f"{'delta':>8} {'pred_cal':>8}")
    for node, (p, y, pop) in pops.items():
        cal[node] = {}
        for b in range(4):
            for s in range(3):
                m = pop & (balls == b) & (strikes == s)
                n = int(m.sum())
                if n < MIN_CELL:
                    continue
                d = fit_logit_shift(p[m], y[m])
                cal[node][f"{b},{s}"] = round(float(d), 6)
                pc = np.clip(p[m].astype(np.float64), EPS, 1 - EPS)
                p_cal = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + d)))
                print(f"{node:>14}  {b}-{s} {n:>7,}  {p[m].mean():>7.4f} "
                      f"{y[m].mean():>7.4f} {d:>+8.4f} {p_cal.mean():>8.4f}")

    out = Path(args.hitter_dir) / "count_calibration.json"
    payload = {"window": [args.start, args.end], "deltas": cal,
               "n_pitches": int(len(df))}
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
