"""Whiff isolation: is the CASCADE's swing-and-miss response calibrated on
REAL held-out pitches, count by count?

Context (2026-06-11): the simulator under-produces strikeouts by ~2.2pp of
PA even after the pitch model's putaway locations were proven real-identical
and the spin axis was supplied. This diagnostic feeds the cascade REAL
held-out pitches (real features, known outcomes) and compares its predicted
swing / whiff-given-swing / called-strike-given-take rates to what actually
happened — by count, with 2-strike counts the ones that matter for K.

If predicted whiff < realized on REAL 2-strike pitches, the leak is in the
batter model itself (fix = count-conditional recalibration and/or retrain
with deception features), and nothing upstream could ever have fixed it.

Run: python -m scripts.hitter.diagnose_whiff_isolation --months 2024-08 2024-09
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from hitter.labels import is_swing, is_whiff_given_swing
from hitter.rollout import load_hitter_ctx
from hitter.train import build_inference_features

_CALLED_STRIKE_DESCRS = {"called_strike"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2024-08", "2024-09"])
    ap.add_argument("--aug-dir", default="data/augmented")
    ap.add_argument("--hitter-dir", default="checkpoints/hitter")
    ap.add_argument("--max-pitches", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = []
    for m in args.months:
        files += sorted(glob.glob(f"{args.aug_dir}/{m[:4]}/{m}-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if len(df) > args.max_pitches:
        rng = np.random.default_rng(args.seed)
        df = df.iloc[rng.permutation(len(df))[: args.max_pitches]].reset_index(drop=True)
    print(f"{len(df):,} real held-out pitches ({'+'.join(args.months)})")

    ctx = load_hitter_ctx(args.hitter_dir)
    # build_base_features RE-SORTS the frame for lag features — labels and
    # counts MUST come from X (the sorted frame), never from df positionally.
    # (Misalignment here produced flat-by-count predictions on first run —
    # the 2026-06-05 measurement-bug class again.)
    X = build_inference_features(df, ctx["bc"], ctx["pc"])
    assert len(X) == len(df)
    casc = ctx["hm"].predict_cascade(X)
    p_swing = np.clip(casc["swing"], 0, 1)
    p_whiff = np.clip(casc["whiff"], 0, 1)          # P(whiff | swing)
    p_cs = np.clip(casc["called_strike"], 0, 1)     # P(called strike | take)

    real_swing = is_swing(X["description"]).to_numpy().astype(bool)
    real_whiff = is_whiff_given_swing(X["description"]).to_numpy().astype(bool)
    real_cs = X["description"].isin(_CALLED_STRIKE_DESCRS).to_numpy()

    balls = X["balls"].to_numpy().astype(int)
    strikes = X["strikes"].to_numpy().astype(int)

    print(f"\n=== cascade on REAL pitches: predicted vs realized, by count ===")
    print(f"{'count':>5} {'n':>7}  {'swing_pred':>10} {'swing_real':>10}  "
          f"{'whiff|sw_pred':>13} {'whiff|sw_real':>13}  {'cs|take_pred':>12} {'cs|take_real':>12}")

    def row(mask, label):
        n = int(mask.sum())
        if n < 300:
            return
        sw_m = mask & real_swing
        tk_m = mask & ~real_swing
        # predicted whiff|swing evaluated on REALIZED swings (the population
        # whose realized whiff rate we can measure)
        print(f"{label:>5} {n:>7,}  {p_swing[mask].mean():>10.3f} "
              f"{real_swing[mask].mean():>10.3f}  "
              f"{p_whiff[sw_m].mean():>13.3f} {real_whiff[sw_m].mean():>13.3f}  "
              f"{p_cs[tk_m].mean():>12.3f} {real_cs[tk_m].mean():>12.3f}")

    for b in range(4):
        for s in range(3):
            row((balls == b) & (strikes == s), f"{b}-{s}")
    row(strikes == 2, "ALL-2s")
    row(np.ones(len(df), dtype=bool), "ALL")

    # ---- The K-critical composite: P(strike-3 event) on 2-strike pitches ----
    # A 2-strike pitch ends in a K iff whiff OR called strike (fouls continue).
    m2 = strikes == 2
    pred_k_pitch = (p_swing * p_whiff + (1 - p_swing) * p_cs)[m2]
    real_k_pitch = (real_whiff & real_swing)[m2] | real_cs[m2]
    print(f"\n=== K-ending pitch rate at 2 strikes (the number that drives K%) ===")
    print(f"  predicted = {pred_k_pitch.mean():.4f}")
    print(f"  realized  = {real_k_pitch.mean():.4f}")
    print(f"  shortfall = {real_k_pitch.mean() - pred_k_pitch.mean():+.4f} per 2-strike pitch")
    print(f"  (with ~2.4 two-strike pitches per K-bound AB, per-pitch shortfall")
    print(f"   compounds: this is the cascade-side leak if positive)")


if __name__ == "__main__":
    main()
