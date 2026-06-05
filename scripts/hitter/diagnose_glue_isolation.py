"""Which simulator approximation strangles the ball rate? Isolate each.

On REAL held-out pitches (cascade validated here), feed the cascade the real
features, then swap in each simulator approximation one at a time and measure the
effect on ball rate / swing rate / whiff rate:

  loc -> zone centroid      (build_step_features uses the zone center)
  velo -> per-type mean
  spin_axis -> 0,0          (blank)
  ALL three together        (= what the rollout actually feeds)

Real per-pitch reference: ball 0.350, swing ~0.484, whiff(of swings) ~0.26.
Rollout gave ball 0.305. Whichever swap drops ball toward 0.305 is the culprit;
that's the thing to fix (e.g. sample a real location within the zone).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from hitter.rollout import load_hitter_ctx
from hitter.train import build_inference_features

N = 25000


def rates(ctx, X):
    c = ctx["hm"].predict_cascade(X)
    s = np.clip(c["swing"], 0, 1); cs = np.clip(c["called_strike"], 0, 1)
    w = np.clip(c["whiff"], 0, 1)
    ball = (1 - s) * (1 - cs)
    return float(ball.mean()), float(s.mean()), float(w.mean())


def main():
    ctx = load_hitter_ctx("checkpoints/hitter")
    import json
    cent = {int(k): v for k, v in json.load(open("checkpoints/hitter/zone_centroids.json")).items()}
    fs = sorted(glob.glob("data/augmented/2024/2024-08-0[1-3]*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    rng = np.random.default_rng(0)
    df = df.iloc[rng.permutation(len(df))[:N]].reset_index(drop=True)
    real_ball = float((df["description"].isin(["ball", "blocked_ball", "pitchout"])).mean())

    X = build_inference_features(df, ctx["bc"], ctx["pc"])
    fz = X["feature_zone"].astype(int).to_numpy() if "feature_zone" in X else df["feature_zone"].astype(int).to_numpy()
    tid = X["type_id"].astype(int).to_numpy() if "type_id" in X else df["type_id"].astype(int).to_numpy()
    # per-type means for the velo/spin swaps
    velo_mean = pd.Series(X["release_speed"].to_numpy()).groupby(tid).transform("mean").to_numpy()
    spin_mean = pd.Series(X["release_spin_rate"].to_numpy()).groupby(tid).transform("mean").to_numpy()

    def swap(loc=False, velo=False, spin=False):
        Y = X.copy()
        if loc:
            Y["plate_x"] = np.array([cent.get(z, [0.0, 2.5])[0] for z in fz], "float32")
            Y["plate_z"] = np.array([cent.get(z, [0.0, 2.5])[1] for z in fz], "float32")
            if "in_zone" in Y: Y["in_zone"] = (fz < 9).astype("int8")
        if velo:
            Y["release_speed"] = velo_mean.astype("float32")
        if spin:
            if "spin_axis_sin" in Y: Y["spin_axis_sin"] = np.zeros(len(Y), "float32")
            if "spin_axis_cos" in Y: Y["spin_axis_cos"] = np.zeros(len(Y), "float32")
            Y["release_spin_rate"] = spin_mean.astype("float32")
        return Y

    print(f"real ball rate (ground truth) = {real_ball:.3f}   rollout ball rate ~ 0.305\n")
    print(f"{'variant':>26} {'ball':>7} {'swing':>7} {'whiff':>7}")
    for name, kw in [("ALL REAL (baseline)", {}),
                     ("loc=centroid", dict(loc=True)),
                     ("velo=type-mean", dict(velo=True)),
                     ("spin=blank", dict(spin=True)),
                     ("ALL glue (rollout-like)", dict(loc=True, velo=True, spin=True))]:
        b, s, w = rates(ctx, swap(**kw))
        print(f"{name:>26} {b:>7.3f} {s:>7.3f} {w:>7.3f}")
    print("\n  whichever swap drops 'ball' toward 0.305 (and raises swing) is the culprit.")


if __name__ == "__main__":
    main()
