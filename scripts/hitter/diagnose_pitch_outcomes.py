"""Clean per-pitch outcome comparison: simulator (live at-bats only) vs real.

Captures, inside g_compute over ACTIVE paths, the per-pitch result distribution
(ball / called-strike / swinging-strike / foul / in-play) and in-zone rate, then
compares to the REAL per-pitch distribution from Statcast `description`.

The walk count is driven by the BALL rate. If the simulator's ball rate is far
below real (~0.33), that's the walk-killer; where it goes instead (called strikes?
swings? fouls?) tells us why.
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import causal.g_computation as gc
from causal.g_computation import g_compute
from causal.nuisance import NuisanceModels
from hitter.rollout import build_cell_step_fn, load_hitter_ctx
from mcsim.state import ReferenceContext
from mcsim.matchup_card import build_synthetic_ab

CKPT = Path("checkpoints_modal/small-fold0-v7/checkpoint_calibrated.pt")
# RESULT_ORDER: 0 ball,1 called_strike,2 swinging_strike,3 foul,4 in_play_out,5 in_play_hit,6 in_play_hr
CLASS5 = ["ball", "called_strike", "swinging_strike", "foul", "in_play"]

_DESC_TO5 = {
    "ball": "ball", "blocked_ball": "ball", "pitchout": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike", "swinging_strike_blocked": "swinging_strike",
    "foul_tip": "swinging_strike", "missed_bunt": "swinging_strike",
    "foul": "foul", "foul_bunt": "foul",
    "hit_into_play": "in_play",
}


def real_dist():
    fs = sorted(glob.glob("data/augmented/2024/2024-08-0[1-3]*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    c5 = df["description"].map(_DESC_TO5).dropna()
    vc = c5.value_counts(normalize=True)
    return {k: float(vc.get(k, 0.0)) for k in CLASS5}, float((df["feature_zone"] < 9).mean())


def rollout_dist(nz, ctx, n_matchups=4):
    df = pd.read_parquet(sorted(glob.glob("data/augmented/2024/2024-08-01*.parquet"))[0])
    df = df[df["pitch_number"] == 1].drop_duplicates(["pitcher", "batter"]).head(n_matchups)
    gc._INZONE_CAPTURE.clear()
    for _, r in df.iterrows():
        step = build_cell_step_fn(ctx, pitcher_id=int(r["pitcher"]), batter_id=int(r["batter"]),
                                  stand=str(r["stand"]), throws=str(r["p_throws"]),
                                  game_date="2024-08-01")
        ab = build_synthetic_ab(pitcher_id=int(r["pitcher"]), batter_id=int(r["batter"]),
                                game_date="2024-08-01", pitcher_throws=str(r["p_throws"]),
                                batter_stand=str(r["stand"]), ballpark_id=int(r["ballpark_id"]),
                                umpire_id=int(r["umpire_id"]), catcher_id=int(r["catcher_id"]),
                                context=ReferenceContext(), game_pk=1)
        g_compute(nz, ab, intervention_position=0, intervention_type=None,
                  n_paths=400, rng_seed=1, outcome_model="hitter", hitter_step_fn=step)
    tot_active = 0; rc = np.zeros(7); in_zone = 0
    for (s, nact, nin, rcounts) in gc._INZONE_CAPTURE:
        tot_active += nact; in_zone += nin; rc += np.array(rcounts)
    # collapse 7 -> 5 (in_play = 4+5+6)
    d5 = {"ball": rc[0], "called_strike": rc[1], "swinging_strike": rc[2],
          "foul": rc[3], "in_play": rc[4] + rc[5] + rc[6]}
    s = sum(d5.values())
    return {k: float(d5[k] / s) for k in CLASS5}, in_zone / max(tot_active, 1), int(tot_active)


def main():
    nz = NuisanceModels(CKPT, device="cpu")
    ctx = load_hitter_ctx("checkpoints/hitter")
    real, real_inzone = real_dist()
    sim, sim_inzone, n = rollout_dist(nz, ctx)
    print(f"=== PER-PITCH OUTCOME: simulator (live, n={n}) vs REAL ===")
    print(f"{'class':>16} {'SIM':>8} {'REAL':>8} {'diff':>8}")
    for k in CLASS5:
        print(f"{k:>16} {sim[k]:>8.3f} {real[k]:>8.3f} {sim[k]-real[k]:>+8.3f}")
    print(f"\n  in-zone rate: SIM={sim_inzone:.3f}  REAL={real_inzone:.3f}")
    print(f"\n  KEY: ball rate SIM {sim['ball']:.3f} vs REAL {real['ball']:.3f} "
          f"-> {'TOO LOW (walk-killer)' if sim['ball'] < real['ball'] - 0.03 else 'about right'}")


if __name__ == "__main__":
    main()
