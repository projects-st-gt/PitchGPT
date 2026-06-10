"""Compare V2 ROLLOUT pitch marginals against real held-out pitches.

The v9 diagnostic (diagnose_rollout_marginals) compared TEACHER-FORCED
marginals; this one instruments the actual closed-loop rollout via
``g_compute_v2(step_capture_fn=...)`` — the regime the backtest scores.
All captures are masked to ACTIVE paths (the 2026-06-05 measurement bug).

Named outputs:
  - per-count pitch-type distribution: rollout vs real (delta + entropy)
  - in-zone rate by count: rollout vs real
  - per-pitch ball rate: rollout (cascade result probs) vs real
  - in-play expected outcome dist + xwOBA-map bin occupancy: rollout vs the
    map's design assumption (uniform over predicted-xwOBA quantile bins)

Run: python -m scripts.hitter.diagnose_v2_rollout_marginals \
        --dists data/backtests/v2_n800_p300_seed0.json --n-pas 60 --n-paths 200
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal.nuisance_v2 import NuisanceModelsV2
from causal.g_computation_v2 import g_compute_v2
from data.dataset import PITCH_TYPES
from hitter.rollout import load_hitter_ctx, build_cell_step_fn
from mcsim.state import ReferenceContext, build_synthetic_ab

BALL_COL = 0  # RESULT_ORDER index of "ball" in cascade result probs


def load_real_reference(aug_dir: str, months: list[str]) -> pd.DataFrame:
    files = []
    for m in months:
        files += sorted(glob.glob(f"{aug_dir}/{m[:4]}/{m}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets for months {months} under {aug_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["in_zone"] = ((df["plate_x"].abs() <= 0.83)
                     & (df["plate_z"] >= 1.5) & (df["plate_z"] <= 3.5))
    df["is_ball"] = df["result_id"] == 1  # RESULT_ID_OFFSET: ball = 1
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", default="data/backtests/v2_n800_p300_seed0.json",
                    help="saved backtest JSON — provides the PA sample to roll out")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints_modal/tiny-v1c-base/checkpoint_calibrated.pt"))
    ap.add_argument("--n-pas", type=int, default=60)
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--real-months", nargs="+",
                    default=["2024-08", "2024-09"],
                    help="held-out months for the real reference marginals")
    ap.add_argument("--aug-dir", default="data/augmented")
    ap.add_argument("--hitter-dir", default="checkpoints/hitter")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--real-handedness", action="store_true",
                    help="rebuild the deterministic PA sample to recover each "
                         "matchup's true throws/stand (the saved JSON lacks them; "
                         "without this flag every matchup is rolled out as R-vs-R)")
    args = ap.parse_args()

    payload = json.load(open(args.dists))
    pas = payload["pas"][: args.n_pas]
    print(f"rolling out {len(pas)} PAs x {args.n_paths} paths from {args.dists}")

    hands: dict[int, tuple[str, str]] = {}
    if args.real_handedness:
        from hitter.backtest import load_terminal_pas, sample_pas
        meta = payload["meta"]
        full = load_terminal_pas(args.aug_dir, start=meta["window"][0],
                                 end=meta["window"][1], verbose=False)
        smp = sample_pas(full, meta["n"], seed=meta["seed"], max_per_matchup=3)
        for i, (_, r) in enumerate(smp.iterrows()):
            assert int(r["pitcher"]) == pas[i]["pitcher"] if i < len(pas) else True, \
                "rebuilt sample does not match saved JSON order"
            hands[i] = (str(r["p_throws"]), str(r["stand"]))
        print(f"recovered real handedness for {len(hands)} PAs "
              f"(deterministic resample, seed={meta['seed']})")

    nz = NuisanceModelsV2(args.ckpt)
    ctx = load_hitter_ctx(args.hitter_dir)
    xmap = json.load(open(f"{args.hitter_dir}/xwoba_outcome_map.json"))
    map_rows = np.asarray(xmap["dist"], dtype=float)          # (n_bins, 5)
    n_bins = len(map_rows)

    caps: list[dict] = []
    context = ReferenceContext()
    import time
    t0 = time.time()
    for i, p in enumerate(pas, 1):
        throws, stand = hands.get(i - 1, ("R", "R"))
        throws = throws if throws in ("R", "L") else "R"
        stand = stand if stand in ("R", "L") else "R"
        ab = build_synthetic_ab(
            pitcher_id=p["pitcher"], batter_id=p["batter"], game_date=p["game_date"],
            pitcher_throws=throws, batter_stand=stand,
            context=context, game_pk=p["game_pk"])
        step_fn = build_cell_step_fn(
            ctx, pitcher_id=p["pitcher"], batter_id=p["batter"],
            stand=stand, throws=throws, game_date=p["game_date"])
        g_compute_v2(nz, ab, n_paths=args.n_paths, rng_seed=args.seed + i,
                     hitter_step_fn=step_fn, step_capture_fn=caps.append)
        if i % 10 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(pas)}] {el:.0f}s elapsed, "
                  f"ETA {el / i * (len(pas) - i):.0f}s", flush=True)

    # ---- Flatten captures, ACTIVE-masked ---------------------------------
    def cat(key):
        return np.concatenate([c[key][c["active"]] for c in caps])

    types = cat("type_1idx")            # 1..7
    balls_pre = cat("balls"); strikes_pre = cat("strikes")
    px = cat("plate_x"); pz = cat("plate_z")
    rp = np.concatenate([c["result_probs"][c["active"]] for c in caps])
    oc5 = np.concatenate([c["outcome5"][c["active"]] for c in caps])
    in_zone = (np.abs(px) <= 0.83) & (pz >= 1.5) & (pz <= 3.5)
    n_pitches = len(types)
    print(f"\ncaptured {n_pitches:,} active rollout pitch-paths")

    real = load_real_reference(args.aug_dir, args.real_months)
    print(f"real reference: {len(real):,} pitches ({'+'.join(args.real_months)})")

    # ---- 1. per-count type distribution ----------------------------------
    print("\n=== per-count pitch-type distribution: rollout - real (pp) ===")
    hdr = "count   n_roll" + "".join(f"  {t:>6}" for t in PITCH_TYPES) + "   |delta|"
    print(hdr)
    total_abs = 0.0; n_counts = 0
    for b in range(4):
        for s in range(3):
            m_roll = (balls_pre == b) & (strikes_pre == s)
            r_rows = real[(real["count_state"] == b * 3 + s)]
            if m_roll.sum() < 50 or len(r_rows) < 200:
                continue
            roll_d = np.array([(types[m_roll] == t + 1).mean() for t in range(7)])
            real_d = np.array([(r_rows["type_id"] == t + 1).mean() for t in range(7)])
            delta = roll_d - real_d
            total_abs += np.abs(delta).sum(); n_counts += 1
            print(f"  {b}-{s}  {int(m_roll.sum()):>7,}"
                  + "".join(f"  {d*100:>+6.1f}" for d in delta)
                  + f"   {np.abs(delta).sum()*100:>6.1f}pp")
    print(f"  mean total |delta| per count = {total_abs / max(n_counts,1) * 100:.1f}pp")

    # ---- 2. in-zone + ball rate by count ----------------------------------
    print("\n=== in-zone rate + ball rate by count: rollout vs real ===")
    print("count   inzone_roll  inzone_real     ball_roll  ball_real")
    for b in range(4):
        for s in range(3):
            m_roll = (balls_pre == b) & (strikes_pre == s)
            r_rows = real[(real["count_state"] == b * 3 + s)]
            if m_roll.sum() < 50 or len(r_rows) < 200:
                continue
            print(f"  {b}-{s}   {in_zone[m_roll].mean():>10.3f}  {r_rows['in_zone'].mean():>11.3f}"
                  f"   {rp[m_roll, BALL_COL].mean():>11.3f}  {r_rows['is_ball'].mean():>9.3f}")
    print(f"  ALL   {in_zone.mean():>10.3f}  {real['in_zone'].mean():>11.3f}"
          f"   {rp[:, BALL_COL].mean():>11.3f}  {real['is_ball'].mean():>9.3f}")

    # ---- 3. in-play split: bin occupancy + expected outcome dist ----------
    # Each oc5 row IS a map row; identify the bin by nearest-row match.
    d2 = ((oc5[:, None, :] - map_rows[None, :, :]) ** 2).sum(-1)
    bins = d2.argmin(1)
    w = rp[:, 4:].sum(1)                                   # in-play prob weight
    occ = np.bincount(bins, weights=w, minlength=n_bins) / max(w.sum(), 1e-12)
    print(f"\n=== xwOBA-map bin occupancy (in-play-weighted) ===")
    print(f"  map design: ~uniform 1/{n_bins} = {1/n_bins:.3f} per bin (quantiles of the")
    print(f"  2023 real-pitch predicted-xwOBA population)")
    lo, mid, hi = occ[: n_bins // 3].sum(), occ[n_bins // 3: 2 * n_bins // 3].sum(), occ[2 * n_bins // 3:].sum()
    print(f"  rollout occupancy: low-third {lo:.3f}  mid-third {mid:.3f}  high-third {hi:.3f}")
    print(f"  (uniform would be 0.333 each; high-third > 0.40 = contact-quality skew")
    print(f"   -> inflated 2B/HR at the expense of outs)")
    exp_ip = (oc5 * w[:, None]).sum(0) / max(w.sum(), 1e-12)
    print(f"  expected in-play dist [out,1B,2B,3B,HR] = "
          f"{np.round(exp_ip, 4).tolist()}")
    real_ip = real[real["result_id"].isin([5, 6, 7])]
    print(f"  (real in-play outcome dist printed by the backtest aggregates; "
          f"n real in-play rows here = {len(real_ip):,})")


if __name__ == "__main__":
    main()
