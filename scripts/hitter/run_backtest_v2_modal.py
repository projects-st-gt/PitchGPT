"""V2 three-way per-PA backtest: PitchGPTV2+cascade vs lookup+cascade vs baseline.

Same harness as ``scripts.hitter.run_backtest_modal`` (same PA sampling, same
7-class vocab, same cascade + outcome map, same scoring) — only the pitchGPT
pitch source changes: PitchGPTV2 (adaLN + GMM) via ``backtest_v2_remote``.

GATE (from the base-v1c spec): the V2 path must
  1. beat the SAME-SAMPLE lookup log-loss (historical anchor 1.4435), and
  2. land BB%% within 1pp of real (~9.25%%).

    # full three-way run on Modal:
    python -m scripts.hitter.run_backtest_v2_modal --n 800 --n-paths 300

Real Statcast only.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from hitter.backtest import (
    PA_CLASSES,
    aggregate_calibration,
    league_baseline_dist,
    load_terminal_pas,
    sample_pas,
)
from scripts.hitter.run_backtest_modal import (
    AUG_DIR,
    HITTER_DIR,
    _fill_none,
    _report,
    compute_lookup_dists,
)

GATE_LOOKUP_LOGLOSS = 1.4435   # historical lookup anchor (2026-06-05 run)
GATE_BB_TOLERANCE = 0.01       # BB%% must be within 1pp of real


def main() -> None:
    ap = argparse.ArgumentParser(description="V2 three-way per-PA backtest")
    ap.add_argument("--n", type=int, default=800, help="sampled PAs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-start", default="2024-07-16")
    ap.add_argument("--window-end", default="2025-12-31")
    ap.add_argument("--baseline-year", default="2023")
    ap.add_argument("--max-per-matchup", type=int, default=3)
    ap.add_argument("--outcome-mode", default="xwoba")
    ap.add_argument("--n-paths", type=int, default=300, help="MC paths/PA (V2)")
    ap.add_argument("--chunk", type=int, default=25, help="PAs per Modal task")
    ap.add_argument("--ckpt", default="/data/checkpoints/tiny-v1c-base/checkpoint_calibrated.pt",
                    help="V2 calibrated checkpoint path ON THE MODAL VOLUME")
    ap.add_argument("--skip-lookup", action="store_true",
                    help="skip the lookup anchor (faster; gate vs historical only)")
    args = ap.parse_args()

    t0 = time.time()
    # 1) sample held-out PAs (the shared evaluation set)
    print("=== PHASE 1/3: sample held-out PAs + baseline (LOCAL) ===", flush=True)
    pas = load_terminal_pas(AUG_DIR, start=args.window_start, end=args.window_end)
    sample = sample_pas(pas, args.n, seed=args.seed,
                        max_per_matchup=args.max_per_matchup)
    actuals = list(sample["outcome"])
    print(f"[sample] {len(sample)} PAs  ({time.time()-t0:.0f}s)")
    print("[sample] outcome marginal:",
          {k: round(actuals.count(k) / len(actuals), 4) for k in PA_CLASSES})

    train_pas = load_terminal_pas(AUG_DIR, start=f"{args.baseline_year}-01-01",
                                  end=f"{args.baseline_year}-12-31", verbose=False)
    baseline = league_baseline_dist(train_pas)
    print(f"[baseline] {args.baseline_year} marginal:",
          {k: round(baseline[k], 4) for k in PA_CLASSES})
    base_dists = [baseline] * len(actuals)

    # 2) lookup + cascade (local, analytic) — the same-sample gate anchor
    lookup_dists = None
    if not args.skip_lookup:
        print("\n=== PHASE 2/3: lookup + cascade (LOCAL, ~2.5s/PA) ===", flush=True)
        from hitter.rollout import load_hitter_ctx
        ctx = load_hitter_ctx(HITTER_DIR)
        t1 = time.time()
        lookup_raw = compute_lookup_dists(
            sample, ctx, start=args.window_start, end=args.window_end,
            outcome_mode=args.outcome_mode)
        lookup_dists = _fill_none(lookup_raw, baseline)
        print(f"[lookup] computed {len(lookup_dists)} dists ({time.time()-t1:.0f}s)")

    # 3) PitchGPTV2 + cascade (Modal fan-out, capped at 10 containers)
    print("\n=== PHASE 3/3: PitchGPTV2 + cascade (MODAL T4, cap 10) ===", flush=True)
    from modal_app import app, backtest_v2_remote
    specs = [{"idx": i, "pitcher_id": int(r["pitcher"]), "batter_id": int(r["batter"]),
              "throws": str(r["p_throws"]), "stand": str(r["stand"]),
              "game_date": pd.Timestamp(r["game_date"]).strftime("%Y-%m-%d"),
              "game_pk": int(r["game_pk"])}
             for i, (_, r) in enumerate(sample.iterrows())]
    tasks = [{"specs": specs[i:i + args.chunk], "n_paths": args.n_paths,
              "rng_seed": 1000 + i, "ckpt": args.ckpt}
             for i in range(0, len(specs), args.chunk)]
    print(f"[v2] ckpt={args.ckpt}")
    print(f"[v2] {len(specs)} PAs in {len(tasks)} Modal tasks "
          f"(n_paths={args.n_paths}, cap 10 containers)")

    v2_dists: list[dict | None] = [None] * len(specs)
    t2 = time.time()
    done = 0
    with app.run():
        for res in backtest_v2_remote.map(tasks):
            for item in res:
                d = {k: float(item["dist"].get(k, 0.0)) for k in PA_CLASSES}
                vals = np.array(list(d.values()))
                if np.all(np.isfinite(vals)) and vals.sum() > 0:
                    v2_dists[item["idx"]] = {k: v / vals.sum() for k, v in d.items()}
            done += len(res)
            el = time.time() - t2
            eta = el / done * (len(specs) - done) if done else 0
            print(f"  [v2 {done}/{len(specs)} PAs] {el:.0f}s elapsed, "
                  f"ETA {eta:.0f}s", flush=True)
    missing = sum(d is None for d in v2_dists)
    if missing:
        print(f"[v2] WARNING {missing} PAs missing a dist — filling baseline")
    v2_filled = _fill_none(v2_dists, baseline)

    # 4) scoring + gate verdict
    print("\n=== LOG-LOSS (lower = better) ===")
    _report("baseline (league avg)", base_dists, actuals)
    lookup_res = None
    if lookup_dists is not None:
        lookup_res = _report("lookup + cascade", lookup_dists, actuals)
    v2_res = _report("PitchGPTV2 + cascade", v2_filled, actuals)

    print("\n=== V2 calibration-in-aggregate (mean pred vs real) ===")
    print(aggregate_calibration(v2_filled, actuals).to_string(index=False))

    bb_pred = float(np.mean([d["BB"] for d in v2_filled]))
    bb_real = actuals.count("BB") / len(actuals)
    lookup_anchor = lookup_res["logloss"] if lookup_res else GATE_LOOKUP_LOGLOSS
    anchor_name = "same-sample lookup" if lookup_res else f"historical lookup {GATE_LOOKUP_LOGLOSS}"
    pass_ll = v2_res["logloss"] < lookup_anchor
    pass_bb = abs(bb_pred - bb_real) <= GATE_BB_TOLERANCE
    print("\n=== GATE ===")
    print(f"  log-loss: V2 {v2_res['logloss']:.4f} vs {anchor_name} "
          f"{lookup_anchor:.4f} -> {'PASS' if pass_ll else 'FAIL'}")
    print(f"  BB%:      V2 {bb_pred:.3f} vs real {bb_real:.3f} "
          f"(|diff| {abs(bb_pred-bb_real):.3f}, tol {GATE_BB_TOLERANCE}) "
          f"-> {'PASS' if pass_bb else 'FAIL'}")
    print(f"  VERDICT: {'PASS — scale to small' if (pass_ll and pass_bb) else 'FAIL — diagnose'}")
    print(f"\n[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
