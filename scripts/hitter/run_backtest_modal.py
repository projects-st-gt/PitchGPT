"""Three-way per-PA backtest driver: pitchGPT+cascade vs lookup+cascade vs baseline.

Grades the simulator's PITCH SOURCE on real held-out 2024H2 at-bats with a proper
score (per-PA log-loss, lower=better). All three are scored on the SAME sampled
PAs, SAME 7-class vocab, SAME cascade + outcome map — only the pitch source differs.

The lookup path (analytic count tree) and the league baseline run locally on CPU.
The pitchGPT path (MC rollout) fans out to Modal T4s (``backtest_remote``, capped at
10 containers).

    # local sanity FIRST — must land near lookup 1.405 / baseline 1.452:
    python -m scripts.hitter.run_backtest_modal --sanity-local --n 120 \
        --window-start 2024-08-01 --window-end 2024-08-31

    # full three-way run on Modal:
    python -m scripts.hitter.run_backtest_modal --n 1200 --n-paths 300

Real Statcast only.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hitter.backtest import (
    PA_CLASSES,
    aggregate_calibration,
    bootstrap_logloss_ci,
    league_baseline_dist,
    load_terminal_pas,
    lookup_dist_for_matchup,
    sample_pas,
)

AUG_DIR = "data/augmented"
HITTER_DIR = "checkpoints/hitter"


def _aug_files(start: str, end: str) -> list[str]:
    y0, y1 = int(start[:4]), int(end[:4])
    files = []
    for y in range(y0, y1 + 1):
        files += sorted(glob.glob(f"{AUG_DIR}/{y}/*.parquet"))
    return files


def load_pitcher_pitches(pitcher_ids: set[int], start: str, end: str
                         ) -> dict[int, pd.DataFrame]:
    """Load each sampled pitcher's augmented pitches over [start, end], once."""
    want = set(int(p) for p in pitcher_ids)
    acc: dict[int, list[pd.DataFrame]] = {p: [] for p in want}
    for f in _aug_files(start, end):
        df = pd.read_parquet(f)
        df = df[df["pitcher"].isin(want)]
        if df.empty:
            continue
        for pid, g in df.groupby("pitcher"):
            acc[int(pid)].append(g)
    out = {p: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
           for p, v in acc.items()}
    return out


def compute_lookup_dists(sample: pd.DataFrame, ctx: dict, *, start: str, end: str,
                         outcome_mode: str = "xwoba") -> list[dict]:
    """Lookup+cascade per-PA dist for each sampled PA (cached per matchup)."""
    print(f"[lookup] loading in-window pitches for "
          f"{sample['pitcher'].nunique()} pitchers...", flush=True)
    t_load = time.time()
    pp_by_pitcher = load_pitcher_pitches(set(sample["pitcher"]), start, end)
    print(f"[lookup] pitch load done ({time.time()-t_load:.0f}s); "
          f"scoring {len(sample)} matchups...", flush=True)
    cache: dict[tuple[int, int], dict] = {}
    dists: list[dict] = []
    n_no_pitches = 0
    n_degenerate = 0
    t_loop = time.time()
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        if i % 50 == 0 or i == len(sample):
            el = time.time() - t_loop
            eta = el / i * (len(sample) - i)
            print(f"  [lookup {i}/{len(sample)}] {el:.0f}s elapsed, ETA {eta:.0f}s "
                  f"({n_no_pitches + n_degenerate} fallbacks so far)", flush=True)
        pid, bid = int(r["pitcher"]), int(r["batter"])
        key = (pid, bid)
        if key not in cache:
            pp = pp_by_pitcher.get(pid)
            if pp is None or pp.empty:
                # pitcher has no augmented pitches in-window — fall back to the
                # league baseline for this PA rather than fabricate a dist.
                cache[key] = None
                n_no_pitches += 1
            else:
                d = lookup_dist_for_matchup(ctx, pp, bid, outcome_mode=outcome_mode)
                if d is None:           # degenerate count-tree solve (NaN/inf)
                    n_degenerate += 1
                cache[key] = d
        dists.append(cache[key])
    n_fb = sum(d is None for d in dists)
    if n_fb:
        print(f"[lookup] {n_fb}/{len(dists)} PAs fell back to baseline "
              f"({n_no_pitches} matchups w/ no in-window pitches, "
              f"{n_degenerate} degenerate solves)")
    return dists


def _fill_none(dists: list[dict], fallback: dict) -> list[dict]:
    return [d if d is not None else fallback for d in dists]


def _report(name: str, dists: list[dict], actuals: list[str]) -> dict:
    res = bootstrap_logloss_ci(dists, actuals)
    print(f"  {name:24s} log-loss = {res['logloss']:.4f}  "
          f"[95% CI {res['ci_lo']:.4f}, {res['ci_hi']:.4f}]  n={res['n']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-way per-PA backtest")
    ap.add_argument("--n", type=int, default=1200, help="sampled PAs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-start", default="2024-07-16")
    ap.add_argument("--window-end", default="2025-12-31")
    ap.add_argument("--baseline-year", default="2023",
                    help="train-split season for the league-average baseline marginal")
    ap.add_argument("--max-per-matchup", type=int, default=3)
    ap.add_argument("--outcome-mode", default="xwoba")
    ap.add_argument("--n-paths", type=int, default=300, help="MC paths/PA (pitchGPT)")
    ap.add_argument("--chunk", type=int, default=25, help="PAs per Modal task")
    ap.add_argument("--sanity-local", action="store_true",
                    help="lookup + baseline only, no Modal (anchor check)")
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

    # 2) league baseline (train-split marginal) — leakage-safe constant predictor
    train_pas = load_terminal_pas(AUG_DIR, start=f"{args.baseline_year}-01-01",
                                  end=f"{args.baseline_year}-12-31", verbose=False)
    baseline = league_baseline_dist(train_pas)
    print(f"[baseline] {args.baseline_year} marginal:",
          {k: round(baseline[k], 4) for k in PA_CLASSES})
    base_dists = [baseline] * len(actuals)

    # 3) lookup + cascade (local, analytic)
    print("\n=== PHASE 2/3: lookup + cascade (LOCAL, ~2.5s/PA) ===", flush=True)
    from hitter.rollout import load_hitter_ctx
    ctx = load_hitter_ctx(HITTER_DIR)
    t1 = time.time()
    lookup_raw = compute_lookup_dists(
        sample, ctx, start=args.window_start, end=args.window_end,
        outcome_mode=args.outcome_mode)
    lookup_dists = _fill_none(lookup_raw, baseline)
    print(f"[lookup] computed {len(lookup_dists)} dists ({time.time()-t1:.0f}s)")

    print("\n=== LOG-LOSS (lower = better) ===")
    _report("baseline (league avg)", base_dists, actuals)
    _report("lookup + cascade", lookup_dists, actuals)

    print("\n=== lookup calibration-in-aggregate (mean pred vs real) ===")
    print(aggregate_calibration(lookup_dists, actuals).to_string(index=False))

    if args.sanity_local:
        print(f"\n[sanity] done in {time.time()-t0:.0f}s. "
              f"Expect lookup near 1.405, baseline near 1.452.")
        return

    # 4) pitchGPT + cascade (Modal fan-out, capped at 10 containers)
    print("\n=== PHASE 3/3: pitchGPT + cascade (MODAL T4, cap 10) ===", flush=True)
    from modal_app import app, backtest_remote
    specs = [{"idx": i, "pitcher_id": int(r["pitcher"]), "batter_id": int(r["batter"]),
              "throws": str(r["p_throws"]), "stand": str(r["stand"]),
              "game_date": pd.Timestamp(r["game_date"]).strftime("%Y-%m-%d"),
              "game_pk": int(r["game_pk"])}
             for i, (_, r) in enumerate(sample.iterrows())]
    tasks = [{"specs": specs[i:i + args.chunk], "n_paths": args.n_paths,
              "rng_seed": 1000 + i}
             for i in range(0, len(specs), args.chunk)]
    print(f"\n[pitchGPT] {len(specs)} PAs in {len(tasks)} Modal tasks "
          f"(n_paths={args.n_paths}, cap 10 containers)")

    pg_dists: list[dict | None] = [None] * len(specs)
    t2 = time.time()
    done = 0
    with app.run():
        for res in backtest_remote.map(tasks):
            for item in res:
                d = {k: float(item["dist"].get(k, 0.0)) for k in PA_CLASSES}
                vals = np.array(list(d.values()))
                if np.all(np.isfinite(vals)) and vals.sum() > 0:
                    pg_dists[item["idx"]] = {k: v / vals.sum() for k, v in d.items()}
            done += len(res)
            el = time.time() - t2
            eta = el / done * (len(specs) - done) if done else 0
            print(f"  [pitchGPT {done}/{len(specs)} PAs] {el:.0f}s elapsed, "
                  f"ETA {eta:.0f}s", flush=True)
    missing = sum(d is None for d in pg_dists)
    if missing:
        print(f"[pitchGPT] WARNING {missing} PAs missing a dist — filling baseline")
    pg_filled = _fill_none(pg_dists, baseline)

    print("\n=== LOG-LOSS (lower = better) ===")
    _report("baseline (league avg)", base_dists, actuals)
    _report("lookup + cascade", lookup_dists, actuals)
    _report("pitchGPT + cascade", pg_filled, actuals)

    print("\n=== pitchGPT calibration-in-aggregate (mean pred vs real) ===")
    print(aggregate_calibration(pg_filled, actuals).to_string(index=False))
    print(f"\n[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
