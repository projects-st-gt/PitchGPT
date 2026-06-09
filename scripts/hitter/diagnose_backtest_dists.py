"""Decompose a saved backtest's log-loss gap: which PAs / classes drive it?

Reads the per-PA dists JSON written by ``run_backtest_v2_modal --save-dists``
and answers, with named numbers:

  1. Floor events: how many PAs got ~zero predicted mass on the ACTUAL
     outcome, per source? (A 300-path MC estimate hits exact zeros; the
     analytic lookup never does. At eps=1e-9 one zero costs ~20.7/n nats.)
  2. Laplace re-score: re-score the MC source with add-one smoothing over
     paths ((p*n_paths + 1) / (n_paths + 7)) — the statistically honest
     posterior-mean for a finite-sample categorical estimate. How much of
     the gap is finite-path artifact vs real signal?
  3. Per-class decomposition: mean NLL contribution by actual class.
  4. Discrimination: entropy + spread of P(class) across PAs per source —
     is V2 overconfident relative to how much matchups really differ?

Run: python -m scripts.hitter.diagnose_backtest_dists \
        --dists data/backtests/v2_n800_p300_seed0.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from hitter.backtest import PA_CLASSES

EPS = 1e-9


def _nll(dists: list[dict], actuals: list[str]) -> np.ndarray:
    return np.array([-np.log(max(d.get(a, 0.0), EPS))
                     for d, a in zip(dists, actuals)])


def _laplace(d: dict, n_paths: int) -> dict:
    # counts = p * n_paths; posterior mean with Dirichlet(1) prior.
    sm = {k: (d.get(k, 0.0) * n_paths + 1.0) / (n_paths + len(PA_CLASSES))
          for k in PA_CLASSES}
    s = sum(sm.values())
    return {k: v / s for k, v in sm.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", required=True)
    ap.add_argument("--floor-thresh", type=float, default=1e-6,
                    help="prob below this on the actual outcome counts as a floor event")
    args = ap.parse_args()

    with open(args.dists) as f:
        payload = json.load(f)
    pas = payload["pas"]
    n_paths = int(payload["meta"]["n_paths"])
    actuals = [p["actual"] for p in pas]
    n = len(pas)
    sources = [s for s in ("v2", "lookup", "baseline") if pas[0].get(s)]
    print(f"{args.dists}: {n} PAs, n_paths={n_paths}, sources={sources}")

    # ---- 1. headline + floor events -------------------------------------
    print(f"\n=== headline log-loss (eps={EPS}) ===")
    nlls = {}
    for s in sources:
        dists = [p[s] for p in pas]
        nlls[s] = _nll(dists, actuals)
        floor = [(i, p) for i, p in enumerate(pas)
                 if p[s].get(p["actual"], 0.0) < args.floor_thresh]
        floor_cost = sum(-np.log(max(p[s].get(p["actual"], 0.0), EPS))
                         for _, p in floor) / n
        print(f"  {s:<9} log-loss = {nlls[s].mean():.4f}   "
              f"floor events (p<{args.floor_thresh:g} on actual) = {len(floor)}  "
              f"costing {floor_cost:.4f} of the total")
        for i, p in floor[:8]:
            print(f"      idx {p['idx']}: actual={p['actual']}  "
                  f"p={p[s].get(p['actual'], 0.0):.2e}  "
                  f"pitcher {p['pitcher']} vs batter {p['batter']}")

    # ---- 2. Laplace re-score for the MC source --------------------------
    if "v2" in sources:
        v2_sm = [_laplace(p["v2"], n_paths) for p in pas]
        nll_sm = _nll(v2_sm, actuals)
        print(f"\n=== Laplace add-one re-score (MC finite-path correction) ===")
        print(f"  v2 raw      = {nlls['v2'].mean():.4f}")
        print(f"  v2 smoothed = {nll_sm.mean():.4f}   "
              f"(gap explained by floor artifact: "
              f"{nlls['v2'].mean() - nll_sm.mean():.4f})")
        if "lookup" in sources:
            lk_sm = [_laplace(p["lookup"], n_paths) for p in pas]
            print(f"  lookup smoothed (same treatment, fairness) = "
                  f"{_nll(lk_sm, actuals).mean():.4f}  "
                  f"(raw {nlls['lookup'].mean():.4f})")

    # ---- 3. per-class NLL decomposition ----------------------------------
    print(f"\n=== mean NLL contribution by actual class (nats/PA) ===")
    hdr = "class  n_real" + "".join(f"  {s:>10}" for s in sources)
    if "v2" in sources and "lookup" in sources:
        hdr += "   v2-lookup"
    print(hdr)
    for c in PA_CLASSES:
        idx = [i for i, a in enumerate(actuals) if a == c]
        row = f"{c:>5}  {len(idx):>6}"
        vals = {}
        for s in sources:
            contrib = nlls[s][idx].sum() / n if idx else 0.0
            vals[s] = contrib
            row += f"  {contrib:>10.4f}"
        if "v2" in vals and "lookup" in vals:
            row += f"   {vals['v2'] - vals['lookup']:>+9.4f}"
        print(row)

    # ---- 4. discrimination / confidence ----------------------------------
    print(f"\n=== discrimination: per-source entropy + P(class) spread ===")
    for s in sources:
        dists = [p[s] for p in pas]
        P = np.array([[d.get(k, 0.0) for k in PA_CLASSES] for d in dists])
        ent = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        print(f"  {s:<9} mean entropy = {ent.mean():.4f} nats   "
              f"std P(out) = {P[:, PA_CLASSES.index('out')].std():.4f}   "
              f"std P(K) = {P[:, PA_CLASSES.index('K')].std():.4f}   "
              f"mean p(actual) = {np.array([d.get(a, 0.0) for d, a in zip(dists, actuals)]).mean():.4f}")


if __name__ == "__main__":
    main()
