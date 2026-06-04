"""Hitter-model evaluation — the compression diagnostic (the acceptance gate).

MODEL_DESIGN §10 / Hitter_Swing_Model §1: the transformer compresses hitter OPS
~6.8× (real std 0.261 vs model 0.038, Pearson r=0.66) — it regresses every hitter
toward ~.700. The dedicated cascade must recover the spread: target ratio ≈ 1×.

This module computes:
- ``real_ops_by_batter``  : real AVG/OBP/SLG/OPS per batter from held-out events.
- ``select_batter_panel`` : the 6-best + 6-worst (≥min PA) panel.
- model OPS per batter via the cascade + analytic count-tree (compose), against a
  fixed reference pitcher (so all cross-batter spread flows through the cascade).
- ``spread_diagnostic``   : std ratio + Pearson r between real and model OPS.

The pure pieces (real OPS, spread stats, panel) are unit-tested on synthetic data;
the model-OPS path is exercised by ``run_compression_diagnostic`` once the cascade
is trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hitter.train import (
    NODE_OBJECTIVE, binary_ece, node_population,
)

# Event -> per-PA accounting. Each terminal `events` value contributes to the
# AVG/OBP/SLG ledger. Unknown/None events are treated as in-play outs (PA+AB,
# no hit) only if the row is a terminal (events not null).
_HIT_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
_BB = {"walk"}
_HBP = {"hit_by_pitch"}
_SF = {"sac_fly"}
_NON_AB_NON_PA = {"sac_bunt", "catcher_interf", "caught_stealing_2b",
                  "caught_stealing_3b", "caught_stealing_home", "pickoff_1b",
                  "pickoff_2b", "pickoff_3b", "stolen_base_2b", "runner_double_play"}


def real_ops_by_batter(pitches: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Real AVG/OBP/SLG/OPS/PA per batter from terminal-pitch ``events``.

    A PA is a row with non-null ``events`` (the last pitch of the AB). Standard
    sabermetric accounting: AB = PA - BB - HBP - SF - (non-AB events);
    AVG = H/AB, OBP = (H+BB+HBP)/(AB+BB+HBP+SF), SLG = TB/AB, OPS = OBP+SLG.
    """
    term = pitches[pitches["events"].notna()].copy()
    out: dict[int, dict[str, float]] = {}
    for batter, grp in term.groupby("batter"):
        ev = grp["events"].astype(str)
        pa = len(ev)
        bb = ev.isin(_BB).sum()
        hbp = ev.isin(_HBP).sum()
        sf = ev.isin(_SF).sum()
        non_ab = ev.isin(_NON_AB_NON_PA).sum()
        bases = ev.map(_HIT_BASES)
        h = bases.notna().sum()
        tb = bases.fillna(0).sum()
        ab = pa - bb - hbp - sf - non_ab
        obp_den = ab + bb + hbp + sf
        avg = float(h / ab) if ab > 0 else 0.0
        obp = float((h + bb + hbp) / obp_den) if obp_den > 0 else 0.0
        slg = float(tb / ab) if ab > 0 else 0.0
        out[int(batter)] = {"AVG": avg, "OBP": obp, "SLG": slg,
                            "OPS": obp + slg, "PA": int(pa)}
    return out


def held_out_node_metrics(
    model, test_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Per-node AUC / **ECE on HELD-OUT test data** (+ regression RMSE/r).

    This is the HONEST calibration number: the isotonic calibrator was fit on the
    val set, so train-time ECE (reported in meta.json) is ~0 by construction. Here
    we predict on the test population (2024H2+2025) the calibrator never saw, so
    the ECE actually measures generalization (calibration is a project primary
    metric — eval-protocol).
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error

    out: dict[str, dict[str, float]] = {}
    for node in model.nodes:
        X, y = node_population(test_df, node)
        if len(y) == 0:
            continue
        pred = model.predict_node(node, X)
        yv = y.to_numpy()
        if NODE_OBJECTIVE[node] == "binary":
            out[node] = {
                "n": int(len(yv)),
                "auc": float(roc_auc_score(yv, pred)),
                "ece": binary_ece(pred, yv),
                "base_rate": float(yv.mean()),
            }
        else:
            out[node] = {
                "n": int(len(yv)),
                "rmse": float(np.sqrt(mean_squared_error(yv, pred))),
                "pearson": float(np.corrcoef(pred, yv)[0, 1]),
                "mean_target": float(yv.mean()),
            }
    return out


#: Real `events` -> the 7-class PA outcome vocab (compose's TERMINALS). Outcomes
#: outside the vocab (HBP, sac, interference) return None and are excluded from
#: scoring (they're not things the cascade models in v0).
_EVENT_TO_PA_CLASS = {
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "walk": "BB", "intent_walk": "BB",
    "strikeout": "K", "strikeout_double_play": "K",
}
_PA_VOCAB = ["BB", "K", "out", "1B", "2B", "3B", "HR"]
_EXCLUDE_PA = {"hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
               "catcher_interf", "sac_bunt_double_play"}


def pa_outcome_class(events: str) -> str | None:
    """Map a real ``events`` value to the 7-class PA vocab, or None to exclude.

    Hits/walks/Ks map directly; any other at-bat-ending event (field_out, GIDP,
    fielders_choice, error, force_out, …) is an ``out``; HBP/sac/interference are
    excluded (not in the cascade's v0 vocabulary).
    """
    if events in _EXCLUDE_PA:
        return None
    if events in _EVENT_TO_PA_CLASS:
        return _EVENT_TO_PA_CLASS[events]
    return "out"          # any other terminal event is an out


def pa_logloss(pred_dists: list[dict], actuals: list[str]) -> float:
    """Mean negative log-likelihood of the actual PA outcomes under the model's
    predicted per-PA distributions (proper score; lower is better)."""
    eps = 1e-9
    tot = 0.0
    for d, a in zip(pred_dists, actuals):
        tot += -np.log(max(d.get(a, 0.0), eps))
    return float(tot / len(actuals)) if actuals else float("nan")


def select_batter_panel(
    real_ops: dict[int, dict | float],
    pa_by_batter: dict[int, int],
    *,
    n_each: int = 6,
    min_pa: int = 150,
) -> list[int]:
    """The n-best + n-worst batters by OPS among those with >= min_pa.

    ``real_ops`` may map batter -> OPS float or -> the metric dict (uses 'OPS').
    """
    def ops(v):
        return v["OPS"] if isinstance(v, dict) else float(v)

    eligible = [b for b in real_ops if pa_by_batter.get(b, 0) >= min_pa]
    ranked = sorted(eligible, key=lambda b: ops(real_ops[b]))
    if len(ranked) <= 2 * n_each:
        return ranked
    return ranked[:n_each] + ranked[-n_each:]


def build_empirical_pitch_provider(pitcher_pitches, target_batter_id,
                                   batter_cache, pitcher_cache):
    """π̂ provider for compose_pa: the reference pitcher's REAL pitches, grouped
    by count, with the batter swapped to ``target_batter_id`` (so the batter
    profile = target's). Counts the pitcher never threw fall back to his overall
    pitch set. Weights are uniform (each real pitch = one π̂ sample).
    """
    from hitter.train import build_inference_features
    from hitter.compose import COUNTS

    pp = pitcher_pitches.copy()
    pp["batter"] = int(target_batter_id)              # swap -> target's profile
    feats = build_inference_features(pp, batter_cache, pitcher_cache)
    valid = set(COUNTS)
    by_count = {c: g for c, g in feats.groupby(["balls", "strikes"])
                if (int(c[0]), int(c[1])) in valid}
    by_count = {(int(b), int(s)): g for (b, s), g in by_count.items()}

    def provider(count):
        g = by_count.get(count)
        if g is None or len(g) == 0:
            g = feats                                  # fallback: overall mix
        return g, np.ones(len(g))

    return provider


def model_ops_by_batter(panel, pitcher_pitches, hitter_model, xwoba_to_outcome,
                        batter_cache, pitcher_cache, *, outcome_mode="xwoba"):
    """Model OPS per panel batter via the cascade + analytic count-tree, against
    the fixed reference pitcher (so cross-batter spread flows through the cascade).
    ``outcome_mode`` selects the in-play model (multiclass head | xwoba map | auto).
    """
    from hitter.compose import compose_pa

    out: dict[int, dict[str, float]] = {}
    for batter in panel:
        provider = build_empirical_pitch_provider(
            pitcher_pitches, batter, batter_cache, pitcher_cache)
        out[int(batter)] = compose_pa(hitter_model, provider, xwoba_to_outcome,
                                      outcome_mode=outcome_mode)
    return out


def run_compression_diagnostic(
    model, xwoba_to_outcome, test_aug: pd.DataFrame,
    batter_cache, pitcher_cache, *,
    n_each: int = 6, min_pa: int = 150, n_ref_pitchers: int = 5,
) -> dict:
    """THE acceptance gate: real vs cascade-model OPS spread over a batter panel.

    Panel = n_each best + n_each worst by real OPS (>= min_pa). For each of the
    top ``n_ref_pitchers`` (by pitch count) reference pitchers, compute model OPS
    per batter (cascade + analytic count-tree) and the spread ratio + Pearson r,
    then average across reference pitchers (robustness — no single-pitcher fluke).

    Returns the panel, per-(ref) diagnostics, and the averaged spread_ratio / r.
    Transformer baseline: 6.8x / r=0.66; target ~1x.
    """
    real = real_ops_by_batter(test_aug)
    pa = {b: real[b]["PA"] for b in real}
    panel = select_batter_panel(real, pa, n_each=n_each, min_pa=min_pa)
    refs = [int(x) for x in test_aug["pitcher"].value_counts().head(n_ref_pitchers).index]

    per_ref = []
    model_ops_acc: dict[int, list[float]] = {b: [] for b in panel}
    for ref in refs:
        rp = test_aug[test_aug["pitcher"] == ref].copy()
        mops = model_ops_by_batter(panel, rp, model, xwoba_to_outcome,
                                   batter_cache, pitcher_cache)
        d = spread_diagnostic({b: real[b] for b in panel}, mops)
        per_ref.append({"pitcher": ref, "n_pitches": int(len(rp)), **d})
        for b in panel:
            model_ops_acc[b].append(mops[b]["OPS"])

    return {
        "panel": panel,
        "real_ops": {b: real[b]["OPS"] for b in panel},
        "model_ops_mean": {b: float(np.mean(v)) for b, v in model_ops_acc.items()},
        "per_ref": per_ref,
        "spread_ratio_mean": float(np.mean([d["spread_ratio"] for d in per_ref])),
        "pearson_mean": float(np.mean([d["pearson"] for d in per_ref])),
        "real_std": float(np.std([real[b]["OPS"] for b in panel], ddof=1)),
    }


def spread_diagnostic(
    real_ops: dict[int, dict | float],
    model_ops: dict[int, dict | float],
) -> dict[str, float]:
    """Compression diagnostic over batters present in BOTH dicts.

    Returns real_std, model_std, spread_ratio = real_std/model_std (target ≈ 1×;
    transformer = 6.8×), and Pearson r (rank/direction agreement).
    """
    def ops(v):
        return v["OPS"] if isinstance(v, dict) else float(v)

    keys = [b for b in real_ops if b in model_ops]
    r = np.array([ops(real_ops[b]) for b in keys], dtype=float)
    m = np.array([ops(model_ops[b]) for b in keys], dtype=float)
    real_std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    model_std = float(m.std(ddof=1)) if len(m) > 1 else 0.0
    pearson = float(np.corrcoef(r, m)[0, 1]) if len(r) > 1 and model_std > 0 else float("nan")
    return {
        "n_batters": len(keys),
        "real_std": real_std,
        "model_std": model_std,
        "spread_ratio": (real_std / model_std) if model_std > 1e-12 else float("inf"),
        "pearson": pearson,
        "real_mean": float(r.mean()) if len(r) else float("nan"),
        "model_mean": float(m.mean()) if len(m) else float("nan"),
    }


def _print_compression_report(result: dict) -> None:
    print("\nbatter        real_OPS  model_OPS")
    for b in sorted(result["panel"], key=lambda x: result["real_ops"][x], reverse=True):
        print(f"  {b:8d}   {result['real_ops'][b]:.3f}    "
              f"{result['model_ops_mean'][b]:.3f}")
    print("\n=== COMPRESSION DIAGNOSTIC (avg over "
          f"{len(result['per_ref'])} reference pitchers) ===")
    print(f"  real_std        = {result['real_std']:.3f}")
    print(f"  SPREAD RATIO    = {result['spread_ratio_mean']:.2f}x"
          "   (transformer 6.8x; target ~1x)")
    print(f"  Pearson r       = {result['pearson_mean']:.3f}"
          "   (transformer 0.66)")


if __name__ == "__main__":
    import argparse
    import json

    from data.profile_cache_loader import ProfileCache
    from hitter.model import HitterModel
    from hitter.train import load_pitch_frame
    from hitter.compose import xwoba_outcome_fn

    ap = argparse.ArgumentParser(description="Hitter compression diagnostic")
    ap.add_argument("--model-dir", default="checkpoints/hitter")
    ap.add_argument("--test-start", default="2024-07-16")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--n-ref-pitchers", type=int, default=5)
    ap.add_argument("--fold-id", type=int, default=0)
    args = ap.parse_args()

    model = HitterModel(args.model_dir)
    xfn = xwoba_outcome_fn(json.load(
        open(f"{args.model_dir}/xwoba_outcome_map.json")))
    bc = ProfileCache(role="batter", fold_id=args.fold_id)
    pc = ProfileCache(role="pitcher", fold_id=args.fold_id)
    aug, _ = load_pitch_frame(args.test_start, args.test_end)
    print(f"test pitches: {len(aug):,}")
    result = run_compression_diagnostic(
        model, xfn, aug, bc, pc, n_ref_pitchers=args.n_ref_pitchers)
    _print_compression_report(result)
