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
