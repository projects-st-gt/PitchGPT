"""Tests for hitter.eval — real OPS from events + the compression diagnostic.

The compression diagnostic is THE acceptance gate (MODEL_DESIGN §10): real OPS
spread vs model OPS spread across a batter panel. Transformer baseline = 6.8×
(real std 0.261 / model 0.038); target ≈ 1×. Pure parts (real OPS, spread stats,
panel selection) are tested here on synthetic data; the model-OPS path is run
once the cascade is trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from hitter.eval import (
    real_ops_by_batter, spread_diagnostic, select_batter_panel,
    held_out_node_metrics,
)

_HAVE_MODEL = Path("checkpoints/hitter/meta.json").exists()
_HAVE_TEST = bool(sorted(Path("data/augmented/2024").glob("2024-07-2*.parquet")))


def test_pa_outcome_class_maps_events():
    from hitter.eval import pa_outcome_class
    assert pa_outcome_class("single") == "1B"
    assert pa_outcome_class("home_run") == "HR"
    assert pa_outcome_class("walk") == "BB"
    assert pa_outcome_class("strikeout") == "K"
    assert pa_outcome_class("field_out") == "out"
    assert pa_outcome_class("grounded_into_double_play") == "out"
    assert pa_outcome_class("sac_fly") is None        # excluded from scoring vocab
    assert pa_outcome_class("hit_by_pitch") is None


def test_pa_logloss_rewards_probability_on_truth():
    """Lower log-loss when the model puts more probability on what happened."""
    from hitter.eval import pa_logloss
    good = {"BB": .08, "K": .22, "out": .45, "1B": .15, "2B": .05, "3B": .005, "HR": .045}
    # actual outcomes: mostly outs and a couple hits
    actuals = ["out", "out", "1B", "K", "out"]
    ll_good = pa_logloss([good] * 5, actuals)
    flat = {k: 1 / 7 for k in good}
    ll_flat = pa_logloss([flat] * 5, actuals)
    assert ll_good < ll_flat                          # informed beats uniform
    print(f"\nlogloss informed={ll_good:.3f} uniform={ll_flat:.3f}")


def _pa_rows(batter, events):
    """One terminal pitch per PA for a batter (events = the PA result)."""
    return pd.DataFrame({
        "batter": batter, "events": events,
        "game_pk": range(len(events)),
        "at_bat_number": 1, "pitch_number": 1,
    })


def test_real_ops_single_outcomes():
    # 1 HR, 1 strikeout, 1 walk, 1 field_out for batter 1 (4 PA)
    df = _pa_rows(1, ["home_run", "strikeout", "walk", "field_out"])
    r = real_ops_by_batter(df)[1]
    # AB = 4 - 1 walk = 3 ; H = 1 ; AVG = 1/3
    assert r["AVG"] == pytest.approx(1 / 3)
    # OBP = (H + BB)/(PA) = (1+1)/4 = 0.5
    assert r["OBP"] == pytest.approx(0.5)
    # SLG = TB/AB = 4/3
    assert r["SLG"] == pytest.approx(4 / 3)
    assert r["OPS"] == pytest.approx(0.5 + 4 / 3)
    assert r["PA"] == 4


def test_real_ops_ignores_non_terminal_rows():
    """Rows with null events (mid-AB pitches) don't count as PAs."""
    df = pd.DataFrame({
        "batter": [1, 1, 1],
        "events": [None, None, "single"],   # one PA, 3 pitches
        "game_pk": [1, 1, 1], "at_bat_number": [1, 1, 1],
        "pitch_number": [1, 2, 3],
    })
    r = real_ops_by_batter(df)[1]
    assert r["PA"] == 1 and r["AVG"] == pytest.approx(1.0)


def test_spread_diagnostic_ratio_and_pearson():
    real = {1: 1.0, 2: 0.5, 3: 0.8, 4: 0.2}
    # model perfectly correlated but compressed to 1/4 the spread around mean
    mean = np.mean(list(real.values()))
    model = {k: mean + (v - mean) / 4 for k, v in real.items()}
    d = spread_diagnostic(real, model)
    assert d["spread_ratio"] == pytest.approx(4.0, rel=1e-6)
    assert d["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert d["real_std"] > d["model_std"]


def test_select_batter_panel_best_and_worst():
    real = {i: ops for i, ops in zip(range(1, 9),
            [1.1, 1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2])}
    pa = {i: 200 for i in range(1, 9)}
    panel = select_batter_panel(real, pa, n_each=2, min_pa=150)
    # 2 best (1.1, 1.0) + 2 worst (0.2, 0.3)
    assert set(panel) == {1, 2, 7, 8}


def test_select_batter_panel_respects_min_pa():
    real = {1: 1.1, 2: 1.0, 3: 0.2, 4: 0.3}
    pa = {1: 200, 2: 100, 3: 200, 4: 100}     # 2 and 4 below min
    panel = select_batter_panel(real, pa, n_each=1, min_pa=150)
    assert set(panel) == {1, 3}


@pytest.mark.skipif(not (_HAVE_MODEL and _HAVE_TEST),
                    reason="needs trained checkpoints/hitter + 2024H2 test data")
def test_held_out_calibration_is_honest_and_good():
    """ECE in meta.json is ~0 (in-sample isotonic artifact); the honest held-out
    ECE must be a real, small number — not 0, and well-calibrated (<0.05)."""
    from data.profile_cache_loader import ProfileCache
    from hitter.model import HitterModel
    from hitter.train import build_training_frame, load_pitch_frame
    hm = HitterModel("checkpoints/hitter")
    b = ProfileCache(role="batter", fold_id=0)
    p = ProfileCache(role="pitcher", fold_id=0)
    test_df = build_training_frame(*load_pitch_frame("2024-07-19", "2024-07-22"), b, p)
    m = held_out_node_metrics(hm, test_df)
    assert "swing" in m and "auc" in m["swing"]
    for node in ("swing", "called_strike", "whiff"):
        assert 0.0 < m[node]["ece"] < 0.05, f"{node} ECE not honest/good: {m[node]['ece']}"
    print("\nhonest held-out ECE:",
          {k: round(v["ece"], 4) for k, v in m.items() if "ece" in v})
