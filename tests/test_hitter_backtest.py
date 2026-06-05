"""Unit tests for hitter/backtest.py pure functions (synthetic fixtures only)."""
import numpy as np
import pandas as pd
import pytest

from hitter.backtest import (
    PA_CLASSES,
    aggregate_calibration,
    bootstrap_logloss_ci,
    league_baseline_dist,
    sample_pas,
    score_path,
)


def _toy_pas(n=100, seed=0):
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(PA_CLASSES, size=n,
                          p=[0.23, 0.08, 0.15, 0.05, 0.005, 0.035, 0.45])
    return pd.DataFrame({
        "pitcher": rng.integers(1, 6, n),
        "batter": rng.integers(1, 6, n),
        "p_throws": "R", "stand": "R",
        "game_date": pd.Timestamp("2024-08-01"),
        "game_pk": 1,
        "outcome": outcomes,
    })


def test_league_baseline_normalizes_and_covers_all_classes():
    base = league_baseline_dist(_toy_pas(2000))
    assert set(base) == set(PA_CLASSES)
    assert abs(sum(base.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in base.values())
    # 'out' is the plurality class in the toy mix
    assert max(base, key=base.get) == "out"


def test_league_baseline_raises_on_empty():
    with pytest.raises(ValueError):
        league_baseline_dist(pd.DataFrame({"outcome": []}))


def test_sample_caps_per_matchup():
    pas = _toy_pas(500, seed=1)
    s = sample_pas(pas, 1000, seed=2, max_per_matchup=2)
    # with the cap, no (pitcher,batter) pair appears more than twice
    assert s.groupby(["pitcher", "batter"]).size().max() <= 2


def test_sample_is_deterministic_given_seed():
    pas = _toy_pas(300, seed=3)
    a = sample_pas(pas, 50, seed=7, max_per_matchup=0)
    b = sample_pas(pas, 50, seed=7, max_per_matchup=0)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50


def test_score_path_perfect_vs_wrong():
    # A confident-correct predictor scores near 0; a confident-wrong one scores high.
    actuals = ["HR", "K", "1B"]
    perfect = [{c: (1.0 if c == a else 0.0) for c in PA_CLASSES} for a in actuals]
    # clamp handled by eps inside pa_logloss; use near-one-hot to avoid -inf
    perfect = [{c: (0.999 if c == a else 0.001 / 6) for c in PA_CLASSES}
               for a in actuals]
    wrong = [{c: (0.001 / 6 if c == a else 0.999 / 6) for c in PA_CLASSES}
             for a in actuals]
    assert score_path(perfect, actuals) < 0.1
    assert score_path(wrong, actuals) > score_path(perfect, actuals)


def test_baseline_beats_uniform_on_its_own_marginal():
    # Scoring the league-baseline constant predictor against PAs drawn from the
    # same marginal must beat a uniform-over-7 predictor (entropy argument).
    pas = _toy_pas(4000, seed=5)
    base = league_baseline_dist(pas)
    actuals = list(pas["outcome"])
    base_dists = [base] * len(actuals)
    uniform = [{c: 1 / 7 for c in PA_CLASSES}] * len(actuals)
    assert score_path(base_dists, actuals) < score_path(uniform, actuals)
    # log(7) is the uniform predictor's loss
    assert abs(score_path(uniform, actuals) - np.log(7)) < 1e-6


def test_aggregate_calibration_recovers_rates():
    actuals = ["HR"] * 10 + ["K"] * 90
    # a predictor that always says P(HR)=0.2 over-predicts HR by +0.10
    dists = [{c: (0.2 if c == "HR" else 0.8 / 6) for c in PA_CLASSES}] * 100
    cal = aggregate_calibration(dists, actuals)
    hr = cal.set_index("class").loc["HR"]
    assert abs(hr["real_freq"] - 0.10) < 1e-9
    assert abs(hr["mean_pred"] - 0.20) < 1e-9
    assert abs(hr["pred_minus_real"] - 0.10) < 1e-9


def test_bootstrap_ci_brackets_point_estimate():
    actuals = ["out"] * 50 + ["1B"] * 50
    base = {"out": 0.5, "1B": 0.5, "K": 0, "BB": 0, "2B": 0, "3B": 0, "HR": 0}
    dists = [base] * 100
    res = bootstrap_logloss_ci(dists, actuals, n_boot=500, seed=1)
    assert res["ci_lo"] <= res["logloss"] <= res["ci_hi"]
    assert abs(res["logloss"] - np.log(2)) < 1e-6  # each PA has p=0.5
    assert res["n"] == 100
