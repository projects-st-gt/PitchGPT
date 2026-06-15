"""Tests for hitter.train — cascade label assembly, conditional populations,
the count-constant foul rate, and the xwOBA-on-contact target join.

Pure-function tests are synthetic + fast. A real-data smoke test (slow) lives at
the bottom behind a marker and prints named numerical outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hitter.train import (
    add_cascade_labels,
    node_population,
    foul_rate_by_count,
    attach_xwoba_target,
    attach_pitcher_profile,
    PITCHER_STUFF_FEATURES,
    train_node,
)


def _pitches():
    """Six pitches covering every cascade branch, in one frame.

    row: description / events                       -> branch
    0:   called_strike                              -> take, called strike
    1:   ball                                        -> take, ball
    2:   swinging_strike                             -> swing, whiff
    3:   foul                                        -> swing, contact, foul (s<2)
    4:   hit_into_play / single                      -> swing, contact, fair, 1B
    5:   hit_into_play / field_out                   -> swing, contact, fair, out
    """
    return pd.DataFrame({
        "game_pk": [1] * 6,
        "at_bat_number": [1] * 6,
        "pitch_number": [1, 2, 3, 4, 5, 6],
        "balls": [0, 0, 0, 0, 0, 0],
        "strikes": [0, 0, 0, 1, 1, 1],
        "description": [
            "called_strike", "ball", "swinging_strike",
            "foul", "hit_into_play", "hit_into_play",
        ],
        "events": [None, None, None, None, "single", "field_out"],
        "estimated_woba_using_speedangle": [
            np.nan, np.nan, np.nan, np.nan, 0.9, 0.05,
        ],
    })


def test_cascade_labels_swing_take():
    out = add_cascade_labels(_pitches())
    assert out["swing"].tolist() == [0, 0, 1, 1, 1, 1]


def test_cascade_labels_whiff_only_defined_on_swings():
    out = add_cascade_labels(_pitches())
    # whiff is NaN on takes (rows 0,1), 1 on the whiff (row 2), 0 on contact
    w = out["whiff"]
    assert np.isnan(w.iloc[0]) and np.isnan(w.iloc[1])
    assert w.iloc[2] == 1
    assert w.iloc[3] == 0 and w.iloc[4] == 0 and w.iloc[5] == 0


def test_cascade_labels_fair_only_defined_on_contact():
    out = add_cascade_labels(_pitches())
    f = out["fair"]
    # NaN on takes (0,1) and on the whiff (2); foul (3)=0; in-play (4,5)=1
    assert np.isnan(f.iloc[0]) and np.isnan(f.iloc[2])
    assert f.iloc[3] == 0
    assert f.iloc[4] == 1 and f.iloc[5] == 1


def test_cascade_labels_called_strike_only_defined_on_takes():
    out = add_cascade_labels(_pitches())
    cs = out["called_strike"]
    assert cs.iloc[0] == 1   # called_strike
    assert cs.iloc[1] == 0   # ball
    assert np.isnan(cs.iloc[2]) and np.isnan(cs.iloc[4])  # NaN on swings


def test_node_population_swing_is_all_rows():
    df = add_cascade_labels(_pitches())
    sub, y = node_population(df, "swing")
    assert len(sub) == 6
    assert y.tolist() == [0, 0, 1, 1, 1, 1]


def test_node_population_whiff_is_swings_only():
    df = add_cascade_labels(_pitches())
    sub, y = node_population(df, "whiff")
    assert len(sub) == 4                       # the four swings
    assert set(y.tolist()) == {0, 1}
    assert y.tolist() == [1, 0, 0, 0]


def test_node_population_called_strike_is_takes_only():
    df = add_cascade_labels(_pitches())
    sub, y = node_population(df, "called_strike")
    assert len(sub) == 2
    assert y.tolist() == [1, 0]


def test_node_population_contact_quality_is_balls_in_play():
    df = add_cascade_labels(_pitches())
    sub, y = node_population(df, "contact_quality")
    assert len(sub) == 2                        # the two in-play
    assert y.tolist() == [0.9, 0.05]            # xwOBA target


def test_foul_rate_by_count_is_empirical_contact_split():
    """Foul rate per (balls,strikes) = fouls / (fouls + fair) among contact."""
    # 0-1 count: 1 foul, 2 fair -> foul rate 1/3
    df = add_cascade_labels(_pitches())
    rates = foul_rate_by_count(df)
    assert rates[(0, 1)] == pytest.approx(1 / 3)


def test_attach_xwoba_target_joins_from_raw():
    """attach_xwoba_target pulls estimated_woba_using_speedangle from a raw
    frame, keyed on (game_pk, at_bat_number, pitch_number)."""
    base = _pitches().drop(columns=["estimated_woba_using_speedangle"])
    raw = _pitches()[["game_pk", "at_bat_number", "pitch_number",
                      "estimated_woba_using_speedangle"]]
    out = attach_xwoba_target(base, raw)
    assert out["estimated_woba_using_speedangle"].iloc[4] == pytest.approx(0.9)
    assert out["estimated_woba_using_speedangle"].iloc[5] == pytest.approx(0.05)


def test_pitcher_stuff_features_drops_heatmap_and_count_arsenal():
    """The compact subset is stuff-quality only: no per-type zone heatmap, no
    per-count arsenal (pitch selection is PitchGPT's job, not the hitter's)."""
    assert all("heatmap" not in f for f in PITCHER_STUFF_FEATURES)
    assert all("_b0s0" not in f and "_b3s2" not in f for f in PITCHER_STUFF_FEATURES)
    for f in ("arsenal_FF", "mean_velo_FF", "mean_spin_SL", "recent_30d_xwoba"):
        assert f in PITCHER_STUFF_FEATURES
    assert 40 <= len(PITCHER_STUFF_FEATURES) <= 80


def _synth_binary(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 2.5 * x1 - 1.0 * x2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y)


def test_train_node_binary_learns_and_calibrates():
    Xtr, ytr = _synth_binary(seed=0)
    Xva, yva = _synth_binary(seed=1)
    res = train_node(Xtr, ytr, Xva, yva, objective="binary",
                     feature_names=["x1", "x2"])
    # learned signal
    assert res["metrics"]["auc"] > 0.85
    # isotonic calibration produced a calibrator + a sane ECE
    assert res["calibrator"] is not None
    assert res["metrics"]["ece"] < 0.06
    # predict() returns calibrated probabilities in [0,1]
    p = res["predict"](Xva)
    assert p.min() >= 0.0 and p.max() <= 1.0
    print(f"\n[binary] AUC={res['metrics']['auc']:.3f} "
          f"logloss={res['metrics']['logloss']:.3f} ECE={res['metrics']['ece']:.3f}")


def test_train_node_multiclass_learns_and_returns_distribution():
    """Multiclass node (contact outcome) returns per-row (n, n_classes) probs."""
    rng = np.random.default_rng(0)
    n = 5000
    x1 = rng.normal(size=n)
    # 5 classes whose likelihood shifts with x1 (ordinal-ish: low x1 -> class 0)
    y = np.clip(((x1 + 3) / 1.2).astype(int), 0, 4)
    X = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n)})
    res = train_node(X.iloc[:4000], pd.Series(y[:4000]),
                     X.iloc[4000:], pd.Series(y[4000:]),
                     objective="multiclass", feature_names=["x1", "x2"],
                     n_classes=5)
    P = res["predict"](X.iloc[4000:])
    assert P.shape == (1000, 5)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    assert res["metrics"]["accuracy"] > 0.5
    print(f"\n[multiclass] acc={res['metrics']['accuracy']:.3f}")


def test_train_node_regression_learns():
    rng = np.random.default_rng(0)
    n = 4000
    x1 = rng.normal(size=n)
    y = 0.3 + 0.2 * x1 + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"x1": x1})
    res = train_node(X.iloc[:3000], pd.Series(y[:3000]),
                     X.iloc[3000:], pd.Series(y[3000:]),
                     objective="regression", feature_names=["x1"])
    assert res["metrics"]["pearson"] > 0.7
    print(f"\n[reg] RMSE={res['metrics']['rmse']:.3f} r={res['metrics']['pearson']:.3f}")


def test_train_node_categorical_handles_unseen_val_category():
    """Val categories not present in train must not crash (the 2023-train /
    2024-val reality): fit category set on train, unseen-in-val -> missing."""
    Xtr, ytr = _synth_binary(seed=4)
    Xva, yva = _synth_binary(seed=5)
    # a categorical id col: train sees ids 0..9, val sees an unseen id 99
    Xtr = Xtr.assign(cat_id=np.random.default_rng(0).integers(0, 10, len(Xtr)))
    Xva = Xva.assign(cat_id=np.random.default_rng(1).integers(0, 10, len(Xva)))
    Xva.iloc[0, Xva.columns.get_loc("cat_id")] = 99    # unseen in train
    res = train_node(Xtr, ytr, Xva, yva, objective="binary",
                     feature_names=["x1", "x2", "cat_id"], categorical=["cat_id"])
    assert res["metrics"]["auc"] > 0.8
    # predict on a frame with an unseen category also works
    p = res["predict"](Xva)
    assert len(p) == len(Xva) and p.min() >= 0 and p.max() <= 1


def test_train_node_monotone_increasing_constraint_respected():
    """With a +1 monotone constraint on x1, calibrated P must be non-decreasing
    in x1 (holding nothing else — single feature)."""
    Xtr, ytr = _synth_binary(seed=2)
    Xva, yva = _synth_binary(seed=3)
    res = train_node(Xtr[["x1"]], ytr, Xva[["x1"]], yva, objective="binary",
                     feature_names=["x1"], monotone={"x1": 1})
    grid = pd.DataFrame({"x1": np.linspace(-2, 2, 50)})
    p = res["predict"](grid)
    assert np.all(np.diff(p) >= -1e-6), "monotone +1 constraint violated"


VAL = sorted(Path("data/augmented/2024").glob("2024-*.parquet"))
requires_data = pytest.mark.skipif(not VAL, reason="no augmented val data")


TRAIN23 = sorted(Path("data/augmented/2023").glob("2023-09-*.parquet"))
RAW23 = sorted(Path("data/raw/2023").glob("2023-09-*.parquet"))
requires_train = pytest.mark.skipif(
    not (TRAIN23 and RAW23 and VAL), reason="needs 2023 aug+raw and 2024 val")


@requires_train
def test_build_training_frame_and_train_all_nodes_real_slice():
    """End-to-end pipeline on a small real slice: build the feature frame, train
    all 4 ML nodes + foul rates. Asserts structure + prints named per-node
    numbers. Not the full train — a wiring check."""
    from data.profile_cache_loader import ProfileCache
    from hitter.train import (
        build_training_frame, train_all_nodes, load_pitch_frame,
        NODE_FEATURES, _BATTER_COLS, _PITCHER_COLS,
    )
    bcache = ProfileCache(role="batter", fold_id=0)
    pcache = ProfileCache(role="pitcher", fold_id=0)
    # use the real loader: a week of 2023-09 train, a week of 2024-04 val
    # (raw alignment guaranteed; enough in-play rows for the S3 val population).
    tr_aug, tr_raw = load_pitch_frame("2023-09-01", "2023-09-07")
    va_aug, va_raw = load_pitch_frame("2024-04-01", "2024-04-07")

    train_df = build_training_frame(tr_aug, tr_raw, bcache, pcache)
    val_df = build_training_frame(va_aug, va_raw, bcache, pcache)
    # profile cols present
    assert all(c in train_df.columns for c in _BATTER_COLS)
    assert all(c in train_df.columns for c in _PITCHER_COLS)
    # xwOBA target joined for in-play rows
    inplay = train_df[train_df["fair"] == 1]
    assert inplay["estimated_woba_using_speedangle"].notna().mean() > 0.8

    print(f"\ntrain rows={len(train_df):,} val rows={len(val_df):,}")
    bundle = train_all_nodes(train_df, val_df, n_estimators=80)
    assert set(bundle["nodes"]) == {"swing", "whiff", "called_strike",
                                    "contact_quality"}
    # the swing node should clear a trivial bar even on this tiny slice
    assert bundle["nodes"]["swing"]["metrics"]["auc"] > 0.6
    assert len(bundle["foul_rate_by_count"]) >= 8   # most of the 12 counts seen


@requires_data
def test_attach_pitcher_profile_differs_across_pitchers():
    """Mirror of the batter join: different pitchers get different stuff vectors
    (98 mph guy != 91 mph guy)."""
    from data.profile_cache_loader import ProfileCache
    df = pd.read_parquet(VAL[0])
    pids = [int(x) for x in df["pitcher"].drop_duplicates().head(2)]
    sub = df[df["pitcher"].isin(pids)].copy()
    cache = ProfileCache(role="pitcher", fold_id=0)
    out = attach_pitcher_profile(sub, cache)
    pcols = [f"p{i}" for i in range(len(PITCHER_STUFF_FEATURES))]
    assert all(c in out.columns for c in pcols)
    v0 = out[out["pitcher"] == pids[0]][pcols].iloc[0].to_numpy()
    v1 = out[out["pitcher"] == pids[1]][pcols].iloc[0].to_numpy()
    assert not np.allclose(v0, v1), "two different pitchers got identical stuff!"
    print(f"\npitcher {pids[0]} vs {pids[1]}: stuff L2 diff = "
          f"{np.linalg.norm(v0 - v1):.2f} ({len(pcols)} dims)")
