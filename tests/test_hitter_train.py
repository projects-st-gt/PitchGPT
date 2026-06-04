"""Tests for hitter.train — cascade label assembly, conditional populations,
the count-constant foul rate, and the xwOBA-on-contact target join.

Pure-function tests are synthetic + fast. A real-data smoke test (slow) lives at
the bottom behind a marker and prints named numerical outputs (per CLAUDE.md).
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


VAL = sorted(Path("data/augmented/2024").glob("2024-*.parquet"))
requires_data = pytest.mark.skipif(not VAL, reason="no augmented val data")


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
