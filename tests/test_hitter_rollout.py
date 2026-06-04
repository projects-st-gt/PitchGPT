"""Tests for hitter.rollout — bridging the cascade into the pitchGPT MC rollout.

cascade_to_result_probs translates the cascade's per-pitch response (swing / whiff
/ foul / how-hard-hit) into the simulator's 7-class per-pitch result vocab
(ball, called_strike, swinging_strike, foul, in_play_out/hit/hr), so the existing
count machine + termination logic run unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from hitter.rollout import cascade_to_result_probs, RESULT_ORDER, build_step_features


def _one(p_swing, p_cs, p_whiff, foul_rate, outcome5):
    return cascade_to_result_probs(
        np.array([p_swing]), np.array([p_cs]), np.array([p_whiff]),
        foul_rate, np.array([outcome5]))[0]


def test_result_order_matches_simulator():
    assert RESULT_ORDER == ["ball", "called_strike", "swinging_strike", "foul",
                            "in_play_out", "in_play_hit", "in_play_hr"]


def test_rows_are_valid_distributions():
    r = _one(0.5, 0.4, 0.3, 0.5, [0.6, 0.2, 0.1, 0.0, 0.1])
    assert r.sum() == pytest.approx(1.0)
    assert (r >= 0).all()


def test_pure_take_called_strike():
    r = _one(0.0, 1.0, 0.0, 0.5, [1, 0, 0, 0, 0])
    assert r[RESULT_ORDER.index("called_strike")] == pytest.approx(1.0)


def test_pure_take_ball():
    r = _one(0.0, 0.0, 0.0, 0.5, [1, 0, 0, 0, 0])
    assert r[RESULT_ORDER.index("ball")] == pytest.approx(1.0)


def test_swing_and_miss():
    r = _one(1.0, 0.0, 1.0, 0.5, [1, 0, 0, 0, 0])
    assert r[RESULT_ORDER.index("swinging_strike")] == pytest.approx(1.0)


def test_swing_contact_foul():
    # swing=1, whiff=0, foul_rate=1 -> all foul
    r = _one(1.0, 0.0, 0.0, 1.0, [1, 0, 0, 0, 0])
    assert r[RESULT_ORDER.index("foul")] == pytest.approx(1.0)


def test_in_play_splits_hr_hit_out():
    # swing=1, whiff=0, foul_rate=0 -> all in play; outcome 50% HR, 30% 1B, 20% out
    r = _one(1.0, 0.0, 0.0, 0.0, [0.2, 0.3, 0.0, 0.0, 0.5])  # [out,1B,2B,3B,HR]
    assert r[RESULT_ORDER.index("in_play_hr")] == pytest.approx(0.5)
    assert r[RESULT_ORDER.index("in_play_hit")] == pytest.approx(0.3)  # 1B+2B+3B
    assert r[RESULT_ORDER.index("in_play_out")] == pytest.approx(0.2)


def test_build_step_features_columns_and_values():
    """The feature frame has every column the cascade nodes need, with location
    coming from the zone centroid and velo from the pitcher's per-type mean."""
    from hitter.train import NODE_FEATURES
    import numpy as np
    centroids = {7: [0.0, 1.9], 4: [0.01, 2.47]}            # zone 7 (low-mid), 4 (mid)
    velo_by_type = {1: 95.0, 4: 84.0}
    spin_by_type = {1: 2300.0, 4: 2500.0}
    df = build_step_features(
        type_ids=np.array([1, 4]), zone_ids=np.array([7, 4]),
        balls=np.array([1, 0]), strikes=np.array([2, 0]),
        prev_type_ids=np.array([0, 1]), prev_zone_ids=np.array([-1, 7]),
        n_prev=np.array([0, 1]),
        batter_vec=np.arange(57, dtype="float32"),
        pitcher_stuff_vec=np.arange(53, dtype="float32"),
        velo_by_type=velo_by_type, spin_by_type=spin_by_type,
        same_hand=1, centroids=centroids,
        ctx_cat={"umpire_id": 5, "catcher_id": 9, "ballpark_id": 3,
                 "roof_state": 1, "temp_bucket": 4})
    # all columns any node needs are present
    need = set().union(*NODE_FEATURES.values())
    assert need.issubset(set(df.columns)), need - set(df.columns)
    # location from centroid; velo from per-type mean; in_zone (7,4 both in grid)
    assert df["plate_z"].tolist() == pytest.approx([1.9, 2.47], abs=1e-5)
    assert df["release_speed"].tolist() == [95.0, 84.0]
    assert df["in_zone"].tolist() == [1, 1]
    assert df["prev_type_id"].tolist() == [0, 1]           # the sequence lag
    assert df["b5"].tolist() == [5.0, 5.0]                  # batter profile broadcast
