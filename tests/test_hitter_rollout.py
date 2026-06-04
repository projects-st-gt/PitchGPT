"""Tests for hitter.rollout — bridging the cascade into the pitchGPT MC rollout.

cascade_to_result_probs translates the cascade's per-pitch response (swing / whiff
/ foul / how-hard-hit) into the simulator's 7-class per-pitch result vocab
(ball, called_strike, swinging_strike, foul, in_play_out/hit/hr), so the existing
count machine + termination logic run unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from hitter.rollout import cascade_to_result_probs, RESULT_ORDER


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
