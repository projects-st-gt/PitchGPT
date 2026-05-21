"""Unit tests for run-value tables and lookups."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.run_value import (
    base_state_from_columns,
    compute_count_value,
    compute_in_play_woba_to_runs_slope,
    compute_re24,
    delta_v,
    in_play_run_value,
    state_value,
)


# ---------- base_state encoding ----------


def test_base_state_empty():
    df = pd.DataFrame({"on_1b": [None], "on_2b": [None], "on_3b": [None]})
    assert list(base_state_from_columns(df)) == [0]


def test_base_state_runner_on_first_only():
    df = pd.DataFrame({"on_1b": [12345], "on_2b": [None], "on_3b": [None]})
    assert list(base_state_from_columns(df)) == [4]  # bit 2 set = 100b = 4


def test_base_state_bases_loaded():
    df = pd.DataFrame({"on_1b": [1], "on_2b": [2], "on_3b": [3]})
    assert list(base_state_from_columns(df)) == [7]


# ---------- compute_re24 ----------


def _half_inning_pitches(game_pk, inning, half, pitches):
    """Build pitch rows for one half-inning. ``pitches`` is list of
    ``(on_1b, on_2b, on_3b, outs, bat_score)``.
    """
    return [
        {
            "game_pk": game_pk,
            "inning": inning,
            "inning_topbot": half,
            "on_1b": p[0],
            "on_2b": p[1],
            "on_3b": p[2],
            "outs_when_up": p[3],
            "bat_score": p[4],
        }
        for p in pitches
    ]


def test_compute_re24_yields_one_row_per_observed_state():
    # One half-inning: bases empty 0 outs (2 pitches), then bases empty 0 outs again
    # then 1B 0 outs (1 pitch), end-of-inning bat_score = 3.
    rows = _half_inning_pitches(
        game_pk=1, inning=1, half="Top",
        pitches=[
            (None, None, None, 0, 0),
            (None, None, None, 0, 0),
            (12345, None, None, 0, 0),
            (12345, None, None, 0, 3),  # final pitch's pre-pitch bat_score=3
        ],
    )
    df = pd.DataFrame(rows)
    table = compute_re24(df)
    # Two cells should appear: (0, 0) and (4, 0).
    cells = set(zip(table["base_state"], table["outs"]))
    assert (0, 0) in cells
    assert (4, 0) in cells


def test_compute_re24_uses_max_bat_score_for_runs_remainder():
    # Two pitches, both at base_state=0 outs=0. bat_score 0 → 5 over the half-inning.
    rows = _half_inning_pitches(
        game_pk=1, inning=1, half="Top",
        pitches=[(None, None, None, 0, 0), (None, None, None, 0, 5)],
    )
    df = pd.DataFrame(rows)
    table = compute_re24(df)
    re_00 = table[(table["base_state"] == 0) & (table["outs"] == 0)]
    # Mean of (5-0, 5-5) = (5, 0) → 2.5
    assert math.isclose(float(re_00["expected_runs"].iloc[0]), 2.5)


def test_compute_re24_raises_on_missing_columns():
    with pytest.raises(KeyError, match="bat_score"):
        compute_re24(pd.DataFrame({
            "game_pk": [1], "inning": [1], "inning_topbot": ["Top"],
            "on_1b": [None], "on_2b": [None], "on_3b": [None], "outs_when_up": [0],
        }))


# ---------- compute_count_value ----------


def test_compute_count_value_groups_by_count():
    df = pd.DataFrame({
        "balls": [0, 0, 3, 3],
        "strikes": [0, 0, 2, 2],
        "delta_run_exp": [0.01, 0.03, -0.05, -0.07],
    })
    table = compute_count_value(df)
    cv_00 = table[(table["balls"] == 0) & (table["strikes"] == 0)]
    cv_32 = table[(table["balls"] == 3) & (table["strikes"] == 2)]
    assert math.isclose(float(cv_00["count_value"].iloc[0]), 0.02)
    assert math.isclose(float(cv_32["count_value"].iloc[0]), -0.06)


# ---------- state_value / delta_v ----------


def _re24_table():
    return pd.DataFrame(
        [
            {"base_state": 0, "outs": 0, "expected_runs": 0.50},
            {"base_state": 0, "outs": 1, "expected_runs": 0.27},
            {"base_state": 4, "outs": 0, "expected_runs": 0.86},
        ]
    )


def _count_value_table():
    return pd.DataFrame(
        [
            {"balls": 0, "strikes": 0, "count_value": 0.0},
            {"balls": 0, "strikes": 1, "count_value": -0.04},
            {"balls": 1, "strikes": 0, "count_value": 0.03},
        ]
    )


def test_state_value_combines_re24_and_count_value_additively():
    re = _re24_table()
    cv = _count_value_table()
    v = state_value(re, cv, base_state=0, outs=0, balls=0, strikes=1)
    assert math.isclose(v, 0.50 + (-0.04))


def test_state_value_raises_on_unseen_state():
    re = _re24_table()
    cv = _count_value_table()
    with pytest.raises(KeyError, match="RE24"):
        state_value(re, cv, base_state=5, outs=0, balls=0, strikes=0)


def test_delta_v():
    re = _re24_table()
    cv = _count_value_table()
    # 0-0 (bases empty, 0 outs) → 0-1 (bases empty, 0 outs)
    d = delta_v(re, cv, before=(0, 0, 0, 0), after=(0, 0, 0, 1))
    assert math.isclose(d, (0.50 - 0.04) - (0.50 + 0.0))


# ---------- in_play_run_value ----------


def test_in_play_run_value_multiplies_by_supplied_slope():
    rv = in_play_run_value(0.4, slope=0.49)
    assert math.isclose(rv, 0.4 * 0.49)


def test_in_play_run_value_handles_nan():
    rv = in_play_run_value(float("nan"), slope=0.49)
    assert math.isnan(rv)


# ---------- compute_in_play_woba_to_runs_slope ----------


def test_in_play_woba_to_runs_slope_recovers_synthetic_slope():
    # Synthetic data with a known per-pitch slope of 0.5; recovery should be tight.
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 0.7, size=500)
    true_slope = 0.5
    y = true_slope * x + rng.normal(0, 0.01, size=500)
    df = pd.DataFrame({
        "estimated_woba_using_speedangle": x,
        "delta_run_exp": y,
    })
    k = compute_in_play_woba_to_runs_slope(df)
    assert abs(k - true_slope) < 0.05, f"recovered slope {k:.4f} too far from {true_slope}"


def test_in_play_woba_to_runs_slope_raises_when_no_data():
    df = pd.DataFrame({
        "estimated_woba_using_speedangle": [None, None],
        "delta_run_exp": [None, None],
    })
    with pytest.raises(ValueError, match="no rows"):
        compute_in_play_woba_to_runs_slope(df)
