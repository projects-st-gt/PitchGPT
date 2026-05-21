"""Unit tests for XGBoost feature engineering."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.dataset import PITCH_TYPES, PITCH_TYPE_TO_ID
from data.xgboost_features import (
    DAYS_REST_NO_PRIOR_SENTINEL,
    PREV_PITCH_BEGIN,
    _add_days_rest,
    _add_prev_pitch_in_ab,
    build_xgboost_features,
    compute_pitcher_arsenal_encoding,
)


def _pitches(rows):
    """rows: list of dicts. Returns DataFrame with the column set required
    by `build_xgboost_features`."""
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _meta(rows):
    return pd.DataFrame(rows)


# ---------- previous pitch within AB ----------


def test_prev_pitch_first_in_ab_is_BEGIN():
    df = _pitches([
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-01", "pitcher": 100, "batter": 200,
         "pitch_type_canonical": "FF",
         "balls": 0, "strikes": 0, "outs_when_up": 0, "inning": 1,
         "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
    ])
    out = _add_prev_pitch_in_ab(df)
    assert out["prev_pitch_id"].iloc[0] == PREV_PITCH_BEGIN


def test_prev_pitch_within_same_ab():
    df = _pitches([
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-01", "pitch_type_canonical": "FF",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 2,
         "game_date": "2024-04-01", "pitch_type_canonical": "SL",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 1,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
    ])
    out = _add_prev_pitch_in_ab(df).sort_values("pitch_number")
    assert out["prev_pitch_id"].iloc[0] == PREV_PITCH_BEGIN
    assert out["prev_pitch_id"].iloc[1] == PITCH_TYPE_TO_ID["FF"]


def test_prev_pitch_resets_per_ab():
    df = _pitches([
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-01", "pitch_type_canonical": "FF",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
        {"game_pk": 1, "at_bat_number": 2, "pitch_number": 1,
         "game_date": "2024-04-01", "pitch_type_canonical": "SL",
         "pitcher": 100, "batter": 201, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
    ])
    out = _add_prev_pitch_in_ab(df).sort_values(["at_bat_number", "pitch_number"])
    # Both first-of-AB → both BEGIN
    assert (out["prev_pitch_id"] == PREV_PITCH_BEGIN).all()


def test_prev_pitch_preserves_input_order():
    """Caller depends on row-order preservation — the bug we fixed in
    pitcher_ngram applies equally here."""
    df = _pitches([
        {"game_pk": 2, "at_bat_number": 5, "pitch_number": 1,
         "game_date": "2024-05-01", "pitch_type_canonical": "FF",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-01", "pitch_type_canonical": "SL",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
    ])
    out = _add_prev_pitch_in_ab(df)
    # Input row 0 had pitch_type FF; should still be at index 0
    assert out.iloc[0]["pitch_type_canonical"] == "FF"
    assert out.iloc[1]["pitch_type_canonical"] == "SL"


# ---------- days rest ----------


def test_days_rest_no_prior_uses_sentinel():
    df = _pitches([
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-01", "pitch_type_canonical": "FF",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
    ])
    out = _add_days_rest(df)
    assert out["days_rest"].iloc[0] == DAYS_REST_NO_PRIOR_SENTINEL


def test_days_rest_counts_gap_between_appearances():
    df = _pitches([
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-01", "pitch_type_canonical": "FF",
         "pitcher": 100, "batter": 200, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
        {"game_pk": 2, "at_bat_number": 1, "pitch_number": 1,
         "game_date": "2024-04-06", "pitch_type_canonical": "SL",
         "pitcher": 100, "batter": 201, "balls": 0, "strikes": 0,
         "outs_when_up": 0, "inning": 1, "p_throws": "R", "stand": "L",
         "on_1b": None, "on_2b": None, "on_3b": None},
    ])
    out = _add_days_rest(df).sort_values("game_date")
    assert out["days_rest"].iloc[0] == DAYS_REST_NO_PRIOR_SENTINEL
    # 5 days between 2024-04-01 and 2024-04-06
    assert out["days_rest"].iloc[1] == 5


# ---------- pitcher arsenal target encoding ----------


def test_pitcher_arsenal_encoding_smooths_low_n_to_league():
    rows = []
    # Pitcher 1: 100 FF in training
    for _ in range(100):
        rows.append({"pitcher": 1, "pitch_type_canonical": "FF"})
    # Pitcher 2: 5 SL in training (small sample)
    for _ in range(5):
        rows.append({"pitcher": 2, "pitch_type_canonical": "SL"})
    df = pd.DataFrame(rows)

    enc = compute_pitcher_arsenal_encoding(df, alpha=10.0)
    # Pitcher 1: 100 FFs, alpha=10, league prior FF = 100/105 ≈ 0.952
    p1 = enc[enc["pitcher"] == 1].iloc[0]
    # Smoothed: (100 + 10*0.952) / (100+10) ≈ 0.991
    assert p1["pt_FF_rate"] > 0.95

    # Pitcher 2: only 5 SL, alpha=10 dominates → SL rate < the 100% raw observation
    p2 = enc[enc["pitcher"] == 2].iloc[0]
    # Raw rate would be 100% SL; smoothed pulls toward league prior (0% SL)
    assert p2["pt_SL_rate"] < 1.0


def test_pitcher_arsenal_encoding_columns_match_pitch_types():
    df = pd.DataFrame({
        "pitcher": [1, 1, 2],
        "pitch_type_canonical": ["FF", "SL", "FF"],
    })
    enc = compute_pitcher_arsenal_encoding(df)
    expected_cols = {f"pt_{t}_rate" for t in PITCH_TYPES} | {
        "pitcher", "arsenal_n_pitches"
    }
    assert set(enc.columns) == expected_cols


def test_pitcher_arsenal_rates_sum_to_one():
    df = pd.DataFrame({
        "pitcher": [1] * 100 + [2] * 50,
        "pitch_type_canonical": ["FF"] * 80 + ["SL"] * 20 + ["FF"] * 25 + ["CH"] * 25,
    })
    enc = compute_pitcher_arsenal_encoding(df, alpha=0.0)  # no smoothing → exact rates
    rate_cols = [f"pt_{t}_rate" for t in PITCH_TYPES]
    sums = enc[rate_cols].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


# ---------- end-to-end build ----------


def _full_pitch_row(**kwargs):
    base = {
        "game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
        "game_date": "2024-04-01",
        "pitcher": 100, "batter": 200,
        "pitch_type_canonical": "FF",
        "balls": 0, "strikes": 0, "outs_when_up": 0, "inning": 1,
        "p_throws": "R", "stand": "L",
        "on_1b": None, "on_2b": None, "on_3b": None,
    }
    base.update(kwargs)
    return base


def test_build_features_includes_metadata_columns():
    pitches = _pitches([_full_pitch_row()])
    metadata = _meta([
        {"game_pk": 1, "hp_umpire_id": 999, "temp_f": 70,
         "roof_closed": False, "wind_speed_mph": 5, "venue_id": 4001},
    ])
    out = build_xgboost_features(pitches, metadata)
    assert out["hp_umpire_id"].iloc[0] == 999
    assert out["temp_f"].iloc[0] == 70
    assert out["roof_closed"].iloc[0] == 0  # False → 0
    assert out["wind_speed_mph"].iloc[0] == 5
    assert out["venue_id"].iloc[0] == 4001


def test_build_features_with_arsenal_encoding():
    pitches = _pitches([_full_pitch_row()])
    metadata = _meta([
        {"game_pk": 1, "hp_umpire_id": 999, "temp_f": 70,
         "roof_closed": False, "wind_speed_mph": 5, "venue_id": 4001},
    ])
    enc = compute_pitcher_arsenal_encoding(
        pd.DataFrame({"pitcher": [100] * 50, "pitch_type_canonical": ["FF"] * 50}),
        alpha=0.0,
    )
    out = build_xgboost_features(pitches, metadata, arsenal_encoding=enc)
    assert "pt_FF_rate" in out.columns
    # Pitcher 100 always threw FF → rate ≈ 1.0
    assert math.isclose(float(out["pt_FF_rate"].iloc[0]), 1.0, abs_tol=1e-6)


def test_build_features_handles_handedness_encoding():
    pitches = _pitches([
        _full_pitch_row(p_throws="R", stand="L"),
        _full_pitch_row(at_bat_number=2, p_throws="L", stand="R"),
    ])
    metadata = _meta([
        {"game_pk": 1, "hp_umpire_id": 999, "temp_f": 70,
         "roof_closed": False, "wind_speed_mph": 5, "venue_id": 4001},
    ])
    out = build_xgboost_features(pitches, metadata)
    assert out["p_throws_L"].tolist() == [0, 1]
    assert out["stand_L"].tolist() == [1, 0]
