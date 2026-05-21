"""Unit tests for the profile cache schema and flatteners.

These tests lock in the contract between cache writer, cache reader, and
model: the slot order in the flat profile vector. If any of these tests
fails after a refactor, the schema changed — bump ``PROFILE_SCHEMA_VERSION``
and consider whether existing caches need to be rebuilt.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.dataset import N_PITCH_TYPES, PITCH_TYPES
from data.player_profiles import N_IN_ZONE_CELLS
from data.profile_cache import (
    BATTER_FEATURE_INDEX,
    BATTER_FEATURE_NAMES,
    BATTER_VECTOR_LEN,
    COUNT_STATES,
    N_COUNT_STATES,
    PITCHER_FEATURE_INDEX,
    PITCHER_FEATURE_NAMES,
    PITCHER_VECTOR_LEN,
    PROFILE_SCHEMA_VERSION,
    blend_with_league_mean,
    build_batter_profile_vector,
    build_pitcher_profile_vector,
    compute_league_means,
)


# ============================================================
# Schema invariants
# ============================================================


def test_pitcher_vector_length_matches_documented_size():
    # v6 (2026-05-17): drop 12 entropy dims; add 84 per-(type x count),
    # 14 per-(type x stand), 14 movement dims. Net 118 → 218.
    expected = (
        N_PITCH_TYPES  # arsenal
        + N_PITCH_TYPES  # mean_velo
        + N_PITCH_TYPES  # mean_spin
        + N_PITCH_TYPES * N_IN_ZONE_CELLS  # heatmap (7 × 9 = 63)
        + 6  # recent_30d_xwoba, n_pitches; days_since; recent_3s_xwoba, n; profile_conf
        + 2  # long_window_span_days, long_window_pct_current_season (v2)
        + N_PITCH_TYPES  # has_pitch flags
        + N_PITCH_TYPES  # arm_slot per pitch type (v4)
        + N_PITCH_TYPES * N_COUNT_STATES  # arsenal_{pt}_b{b}s{s} per-count (v6)  84
        + N_PITCH_TYPES * 2  # arsenal_{pt}_vs{stand} per-stand (v6)  14
        + N_PITCH_TYPES  # mean_pfx_x_{pt} (v6)  7
        + N_PITCH_TYPES  # mean_pfx_z_{pt} (v6)  7
    )
    assert PITCHER_VECTOR_LEN == expected
    assert len(PITCHER_FEATURE_NAMES) == expected
    assert len(PITCHER_FEATURE_INDEX) == expected


def test_batter_vector_length_matches_documented_size():
    # 2 (recent_14d) + 25 swing + 25 whiff + 25 xba + 7 chase + 4 outcomes
    # + 1 confidence + 2 staleness (v2) + 7 whiff_on_swing + 7 swing_rate (v3, ADR 013)
    expected = 2 + 3 * N_IN_ZONE_CELLS + N_PITCH_TYPES + 4 + 1 + 2 + 2 * N_PITCH_TYPES
    assert BATTER_VECTOR_LEN == expected
    assert len(BATTER_FEATURE_NAMES) == expected
    assert len(BATTER_FEATURE_INDEX) == expected


def test_schema_version_is_6():
    """Reminder to bump on feature changes; currently at 6 since the v6
    profile-feature expansion (drop 12 entropy dims, add 84 per-(type×count)
    + 14 per-(type×stand) + 14 movement; pitcher profile 118 → 218)."""
    assert PROFILE_SCHEMA_VERSION == 6


def test_schema_version_is_an_integer():
    assert isinstance(PROFILE_SCHEMA_VERSION, int)
    assert PROFILE_SCHEMA_VERSION >= 1


def test_count_states_cover_all_legal_balls_strikes():
    # 4 ball values (0-3) × 3 strike values (0-2) = 12
    assert N_COUNT_STATES == 12
    assert len(COUNT_STATES) == 12
    for b, s in COUNT_STATES:
        assert 0 <= b <= 3
        assert 0 <= s <= 2


def test_pitcher_feature_names_are_unique():
    assert len(PITCHER_FEATURE_NAMES) == len(set(PITCHER_FEATURE_NAMES))


def test_batter_feature_names_are_unique():
    assert len(BATTER_FEATURE_NAMES) == len(set(BATTER_FEATURE_NAMES))


# ============================================================
# Pitcher flattener
# ============================================================


def _make_pitcher_pitches(n_each_type: dict[str, int], game_date: str = "2024-04-01"):
    """Build pitches for a single pitcher with arsenal as specified by counts."""
    rows = []
    for pt, n in n_each_type.items():
        for _ in range(n):
            rows.append(
                {
                    "pitch_type_canonical": pt,
                    "feature_zone": 4,  # all middle-middle for simplicity
                    "release_speed": 95.0,
                    "release_spin_rate": 2200.0,
                    "balls": 0,
                    "strikes": 0,
                    "game_date": pd.to_datetime(game_date),
                    "game_pk": 1,
                    "game_num": 1,
                    "estimated_woba_using_speedangle": 0.30,
                }
            )
    return pd.DataFrame(rows)


def test_pitcher_vector_arsenal_slot_matches_input():
    # 60 FF, 30 SL, 10 CH; arsenal_pct should reflect those fractions.
    pitches = _make_pitcher_pitches({"FF": 60, "SL": 30, "CH": 10})
    vec = build_pitcher_profile_vector(pitches, asof_game_date="2024-05-01")

    assert math.isclose(vec[PITCHER_FEATURE_INDEX["arsenal_FF"]], 0.6, abs_tol=1e-5)
    assert math.isclose(vec[PITCHER_FEATURE_INDEX["arsenal_SL"]], 0.3, abs_tol=1e-5)
    assert math.isclose(vec[PITCHER_FEATURE_INDEX["arsenal_CH"]], 0.1, abs_tol=1e-5)
    # Pitch types not thrown should be 0 in arsenal_pct
    assert vec[PITCHER_FEATURE_INDEX["arsenal_SI"]] == 0.0
    assert vec[PITCHER_FEATURE_INDEX["arsenal_FS"]] == 0.0


def test_pitcher_vector_has_pitch_mask_distinguishes_zero_vs_unseen():
    """The 'has thrown this pitch' mask is the key disambiguation feature:
    0% in the arsenal slot means very different things for 'never thrown'
    vs 'rarely thrown'. The mask should fire for thrown types only.
    """
    pitches = _make_pitcher_pitches({"FF": 60, "SL": 30, "CH": 10})
    vec = build_pitcher_profile_vector(pitches, asof_game_date="2024-05-01")

    assert vec[PITCHER_FEATURE_INDEX["has_pitch_FF"]] == 1.0
    assert vec[PITCHER_FEATURE_INDEX["has_pitch_SL"]] == 1.0
    assert vec[PITCHER_FEATURE_INDEX["has_pitch_CH"]] == 1.0
    # Never thrown
    assert vec[PITCHER_FEATURE_INDEX["has_pitch_SI"]] == 0.0
    assert vec[PITCHER_FEATURE_INDEX["has_pitch_FS"]] == 0.0


def test_pitcher_vector_empty_input_returns_safe_defaults():
    """Debut pitcher case: no prior pitches → all-zeros arsenal/has_pitch,
    NaN/0 for stats that require data, profile_confidence == 0.
    """
    empty = pd.DataFrame(
        columns=[
            "pitch_type_canonical", "feature_zone", "release_speed",
            "release_spin_rate", "balls", "strikes", "game_date",
            "estimated_woba_using_speedangle",
        ]
    )
    vec = build_pitcher_profile_vector(empty, asof_game_date="2024-04-01")

    assert vec.shape == (PITCHER_VECTOR_LEN,)
    # Arsenal defaults to 0.0
    assert vec[PITCHER_FEATURE_INDEX["arsenal_FF"]] == 0.0
    # has_pitch all 0
    for pt in PITCH_TYPES:
        assert vec[PITCHER_FEATURE_INDEX[f"has_pitch_{pt}"]] == 0.0
    # Confidence = 0
    assert vec[PITCHER_FEATURE_INDEX["profile_confidence"]] == 0.0
    # Recent-form is NaN for empty data
    assert np.isnan(vec[PITCHER_FEATURE_INDEX["recent_30d_xwoba"]])
    # days_since is NaN for debut
    assert np.isnan(vec[PITCHER_FEATURE_INDEX["days_since_last_appearance"]])


def test_pitcher_vector_recent_form_populated():
    pitches = _make_pitcher_pitches({"FF": 100}, game_date="2024-04-15")
    vec = build_pitcher_profile_vector(pitches, asof_game_date="2024-05-01")
    # 16 days between 4-15 and 5-01 → within 30-day window
    assert vec[PITCHER_FEATURE_INDEX["recent_30d_n_pitches"]] == 100
    # xwOBA in synthetic data is uniformly 0.30
    assert math.isclose(
        vec[PITCHER_FEATURE_INDEX["recent_30d_xwoba"]], 0.30, abs_tol=1e-5
    )
    # Layoff: 16 days
    assert vec[PITCHER_FEATURE_INDEX["days_since_last_appearance"]] == 16


def test_pitcher_vector_staleness_zero_span_when_pitches_are_recent():
    """All synthetic pitches on a single date → span_days = days from that
    date to asof, pct_current_season = 1.0 since same year."""
    pitches = _make_pitcher_pitches({"FF": 100}, game_date="2024-04-15")
    vec = build_pitcher_profile_vector(pitches, asof_game_date="2024-05-01")
    assert vec[PITCHER_FEATURE_INDEX["long_window_span_days"]] == 16
    assert math.isclose(
        vec[PITCHER_FEATURE_INDEX["long_window_pct_current_season"]], 1.0
    )


def test_pitcher_vector_staleness_flags_cross_season():
    """The canonical cross-season case: April-1 AB whose 1000-pitch window
    is mostly previous-October data."""
    # 800 pitches in October 2023, 200 in April 2024 (asof-eve)
    rows = []
    for i in range(800):
        rows.append({
            "pitch_type_canonical": "FF", "feature_zone": 4,
            "release_speed": 95.0, "release_spin_rate": 2200.0,
            "balls": 0, "strikes": 0,
            "game_date": pd.to_datetime("2023-10-15"),
            "game_pk": 1000 + i,
            "game_num": 1,
            "estimated_woba_using_speedangle": 0.30,
        })
    for i in range(200):
        rows.append({
            "pitch_type_canonical": "FF", "feature_zone": 4,
            "release_speed": 95.0, "release_spin_rate": 2200.0,
            "balls": 0, "strikes": 0,
            "game_date": pd.to_datetime("2024-04-01"),
            "game_pk": 2000 + i,
            "game_num": 1,
            "estimated_woba_using_speedangle": 0.30,
        })
    pitches = pd.DataFrame(rows)
    vec = build_pitcher_profile_vector(pitches, asof_game_date="2024-04-15")
    # Span: from oldest (2023-10-15) to asof (2024-04-15) = ~183 days
    assert vec[PITCHER_FEATURE_INDEX["long_window_span_days"]] >= 180
    # Current season fraction: 200 / 1000 = 0.2 (vector is float32; allow rounding)
    assert math.isclose(
        float(vec[PITCHER_FEATURE_INDEX["long_window_pct_current_season"]]),
        0.2,
        abs_tol=1e-5,
    )


def test_pitcher_vector_staleness_nan_for_empty_window():
    empty = pd.DataFrame(
        columns=[
            "pitch_type_canonical", "feature_zone", "release_speed",
            "release_spin_rate", "balls", "strikes", "game_date",
            "estimated_woba_using_speedangle",
        ]
    )
    vec = build_pitcher_profile_vector(empty, asof_game_date="2024-04-01")
    assert np.isnan(vec[PITCHER_FEATURE_INDEX["long_window_span_days"]])
    assert np.isnan(
        vec[PITCHER_FEATURE_INDEX["long_window_pct_current_season"]]
    )


# ============================================================
# Batter flattener
# ============================================================


def _make_batter_pitches_seen(n: int = 100, game_date: str = "2024-04-01"):
    return pd.DataFrame(
        {
            "feature_zone": [4] * n,
            "description": ["ball"] * (n // 2) + ["swinging_strike"] * (n - n // 2),
            "estimated_ba_using_speedangle": [None] * n,
            "pitch_type_canonical": ["FF"] * n,
            "game_date": pd.to_datetime([game_date] * n),
            "game_pk": [1] * n,
            "game_num": [1] * n,
        }
    )


def _make_batter_pas(n_pas: int = 50, game_date: str = "2024-04-15"):
    """Create n_pas with mix of K, BB, single events."""
    events = (["strikeout"] * (n_pas // 4)
              + ["walk"] * (n_pas // 4)
              + ["single"] * (n_pas // 4)
              + ["field_out"] * (n_pas - 3 * (n_pas // 4)))
    event_to_woba = {"strikeout": 0.0, "walk": 0.7, "single": 0.9, "field_out": 0.0}
    return pd.DataFrame(
        {
            "events": events,
            "game_date": pd.to_datetime([game_date] * n_pas),
            "woba_value": [event_to_woba[e] for e in events],
            "launch_speed": [None] * n_pas,
        }
    )


def test_batter_vector_shape():
    seen = _make_batter_pitches_seen()
    pas = _make_batter_pas()
    vec = build_batter_profile_vector(seen, pas, asof_game_date="2024-05-01")
    assert vec.shape == (BATTER_VECTOR_LEN,)


def test_batter_vector_outcome_rates():
    seen = _make_batter_pitches_seen()
    pas = _make_batter_pas(n_pas=40)  # 10 K, 10 BB, 10 single, 10 field_out
    vec = build_batter_profile_vector(seen, pas, asof_game_date="2024-05-01")
    assert math.isclose(vec[BATTER_FEATURE_INDEX["k_pct"]], 0.25)
    assert math.isclose(vec[BATTER_FEATURE_INDEX["bb_pct"]], 0.25)
    assert vec[BATTER_FEATURE_INDEX["n_pas"]] == 40


def test_batter_vector_swing_grid_in_correct_slot():
    seen = _make_batter_pitches_seen(n=100, game_date="2024-04-01")
    # _make_batter_pitches_seen alternates ball / swinging_strike → swing rate ~50%
    pas = _make_batter_pas()
    vec = build_batter_profile_vector(seen, pas, asof_game_date="2024-05-01")
    # All seen pitches were in cell 4 (middle in-zone under v5) → that slot has
    # the swing rate; others NaN.
    assert math.isclose(vec[BATTER_FEATURE_INDEX["swing_z4"]], 0.5, abs_tol=1e-5)
    # Other zones got no pitches → NaN
    assert np.isnan(vec[BATTER_FEATURE_INDEX["swing_z0"]])


def test_batter_vector_empty_input_returns_safe_defaults():
    empty_seen = pd.DataFrame(
        columns=["feature_zone", "description", "estimated_ba_using_speedangle",
                 "pitch_type_canonical"]
    )
    empty_pas = pd.DataFrame(columns=["events"])
    vec = build_batter_profile_vector(
        empty_seen, empty_pas, asof_game_date="2024-04-01"
    )
    assert vec.shape == (BATTER_VECTOR_LEN,)
    assert vec[BATTER_FEATURE_INDEX["n_pas"]] == 0
    assert vec[BATTER_FEATURE_INDEX["profile_confidence"]] == 0.0
    assert np.isnan(vec[BATTER_FEATURE_INDEX["recent_14d_woba"]])
    assert np.isnan(vec[BATTER_FEATURE_INDEX["k_pct"]])


# ============================================================
# Heatmap layout matches PITCH_TYPES × cell ordering
# ============================================================


def test_heatmap_indexing_is_canonical():
    """Verify heatmap slot ordering: heatmap_FF_z0 starts after 7+7+7 scalar
    slots; consecutive in-zone cells (per pitch type), then onto the next pitch
    type. Under v5 (SIS 14-zone) the in-zone is 3×3 = 9 cells, so the last
    in-zone cell per type is ``z8``.
    """
    base = 3 * N_PITCH_TYPES  # 21
    last_cell_idx = N_IN_ZONE_CELLS - 1  # 8 under v5
    assert PITCHER_FEATURE_INDEX["heatmap_FF_z0"] == base
    assert PITCHER_FEATURE_INDEX[f"heatmap_FF_z{last_cell_idx}"] == base + last_cell_idx
    assert PITCHER_FEATURE_INDEX["heatmap_SI_z0"] == base + N_IN_ZONE_CELLS
    assert PITCHER_FEATURE_INDEX[f"heatmap_FS_z{last_cell_idx}"] == base + N_PITCH_TYPES * N_IN_ZONE_CELLS - 1


# ============================================================
# League-mean aggregation
# ============================================================


def _player_cache_row(player_id, asof_date, asof_game_num, fold_id, vector):
    return {
        "player_id": int(player_id),
        "asof_date": asof_date,
        "asof_game_num": int(asof_game_num),
        "fold_id": int(fold_id),
        "schema_version": PROFILE_SCHEMA_VERSION,
        "vector": vector.tolist() if hasattr(vector, "tolist") else list(vector),
    }


def test_compute_league_means_averages_per_asof():
    df = pd.DataFrame([
        _player_cache_row(1, "2024-04-01", 1, 0, np.array([1.0, 2.0, 3.0])),
        _player_cache_row(2, "2024-04-01", 1, 0, np.array([3.0, 4.0, 5.0])),
        _player_cache_row(3, "2024-04-02", 1, 0, np.array([10.0, 20.0, 30.0])),
    ])
    league = compute_league_means(df)
    # Two distinct asof keys → two league rows
    assert len(league) == 2
    row_0401 = league[league["asof_date"] == "2024-04-01"].iloc[0]
    np.testing.assert_allclose(row_0401["vector"], [2.0, 3.0, 4.0])
    assert int(row_0401["n_players_in_mean"]) == 2

    row_0402 = league[league["asof_date"] == "2024-04-02"].iloc[0]
    np.testing.assert_allclose(row_0402["vector"], [10.0, 20.0, 30.0])
    assert int(row_0402["n_players_in_mean"]) == 1


def test_compute_league_means_handles_nans_with_nanmean():
    """NaN slots in some players shouldn't drag the league mean to NaN
    unless ALL players are NaN in that slot."""
    df = pd.DataFrame([
        _player_cache_row(1, "2024-04-01", 1, 0, np.array([1.0, np.nan, 3.0])),
        _player_cache_row(2, "2024-04-01", 1, 0, np.array([3.0, 4.0, np.nan])),
        _player_cache_row(3, "2024-04-01", 1, 0, np.array([np.nan, np.nan, np.nan])),
    ])
    league = compute_league_means(df)
    vec = np.array(league["vector"].iloc[0])
    # Slot 0: mean of (1, 3, NaN) = 2
    assert math.isclose(vec[0], 2.0)
    # Slot 1: mean of (NaN, 4, NaN) = 4
    assert math.isclose(vec[1], 4.0)
    # Slot 2: mean of (3, NaN, NaN) = 3
    assert math.isclose(vec[2], 3.0)


def test_compute_league_means_all_nan_cell_stays_nan():
    df = pd.DataFrame([
        _player_cache_row(1, "2024-04-01", 1, 0, np.array([np.nan, 1.0])),
        _player_cache_row(2, "2024-04-01", 1, 0, np.array([np.nan, 2.0])),
    ])
    league = compute_league_means(df)
    vec = np.array(league["vector"].iloc[0])
    assert np.isnan(vec[0])
    assert math.isclose(vec[1], 1.5)


def test_compute_league_means_empty_input():
    league = compute_league_means(pd.DataFrame(
        columns=["player_id", "asof_date", "asof_game_num", "fold_id",
                 "schema_version", "vector"]
    ))
    assert len(league) == 0


def test_compute_league_means_rejects_mixed_schemas():
    df = pd.DataFrame([
        {**_player_cache_row(1, "2024-04-01", 1, 0, np.array([1.0])),
         "schema_version": 1},
        {**_player_cache_row(2, "2024-04-01", 1, 0, np.array([2.0])),
         "schema_version": 2},
    ])
    with pytest.raises(ValueError, match="schema versions"):
        compute_league_means(df)


# ============================================================
# blend_with_league_mean
# ============================================================


def test_blend_with_full_confidence_returns_per_player():
    per = np.array([1.0, 2.0, 3.0])
    league = np.array([10.0, 20.0, 30.0])
    out = blend_with_league_mean(per, league, profile_confidence=1.0)
    np.testing.assert_allclose(out, per)


def test_blend_with_zero_confidence_returns_league():
    per = np.array([1.0, 2.0, 3.0])
    league = np.array([10.0, 20.0, 30.0])
    out = blend_with_league_mean(per, league, profile_confidence=0.0)
    np.testing.assert_allclose(out, league)


def test_blend_with_half_confidence_is_average():
    per = np.array([1.0, 2.0, 3.0])
    league = np.array([10.0, 20.0, 30.0])
    out = blend_with_league_mean(per, league, profile_confidence=0.5)
    np.testing.assert_allclose(out, [5.5, 11.0, 16.5])


def test_blend_nan_slots_in_per_player_use_league_value():
    """The canonical debut-player case: per-player has NaN where the
    player has no data; league mean fills in; weighted blend uses league
    on those slots."""
    per = np.array([1.0, np.nan, 3.0])
    league = np.array([10.0, 20.0, 30.0])
    # With confidence 0.5: filled = [1, 20, 3]; blend = [5.5, 20.0, 16.5]
    out = blend_with_league_mean(per, league, profile_confidence=0.5)
    np.testing.assert_allclose(out, [5.5, 20.0, 16.5])


def test_blend_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        blend_with_league_mean(
            np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), profile_confidence=0.5
        )
