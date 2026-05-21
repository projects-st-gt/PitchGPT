"""Unit tests for preprocessing — harmonize_and_tag, velocity binning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.preprocess import (
    bin_velo_z_score,
    harmonize_and_tag,
    type_relative_velocity_bin,
)


# ---------- harmonize_and_tag ----------


def _ab_row(game_pk, ab, pitch_type, plate_x=0.0, plate_z=2.5,
            sz_top=3.5, sz_bot=1.5, p_throws="R", zone=5):
    # Default zone=5 (SIS middle-of-zone) matches the default plate_x=0, plate_z=2.5.
    return {
        "game_pk": game_pk,
        "at_bat_number": ab,
        "pitch_type": pitch_type,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "sz_top": sz_top,
        "sz_bot": sz_bot,
        "p_throws": p_throws,
        "zone": zone,
    }


def test_harmonize_and_tag_drops_unmapped_at_bat():
    df = pd.DataFrame(
        [
            _ab_row(1, 1, "FF"),
            _ab_row(1, 1, "SL"),
            _ab_row(1, 2, "EP"),  # eephus → drop AB (1, 2)
            _ab_row(1, 2, "FF"),
            _ab_row(2, 1, "FF"),
        ]
    )
    out = harmonize_and_tag(df)
    surviving = set(zip(out["game_pk"], out["at_bat_number"]))
    assert surviving == {(1, 1), (2, 1)}
    assert "action_zone" in out.columns
    assert "feature_zone" in out.columns
    assert "pitch_type_canonical" in out.columns


def test_harmonize_and_tag_drops_invalid_zone_height():
    df = pd.DataFrame(
        [
            _ab_row(1, 1, "FF", sz_top=3.5, sz_bot=1.5),  # height 2.0, OK
            _ab_row(2, 1, "FF", sz_top=2.4, sz_bot=1.5),  # height 0.9, drop
        ]
    )
    out = harmonize_and_tag(df)
    assert list(out["game_pk"]) == [1]


# ---------- bin_velo_z_score ----------


def test_bin_velo_z_score_center_to_middle_decile():
    z = pd.Series([0.0])
    out = bin_velo_z_score(z)
    # 0.0 sits exactly on the boundary between deciles 4 and 5; cut() with
    # right=True (default) puts it in 4.
    assert int(out.iloc[0]) in (4, 5)


def test_bin_velo_z_score_top_decile():
    z = pd.Series([3.0])
    out = bin_velo_z_score(z)
    assert int(out.iloc[0]) == 9


def test_bin_velo_z_score_bottom_decile():
    z = pd.Series([-3.0])
    out = bin_velo_z_score(z)
    assert int(out.iloc[0]) == 0


def test_bin_velo_z_score_preserves_nan():
    z = pd.Series([0.5, np.nan, -0.5])
    out = bin_velo_z_score(z)
    assert pd.isna(out.iloc[1])


# ---------- type_relative_velocity_bin ----------


def _make_velo_test_inputs():
    """Two pitchers, one date. Pitcher 1 has enough data; pitcher 2 doesn't."""
    df = pd.DataFrame(
        {
            "pitcher": [1001, 1001, 2002, 2002],
            "pitch_type_canonical": ["FF", "FF", "FF", "FF"],
            "release_speed": [95.0, 96.0, 89.0, 91.0],
            "asof_date": ["2024-04-15"] * 4,
        }
    )
    velo_stats = pd.DataFrame(
        {
            "pitcher": [1001, 2002],
            "pitch_type_canonical": ["FF", "FF"],
            "asof_date": ["2024-04-15", "2024-04-15"],
            "mean": [95.0, 90.0],
            "std": [1.0, 1.0],
            "n": [500, 5],  # pitcher 2002 below min 30 → fallback
        }
    )
    league_means = pd.DataFrame(
        {
            "pitch_type_canonical": ["FF"],
            "asof_date": ["2024-04-15"],
            "mean": [93.0],
            "std": [2.0],
        }
    )
    return df, velo_stats, league_means


def test_type_relative_velocity_uses_pitcher_stats_when_n_sufficient():
    df, velo_stats, league_means = _make_velo_test_inputs()
    out = type_relative_velocity_bin(
        df, velo_stats=velo_stats, league_means=league_means, min_pitches_for_pitcher=30
    )
    pitcher1 = out[out["pitcher"] == 1001]
    # release_speed_z = (95 - 95) / 1 = 0; (96 - 95) / 1 = 1
    assert pitcher1["release_speed_z"].tolist() == [0.0, 1.0]
    assert not pitcher1["velo_used_fallback"].any()


def test_type_relative_velocity_falls_back_to_league_for_low_n_pitcher():
    df, velo_stats, league_means = _make_velo_test_inputs()
    out = type_relative_velocity_bin(
        df, velo_stats=velo_stats, league_means=league_means, min_pitches_for_pitcher=30
    )
    pitcher2 = out[out["pitcher"] == 2002]
    # Should use league: mean 93, std 2 → (89-93)/2 = -2; (91-93)/2 = -1
    assert pitcher2["release_speed_z"].tolist() == [-2.0, -1.0]
    assert pitcher2["velo_used_fallback"].all()


def test_type_relative_velocity_handles_missing_pitcher_entry():
    df, _, league_means = _make_velo_test_inputs()
    empty_stats = pd.DataFrame(columns=["pitcher", "pitch_type_canonical", "asof_date",
                                        "mean", "std", "n"])
    out = type_relative_velocity_bin(
        df, velo_stats=empty_stats, league_means=league_means
    )
    # All rows should fall back to league stats.
    assert out["velo_used_fallback"].all()
