"""Unit tests for player_profiles, with the synthetic-marker leakage test
as the centerpiece. ADR 003 / statcast-pipeline skill enforce the same
window discipline; if any of these fail, leakage is silently entering the
training pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.dataset import PITCH_TYPES
from data.player_profiles import (
    N_IN_ZONE_CELLS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    batter_chase_rate_by_type,
    batter_last_n_days_woba,
    batter_outcome_rates,
    batter_zone_grids,
    before_asof,
    compute_league_velo_means,
    pitcher_arsenal,
    pitcher_arsenal_by_count,
    pitcher_arsenal_by_stand,
    pitcher_count_conditional_entropy,
    pitcher_days_since_last_appearance,
    pitcher_last_n_days_xwoba,
    pitcher_last_n_starts_xwoba,
    pitcher_movement_by_type,
    pitcher_velo_stats,
    pitcher_zone_heatmap_by_type,
    window_freshness,
)


# ============================================================
# CENTERPIECE: leakage primitive must reject same-game and later content
# ============================================================


def test_before_asof_rejects_same_game_same_num():
    """A pitch in the same (game_date, game_num) must NOT be in the window."""
    df = pd.DataFrame(
        {
            "pitcher": [1, 1],
            "game_date": pd.to_datetime(["2024-05-15", "2024-05-15"]),
            "game_num": [1, 1],
            "marker": ["EARLIER_AB", "TARGET_AB"],
        }
    )
    out = before_asof(df, "2024-05-15", asof_game_num=1)
    # Both rows are in the same game as asof — neither should be in the window.
    assert len(out) == 0


def test_before_asof_includes_strictly_earlier_dates():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-05-14", "2024-05-15", "2024-05-16"]),
            "game_num": [1, 1, 1],
            "marker": ["BEFORE", "CURRENT", "AFTER"],
        }
    )
    out = before_asof(df, "2024-05-15", asof_game_num=1)
    assert list(out["marker"]) == ["BEFORE"]


def test_before_asof_handles_doubleheader_g1_correctly():
    """Same-date G1 (lower game_num) must be in G2's trailing window."""
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-05-15", "2024-05-15", "2024-05-15"]),
            "game_num": [1, 2, 3],  # G1, G2, G3 of the same date
            "marker": ["G1", "G2", "G3"],
        }
    )
    # asof = G2 → only G1 should be in window
    out = before_asof(df, "2024-05-15", asof_game_num=2)
    assert list(out["marker"]) == ["G1"]


def test_synthetic_marker_pitch_does_not_leak_into_window():
    """Centerpiece leakage test. If this fails, current-AB content is in the
    training-time profile — the model would peek at the answer."""
    df = pd.DataFrame(
        {
            "pitcher": [1] * 5,
            "game_pk": [99, 99, 100, 100, 101],
            "game_date": pd.to_datetime(
                ["2024-05-14", "2024-05-14", "2024-05-15", "2024-05-15", "2024-05-16"]
            ),
            "game_num": [1, 1, 1, 1, 1],
            "pitch_type_canonical": ["FF", "SL", "MARKER", "MARKER", "FF"],
            "release_speed": [95.0, 85.0, 99.0, 99.0, 96.0],
            "estimated_woba_using_speedangle": [0.3, 0.0, 0.5, 0.5, 0.4],
        }
    )
    # Predicting an AB on game_pk=100 (date 2024-05-15, num 1)
    earlier = before_asof(df, "2024-05-15", asof_game_num=1)
    # MARKER pitches must be excluded
    assert "MARKER" not in set(earlier["pitch_type_canonical"])
    # Future-game pitch (101 on 2024-05-16) must also be excluded
    assert (earlier["game_pk"] == 101).sum() == 0
    # Earlier-game pitches must be present
    assert (earlier["game_pk"] == 99).sum() == 2


# ============================================================
# pitcher_arsenal
# ============================================================


def test_pitcher_arsenal_empty_input_returns_safe_defaults():
    out = pitcher_arsenal(pd.DataFrame(columns=["pitch_type_canonical", "release_speed"]))
    assert out["arsenal_pct"] == {}
    assert out["n_pitches"] == 0
    assert out["profile_confidence"] == 0.0


def test_pitcher_arsenal_computes_fractions_and_means():
    pitches = pd.DataFrame(
        {
            "pitch_type_canonical": ["FF"] * 3 + ["SL"] * 1,
            "release_speed": [95, 95, 96, 84],
        }
    )
    out = pitcher_arsenal(pitches, window_pitches=10)
    assert math.isclose(out["arsenal_pct"]["FF"], 0.75)
    assert math.isclose(out["arsenal_pct"]["SL"], 0.25)
    assert math.isclose(out["mean_velo_by_type"]["FF"], 95 + 1/3)
    assert out["n_pitches"] == 4
    # Below 10-pitch window → confidence < 1
    assert out["profile_confidence"] == 0.4


def test_pitcher_arsenal_takes_only_last_window_pitches():
    pitches = pd.DataFrame(
        {
            "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 5,
            "release_speed": [95] * 5 + [85] * 5,
        }
    )
    # Window=3 should pick only the last 3 (all SL).
    out = pitcher_arsenal(pitches, window_pitches=3)
    assert out["arsenal_pct"] == {"SL": 1.0}


# -----------------------------------------------------------------
# pitcher_arsenal_by_count (v6 / 2026-05-17)
# -----------------------------------------------------------------


def test_pitcher_arsenal_by_count_returns_zero_for_unthrown_types_in_observed_cells():
    """A cell with at least one pitch produces entries for ALL 7 types (0.0 for unthrown)."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL", "SL"],
        "balls": [0, 0, 0, 0],
        "strikes": [0, 0, 2, 2],
    })
    out = pitcher_arsenal_by_count(pitches)
    # (0, 0) was observed: 2 FF, 0 SL. (0, 2) was observed: 0 FF, 2 SL.
    assert out[(0, 0, "FF")] == 1.0
    assert out[(0, 0, "SL")] == 0.0
    assert out[(0, 2, "FF")] == 0.0
    assert out[(0, 2, "SL")] == 1.0
    # Verify ALL 7 pitch types appear in each observed cell (the contract).
    for cell in [(0, 0), (0, 2)]:
        for pt in PITCH_TYPES:
            assert (cell[0], cell[1], pt) in out, f"missing {(cell[0], cell[1], pt)}"


def test_pitcher_arsenal_by_count_omits_unobserved_cells():
    """A (b, s) cell with zero observations produces no entries at all."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"],
        "balls": [0],
        "strikes": [0],
    })
    out = pitcher_arsenal_by_count(pitches)
    assert (0, 0, "FF") in out
    assert (3, 0, "FF") not in out  # No 3-0 observations
    assert (1, 1, "FF") not in out


def test_pitcher_arsenal_by_count_empty_input_returns_empty():
    out = pitcher_arsenal_by_count(pd.DataFrame(columns=["pitch_type_canonical", "balls", "strikes"]))
    assert out == {}


def test_pitcher_arsenal_by_count_respects_trailing_window():
    """Only the last window_pitches matter."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 3,
        "balls": [0] * 8,
        "strikes": [0] * 8,
    })
    out = pitcher_arsenal_by_count(pitches, window_pitches=3)
    # Last 3 pitches were all SL on 0-0
    assert out[(0, 0, "SL")] == 1.0
    assert out[(0, 0, "FF")] == 0.0


def test_pitcher_arsenal_by_count_raises_on_missing_columns():
    pitches = pd.DataFrame({"pitch_type_canonical": ["FF"], "balls": [0]})  # no strikes
    with pytest.raises(KeyError, match="strikes"):
        pitcher_arsenal_by_count(pitches)


# -----------------------------------------------------------------
# pitcher_arsenal_by_stand (v6 / 2026-05-17)
# -----------------------------------------------------------------


def test_pitcher_arsenal_by_stand_returns_zero_for_unthrown_types_in_observed_cells():
    """A stand with at least one pitch produces entries for ALL 7 types (0.0 for unthrown)."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL"],
        "stand": ["R", "R", "L"],
    })
    out = pitcher_arsenal_by_stand(pitches)
    assert out[("R", "FF")] == 1.0
    assert out[("R", "SL")] == 0.0
    assert out[("L", "FF")] == 0.0
    assert out[("L", "SL")] == 1.0
    # Verify ALL 7 pitch types appear in each observed stand (the contract).
    for stand in ["R", "L"]:
        for pt in PITCH_TYPES:
            assert (stand, pt) in out, f"missing {(stand, pt)}"


def test_pitcher_arsenal_by_stand_omits_unobserved_stands():
    """If a pitcher only faced RHB, no L entries appear."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"],
        "stand": ["R"],
    })
    out = pitcher_arsenal_by_stand(pitches)
    assert ("R", "FF") in out
    assert ("L", "FF") not in out


def test_pitcher_arsenal_by_stand_empty_input_returns_empty():
    out = pitcher_arsenal_by_stand(pd.DataFrame(columns=["pitch_type_canonical", "stand"]))
    assert out == {}


def test_pitcher_arsenal_by_stand_respects_trailing_window():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 3,
        "stand": ["R"] * 8,
    })
    out = pitcher_arsenal_by_stand(pitches, window_pitches=3)
    assert out[("R", "SL")] == 1.0
    assert out[("R", "FF")] == 0.0


def test_pitcher_arsenal_by_stand_raises_on_missing_columns():
    pitches = pd.DataFrame({"pitch_type_canonical": ["FF"]})
    with pytest.raises(KeyError, match="stand"):
        pitcher_arsenal_by_stand(pitches)


# -----------------------------------------------------------------
# pitcher_movement_by_type (v6 / 2026-05-17)
# -----------------------------------------------------------------


def test_pitcher_movement_by_type_computes_per_type_means():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL"],
        "pfx_x": [-1.0, 1.0, 5.0],
        "pfx_z": [10.0, 12.0, 2.0],
    })
    out = pitcher_movement_by_type(pitches)
    assert out["FF"]["pfx_x"] == 0.0   # mean of -1, 1
    assert out["FF"]["pfx_z"] == 11.0  # mean of 10, 12
    assert out["SL"]["pfx_x"] == 5.0
    assert out["SL"]["pfx_z"] == 2.0


def test_pitcher_movement_by_type_omits_unthrown_types():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"],
        "pfx_x": [-1.0],
        "pfx_z": [10.0],
    })
    out = pitcher_movement_by_type(pitches)
    assert "FF" in out
    assert "SL" not in out
    assert "CU" not in out


def test_pitcher_movement_by_type_drops_nan_pitches_per_type():
    """A type with all-NaN pfx values is omitted; partial NaN uses nanmean."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL", "SL"],
        "pfx_x": [1.0, np.nan, np.nan, np.nan],
        "pfx_z": [10.0, np.nan, np.nan, np.nan],
    })
    out = pitcher_movement_by_type(pitches)
    assert out["FF"]["pfx_x"] == 1.0  # nanmean of [1.0, NaN] = 1.0
    assert out["FF"]["pfx_z"] == 10.0
    assert "SL" not in out  # all SL pfx values were NaN → omitted


def test_pitcher_movement_by_type_empty_input_returns_empty():
    out = pitcher_movement_by_type(pd.DataFrame(columns=["pitch_type_canonical", "pfx_x", "pfx_z"]))
    assert out == {}


def test_pitcher_movement_by_type_respects_trailing_window():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 3,
        "pfx_x": [1.0] * 5 + [5.0] * 3,
        "pfx_z": [10.0] * 5 + [2.0] * 3,
    })
    out = pitcher_movement_by_type(pitches, window_pitches=3)
    # Last 3 pitches were all SL
    assert "SL" in out
    assert out["SL"]["pfx_x"] == 5.0
    assert "FF" not in out


def test_pitcher_movement_by_type_raises_on_missing_columns():
    pitches = pd.DataFrame({"pitch_type_canonical": ["FF"], "pfx_x": [1.0]})  # no pfx_z
    with pytest.raises(KeyError, match="pfx_z"):
        pitcher_movement_by_type(pitches)


# ============================================================
# pitcher_last_n_starts_xwoba
# ============================================================


def test_pitcher_last_3_starts_picks_3_most_recent_by_game_date():
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 1, 2, 3, 4, 5],
            "game_date": pd.to_datetime(
                ["2024-05-01", "2024-05-01", "2024-05-08", "2024-05-15",
                 "2024-05-22", "2024-05-29"]
            ),
            "game_num": [1] * 6,
            "estimated_woba_using_speedangle": [0.3, 0.4, 0.2, 0.5, 0.1, 0.6],
        }
    )
    out = pitcher_last_n_starts_xwoba(pitches, n_starts=3)
    # Last 3 starts are game_pks 3, 4, 5 with xwOBAs 0.5, 0.1, 0.6 → mean = 0.4
    assert math.isclose(out["mean_xwoba"], 0.4)
    assert out["n_starts"] == 3


def test_pitcher_last_n_starts_handles_fewer_starts_than_n():
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 1],
            "game_date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
            "game_num": [1, 1],
            "estimated_woba_using_speedangle": [0.3, 0.4],
        }
    )
    out = pitcher_last_n_starts_xwoba(pitches, n_starts=3)
    assert out["n_starts"] == 1
    assert math.isclose(out["mean_xwoba"], 0.35)


def test_pitcher_last_n_starts_empty_input():
    out = pitcher_last_n_starts_xwoba(pd.DataFrame(
        columns=["game_pk", "game_date", "game_num", "estimated_woba_using_speedangle"]
    ))
    assert out["n_starts"] == 0
    assert math.isnan(out["mean_xwoba"])


# ============================================================
# batter_last_n_days_woba
# ============================================================


def test_batter_last_n_days_woba_filters_by_window():
    pas = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2024-04-30", "2024-05-10", "2024-05-13"]
            ),
            "woba_value": [0.5, 0.3, 0.7],
        }
    )
    # asof = 2024-05-15, 14-day window cuts at 2024-05-01
    out = batter_last_n_days_woba(pas, "2024-05-15", days=14)
    # Only the 2024-05-10 and 2024-05-13 PAs are in window → mean (0.3, 0.7) = 0.5
    assert out["n_pas"] == 2
    assert math.isclose(out["mean_woba"], 0.5)


def test_batter_last_n_days_woba_uses_xwoba_fallback_column():
    pas = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-05-13"]),
            "estimated_woba_using_speedangle": [0.42],
        }
    )
    out = batter_last_n_days_woba(pas, "2024-05-15")
    assert math.isclose(out["mean_woba"], 0.42)


def test_batter_last_n_days_woba_raises_when_no_woba_column():
    pas = pd.DataFrame({"game_date": pd.to_datetime(["2024-05-13"])})
    with pytest.raises(KeyError):
        batter_last_n_days_woba(pas, "2024-05-15")


# ============================================================
# pitcher_velo_stats and league fallbacks
# ============================================================


def test_pitcher_velo_stats_per_type():
    pitches = pd.DataFrame(
        {
            "pitch_type_canonical": ["FF", "FF", "FF", "SL"],
            "release_speed": [95.0, 96.0, 94.0, 84.0],
        }
    )
    out = pitcher_velo_stats(pitches, window_pitches=10)
    ff = out[out["pitch_type_canonical"] == "FF"].iloc[0]
    assert math.isclose(float(ff["mean"]), 95.0)
    assert int(ff["n"]) == 3


def test_compute_league_velo_means_aggregates_across_pitchers():
    pitches = pd.DataFrame(
        {
            "pitcher": [1, 1, 2, 2, 3],
            "pitch_type_canonical": ["FF", "FF", "FF", "FF", "SL"],
            "release_speed": [95, 96, 90, 91, 85],
        }
    )
    league = compute_league_velo_means(pitches)
    ff = league[league["pitch_type_canonical"] == "FF"].iloc[0]
    assert math.isclose(float(ff["mean"]), 93.0)
    assert int(ff["n"]) == 4


# ============================================================
# pitcher_zone_heatmap_by_type
# ============================================================


def test_zone_heatmap_returns_in_zone_vector_per_type_summing_to_one():
    # v5 SIS 14-zone: in-zone cells are indices 0..8 (3x3 grid).
    # 4 fastballs in cells {0, 1, 2, 5}, 2 sliders in cell {7, 7}.
    pitches = pd.DataFrame(
        {
            "pitch_type_canonical": ["FF", "FF", "FF", "FF", "SL", "SL"],
            "feature_zone": [0, 1, 2, 5, 7, 7],
        }
    )
    out = pitcher_zone_heatmap_by_type(pitches)
    assert set(out.keys()) == {"FF", "SL"}
    assert out["FF"].shape == (N_IN_ZONE_CELLS,)
    assert math.isclose(out["FF"].sum(), 1.0)
    # 4 FF distributed across 4 cells → each gets 0.25
    assert math.isclose(out["FF"][0], 0.25)
    assert math.isclose(out["FF"][5], 0.25)
    # All SL in cell 7
    assert math.isclose(out["SL"][7], 1.0)


def test_zone_heatmap_ignores_out_of_zone_cell():
    pitches = pd.DataFrame(
        {
            "pitch_type_canonical": ["FF", "FF", "FF"],
            "feature_zone": [0, 9, 8],  # cell 9 = upper-left OOZ (excluded), 0 and 8 are in-zone
        }
    )
    out = pitcher_zone_heatmap_by_type(pitches)
    # Only the in-zone pitches contribute (cells 0 and 8)
    assert math.isclose(out["FF"][0], 0.5)
    assert math.isclose(out["FF"][8], 0.5)
    assert math.isclose(out["FF"].sum(), 1.0)


def test_zone_heatmap_empty_input():
    out = pitcher_zone_heatmap_by_type(
        pd.DataFrame(columns=["pitch_type_canonical", "feature_zone"])
    )
    assert out == {}


# ============================================================
# pitcher_count_conditional_entropy
# ============================================================


def test_count_entropy_zero_for_deterministic_count():
    # In count 0-2, this pitcher always throws SL.
    pitches = pd.DataFrame(
        {
            "balls": [0, 0, 0],
            "strikes": [2, 2, 2],
            "pitch_type_canonical": ["SL", "SL", "SL"],
        }
    )
    out = pitcher_count_conditional_entropy(pitches)
    assert math.isclose(out[(0, 2)], 0.0)


def test_count_entropy_log2_for_50_50_split():
    # In count 1-1, this pitcher splits 50/50 → entropy = ln(2) ≈ 0.693 nats
    pitches = pd.DataFrame(
        {
            "balls": [1, 1, 1, 1],
            "strikes": [1, 1, 1, 1],
            "pitch_type_canonical": ["FF", "FF", "SL", "SL"],
        }
    )
    out = pitcher_count_conditional_entropy(pitches)
    assert math.isclose(out[(1, 1)], math.log(2), abs_tol=1e-6)


# ============================================================
# batter_zone_grids
# ============================================================


def test_zone_grids_swing_pct():
    # In cell 5: 3 pitches, 2 swings → swing rate = 2/3
    pitches = pd.DataFrame(
        {
            "feature_zone": [5, 5, 5],
            "description": ["ball", "swinging_strike", "hit_into_play"],
        }
    )
    grids = batter_zone_grids(pitches)
    assert math.isclose(grids["swing"][5], 2 / 3)


def test_zone_grids_whiff_pct():
    # 3 swings (whiff, foul, hit_into_play), 1 is whiff → whiff% = 1/3
    # Use cell 6 (in-zone, middle-right under v5 SIS 3x3).
    pitches = pd.DataFrame(
        {
            "feature_zone": [6, 6, 6, 6],
            "description": ["ball", "swinging_strike", "foul", "hit_into_play"],
        }
    )
    grids = batter_zone_grids(pitches)
    # swing% = 3/4 (3 swings out of 4 pitches)
    assert math.isclose(grids["swing"][6], 3 / 4)
    # whiff% = 1/3 (1 swinging_strike out of 3 swings)
    assert math.isclose(grids["whiff"][6], 1 / 3)


def test_zone_grids_xba_uses_contact_only():
    # Cell 4 is the middle of the in-zone 3x3 under v5 SIS scheme.
    pitches = pd.DataFrame(
        {
            "feature_zone": [4, 4, 4],
            "description": ["hit_into_play", "hit_into_play", "swinging_strike"],
            "estimated_ba_using_speedangle": [0.4, 0.6, None],
        }
    )
    grids = batter_zone_grids(pitches)
    # xBA averages over the two contact pitches → 0.5 (whiff is excluded)
    assert math.isclose(grids["xba"][4], 0.5)


def test_zone_grids_empty_input_returns_nans():
    grids = batter_zone_grids(
        pd.DataFrame(columns=["feature_zone", "description",
                              "estimated_ba_using_speedangle"])
    )
    assert grids["swing"].shape == (N_IN_ZONE_CELLS,)
    assert all(np.isnan(grids["swing"]))


# ============================================================
# batter_chase_rate_by_type
# ============================================================


def test_chase_rate_only_counts_out_of_zone_pitches():
    # v5 SIS scheme: in-zone = 0..8, OOZ quadrants = 9..12.
    pitches = pd.DataFrame(
        {
            "pitch_type_canonical": ["SL", "SL", "SL", "FF"],
            "feature_zone": [9, 12, 5, 11],  # 3 OOZ quadrants, 1 in-zone (cell 5)
            "description": ["swinging_strike", "ball", "ball", "ball"],
        }
    )
    out = batter_chase_rate_by_type(pitches)
    # SL: 2 OOZ pitches, 1 swing → 50% chase
    assert math.isclose(out["SL"], 0.5)
    # FF: 1 OOZ pitch, 0 swings → 0%
    assert math.isclose(out["FF"], 0.0)


# ============================================================
# batter_outcome_rates
# ============================================================


def test_outcome_rates_k_and_bb_pcts():
    pas = pd.DataFrame(
        {"events": ["strikeout", "single", "walk", "field_out",
                    "strikeout_double_play", "intent_walk"]}
    )
    out = batter_outcome_rates(pas)
    # 2 K events out of 6 → 2/6
    assert math.isclose(out["k_pct"], 2 / 6)
    # 2 BB events (walk + intent_walk) out of 6 → 2/6
    assert math.isclose(out["bb_pct"], 2 / 6)
    assert out["n_pas"] == 6


def test_outcome_rates_hard_contact_at_95():
    pas = pd.DataFrame(
        {
            "events": ["single"] * 4,
            "launch_speed": [94.0, 96.0, 100.0, None],
        }
    )
    out = batter_outcome_rates(pas)
    # 2 batted balls ≥95 mph out of 3 with launch_speed → 2/3
    assert math.isclose(out["hard_contact_pct"], 2 / 3)


def test_outcome_rates_empty_returns_nans():
    out = batter_outcome_rates(pd.DataFrame(columns=["events"]))
    assert out["n_pas"] == 0
    assert math.isnan(out["k_pct"])


# ============================================================
# leakage discipline still holds across the new features
# ============================================================


def test_zone_heatmap_respects_before_asof_filter():
    """Run the leakage primitive THEN the new feature; marker pitches must
    not appear in the heatmap.
    """
    df = pd.DataFrame(
        {
            "pitcher": [1] * 4,
            "game_pk": [99, 100, 100, 101],
            "game_date": pd.to_datetime(
                ["2024-05-14", "2024-05-15", "2024-05-15", "2024-05-16"]
            ),
            "game_num": [1, 1, 1, 1],
            "pitch_type_canonical": ["FF", "MARKER", "MARKER", "FF"],
            "feature_zone": [0, 4, 4, 0],  # v5: cell 4 is middle in-zone
        }
    )
    earlier = before_asof(df, "2024-05-15", asof_game_num=1)
    out = pitcher_zone_heatmap_by_type(earlier)
    # MARKER pitches were on game 100 (current AB's game) and game 101 (future) —
    # neither should appear in the heatmap.
    assert "MARKER" not in out
    assert "FF" in out  # The 99 game's FF pitch should be there


# ============================================================
# Sanity: SWING_DESCRIPTIONS and WHIFF_DESCRIPTIONS are consistent
# ============================================================


def test_whiff_descriptions_are_subset_of_swing_descriptions():
    assert WHIFF_DESCRIPTIONS.issubset(SWING_DESCRIPTIONS)


# ============================================================
# pitcher_last_n_days_xwoba (the role-uniform recent-form feature)
# ============================================================


def test_last_n_days_xwoba_filters_by_calendar_window():
    pitches = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2024-04-01", "2024-04-15", "2024-05-10", "2024-05-13"]
            ),
            "estimated_woba_using_speedangle": [0.4, 0.5, 0.3, 0.6],
        }
    )
    out = pitcher_last_n_days_xwoba(pitches, "2024-05-15", days=30)
    # 30-day cutoff: 2024-04-15 onward.
    # Three pitches in window (4-15, 5-10, 5-13); xwOBAs (0.5, 0.3, 0.6) → mean = 0.4666...
    assert out["n_pitches"] == 3
    assert math.isclose(out["mean_xwoba"], (0.5 + 0.3 + 0.6) / 3, abs_tol=1e-9)


def test_last_n_days_xwoba_returns_zero_for_empty_window():
    """Injury return / start-of-season case: the 30-day window is empty."""
    pitches = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-04-01"]),  # 75 days before asof
            "estimated_woba_using_speedangle": [0.3],
        }
    )
    out = pitcher_last_n_days_xwoba(pitches, "2024-06-15", days=30)
    assert out["n_pitches"] == 0
    assert math.isnan(out["mean_xwoba"])


def test_last_n_days_xwoba_empty_input():
    out = pitcher_last_n_days_xwoba(
        pd.DataFrame(columns=["game_date", "estimated_woba_using_speedangle"]),
        "2024-05-15",
    )
    assert out["n_pitches"] == 0
    assert math.isnan(out["mean_xwoba"])


# ============================================================
# pitcher_days_since_last_appearance
# ============================================================


def test_days_since_last_appearance_counts_from_most_recent():
    pitches = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2024-03-01", "2024-04-15", "2024-05-10"]
            ),
        }
    )
    days = pitcher_days_since_last_appearance(pitches, "2024-05-15")
    # Most recent prior pitch was 2024-05-10; gap = 5 days
    assert days == 5


def test_days_since_last_appearance_long_layoff():
    """75-day injury layoff — model should be able to see this."""
    pitches = pd.DataFrame(
        {"game_date": pd.to_datetime(["2024-04-01"])}
    )
    days = pitcher_days_since_last_appearance(pitches, "2024-06-15")
    assert days == 75


def test_days_since_last_appearance_returns_none_for_debut():
    """No prior appearances at all (MLB debut) → None."""
    days = pitcher_days_since_last_appearance(
        pd.DataFrame(columns=["game_date"]),
        "2024-04-01",
    )
    assert days is None


# ============================================================
# window_freshness — cross-season staleness signals
# ============================================================


def test_window_freshness_fresh_in_season_window():
    """Mid-season profile: oldest pitch ~50 days back, all current-season."""
    pitches = pd.DataFrame(
        {"game_date": pd.to_datetime(["2024-05-01"] * 100 + ["2024-06-15"] * 100)}
    )
    out = window_freshness(pitches, "2024-07-01")
    assert out["span_days"] == 61  # 2024-05-01 to 2024-07-01
    assert math.isclose(out["pct_current_season"], 1.0)
    assert out["n_pitches_in_window"] == 200


def test_window_freshness_cross_season_april_one():
    """April-1 case: 800 pitches from previous October, 200 from current April."""
    pitches = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2023-10-15"] * 800 + ["2024-04-01"] * 200
            )
        }
    )
    out = window_freshness(pitches, "2024-04-15")
    # Span from 2023-10-15 to 2024-04-15 ≈ 183 days
    assert out["span_days"] >= 180
    # Current-season fraction: 200/1000 = 0.2
    assert math.isclose(out["pct_current_season"], 0.2)


def test_window_freshness_empty_returns_nans():
    out = window_freshness(pd.DataFrame(columns=["game_date"]), "2024-04-01")
    assert math.isnan(out["span_days"])
    assert math.isnan(out["pct_current_season"])
    assert out["n_pitches_in_window"] == 0


def test_window_freshness_respects_window_cap():
    """The window cap (default 1000) limits how many pitches contribute."""
    pitches = pd.DataFrame(
        {"game_date": pd.to_datetime(["2024-05-01"] * 1500)}
    )
    out = window_freshness(pitches, "2024-06-01", window_pitches=1000)
    # tail(1000) → still 2024-05-01 dates only
    assert out["n_pitches_in_window"] == 1000
    assert math.isclose(out["pct_current_season"], 1.0)
