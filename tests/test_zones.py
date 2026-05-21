"""Unit tests for action-zone and feature-zone tagging."""

from __future__ import annotations

import pandas as pd
import pytest

from data.zones import (
    ACTION_ZONES,
    FEATURE_ZONE_OUT_OF_ZONE,
    INTERNAL_TO_SIS,
    N_FEATURE_ZONES_14,
    N_IN_ZONE_CELLS_14,
    SIS_TO_INTERNAL,
    assign_action_zone,
    assign_feature_zone,
    assign_feature_zone_14,
    valid_zone_mask,
)

# A canonical strike-zone for tests: bot=1.5 ft, top=3.5 ft (height = 2.0 ft).
SZ_TOP = 3.5
SZ_BOT = 1.5


def _row(plate_x: float, plate_z: float, p_throws: str = "R") -> dict:
    return {
        "plate_x": plate_x,
        "plate_z": plate_z,
        "sz_top": SZ_TOP,
        "sz_bot": SZ_BOT,
        "p_throws": p_throws,
    }


def _df(rows):
    return pd.DataFrame(rows)


# ---------- action zone ----------


def test_action_zone_top_of_zone_is_up():
    # z_norm = (3.4 - 1.5) / 2.0 = 0.95 > 0.67 → up
    out = assign_action_zone(_df([_row(0.0, 3.4)]))
    assert list(out) == ["up"]


def test_action_zone_bottom_of_zone_is_down():
    # z_norm = (1.6 - 1.5) / 2.0 = 0.05 < 0.33 → down
    out = assign_action_zone(_df([_row(0.0, 1.6)]))
    assert list(out) == ["down"]


def test_action_zone_middle_arm_side_rhp():
    # z_norm = 0.5; x_norm = 0.5/0.83 ≈ 0.6 > 0; RHP arm_sign = +1 → arm-side
    out = assign_action_zone(_df([_row(0.5, 2.5, p_throws="R")]))
    assert list(out) == ["arm-side"]


def test_action_zone_middle_glove_side_rhp():
    # x = -0.5 ft; arm-side signed < 0 for RHP → glove-side
    out = assign_action_zone(_df([_row(-0.5, 2.5, p_throws="R")]))
    assert list(out) == ["glove-side"]


def test_action_zone_arm_side_flips_for_lhp():
    # Same physical x as RHP-arm-side, but LHP's arm-side is negative x.
    # plate_x = +0.5, LHP → arm_sign = -1 → arm-side signed < 0 → glove-side.
    out = assign_action_zone(_df([_row(0.5, 2.5, p_throws="L")]))
    assert list(out) == ["glove-side"]
    # And LHP arm-side at plate_x = -0.5
    out = assign_action_zone(_df([_row(-0.5, 2.5, p_throws="L")]))
    assert list(out) == ["arm-side"]


def test_action_zone_out_of_zone_high():
    # z_norm > 1 → out-of-zone
    out = assign_action_zone(_df([_row(0.0, 4.0)]))
    assert list(out) == ["out-of-zone"]


def test_action_zone_out_of_zone_wide():
    # plate_x = 1.2 ft → x_norm > 1 → out-of-zone
    out = assign_action_zone(_df([_row(1.2, 2.5)]))
    assert list(out) == ["out-of-zone"]


def test_action_zone_categorical_categories_match_canonical():
    out = assign_action_zone(_df([_row(0.0, 2.5)]))
    assert list(out.categories) == ACTION_ZONES


def test_action_zone_raises_on_missing_columns():
    with pytest.raises(KeyError, match="p_throws"):
        assign_action_zone(pd.DataFrame({"plate_x": [0], "plate_z": [2.5],
                                         "sz_top": [3.5], "sz_bot": [1.5]}))


# ---------- feature zone ----------


def test_feature_zone_in_zone_center():
    # z_norm = 0.5 → band 2; x_norm = 0 → x_clipped/3*5 = 1.5/3*5=2.5 → band 2
    # cell = 5*2 + 2 = 12
    out = assign_feature_zone(_df([_row(0.0, 2.5)]))
    assert list(out) == [12]


def test_feature_zone_top_right_corner():
    # z_norm ~ 1 → band 4; plate_x = 0.83 → x_clipped+1.5=2.33; /3*5=3.88 → band 3
    # cell = 5*4 + 3 = 23
    out = assign_feature_zone(_df([_row(0.83, 3.49)]))
    assert list(out) == [23]


def test_feature_zone_out_of_zone():
    out = assign_feature_zone(_df([_row(0.0, 5.0)]))
    assert list(out) == [FEATURE_ZONE_OUT_OF_ZONE]


def test_feature_zone_no_p_throws_required():
    # feature zone is batter-symmetric; should not require p_throws.
    df = pd.DataFrame({"plate_x": [0.0], "plate_z": [2.5],
                       "sz_top": [3.5], "sz_bot": [1.5]})
    out = assign_feature_zone(df)
    assert list(out) == [12]


# ---------- valid mask ----------


def test_valid_zone_mask_drops_short_zone():
    df = pd.DataFrame({"sz_top": [3.0, 2.4], "sz_bot": [1.0, 1.5]})
    # heights 2.0, 0.9 — second is below threshold
    mask = valid_zone_mask(df)
    assert list(mask) == [True, False]


# ---------- v2 feature zone (SIS 14-zone via Statcast native zone column) ----------


def test_assign_feature_zone_14_maps_in_zone_sis_labels_to_internal_0_to_8():
    # SIS labels 1..9 → internal indices 0..8 (top-left=0, bottom-right=8).
    df = pd.DataFrame({"zone": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    out = assign_feature_zone_14(df)
    assert list(out) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_assign_feature_zone_14_maps_ooz_sis_labels_to_internal_9_to_12():
    # SIS labels 11..14 → internal indices 9..12 (UL, UR, LL, LR quadrants).
    df = pd.DataFrame({"zone": [11, 12, 13, 14]})
    out = assign_feature_zone_14(df)
    assert list(out) == [9, 10, 11, 12]


def test_assign_feature_zone_14_raises_on_nan():
    # Caller is expected to drop NaN-zone rows upstream (book-keeping artifacts:
    # automatic_ball / intent_walk / pitch-clock). The function refuses NaN.
    df = pd.DataFrame({"zone": [1.0, float("nan"), 5.0]})
    with pytest.raises(ValueError, match="NaN 'zone'"):
        assign_feature_zone_14(df)


def test_assign_feature_zone_14_raises_on_unexpected_label():
    # SIS labels are exactly {1..9, 11..14}; anything else (e.g., 10, 15, 0) is invalid.
    df = pd.DataFrame({"zone": [1, 10, 5]})  # zone 10 doesn't exist in SIS
    with pytest.raises(ValueError, match="unexpected Statcast zone values"):
        assign_feature_zone_14(df)


def test_assign_feature_zone_14_raises_without_zone_column():
    with pytest.raises(KeyError, match="zone"):
        assign_feature_zone_14(pd.DataFrame({"plate_x": [0.0], "plate_z": [2.5]}))


def test_sis_to_internal_is_a_bijection_with_13_entries():
    assert N_FEATURE_ZONES_14 == 13
    assert N_IN_ZONE_CELLS_14 == 9
    assert len(SIS_TO_INTERNAL) == 13
    assert set(SIS_TO_INTERNAL.values()) == set(range(13))
    # Round-trip via the inverse.
    for sis, internal in SIS_TO_INTERNAL.items():
        assert INTERNAL_TO_SIS[internal] == sis
