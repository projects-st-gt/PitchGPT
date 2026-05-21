"""Unit tests for pitch type harmonization."""

from __future__ import annotations

import pandas as pd

from data.harmonization import (
    CANONICAL_TYPES,
    PITCH_TYPE_MAP,
    harmonize_dataframe,
    harmonize_pitch_type,
)


def test_canonical_types_are_consistent_with_map():
    assert set(CANONICAL_TYPES) == set(PITCH_TYPE_MAP.values())


def test_known_pitches_map_to_canonical():
    assert harmonize_pitch_type("FF") == "FF"
    assert harmonize_pitch_type("FA") == "FF"
    assert harmonize_pitch_type("ST") == "SL"  # sweeper
    assert harmonize_pitch_type("KN") == "CU"  # knuckle-curve grouped with CU


def test_unknown_pitch_returns_none():
    assert harmonize_pitch_type("EP") is None
    assert harmonize_pitch_type("IN") is None
    assert harmonize_pitch_type(None) is None


def test_drop_unmapped_drops_entire_at_bat():
    df = pd.DataFrame(
        {
            "game_pk": [1, 1, 1, 1, 2, 2],
            "at_bat_number": [1, 1, 2, 2, 1, 1],
            "pitch_type": ["FF", "SL", "EP", "FF", "FF", "SI"],
        }
    )
    out = harmonize_dataframe(df, drop_unmapped=True)
    surviving = set(zip(out["game_pk"], out["at_bat_number"]))
    # AB (1, 2) had an EP — entire AB dropped, including its FF.
    assert surviving == {(1, 1), (2, 1)}


def test_drop_unmapped_disabled_keeps_rows():
    df = pd.DataFrame(
        {
            "game_pk": [1, 1],
            "at_bat_number": [1, 1],
            "pitch_type": ["FF", "EP"],
        }
    )
    out = harmonize_dataframe(df, drop_unmapped=False)
    assert len(out) == 2
    assert pd.isna(out["pitch_type_canonical"].iloc[1])
