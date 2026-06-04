"""Tests for hitter.features — base/lag features (synthetic) + profile join (real)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hitter.features import (
    add_recent_pitch_lags, add_zone_flag, build_base_features, attach_batter_profile,
)


def _toy():
    # one AB, 3 pitches (game_pk 1, ab 1) — types 1,4,7; locations in/out of zone
    return pd.DataFrame({
        "game_pk": [1, 1, 1],
        "at_bat_number": [1, 1, 1],
        "pitch_number": [1, 2, 3],
        "type_id": [1, 4, 7],
        "plate_x": [0.0, 2.0, 0.1],     # in, out (wide), in
        "plate_z": [2.5, 2.5, 0.2],     # in, in, out (low)
        "release_speed": [95.0, 84.0, 88.0],
        "balls": [0, 1, 1], "strikes": [0, 0, 1],
        "stand": ["R", "R", "R"], "p_throws": ["R", "R", "R"],
        "game_date": ["2024-04-01"] * 3,
        "batter": [100, 100, 100], "pitcher": [200, 200, 200],
    })


def test_zone_flag():
    z = add_zone_flag(_toy())["in_zone"].tolist()
    assert z == [1, 0, 0]  # in, wide, low


def test_recent_pitch_lags_within_ab():
    out = add_recent_pitch_lags(add_zone_flag(_toy()))
    assert out["prev_type_id"].tolist() == [0, 1, 4]   # first=0, then prior types
    assert out["n_prev_pitches"].tolist() == [0, 1, 2]


def test_base_features_has_expected_cols():
    out = build_base_features(_toy())
    for c in ["in_zone", "prev_type_id", "prev_in_zone", "n_prev_pitches", "same_hand"]:
        assert c in out.columns
    assert out["same_hand"].tolist() == [1, 1, 1]  # R vs R


VAL = sorted(Path("data/augmented/2024").glob("2024-*.parquet"))
requires_data = pytest.mark.skipif(not VAL, reason="no augmented val data")


@requires_data
def test_attach_batter_profile_differs_across_batters():
    """The whole point: different batters must get DIFFERENT profile vectors
    (else the hitter model can't discriminate, same bug as the transformer)."""
    from data.profile_cache_loader import ProfileCache
    df = pd.read_parquet(VAL[0])
    bids = [int(x) for x in df["batter"].drop_duplicates().head(2)]
    sub = df[df["batter"].isin(bids)].copy()
    cache = ProfileCache(role="batter", fold_id=0)
    out = attach_batter_profile(build_base_features(sub), cache)
    bcols = [c for c in out.columns if c.startswith("b") and c[1:].isdigit()]
    assert len(bcols) > 50, "profile vector should add many columns"
    v0 = out[out["batter"] == bids[0]][bcols].iloc[0].to_numpy()
    v1 = out[out["batter"] == bids[1]][bcols].iloc[0].to_numpy()
    assert not np.allclose(v0, v1), "two different batters got identical profiles!"
    print(f"\nbatter {bids[0]} vs {bids[1]}: profile L2 diff = {np.linalg.norm(v0 - v1):.1f} ({len(bcols)} dims)")
