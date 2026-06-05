"""TDD tests for the location MDN target in PitchGPTAtBatDataset.

Verifies that:
- item["targets"]["location"] exists, shape (T, 2), dtype float32
- last position is NaN (left-shift: no successor for the final pitch)
- all non-terminal positions are finite (real Statcast plate_x/plate_z)
- collate pads location targets with NaN and produces shape (B, max_T, 2)

Tests use a synthetic fixture with real-looking column structure — no live
augmented parquets required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from data.profile_cache import (
    BATTER_VECTOR_LEN,
    PITCHER_FEATURE_INDEX,
    PITCHER_VECTOR_LEN,
    PROFILE_SCHEMA_VERSION,
)
from data.profile_cache_loader import ProfileCache
from model.pitchgpt_dataset import PitchGPTAtBatDataset, collate_pitchgpt_at_bats


# ============================================================
# Synthetic fixture helpers
# ============================================================


def _make_augmented_pitches(
    *,
    ab_id: int,
    n_pitches: int,
    pitcher_id: int,
    batter_id: int,
    game_pk: int = 100,
    game_date: str = "2023-04-01",
    plate_x_vals: list[float] | None = None,
    plate_z_vals: list[float] | None = None,
) -> pd.DataFrame:
    """Build a per-pitch DataFrame that satisfies REQUIRED_AUG_COLS + plate_x/plate_z."""
    rng = np.random.default_rng(42)
    n = n_pitches
    px = plate_x_vals if plate_x_vals is not None else rng.uniform(-1.5, 1.5, n).tolist()
    pz = plate_z_vals if plate_z_vals is not None else rng.uniform(1.0, 4.0, n).tolist()
    return pd.DataFrame({
        "game_pk": [game_pk] * n,
        "at_bat_number": [ab_id] * n,
        "pitch_number": list(range(1, n + 1)),
        "game_date": pd.to_datetime([game_date] * n),
        "game_num": [1] * n,
        "pitcher": [pitcher_id] * n,
        "batter": [batter_id] * n,
        # Pitch-type integer factors
        "type_id": [1, 4, 2, 3, 5][:n],          # FF, SL, SI, CH, CU
        "feature_zone": [5, 7, 5, 11, 8][:n],
        "velo_bin": [5, 4, 5, 3, 2][:n],
        "spin_rate_bin": [3, 2, 3, 1, 4][:n],
        "result_id": [1, 2, 1, 3, 5][:n],
        "count_state": [0, 1, 1, 2, 5][:n],
        "runners_state": [0] * n,
        "outs_state": [1] * n,
        "pos": list(range(n)),
        "pitcher_fatigue_bucket": [2] * n,
        # Spin axis
        "spin_axis_sin": rng.uniform(-1, 1, n).tolist(),
        "spin_axis_cos": rng.uniform(-1, 1, n).tolist(),
        # Categorical context (constant within AB)
        "p_throws_id": [1] * n,
        "stand_id": [2] * n,
        "ballpark_id": [3] * n,
        "umpire_id": [10] * n,
        "catcher_id": [20] * n,
        "inning_bucket": [1] * n,
        "score_diff_bucket": [5] * n,
        "inning_half": [0] * n,
        "days_rest_bucket": [4] * n,
        "tto_bucket": [1] * n,
        "temp_bucket": [3] * n,
        "roof_state": [1] * n,
        # Text cols
        "description": ["ball", "called_strike", "foul", "swinging_strike", "ball"][:n],
        "events": [None, None, None, None, "strikeout"][:n],
        # Location target cols
        "plate_x": px,
        "plate_z": pz,
    })


def _write_minimal_profile_cache(tmp_path, pitcher_id: int, batter_id: int,
                                  game_date: str = "2023-04-01"):
    """Write per-player + league profile caches for one pitcher + one batter."""
    pitcher_vec = np.ones(PITCHER_VECTOR_LEN, dtype=np.float32)
    # confidence = 1.0 so the blended result is pure per-player
    pitcher_vec[PITCHER_FEATURE_INDEX["profile_confidence"]] = 1.0
    batter_vec = np.full(BATTER_VECTOR_LEN, 0.5, dtype=np.float32)

    for role, pid, vec_len, vec in [
        ("pitcher", pitcher_id, PITCHER_VECTOR_LEN, pitcher_vec),
        ("batter", batter_id, BATTER_VECTOR_LEN, batter_vec),
    ]:
        player_df = pd.DataFrame([{
            "player_id": int(pid),
            "asof_date": pd.Timestamp(game_date),
            "asof_game_num": 1,
            "fold_id": 0,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "vector": list(vec),
        }])
        player_df.to_parquet(tmp_path / f"{role}_fold_0.parquet", index=False)

        league_vec = np.zeros(vec_len, dtype=np.float32)
        league_df = pd.DataFrame([{
            "asof_date": pd.Timestamp(game_date),
            "asof_game_num": 1,
            "fold_id": 0,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "vector": list(league_vec),
            "n_players_in_mean": 99,
        }])
        league_df.to_parquet(tmp_path / f"league_{role}_fold_0.parquet", index=False)


def _build_dataset(tmp_path, pitches: pd.DataFrame, pitcher_id: int, batter_id: int,
                   game_date: str = "2023-04-01") -> PitchGPTAtBatDataset:
    _write_minimal_profile_cache(tmp_path, pitcher_id, batter_id, game_date)
    pc_p = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    pc_b = ProfileCache(role="batter",  fold_id=0, profiles_dir=tmp_path)
    return PitchGPTAtBatDataset(
        pitches=pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
    )


# ============================================================
# Tests
# ============================================================


def test_dataset_emits_location_target(tmp_path):
    """Location target (plate_x, plate_z) is present, left-shifted, NaN-padded."""
    pitcher_id, batter_id = 555, 777
    n_pitches = 4
    px_vals = [0.867, 0.083, 0.102, -0.467]
    pz_vals = [1.032, 3.823, 1.201, 1.135]

    pitches = _make_augmented_pitches(
        ab_id=1, n_pitches=n_pitches,
        pitcher_id=pitcher_id, batter_id=batter_id,
        plate_x_vals=px_vals, plate_z_vals=pz_vals,
    )
    ds = _build_dataset(tmp_path, pitches, pitcher_id, batter_id)
    item = ds[0]

    assert "location" in item["targets"], "'location' key missing from targets"

    loc = item["targets"]["location"]
    T = len(item["padding_mask"])

    assert loc.shape == (T, 2), f"expected ({T}, 2), got {loc.shape}"
    assert loc.dtype == torch.float32, f"expected float32, got {loc.dtype}"

    # Last position must be NaN — no successor pitch to predict
    assert torch.isnan(loc[-1]).all(), (
        f"last position should be NaN (no successor), got {loc[-1]}"
    )

    # All non-terminal positions must be finite real coordinates
    assert torch.isfinite(loc[:-1]).all(), (
        f"non-terminal positions should be finite, got {loc[:-1]}"
    )

    # Verify left-shift: position 0 should hold pitch 1's coordinates
    expected_px_at_0 = px_vals[1]
    expected_pz_at_0 = pz_vals[1]
    assert abs(loc[0, 0].item() - expected_px_at_0) < 1e-4, (
        f"plate_x at position 0 should be pitch[1]'s plate_x={expected_px_at_0:.4f}, "
        f"got {loc[0, 0].item():.4f}"
    )
    assert abs(loc[0, 1].item() - expected_pz_at_0) < 1e-4, (
        f"plate_z at position 0 should be pitch[1]'s plate_z={expected_pz_at_0:.4f}, "
        f"got {loc[0, 1].item():.4f}"
    )

    # Named numerical output (per CLAUDE.md discipline)
    print(f"location target[0] = ({loc[0, 0].item():.3f}, {loc[0, 1].item():.3f})")
    print(f"location target[-1] = ({loc[-1, 0].item()}, {loc[-1, 1].item()})  [NaN expected]")


def test_collate_pads_location_with_nan(tmp_path):
    """Collated location targets are NaN-padded to max_T."""
    pitcher_id, batter_id = 555, 777
    game_date = "2023-04-01"

    # Two ABs of different lengths: 2 and 4 pitches
    ab1 = _make_augmented_pitches(
        ab_id=1, n_pitches=2,
        pitcher_id=pitcher_id, batter_id=batter_id,
        game_pk=100, game_date=game_date,
    )
    ab2 = _make_augmented_pitches(
        ab_id=2, n_pitches=4,
        pitcher_id=pitcher_id, batter_id=batter_id,
        game_pk=100, game_date=game_date,
    )
    pitches = pd.concat([ab1, ab2], ignore_index=True)
    ds = _build_dataset(tmp_path, pitches, pitcher_id, batter_id, game_date)

    assert len(ds) == 2, f"expected 2 ATs, got {len(ds)}"

    items = [ds[0], ds[1]]
    batch = collate_pitchgpt_at_bats(items)

    loc = batch["targets"]["location"]
    assert loc.ndim == 3, f"expected 3D (B, max_T, 2), got {loc.ndim}D"
    assert loc.shape[0] == 2, f"expected B=2, got {loc.shape[0]}"
    assert loc.shape[2] == 2, f"expected last dim=2, got {loc.shape[2]}"

    # max_T should be 4 (the longer AB)
    assert loc.shape[1] == 4, f"expected max_T=4, got {loc.shape[1]}"

    # Padded positions (beyond real length) must be NaN
    mask = batch["padding_mask"]  # (B, max_T) bool
    for i in range(2):
        n_real = int(mask[i].sum().item())
        if n_real < loc.shape[1]:
            padded = loc[i, n_real:]
            assert torch.isnan(padded).all(), (
                f"item {i}: padded positions (after {n_real}) should be NaN, "
                f"got {padded}"
            )

    # Named numerical output
    print(f"collated location shape: {tuple(loc.shape)}")
    print(f"loc[0, 0] = ({loc[0, 0, 0].item():.3f}, {loc[0, 0, 1].item():.3f})")
    print(f"loc[0, 2] = ({loc[0, 2, 0].item()}, {loc[0, 2, 1].item()})  [NaN for short AB]")
