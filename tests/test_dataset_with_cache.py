"""End-to-end integration test: ProfileCache wired into AtBatDataset.

Builds a tiny synthetic corpus (a couple of at-bats), writes a per-player
+ league-mean cache for both roles in a tmp directory, instantiates
ProfileCache for each, and uses ``.lookup`` as the dataset's profile
callable. Then verifies that ``__getitem__`` and ``collate_at_bats``
produce the right tensor shapes — which means the cache + dataset
contract holds end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.dataset import AtBatDataset, collate_at_bats
from data.profile_cache import (
    BATTER_VECTOR_LEN,
    PITCHER_FEATURE_INDEX,
    PITCHER_VECTOR_LEN,
    PROFILE_SCHEMA_VERSION,
)
from data.profile_cache_loader import ProfileCache


def _write_per_role_cache(
    tmp_path,
    role,
    fold_id,
    *,
    player_rows: list[tuple],
    league_rows: list[tuple],
):
    """Write a per-(role, fold) per-player + league parquet pair to tmp."""
    vec_len = PITCHER_VECTOR_LEN if role == "pitcher" else BATTER_VECTOR_LEN

    pdf = pd.DataFrame([
        {
            "player_id": int(pid),
            "asof_date": pd.Timestamp(asof),
            "asof_game_num": int(num),
            "fold_id": int(fold_id),
            "schema_version": PROFILE_SCHEMA_VERSION,
            "vector": list(vec),
        }
        for pid, asof, num, vec in player_rows
    ])
    pdf.to_parquet(tmp_path / f"{role}_fold_{fold_id}.parquet", index=False)

    ldf = pd.DataFrame([
        {
            "asof_date": pd.Timestamp(asof),
            "asof_game_num": int(num),
            "fold_id": int(fold_id),
            "schema_version": PROFILE_SCHEMA_VERSION,
            "vector": list(vec),
            "n_players_in_mean": 99,
        }
        for asof, num, vec in league_rows
    ])
    ldf.to_parquet(tmp_path / f"league_{role}_fold_{fold_id}.parquet", index=False)


def _make_ab_pitches(*, ab_id, n_pitches, pitcher_id, batter_id,
                     game_pk=100, game_date="2024-04-01"):
    """Build a per-pitch DataFrame for one AB matching AtBatDataset's columns."""
    return pd.DataFrame({
        "game_pk": [game_pk] * n_pitches,
        "at_bat_number": [ab_id] * n_pitches,
        "pitch_number": list(range(1, n_pitches + 1)),
        "game_date": pd.to_datetime([game_date] * n_pitches),
        "game_num": [1] * n_pitches,
        "pitcher": [pitcher_id] * n_pitches,
        "batter": [batter_id] * n_pitches,
        "p_throws": ["R"] * n_pitches,
        "stand": ["L"] * n_pitches,
        "pitch_type_canonical": ["FF", "SL", "FF", "CH"][:n_pitches],
        "feature_zone": [5, 7, 5, 11][:n_pitches],  # v5 SIS: in-zone 0-8, OOZ 9-12
        "action_zone": pd.Categorical(
            ["arm-side", "down", "arm-side", "out-of-zone"][:n_pitches],
            categories=["up", "down", "arm-side", "glove-side", "out-of-zone"],
        ),
        "description": ["ball", "called_strike", "foul", "swinging_strike"][:n_pitches],
        "events": [None, None, None, "strikeout"][:n_pitches],
        "balls": [0, 1, 1, 1][:n_pitches],
        "strikes": [0, 0, 1, 2][:n_pitches],
        "outs_when_up": [1] * n_pitches,
        "on_1b": [None] * n_pitches,
        "on_2b": [12345] * n_pitches,
        "on_3b": [None] * n_pitches,
    })


# ============================================================
# Per-AB lookup goes through the cache
# ============================================================


def test_dataset_pulls_profile_via_profile_cache(tmp_path):
    pitcher_id, batter_id = 555, 777
    asof = "2024-04-01"

    # Per-player cache vectors: pitcher gets value 1.0 in every slot,
    # batter gets 0.5. Confidence = 1 so the blended result equals per-player.
    pitcher_vec = np.full(PITCHER_VECTOR_LEN, 1.0, dtype=np.float32)
    pitcher_vec[PITCHER_FEATURE_INDEX["profile_confidence"]] = 1.0
    batter_vec = np.full(BATTER_VECTOR_LEN, 0.5, dtype=np.float32)

    _write_per_role_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(pitcher_id, asof, 1, pitcher_vec)],
        league_rows=[(asof, 1, np.full(PITCHER_VECTOR_LEN, 0.0, dtype=np.float32))],
    )
    _write_per_role_cache(
        tmp_path, "batter", 0,
        player_rows=[(batter_id, asof, 1, batter_vec)],
        league_rows=[(asof, 1, np.full(BATTER_VECTOR_LEN, 0.0, dtype=np.float32))],
    )

    pc_pitcher = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    pc_batter = ProfileCache(role="batter",  fold_id=0, profiles_dir=tmp_path)

    pitches = _make_ab_pitches(
        ab_id=1, n_pitches=4,
        pitcher_id=pitcher_id, batter_id=batter_id,
        game_date=asof,
    )
    ds = AtBatDataset(
        pitches=pitches,
        pitcher_profile_lookup=pc_pitcher.lookup,
        batter_profile_lookup=pc_batter.lookup,
        velo_bin_col=None,
    )

    item = ds[0]
    # Pitcher vec is 1.0s with full confidence → blended = pure per-player
    np.testing.assert_allclose(item["pitcher_profile"].numpy(), pitcher_vec, atol=1e-6)
    # Batter vec is 0.5s with no confidence slot at index 0 → confidence=0 here,
    # so blend pulls all from league (zeros). Just check it's NaN-free.
    assert not np.any(np.isnan(item["batter_profile"].numpy()))


def test_dataset_with_cache_collates_into_batch(tmp_path):
    """Two ABs, different lengths; collate produces a padded batch."""
    pitcher_id, batter_id_1, batter_id_2 = 555, 777, 888
    asof = "2024-04-01"

    _write_per_role_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(pitcher_id, asof, 1, np.full(PITCHER_VECTOR_LEN, 1.0, dtype=np.float32))],
        league_rows=[(asof, 1, np.zeros(PITCHER_VECTOR_LEN, dtype=np.float32))],
    )
    _write_per_role_cache(
        tmp_path, "batter", 0,
        player_rows=[
            (batter_id_1, asof, 1, np.full(BATTER_VECTOR_LEN, 0.5, dtype=np.float32)),
            (batter_id_2, asof, 1, np.full(BATTER_VECTOR_LEN, 0.7, dtype=np.float32)),
        ],
        league_rows=[(asof, 1, np.zeros(BATTER_VECTOR_LEN, dtype=np.float32))],
    )

    pc_p = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    pc_b = ProfileCache(role="batter",  fold_id=0, profiles_dir=tmp_path)

    ab1 = _make_ab_pitches(ab_id=1, n_pitches=2, pitcher_id=pitcher_id,
                           batter_id=batter_id_1, game_pk=100, game_date=asof)
    ab2 = _make_ab_pitches(ab_id=2, n_pitches=4, pitcher_id=pitcher_id,
                           batter_id=batter_id_2, game_pk=100, game_date=asof)
    pitches = pd.concat([ab1, ab2], ignore_index=True)

    ds = AtBatDataset(
        pitches=pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        velo_bin_col=None,
    )
    assert len(ds) == 2

    batch = collate_at_bats([ds[0], ds[1]])
    # Two ABs, max length 4, padding mask True for real positions
    assert batch["padding_mask"].shape == (2, 4)
    assert batch["padding_mask"][0].tolist() == [True, True, False, False]
    assert batch["padding_mask"][1].tolist() == [True, True, True, True]
    assert batch["pitcher_profile"].shape == (2, PITCHER_VECTOR_LEN)
    assert batch["batter_profile"].shape == (2, BATTER_VECTOR_LEN)


def test_dataset_falls_back_to_league_for_unknown_player(tmp_path):
    """Unknown pitcher (not in per-player cache) → dataset still produces a vec."""
    asof = "2024-04-01"
    league_pitcher = np.full(PITCHER_VECTOR_LEN, 0.3, dtype=np.float32)

    _write_per_role_cache(
        tmp_path, "pitcher", 0,
        player_rows=[],  # NO per-player entries!
        league_rows=[(asof, 1, league_pitcher)],
    )
    _write_per_role_cache(
        tmp_path, "batter", 0,
        player_rows=[],
        league_rows=[(asof, 1, np.zeros(BATTER_VECTOR_LEN, dtype=np.float32))],
    )
    pc_p = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    pc_b = ProfileCache(role="batter",  fold_id=0, profiles_dir=tmp_path)

    pitches = _make_ab_pitches(
        ab_id=1, n_pitches=2,
        pitcher_id=99999, batter_id=11111,  # both unknown
        game_date=asof,
    )
    ds = AtBatDataset(
        pitches=pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
        velo_bin_col=None,
    )
    item = ds[0]
    # Unknown pitcher → league_only path → returns the league vector
    np.testing.assert_allclose(item["pitcher_profile"].numpy(), league_pitcher,
                               atol=1e-6)
    # Unknown batter → league_only → all zeros (the league we wrote)
    np.testing.assert_allclose(item["batter_profile"].numpy(),
                               np.zeros(BATTER_VECTOR_LEN), atol=1e-6)
