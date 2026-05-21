"""Unit tests for AtBatDataset, classify_result, and the collate function.

No real Statcast data — all synthetic. The tests check tensor shapes, target
shifting, padding behavior, and that the dataset asks for profiles using the
correct ``(player_id, asof_date, asof_game_num)`` keys.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import (
    ACTION_ZONE_TO_ID,
    N_FEATURE_ZONES,
    N_PITCH_TYPES,
    N_RESULTS,
    PAD_ID,
    PITCH_TYPE_TO_ID,
    RESULT_TO_ID,
    AtBatDataset,
    classify_result,
    collate_at_bats,
)

# ---------- classify_result ----------


@pytest.mark.parametrize(
    "description,events,expected",
    [
        ("ball", None, "ball"),
        ("intent_ball", None, "ball"),
        ("hit_by_pitch", "hit_by_pitch", "ball"),
        ("called_strike", None, "called_strike"),
        ("swinging_strike", None, "swinging_strike"),
        ("swinging_strike_blocked", None, "swinging_strike"),
        ("foul", None, "foul"),
        ("foul_tip", None, "foul"),
        ("hit_into_play", "field_out", "in_play_out"),
        ("hit_into_play", "force_out", "in_play_out"),
        ("hit_into_play", "single", "in_play_hit"),
        ("hit_into_play", "double", "in_play_hit"),
        ("hit_into_play", "triple", "in_play_hit"),
        ("hit_into_play", "home_run", "in_play_hr"),
    ],
)
def test_classify_result_known_cases(description, events, expected):
    assert classify_result(description, events) == RESULT_TO_ID[expected]


def test_classify_result_unknown_returns_none():
    assert classify_result("ejection", None) is None
    assert classify_result(None, None) is None


# ---------- AtBatDataset shape and content ----------


def _make_pitches(*, n_pitches: int = 4, ab_id: int = 1, game_pk: int = 100,
                  game_date: str = "2024-04-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [game_pk] * n_pitches,
            "at_bat_number": [ab_id] * n_pitches,
            "pitch_number": list(range(1, n_pitches + 1)),
            "game_date": pd.to_datetime([game_date] * n_pitches),
            "game_num": [1] * n_pitches,
            "pitcher": [555] * n_pitches,
            "batter": [777] * n_pitches,
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
            "on_1b": [None, None, None, None][:n_pitches],
            "on_2b": [12345, 12345, 12345, 12345][:n_pitches],
            "on_3b": [None, None, None, None][:n_pitches],
        }
    )


def _mock_lookup(d_profile: int = 8):
    """Return a profile callable that records call args and returns a fixed vector."""
    calls = []

    def _lookup(player_id, asof_date, asof_game_num):
        calls.append((int(player_id), pd.Timestamp(asof_date), int(asof_game_num)))
        return {"vector": np.full(d_profile, float(player_id) / 1000.0, dtype=np.float32)}

    return _lookup, calls


def test_dataset_returns_correct_keys_and_shapes():
    df = _make_pitches(n_pitches=4)
    p_lookup, _ = _mock_lookup(d_profile=8)
    b_lookup, _ = _mock_lookup(d_profile=12)
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)

    assert len(ds) == 1
    item = ds[0]

    assert set(item.keys()) == {
        "pitcher_profile", "batter_profile", "context_tokens",
        "pitch_factors", "result_factors", "target_factors", "padding_mask",
    }
    assert item["pitcher_profile"].shape == (8,)
    assert item["batter_profile"].shape == (12,)
    assert item["context_tokens"].shape == (6,)
    assert item["padding_mask"].shape == (4,)
    assert item["padding_mask"].all()
    for v in item["pitch_factors"].values():
        assert v.shape == (4,)


def test_dataset_tokens_match_vocabulary():
    df = _make_pitches(n_pitches=4)
    p_lookup, _ = _mock_lookup()
    b_lookup, _ = _mock_lookup()
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)
    item = ds[0]
    types = item["pitch_factors"]["type"].tolist()
    assert types == [PITCH_TYPE_TO_ID["FF"], PITCH_TYPE_TO_ID["SL"],
                     PITCH_TYPE_TO_ID["FF"], PITCH_TYPE_TO_ID["CH"]]
    actions = item["pitch_factors"]["action_zone"].tolist()
    assert actions == [
        ACTION_ZONE_TO_ID["arm-side"], ACTION_ZONE_TO_ID["down"],
        ACTION_ZONE_TO_ID["arm-side"], ACTION_ZONE_TO_ID["out-of-zone"],
    ]
    fz = item["pitch_factors"]["feature_zone"].tolist()
    assert fz == [5, 7, 5, 11]


def test_target_factors_are_input_factors_shifted_left_by_one():
    df = _make_pitches(n_pitches=4)
    p_lookup, _ = _mock_lookup()
    b_lookup, _ = _mock_lookup()
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)
    item = ds[0]
    inp_types = item["pitch_factors"]["type"].tolist()
    tgt_types = item["target_factors"]["type"].tolist()
    # Position t targets pitch at t+1; last position has PAD_ID (no successor).
    assert tgt_types[:-1] == inp_types[1:]
    assert tgt_types[-1] == PAD_ID


def test_dataset_calls_lookup_with_asof_keys():
    df = _make_pitches()
    p_lookup, p_calls = _mock_lookup()
    b_lookup, b_calls = _mock_lookup()
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)
    _ = ds[0]
    assert p_calls == [(555, pd.Timestamp("2024-04-01"), 1)]
    assert b_calls == [(777, pd.Timestamp("2024-04-01"), 1)]


def test_dataset_validates_required_columns():
    df = pd.DataFrame({"game_pk": [1], "at_bat_number": [1]})
    with pytest.raises(KeyError, match="missing columns"):
        AtBatDataset(df, pitcher_profile_lookup=lambda *a: {"vector": np.zeros(1)},
                     batter_profile_lookup=lambda *a: {"vector": np.zeros(1)})


def test_dataset_groups_by_at_bat():
    # Two at-bats in one game.
    df1 = _make_pitches(n_pitches=3, ab_id=1, game_pk=100)
    df2 = _make_pitches(n_pitches=4, ab_id=2, game_pk=100)
    df = pd.concat([df1, df2], ignore_index=True)
    p_lookup, _ = _mock_lookup()
    b_lookup, _ = _mock_lookup()
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)
    assert len(ds) == 2
    assert ds[0]["padding_mask"].shape == (3,)
    assert ds[1]["padding_mask"].shape == (4,)


# ---------- collate ----------


def test_collate_pads_to_max_length_in_batch():
    df1 = _make_pitches(n_pitches=2, ab_id=1, game_pk=100)
    df2 = _make_pitches(n_pitches=4, ab_id=2, game_pk=100)
    df = pd.concat([df1, df2], ignore_index=True)
    p_lookup, _ = _mock_lookup(d_profile=4)
    b_lookup, _ = _mock_lookup(d_profile=6)
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)
    batch = collate_at_bats([ds[0], ds[1]])

    assert batch["pitcher_profile"].shape == (2, 4)
    assert batch["batter_profile"].shape == (2, 6)
    assert batch["context_tokens"].shape == (2, 6)
    assert batch["padding_mask"].shape == (2, 4)
    # Item 0 had only 2 real pitches — last 2 positions are padding (False)
    assert batch["padding_mask"][0].tolist() == [True, True, False, False]
    assert batch["padding_mask"][1].tolist() == [True, True, True, True]
    # Padded positions in pitch_factors get PAD_ID
    assert batch["pitch_factors"]["type"][0, 2:].tolist() == [PAD_ID, PAD_ID]


def test_collate_targets_also_padded_with_pad_id():
    df1 = _make_pitches(n_pitches=2, ab_id=1, game_pk=100)
    df2 = _make_pitches(n_pitches=4, ab_id=2, game_pk=100)
    df = pd.concat([df1, df2], ignore_index=True)
    p_lookup, _ = _mock_lookup()
    b_lookup, _ = _mock_lookup()
    ds = AtBatDataset(df, pitcher_profile_lookup=p_lookup, batter_profile_lookup=b_lookup,
                      velo_bin_col=None)
    batch = collate_at_bats([ds[0], ds[1]])
    # Padded item: positions 2 and 3 should be PAD_ID in targets
    assert batch["target_factors"]["type"][0, 2:].tolist() == [PAD_ID, PAD_ID]
    # Item 1 (full length 4): last position is PAD_ID (no successor)
    assert batch["target_factors"]["type"][1, -1].item() == PAD_ID


def test_collate_raises_on_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate_at_bats([])
