"""Unit tests for the temporal split per CLAUDE.md."""

from __future__ import annotations

import pandas as pd
import pytest

from data.dataset import (
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    temporal_split,
    temporal_split_mask,
)


def _df(dates):
    return pd.DataFrame({"game_date": pd.to_datetime(dates)})


def test_split_constants_match_claude_md():
    assert TRAIN_END == pd.Timestamp("2023-12-31")
    assert VAL_START == pd.Timestamp("2024-01-01")
    assert VAL_END == pd.Timestamp("2024-07-15")
    assert TEST_START == pd.Timestamp("2024-07-16")


def test_temporal_split_assigns_each_date_to_one_bucket():
    pitches = _df([
        "2017-04-01",  # train
        "2023-12-31",  # train (boundary)
        "2024-01-01",  # val (boundary)
        "2024-07-15",  # val (boundary)
        "2024-07-16",  # test (boundary)
        "2025-09-01",  # test
    ])
    m = temporal_split_mask(pitches)
    assert m["train"].tolist() == [True, True, False, False, False, False]
    assert m["val"].tolist() == [False, False, True, True, False, False]
    assert m["test"].tolist() == [False, False, False, False, True, True]
    # Mutually exclusive, exhaustive
    union = m["train"] | m["val"] | m["test"]
    assert union.all()


def test_temporal_split_returns_view_dataframes():
    pitches = _df([
        "2018-04-01", "2024-03-15", "2024-08-01"
    ])
    out = temporal_split(pitches)
    assert len(out["train"]) == 1
    assert len(out["val"]) == 1
    assert len(out["test"]) == 1


def test_temporal_split_handles_string_game_date_column():
    pitches = pd.DataFrame({"game_date": ["2018-04-01", "2024-08-01"]})
    out = temporal_split(pitches)
    assert len(out["train"]) == 1
    assert len(out["test"]) == 1


def test_temporal_split_validates_required_column():
    with pytest.raises(KeyError, match="game_date"):
        temporal_split_mask(pd.DataFrame({"foo": [1]}))


def test_temporal_split_empty_input():
    pitches = _df([])
    out = temporal_split(pitches)
    assert all(len(out[k]) == 0 for k in ("train", "val", "test"))
