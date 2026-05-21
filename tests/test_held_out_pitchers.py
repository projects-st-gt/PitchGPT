"""Unit tests for the held-out-pitcher cohort filter."""

from __future__ import annotations

import pandas as pd
import pytest

from eval.generalization.held_out_pitchers import (
    held_out_pitcher_ids,
    held_out_pitcher_mask,
)


def _pitches(rows):
    """rows: list of (pitcher_id, game_date_str)."""
    return pd.DataFrame(
        {"pitcher": [r[0] for r in rows],
         "game_date": pd.to_datetime([r[1] for r in rows])}
    )


def test_held_out_picks_post_2024_debuts():
    pitches = _pitches([
        (1, "2017-04-01"),  # debut 2017 → not in cohort
        (1, "2024-08-01"),
        (2, "2024-04-15"),  # debut 2024 → in cohort
        (3, "2025-05-01"),  # debut 2025 → in cohort
        (4, "2023-06-01"),  # debut 2023 → not in cohort
        (4, "2024-09-01"),
    ])
    cohort = held_out_pitcher_ids(pitches, debut_year=2024)
    assert cohort == {2, 3}


def test_held_out_pitcher_mask_filters_pitches_correctly():
    pitches = _pitches([
        (1, "2017-04-01"),
        (1, "2024-08-01"),
        (2, "2024-04-15"),
        (3, "2025-05-01"),
    ])
    mask = held_out_pitcher_mask(pitches, debut_year=2024)
    # Only pitcher 1's pitches are NOT in cohort
    assert mask.tolist() == [False, False, True, True]


def test_held_out_empty_input_returns_empty_set():
    cohort = held_out_pitcher_ids(_pitches([]))
    assert cohort == set()


def test_held_out_validates_required_columns():
    with pytest.raises(KeyError, match="pitcher"):
        held_out_pitcher_ids(pd.DataFrame({"foo": [1]}))


def test_held_out_includes_january_one_debut():
    """Edge case: pitcher debuting on Jan 1 of debut_year is in cohort."""
    pitches = _pitches([
        (1, "2024-01-01"),
        (2, "2023-12-31"),
    ])
    cohort = held_out_pitcher_ids(pitches, debut_year=2024)
    assert cohort == {1}
