"""Unit tests for K-fold assignment.

Covers determinism, balance, the K=5 default, and the FoldAssignments
lookup interface.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from data.folds import (
    FoldAssignments,
    assign_folds,
    fold_balance_deviation,
    load_folds,
    save_folds,
)


def _games(n_per_season=20, seasons=(2017, 2018, 2019)):
    rows = []
    pk = 100
    for season in seasons:
        for _ in range(n_per_season):
            rows.append({"game_pk": pk, "season": season})
            pk += 1
    return pd.DataFrame(rows)


# ---------- assignment basics ----------


def test_assign_folds_returns_required_columns():
    out = assign_folds(_games())
    assert {"game_pk", "season", "fold_id"}.issubset(out.columns)


def test_assign_folds_default_k_is_5():
    out = assign_folds(_games(n_per_season=50))
    assert sorted(out["fold_id"].unique().tolist()) == [0, 1, 2, 3, 4]


def test_assign_folds_is_deterministic_with_seed():
    df = _games(n_per_season=30)
    a = assign_folds(df, seed=42)
    b = assign_folds(df, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_assign_folds_changes_with_seed():
    df = _games(n_per_season=30)
    a = assign_folds(df, seed=42)
    b = assign_folds(df, seed=43)
    # Same set of games, but at least one game's fold changes
    merged = a.merge(b, on=["game_pk", "season"], suffixes=("_42", "_43"))
    assert (merged["fold_id_42"] != merged["fold_id_43"]).any()


def test_assign_folds_every_game_assigned_exactly_once():
    df = _games(n_per_season=20)
    out = assign_folds(df)
    assert len(out) == len(df)
    assert out["game_pk"].nunique() == out["game_pk"].size


def test_assign_folds_dedupes_input():
    df = pd.concat([_games(n_per_season=10), _games(n_per_season=10)],
                   ignore_index=True)
    out = assign_folds(df)
    # Half the rows were duplicates of the same game_pk values
    assert len(out) == 30  # 10 games × 3 seasons


def test_assign_folds_validates_columns():
    with pytest.raises(KeyError, match="game_pk"):
        assign_folds(pd.DataFrame({"foo": [1]}))


def test_assign_folds_rejects_k_below_2():
    with pytest.raises(ValueError, match="k must be >= 2"):
        assign_folds(_games(), k=1)


# ---------- balance ----------


def test_fold_balance_within_2pct_for_clean_input():
    out = assign_folds(_games(n_per_season=100))
    dev = fold_balance_deviation(out)
    # 100 games / 5 folds = 20 per fold per season; mod-K assignment is exact.
    assert dev < 0.05


def test_fold_balance_handles_uneven_seasons():
    # 17 games in one season → folds get 4, 4, 4, 3, 2 → max deviation 47% from 3.4
    out = assign_folds(_games(n_per_season=17, seasons=(2017,)))
    dev = fold_balance_deviation(out)
    assert dev > 0  # not perfectly balanced
    assert dev < 1.0  # but not catastrophic


def test_fold_balance_deviation_zero_for_empty():
    empty = pd.DataFrame(columns=["fold_id", "season", "game_pk"])
    assert fold_balance_deviation(empty) == 0.0


# ---------- save / load ----------


def test_save_then_load_roundtrips(tmp_path):
    out = assign_folds(_games())
    save_folds(out, tmp_path / "folds.parquet")
    loaded = load_folds(tmp_path / "folds.parquet")
    pd.testing.assert_frame_equal(out, loaded)


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    save_folds(assign_folds(_games()), tmp_path / "folds.parquet")
    assert not (tmp_path / "folds.tmp.parquet").exists()


# ---------- FoldAssignments lookup ----------


def test_fold_assignments_lookup_returns_correct_fold():
    out = assign_folds(_games(n_per_season=10))
    fa = FoldAssignments(out)
    assert fa.k == 5
    assert len(fa) == len(out)
    sample = out.iloc[0]
    assert fa.fold_of(int(sample["game_pk"])) == int(sample["fold_id"])


def test_fold_assignments_unknown_game_returns_none():
    fa = FoldAssignments(assign_folds(_games(n_per_season=10)))
    assert fa.fold_of(999_999) is None


def test_fold_assignments_games_in_and_excluding_fold():
    out = assign_folds(_games(n_per_season=20))
    fa = FoldAssignments(out)
    in_fold_0 = set(fa.games_in_fold(0))
    excluding = set(fa.games_excluding_fold(0))
    # Disjoint, union = all games
    assert in_fold_0.isdisjoint(excluding)
    assert in_fold_0 | excluding == set(int(pk) for pk in out["game_pk"])


def test_fold_assignments_validates_columns():
    with pytest.raises(KeyError):
        FoldAssignments(pd.DataFrame({"foo": [1]}))


def test_composite_keys_unit_safe():
    """Regression: 2026-06-12 unit bug. pandas 3.x parses date strings to
    datetime64[us]; us-as-ns collapsed every sort key to ~day 20, degenerating
    every asof cutoff to 'include everything' (frozen profiles + temporal
    leakage). The sort key and asof key must agree for any datetime unit."""
    import numpy as np
    import pandas as pd
    from scripts.build_profile_cache import _composite_asof_key, _composite_sort_key

    dates_us = pd.to_datetime(["2026-05-08", "2026-06-10"]).as_unit("us").to_numpy()
    nums = np.array([1, 1])
    keys = _composite_sort_key(dates_us, nums)
    ak = _composite_asof_key(pd.Timestamp("2026-06-01"), 1)
    assert keys[0] < ak < keys[1], (
        f"asof key {ak} must separate {keys} — unit mismatch regression")
