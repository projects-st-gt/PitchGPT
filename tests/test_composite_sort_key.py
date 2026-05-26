"""Regression test for the composite (date, game_num) sort-key helper.

A silent bug appeared with pandas 3.x parquet defaults producing
``datetime64[us]`` instead of ``[ns]`` — the original helper hard-coded
``// _NS_PER_DAY`` after ``astype("int64")``, which is off by 1000× on us
inputs. The fix forces ``astype("datetime64[ns]")`` before the division so
the helper is unit-safe regardless of the source dtype.

These tests pin both the unit-invariance and the asof-key matching that the
matchup + batter cache builders rely on for correct ``before_asof``-equivalent
filtering via ``np.searchsorted``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_profile_cache import _composite_sort_key, _composite_asof_key


def test_composite_sort_key_unit_invariant_ns_vs_us():
    """Same logical dates yield the same int64 key whether stored as ns or us."""
    dates_ns = pd.Series(
        pd.to_datetime(["2017-04-02", "2017-06-03", "2024-09-30"])
    ).astype("datetime64[ns]")
    dates_us = dates_ns.astype("datetime64[us]")
    nums = np.array([1, 1, 2], dtype=np.int64)

    keys_ns = _composite_sort_key(dates_ns.to_numpy(), nums)
    keys_us = _composite_sort_key(dates_us.to_numpy(), nums)

    assert np.array_equal(keys_ns, keys_us), (
        f"unit mismatch: ns={keys_ns.tolist()} us={keys_us.tolist()}"
    )


def test_composite_keys_searchsorted_excludes_same_game():
    """Searchsorted with side='left' on these keys must match before_asof:
    same-(date, game_num) rows are excluded, strictly-earlier rows kept.

    Pins the contract that ``build_matchup_cache_for_fold_fast`` and
    ``build_batter_cache_for_fold_fast`` rely on.
    """
    dates = pd.Series(
        pd.to_datetime(["2017-04-02", "2017-04-02", "2017-06-03", "2017-06-03"])
    ).astype("datetime64[us]")  # the dtype that ships from pandas 3.x parquets
    nums = np.array([1, 1, 1, 1], dtype=np.int64)
    keys = _composite_sort_key(dates.to_numpy(), nums)

    # asof = the first game on 2017-04-02 — same-game must be excluded
    ak = _composite_asof_key(pd.Timestamp("2017-04-02"), 1)
    cutoff = int(np.searchsorted(keys, ak, side="left"))
    assert cutoff == 0, f"same-game leak: cutoff={cutoff} (must be 0)"

    # asof = the first game on 2017-06-03 — the two 2017-04-02 PAs must be kept
    ak_later = _composite_asof_key(pd.Timestamp("2017-06-03"), 1)
    cutoff_later = int(np.searchsorted(keys, ak_later, side="left"))
    assert cutoff_later == 2, (
        f"strictly-prior rows lost: cutoff={cutoff_later} (must be 2)"
    )


def test_composite_keys_doubleheader_ordering():
    """Doubleheader G1 < G2 on the same date — G1 must be eligible for G2's window."""
    dates = pd.Series(
        pd.to_datetime(["2019-07-13", "2019-07-13"])
    ).astype("datetime64[us]")
    nums = np.array([1, 2], dtype=np.int64)
    keys = _composite_sort_key(dates.to_numpy(), nums)

    ak_g2 = _composite_asof_key(pd.Timestamp("2019-07-13"), 2)
    cutoff = int(np.searchsorted(keys, ak_g2, side="left"))
    assert cutoff == 1, (
        f"doubleheader G1 excluded from G2's window: cutoff={cutoff} (must be 1)"
    )
