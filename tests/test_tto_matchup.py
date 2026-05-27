"""Tests for ``compute_tto_matchup`` (ADR-013 Decision 2).

The matchup TTO must diverge from the batter-only TTO exactly when there
is a pitching change in the game. Pins that contract on a small synthetic
DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocess_pitchgpt import compute_tto_bucket, compute_tto_matchup


def _synthetic_game_with_pitching_change() -> pd.DataFrame:
    """One game, 5 ABs, one pitching change (A→B→A) — the canonical case where
    batter-only TTO and pitcher×batter TTO must disagree.

    Schedule:
      AB 1: Pitcher A vs Batter X (X PA #1, X-vs-A matchup #1)
      AB 2: Pitcher A vs Batter Y (Y PA #1, Y-vs-A matchup #1)
      AB 3: Pitcher A vs Batter X (X PA #2, X-vs-A matchup #2)
      AB 4: Pitcher B vs Batter X (X PA #3, X-vs-B matchup #1)  ← divergence
      AB 5: Pitcher A vs Batter X (X PA #4, X-vs-A matchup #3)  ← divergence
    """
    rows = []
    for ab_n, p, b in [(1, "A", "X"), (2, "A", "Y"), (3, "A", "X"),
                       (4, "B", "X"), (5, "A", "X")]:
        for pi in range(3):  # 3 pitches per AB
            rows.append({
                "game_pk": 100,
                "at_bat_number": ab_n,
                "pitch_number": pi + 1,
                "pitcher": p,
                "batter": b,
            })
    return pd.DataFrame(rows)


def test_tto_matchup_diverges_from_batter_tto_on_pitching_change():
    df = _synthetic_game_with_pitching_change()
    tto = compute_tto_bucket(df)
    tto_m = compute_tto_matchup(df)
    df = df.assign(tto_bucket=tto, tto_matchup=tto_m)

    first_per_ab = (
        df.drop_duplicates("at_bat_number")
          .sort_values("at_bat_number")
          .set_index("at_bat_number")
    )

    # Expected per-AB (batter-TTO, matchup-TTO)
    expected = {
        1: (1, 1),
        2: (1, 1),
        3: (2, 2),
        4: (3, 1),  # X's 3rd PA of game, X-vs-B's 1st encounter
        5: (4, 3),  # X's 4th PA of game, X-vs-A's 3rd encounter
    }
    for ab_n, (tto_e, ttm_e) in expected.items():
        row = first_per_ab.loc[ab_n]
        assert int(row.tto_bucket) == tto_e, (
            f"AB{ab_n}: tto_bucket={row.tto_bucket} (expected {tto_e})"
        )
        assert int(row.tto_matchup) == ttm_e, (
            f"AB{ab_n}: tto_matchup={row.tto_matchup} (expected {ttm_e})"
        )


def test_tto_matchup_4plus_bucket_clipping():
    """6 same-pitcher × same-batter ABs in a game should clip the matchup TTO
    at 4 (the "4+" bucket), same as the batter-only TTO does."""
    rows = []
    for ab_n in range(1, 7):
        for pi in range(2):
            rows.append({
                "game_pk": 200, "at_bat_number": ab_n, "pitch_number": pi + 1,
                "pitcher": "A", "batter": "X",
            })
    df = pd.DataFrame(rows)
    ttm = compute_tto_matchup(df)
    df = df.assign(tto_matchup=ttm)
    per_ab = df.drop_duplicates("at_bat_number").sort_values("at_bat_number")
    assert per_ab["tto_matchup"].tolist() == [1, 2, 3, 4, 4, 4]


def test_tto_matchup_returns_int8_array_aligned_to_pitches():
    """Output length matches pitches; dtype int8 (matching the existing tto)."""
    df = _synthetic_game_with_pitching_change()
    arr = compute_tto_matchup(df)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int8
    assert len(arr) == len(df)
    # Every pitch within an AB shares the same matchup-TTO value
    for ab_n, g in df.assign(ttm=arr).groupby("at_bat_number"):
        assert g["ttm"].nunique() == 1, f"AB{ab_n} matchup-TTO not constant within AB"
