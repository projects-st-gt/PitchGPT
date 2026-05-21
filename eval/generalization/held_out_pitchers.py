"""Held-out pitcher cohort per the ``eval-protocol`` skill.

The cohort is *pitchers whose first MLB pitch is in or after a debut year*
(default 2024). Used as a separate eval cell that tests profile-based
generalization (versus memorization).

Per the skill: if PitchGPT's accuracy collapses on this cohort but
XGBoost's doesn't, the player-profile encoder is doing more memorization
than generalization, and the writeup needs to say so.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_DEBUT_YEAR: int = 2024


def held_out_pitcher_ids(
    pitches: pd.DataFrame,
    debut_year: int = DEFAULT_DEBUT_YEAR,
) -> set[int]:
    """Set of pitcher_ids whose earliest game_date in ``pitches`` is in or
    after ``debut_year``.

    Required columns: ``pitcher``, ``game_date``.
    """
    if not {"pitcher", "game_date"}.issubset(pitches.columns):
        raise KeyError("held_out_pitcher_ids needs 'pitcher' and 'game_date' columns")
    if len(pitches) == 0:
        return set()

    dates = pd.to_datetime(pitches["game_date"])
    first_seen = (
        pd.DataFrame({"pitcher": pitches["pitcher"], "game_date": dates})
        .groupby("pitcher")["game_date"]
        .min()
    )
    cutoff = pd.Timestamp(f"{debut_year}-01-01")
    return {int(pid) for pid in first_seen[first_seen >= cutoff].index}


def held_out_pitcher_mask(
    pitches: pd.DataFrame,
    debut_year: int = DEFAULT_DEBUT_YEAR,
) -> pd.Series:
    """Boolean mask selecting rows where the pitcher debuted in or after
    ``debut_year``."""
    cohort = held_out_pitcher_ids(pitches, debut_year=debut_year)
    return pitches["pitcher"].astype(int).isin(cohort)
