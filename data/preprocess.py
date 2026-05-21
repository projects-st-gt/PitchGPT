"""Preprocessing utilities — harmonization, zone tagging, velocity binning.

Composes primitives from ``data.harmonization`` and ``data.zones`` into a
single entry point for turning a raw Statcast pull into a tagged dataframe.

Type-relative velocity binning is leakage-sensitive — the per-pitcher mean
and std must come from a trailing window that ends strictly before the
current AB. Computing those statistics is the job of
``data.player_profiles``; this module just consumes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.harmonization import harmonize_dataframe
from data.zones import assign_action_zone, assign_feature_zone_14, valid_zone_mask

# Quantile cuts on the standard normal — 10 deciles, symmetric around 0.
# Used to bin velocity z-scores into a categorical feature.
VELO_DECILE_CUTS: list[float] = [
    -np.inf, -1.2816, -0.8416, -0.5244, -0.2533,
    0.0, 0.2533, 0.5244, 0.8416, 1.2816, np.inf,
]


def harmonize_and_tag(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize pitch types and add ``action_zone`` and ``feature_zone`` columns.

    Drops at-bats containing any unmapped pitch type (whole-AB drop, see
    ``harmonize_dataframe``). Drops rows with invalid zone inputs
    (``sz_top - sz_bot < 1.0`` ft). Also drops rows with NaN ``zone`` —
    these are book-keeping artifacts (automatic_ball entries from
    intentional walks since 2017, automatic_strike from pitch-clock
    violations since 2023) with no measured pitch data (plate_x/plate_z,
    pitch_type, velocity all NaN).

    ``feature_zone`` is the Statcast SIS 14-zone scheme (13 actual indices
    via ``assign_feature_zone_14``); ``action_zone`` is the 5-cell scheme
    from ADR 001, computed from plate_x/plate_z.
    """
    df = harmonize_dataframe(df, drop_unmapped=True)
    if "zone" in df.columns:
        df = df.loc[df["zone"].notna()].copy()
    df = df.loc[valid_zone_mask(df)].copy()
    df["action_zone"] = assign_action_zone(df)
    df["feature_zone"] = assign_feature_zone_14(df)
    return df


def bin_velo_z_score(z: pd.Series) -> pd.Series:
    """Bin a velocity z-score Series into 10 deciles (0–9). Preserves NaN."""
    return pd.cut(z, bins=VELO_DECILE_CUTS, labels=False, include_lowest=True).astype("Int8")


def type_relative_velocity_bin(
    df: pd.DataFrame,
    *,
    velo_stats: pd.DataFrame,
    league_means: pd.DataFrame,
    min_pitches_for_pitcher: int = 30,
) -> pd.DataFrame:
    """Add ``release_speed_z`` and ``velo_bin`` columns.

    Args:
        df: pitches with ``pitcher``, ``pitch_type_canonical``, ``release_speed``,
            ``asof_date`` columns. ``asof_date`` is the trailing-window end
            date and matches ADR 003's no-leakage discipline.
        velo_stats: per-(``pitcher``, ``pitch_type_canonical``, ``asof_date``)
            trailing-window stats with columns ``mean``, ``std``, ``n``.
        league_means: per-(``pitch_type_canonical``, ``asof_date``) trailing
            league-wide stats with columns ``mean``, ``std``. Used as fallback
            when the pitcher's trailing window has fewer than
            ``min_pitches_for_pitcher`` examples.

    Returns:
        ``df`` with three new columns:

        - ``release_speed_z``: standardized velocity (NaN if both pitcher
          and league stats are unavailable).
        - ``velo_bin``: deciled z-score (Int8, 0–9, nullable).
        - ``velo_used_fallback``: bool, True when the league fallback was used.
    """
    keys_p = ["pitcher", "pitch_type_canonical", "asof_date"]
    keys_l = ["pitch_type_canonical", "asof_date"]

    keyed = df.merge(
        velo_stats.rename(columns={"mean": "_p_mean", "std": "_p_std", "n": "_p_n"}),
        on=keys_p,
        how="left",
    )
    keyed = keyed.merge(
        league_means.rename(columns={"mean": "_l_mean", "std": "_l_std"}),
        on=keys_l,
        how="left",
    )

    use_league = keyed["_p_n"].fillna(0) < min_pitches_for_pitcher
    mu = np.where(use_league, keyed["_l_mean"], keyed["_p_mean"])
    sigma = np.where(use_league, keyed["_l_std"], keyed["_p_std"])
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(sigma == 0, np.nan, sigma)

    z = (keyed["release_speed"] - mu) / sigma
    keyed["release_speed_z"] = z
    keyed["velo_bin"] = bin_velo_z_score(z)
    keyed["velo_used_fallback"] = use_league.astype(bool)

    return keyed.drop(columns=["_p_mean", "_p_std", "_p_n", "_l_mean", "_l_std"])
