"""Profile cache schema, flattening, and builder.

This module defines the fixed-length numpy vectors that the model consumes
as its pitcher and batter profile features. The schema is the canonical
contract between:

- The cache writer (which produces these vectors from the raw corpus)
- The cache reader (which serves them to the dataset class)
- The model (which expects these slots in this exact order)

Any change to the schema bumps ``PROFILE_SCHEMA_VERSION``. Cache loaders
will refuse to read a stale-version cache; rebuild instead of silently
training on misaligned features.

Per ADR 008, the cache is fold-aware (K=5): the cache key includes a
``fold_id`` and the cache for fold X is built using only pitches from
folds ≠ X (in addition to the ``before_asof`` no-leakage rule).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from data.dataset import PITCH_TYPES, N_PITCH_TYPES
from data.player_profiles import (
    N_IN_ZONE_CELLS,
    batter_chase_rate_by_type,
    batter_last_n_days_woba,
    batter_outcome_rates,
    batter_whiff_and_swing_by_type,
    batter_zone_grids,
    matchup_cumulative_pa_stats,
    matchup_last_face,
    pitcher_arm_slot_by_type,
    pitcher_arsenal,
    pitcher_arsenal_by_count,
    pitcher_arsenal_by_stand,
    pitcher_days_since_last_appearance,
    pitcher_last_n_days_xwoba,
    pitcher_last_n_starts_xwoba,
    pitcher_movement_by_type,
    pitcher_zone_heatmap_by_type,
    window_freshness,
)

# Bump on any feature definition change. The cache loader refuses mismatched
# versions to prevent silent feature-slot misalignment.
#
# v2 (2026-05-09): Added long-window staleness features
#   ``long_window_span_days`` and ``long_window_pct_current_season`` to both
#   pitcher and batter schemas, to expose cross-season profile staleness
#   (e.g. April-1 ABs whose 1000-pitch window is mostly from prior October).
# v3 (2026-05-12, ADR 013 / roadmap A1): Added 14 dims to the BATTER schema —
#   ``whiff_on_swing_{pt}`` (7) and ``swing_rate_{pt}`` (7) per canonical pitch
#   type. The pitcher schema is unchanged at v3 (a single version number is kept
#   for simplicity; rebuilding the full cache re-tags the pitcher vectors v3 too).
# v4 (2026-05-13, Sprint 0b): Added 7 dims to the PITCHER schema — per-pitch-type
#   mean ``arm_angle`` (release-side arm slot, degrees). Statcast publishes this
#   from 2020 onward; pre-2020 entries have NaN per type, which the league-mean
#   fallback handles cleanly via the existing 3-step chain in
#   ``profile_cache_loader.ProfileCache.lookup``. Targets within-family pitch
#   disambiguation (FF vs SI vs FC) and is the primary observable signal for the
#   tipping detector (ADR 005). The batter schema is unchanged at v4.
# v5 (2026-05-14, 14-zone migration): swapped the location feature from the
#   legacy 26-class v1 (5×5 grid + OOZ, computed from plate_x/plate_z) to the
#   Statcast SIS 14-zone v2 (3×3 in-zone + 4 OOZ quadrants, derived from the
#   raw ``zone`` column). N_IN_ZONE_CELLS drops 25→9, so the pitcher heatmap
#   shrinks 7×25=175 → 7×9=63 dims (pitcher vector 230 → 118) and the batter
#   zone grids shrink 3×25=75 → 3×9=27 dims (batter vector 105 → 57). Caches
#   built under v4 are not loadable; full rebuild required.
# v6 (2026-05-17): per-(pitch type × count) and per-(pitch type × batter stand)
#   arsenal usage blocks (84 + 14 dims), plus per-type pfx_x/pfx_z mean (14
#   dims). Drops the 12 entropy_b{b}s{s} dims as redundant given the new
#   conditional distribution. Pitcher vector 118 → 218. Targets the CU/FC
#   under-recall and FF argmax over-prediction diagnosed on the v5 baseline
#   (see docs/superpowers/specs/2026-05-17-v6-profile-features-design.md).
#   The batter schema is unchanged at v6 (rebuild re-tags batter vectors v6
#   for consistency).
PROFILE_SCHEMA_VERSION: int = 6

# Canonical (balls, strikes) ordering for entropy features.
COUNT_STATES: list[tuple[int, int]] = [
    (b, s) for b in range(4) for s in range(3)
]
N_COUNT_STATES: int = len(COUNT_STATES)  # 12


# ---------- pitcher schema ----------

PITCHER_FEATURE_NAMES: list[str] = [
    # 7 per-type usage fractions
    *(f"arsenal_{pt}" for pt in PITCH_TYPES),
    # 7 per-type mean velocities (NaN if pitch type not thrown)
    *(f"mean_velo_{pt}" for pt in PITCH_TYPES),
    # 7 per-type mean spin rates (NaN if not thrown or column missing)
    *(f"mean_spin_{pt}" for pt in PITCH_TYPES),
    # 7 × 25 = 175 in-zone heatmap entries, in PITCH_TYPES × cell-index order
    *(
        f"heatmap_{pt}_z{i}"
        for pt, i in itertools.product(PITCH_TYPES, range(N_IN_ZONE_CELLS))
    ),
    # Recent-form (time-based, primary)
    "recent_30d_xwoba",
    "recent_30d_n_pitches",
    "days_since_last_appearance",
    # Recent-form (start-based, starter-specific; NaN for relievers)
    "recent_3starts_xwoba",
    "recent_3starts_n",
    # Confidence
    "profile_confidence",
    # Cross-season staleness signals (v2). ``long_window_span_days`` is the
    # day-gap between the oldest pitch in the 1000-pitch window and the
    # asof date; ``long_window_pct_current_season`` is the fraction of
    # window pitches whose year matches the asof year. Both NaN for empty
    # windows.
    "long_window_span_days",
    "long_window_pct_current_season",
    # 7 binary "has thrown this pitch type at least once" flags — disambiguates
    # "0 because never thrown" from "0 because rare in window"
    *(f"has_pitch_{pt}" for pt in PITCH_TYPES),
    # v4 / Sprint 0b: per-pitch-type mean arm slot (Statcast ``arm_angle``,
    # release-side angle in degrees). NaN for types never thrown OR for pre-2020
    # pitches (Statcast didn't publish arm_angle before then) — the league-mean
    # fallback handles both cases. Primary observable signal for the tipping
    # detector (ADR 005) and a within-family pitch disambiguator.
    *(f"arm_slot_{pt}" for pt in PITCH_TYPES),
    # v6 (2026-05-17): per-(pitch type × count) usage fraction, 7 × 12 = 84
    # dims. NaN for (b, s) cells with zero observations in the trailing window
    # — league-mean fallback handles. Cells with observations but where a type
    # was not thrown get 0.0. Targets CU/FC under-recall and FF argmax over-
    # prediction by exposing the conditional structure of pitcher arsenals.
    *(
        f"arsenal_{pt}_b{b}s{s}"
        for pt, (b, s) in itertools.product(PITCH_TYPES, COUNT_STATES)
    ),
    # v6: per-(pitch type × batter stand) usage fraction, 7 × 2 = 14 dims.
    # NaN for stands the pitcher hasn't faced — league-mean fallback handles.
    *(
        f"arsenal_{pt}_vs{stand}"
        for pt, stand in itertools.product(PITCH_TYPES, ("L", "R"))
    ),
    # v6: mean pfx_x (horizontal break, feet) per pitch type. NaN for types
    # never thrown or all-NaN window — league-mean fallback handles.
    *(f"mean_pfx_x_{pt}" for pt in PITCH_TYPES),
    # v6: mean pfx_z (vertical break, feet) per pitch type. NaN handling
    # same as pfx_x.
    *(f"mean_pfx_z_{pt}" for pt in PITCH_TYPES),
]
PITCHER_FEATURE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(PITCHER_FEATURE_NAMES)
}
PITCHER_VECTOR_LEN: int = len(PITCHER_FEATURE_NAMES)


# ---------- batter schema ----------

BATTER_FEATURE_NAMES: list[str] = [
    # Recent-form
    "recent_14d_woba",
    "recent_14d_n_pas",
    # 25 in-zone swing-rate cells
    *(f"swing_z{i}" for i in range(N_IN_ZONE_CELLS)),
    # 25 in-zone whiff-rate cells (NaN where no swings)
    *(f"whiff_z{i}" for i in range(N_IN_ZONE_CELLS)),
    # 25 in-zone xBA cells (NaN where no contact)
    *(f"xba_z{i}" for i in range(N_IN_ZONE_CELLS)),
    # 7 per-pitch-type chase rates
    *(f"chase_{pt}" for pt in PITCH_TYPES),
    # Outcome rates
    "k_pct",
    "bb_pct",
    "hard_contact_pct",
    "n_pas",
    # Confidence
    "profile_confidence",
    # Cross-season staleness signals (v2), same semantics as the pitcher
    # vector but computed over the batter's seen-pitches window.
    "long_window_span_days",
    "long_window_pct_current_season",
    # v3 / ADR 013 (roadmap A1): per-pitch-type whiff-on-swing & swing-rate
    # over the seen-pitches window. "Can this batter hit a slider?" lives here.
    *(f"whiff_on_swing_{pt}" for pt in PITCH_TYPES),
    *(f"swing_rate_{pt}" for pt in PITCH_TYPES),
]
BATTER_FEATURE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(BATTER_FEATURE_NAMES)
}
BATTER_VECTOR_LEN: int = len(BATTER_FEATURE_NAMES)


# ---------- matchup schema (ADR 014 / roadmap A3) ----------
#
# Pitcher × batter matchup history features. Keyed by
# ``(pitcher_id, batter_id, asof_date, asof_game_num)`` rather than the
# per-player ``(player_id, asof_date, asof_game_num)`` of the pitcher/batter
# caches. The vector encodes "what has this specific (pitcher, batter)
# encounter looked like up to now" — pitch mix the pitcher uses vs THIS
# batter, this batter's whiff-on-swing vs THIS pitcher, and cumulative
# PA-level stats. All trailing-window, leakage-safe (caller filters with
# ``before_asof``).
#
# Schema version is tracked independently of the pitcher/batter
# ``PROFILE_SCHEMA_VERSION`` since the matchup cache is built into a separate
# parquet (``matchup_fold_{k}.parquet``); rebuilding pitcher/batter doesn't
# invalidate matchup and vice versa.

MATCHUP_SCHEMA_VERSION: int = 1
MATCHUP_WINDOW_PITCHES: int = 50  # last-N-pitches window for matchup pitch-mix
                                  # and whiff-on-swing-by-type

MATCHUP_FEATURE_NAMES: list[str] = [
    # 7 per-pitch-type pitch-mix this pitcher has used vs this batter
    # (over the last MATCHUP_WINDOW_PITCHES pitches in the matchup)
    *(f"mix_{pt}" for pt in PITCH_TYPES),
    # 7 per-pitch-type whiff-on-swing this batter has against this pitcher's
    # type-X pitches (NaN if no swings of that type in window)
    *(f"whiff_on_swing_{pt}" for pt in PITCH_TYPES),
    # Cumulative PA-level signals
    "n_pas",
    "cum_k",
    "cum_bb",
    "cum_hr",
    # Most-recent-face signals
    "last_face_xwoba",
    "days_since_last_face",
    # Confidence (n_pas saturating at 10 PAs)
    "matchup_confidence",
]
MATCHUP_FEATURE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(MATCHUP_FEATURE_NAMES)
}
MATCHUP_VECTOR_LEN: int = len(MATCHUP_FEATURE_NAMES)


# ---------- flatteners ----------


def _safe_get(d: dict, key, default=np.nan) -> float:
    v = d.get(key, default)
    return default if v is None else v


def build_pitcher_profile_vector(
    pitcher_pitches: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    *,
    window_pitches: int = 1000,
    recent_form_days: int = 30,
    recent_form_n_starts: int = 3,
) -> np.ndarray:
    """Compute the flattened pitcher profile vector for a single asof key.

    The caller is responsible for filtering ``pitcher_pitches`` via
    ``before_asof`` (and, per ADR 008, by fold) *before* calling this
    function. This function is purely the dict→vector flattener.

    Returns a ``np.ndarray`` of length ``PITCHER_VECTOR_LEN``. Slot meanings
    are defined by ``PITCHER_FEATURE_NAMES``.
    """
    vec = np.full(PITCHER_VECTOR_LEN, np.nan, dtype=np.float32)

    arsenal = pitcher_arsenal(pitcher_pitches, window_pitches=window_pitches)
    arsenal_pct = arsenal["arsenal_pct"]
    mean_velo = arsenal["mean_velo_by_type"]
    mean_spin = arsenal["mean_spin_by_type"]

    for pt in PITCH_TYPES:
        vec[PITCHER_FEATURE_INDEX[f"arsenal_{pt}"]] = arsenal_pct.get(pt, 0.0)
        vec[PITCHER_FEATURE_INDEX[f"mean_velo_{pt}"]] = _safe_get(mean_velo, pt)
        vec[PITCHER_FEATURE_INDEX[f"mean_spin_{pt}"]] = _safe_get(mean_spin, pt)
        vec[PITCHER_FEATURE_INDEX[f"has_pitch_{pt}"]] = float(pt in arsenal_pct)

    heatmap = pitcher_zone_heatmap_by_type(
        pitcher_pitches, window_pitches=window_pitches
    )
    for pt in PITCH_TYPES:
        cells = heatmap.get(pt)
        if cells is None:
            for i in range(N_IN_ZONE_CELLS):
                vec[PITCHER_FEATURE_INDEX[f"heatmap_{pt}_z{i}"]] = 0.0
        else:
            for i in range(N_IN_ZONE_CELLS):
                vec[PITCHER_FEATURE_INDEX[f"heatmap_{pt}_z{i}"]] = float(cells[i])

    # v6: per-(pitch type × count) usage fraction. Cells with zero observations
    # leave the slot at NaN (vec was initialized to NaN at the top of the
    # function) — the loader's 3-step league-mean fallback handles.
    if "balls" in pitcher_pitches.columns and "strikes" in pitcher_pitches.columns:
        by_count = pitcher_arsenal_by_count(pitcher_pitches, window_pitches=window_pitches)
        for (b, s, pt), frac in by_count.items():
            vec[PITCHER_FEATURE_INDEX[f"arsenal_{pt}_b{b}s{s}"]] = float(frac)

    # v6: per-(pitch type × batter stand) usage fraction. Same NaN-fallback
    # discipline.
    if "stand" in pitcher_pitches.columns:
        by_stand = pitcher_arsenal_by_stand(pitcher_pitches, window_pitches=window_pitches)
        for (stand, pt), frac in by_stand.items():
            vec[PITCHER_FEATURE_INDEX[f"arsenal_{pt}_vs{stand}"]] = float(frac)

    # v6: mean pfx_x / pfx_z per pitch type. NaN-fallback handles types not
    # thrown or all-NaN windows.
    if "pfx_x" in pitcher_pitches.columns and "pfx_z" in pitcher_pitches.columns:
        movement = pitcher_movement_by_type(pitcher_pitches, window_pitches=window_pitches)
        for pt, m in movement.items():
            vec[PITCHER_FEATURE_INDEX[f"mean_pfx_x_{pt}"]] = float(m["pfx_x"])
            vec[PITCHER_FEATURE_INDEX[f"mean_pfx_z_{pt}"]] = float(m["pfx_z"])

    recent_30d = pitcher_last_n_days_xwoba(
        pitcher_pitches, asof_game_date, days=recent_form_days
    )
    vec[PITCHER_FEATURE_INDEX["recent_30d_xwoba"]] = recent_30d["mean_xwoba"]
    vec[PITCHER_FEATURE_INDEX["recent_30d_n_pitches"]] = recent_30d["n_pitches"]

    layoff = pitcher_days_since_last_appearance(pitcher_pitches, asof_game_date)
    vec[PITCHER_FEATURE_INDEX["days_since_last_appearance"]] = (
        np.nan if layoff is None else float(layoff)
    )

    recent_starts = pitcher_last_n_starts_xwoba(
        pitcher_pitches, n_starts=recent_form_n_starts
    )
    vec[PITCHER_FEATURE_INDEX["recent_3starts_xwoba"]] = recent_starts["mean_xwoba"]
    vec[PITCHER_FEATURE_INDEX["recent_3starts_n"]] = recent_starts["n_starts"]

    vec[PITCHER_FEATURE_INDEX["profile_confidence"]] = arsenal["profile_confidence"]

    freshness = window_freshness(
        pitcher_pitches, asof_game_date, window_pitches=window_pitches
    )
    vec[PITCHER_FEATURE_INDEX["long_window_span_days"]] = freshness["span_days"]
    vec[PITCHER_FEATURE_INDEX["long_window_pct_current_season"]] = freshness[
        "pct_current_season"
    ]

    # v4 / Sprint 0b: per-pitch-type mean arm slot. NaN-by-default; league-mean
    # fallback fills the holes for pre-2020 pitchers and types with no measurements.
    if "arm_angle" in pitcher_pitches.columns:
        arm_slot = pitcher_arm_slot_by_type(pitcher_pitches, window_pitches=window_pitches)
        for pt in PITCH_TYPES:
            vec[PITCHER_FEATURE_INDEX[f"arm_slot_{pt}"]] = float(arm_slot.get(pt, np.nan))

    return vec


def build_batter_profile_vector(
    batter_pitches_seen: pd.DataFrame,
    batter_pas: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    *,
    window_pitches: int = 1000,
    recent_form_days: int = 14,
) -> np.ndarray:
    """Compute the flattened batter profile vector for a single asof key.

    Args:
        batter_pitches_seen: pitches this batter has *seen* (one row per
            pitch from the opponent), already filtered via ``before_asof``
            and (per ADR 008) by fold.
        batter_pas: this batter's PAs (one row per PA), already filtered
            similarly.
        asof_game_date: anchor date.

    Returns ``np.ndarray`` of length ``BATTER_VECTOR_LEN``.
    """
    vec = np.full(BATTER_VECTOR_LEN, np.nan, dtype=np.float32)

    recent = batter_last_n_days_woba(
        batter_pas, asof_game_date, days=recent_form_days
    )
    vec[BATTER_FEATURE_INDEX["recent_14d_woba"]] = recent["mean_woba"]
    vec[BATTER_FEATURE_INDEX["recent_14d_n_pas"]] = recent["n_pas"]

    grids = batter_zone_grids(batter_pitches_seen, window_pitches=window_pitches)
    for i in range(N_IN_ZONE_CELLS):
        vec[BATTER_FEATURE_INDEX[f"swing_z{i}"]] = float(grids["swing"][i])
        vec[BATTER_FEATURE_INDEX[f"whiff_z{i}"]] = float(grids["whiff"][i])
        vec[BATTER_FEATURE_INDEX[f"xba_z{i}"]] = float(grids["xba"][i])

    chase = batter_chase_rate_by_type(
        batter_pitches_seen, window_pitches=window_pitches
    )
    for pt in PITCH_TYPES:
        vec[BATTER_FEATURE_INDEX[f"chase_{pt}"]] = float(chase.get(pt, np.nan))

    # v3 / ADR 013: per-pitch-type whiff-on-swing & swing-rate.
    wos = batter_whiff_and_swing_by_type(
        batter_pitches_seen, window_pitches=window_pitches
    )
    for pt in PITCH_TYPES:
        d = wos.get(pt, {})
        vec[BATTER_FEATURE_INDEX[f"whiff_on_swing_{pt}"]] = float(d.get("whiff_on_swing", np.nan))
        vec[BATTER_FEATURE_INDEX[f"swing_rate_{pt}"]] = float(d.get("swing_rate", np.nan))

    outcomes = batter_outcome_rates(batter_pas)
    vec[BATTER_FEATURE_INDEX["k_pct"]] = outcomes["k_pct"]
    vec[BATTER_FEATURE_INDEX["bb_pct"]] = outcomes["bb_pct"]
    vec[BATTER_FEATURE_INDEX["hard_contact_pct"]] = outcomes["hard_contact_pct"]
    vec[BATTER_FEATURE_INDEX["n_pas"]] = float(outcomes["n_pas"])

    n_pitches_seen = len(batter_pitches_seen)
    vec[BATTER_FEATURE_INDEX["profile_confidence"]] = min(
        n_pitches_seen / window_pitches, 1.0
    )

    freshness = window_freshness(
        batter_pitches_seen, asof_game_date, window_pitches=window_pitches
    )
    vec[BATTER_FEATURE_INDEX["long_window_span_days"]] = freshness["span_days"]
    vec[BATTER_FEATURE_INDEX["long_window_pct_current_season"]] = freshness[
        "pct_current_season"
    ]

    return vec


def build_matchup_profile_vector(
    matchup_pitches: pd.DataFrame,
    matchup_pas: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    *,
    window_pitches: int = MATCHUP_WINDOW_PITCHES,
) -> np.ndarray:
    """Compute the flattened matchup profile for one (pitcher, batter, asof) key.

    Pitcher × batter history features (ADR 014 / roadmap A3). The caller is
    responsible for filtering ``matchup_pitches`` and ``matchup_pas`` to this
    specific (pitcher, batter) pair via ``before_asof`` (and, per ADR 008, by
    fold) *before* calling this function.

    Args:
        matchup_pitches: pitches the pitcher threw to this batter, already
            ``before_asof``-filtered, fold-filtered, and sorted chronologically.
            Required columns: ``pitch_type_canonical``, ``description``,
            ``release_speed`` (used by ``pitcher_arsenal``).
        matchup_pas: terminal pitches (``events`` not null) for the same pair,
            same filtering + sort. Required columns: ``events``, ``game_date``,
            ``estimated_woba_using_speedangle``.
        asof_game_date: anchor date.
        window_pitches: last-N-pitches window for the pitch-mix and whiff
            features. Defaults to ``MATCHUP_WINDOW_PITCHES`` (50).

    Returns ``np.ndarray`` of length ``MATCHUP_VECTOR_LEN``.
    """
    vec = np.full(MATCHUP_VECTOR_LEN, np.nan, dtype=np.float32)

    # Pitch mix the pitcher uses vs this batter (per-type fractions).
    if len(matchup_pitches) > 0:
        arsenal = pitcher_arsenal(matchup_pitches, window_pitches=window_pitches)
        for pt in PITCH_TYPES:
            vec[MATCHUP_FEATURE_INDEX[f"mix_{pt}"]] = float(
                arsenal["arsenal_pct"].get(pt, 0.0)
            )
        wos = batter_whiff_and_swing_by_type(
            matchup_pitches, window_pitches=window_pitches
        )
        for pt in PITCH_TYPES:
            d = wos.get(pt, {})
            vec[MATCHUP_FEATURE_INDEX[f"whiff_on_swing_{pt}"]] = float(
                d.get("whiff_on_swing", np.nan)
            )
    else:
        # Empty matchup: zero mix (model can detect via n_pas / confidence),
        # NaN whiff (no swings to whiff on).
        for pt in PITCH_TYPES:
            vec[MATCHUP_FEATURE_INDEX[f"mix_{pt}"]] = 0.0

    cum = matchup_cumulative_pa_stats(matchup_pas)
    vec[MATCHUP_FEATURE_INDEX["n_pas"]] = cum["n_pas"]
    vec[MATCHUP_FEATURE_INDEX["cum_k"]] = cum["cum_k"]
    vec[MATCHUP_FEATURE_INDEX["cum_bb"]] = cum["cum_bb"]
    vec[MATCHUP_FEATURE_INDEX["cum_hr"]] = cum["cum_hr"]

    lf = matchup_last_face(matchup_pas, asof_game_date)
    vec[MATCHUP_FEATURE_INDEX["last_face_xwoba"]] = lf["last_face_xwoba"]
    vec[MATCHUP_FEATURE_INDEX["days_since_last_face"]] = lf["days_since_last_face"]

    # Confidence saturates at 10 PAs (a fully-fleshed-out matchup); zero at debut.
    vec[MATCHUP_FEATURE_INDEX["matchup_confidence"]] = min(cum["n_pas"] / 10.0, 1.0)

    return vec


# ---------- league-mean aggregation ----------


def compute_league_means(player_cache: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-player cache into league-mean profiles.

    Groups by ``(asof_date, asof_game_num, fold_id)`` and takes the
    **NaN-aware** mean of per-player vectors within each group. The result
    is a small "league fallback" cache used by the loader to fill in NaNs
    for sparse per-player rows (debut players, early-corpus rows).

    Per ADR 008, fold-awareness is preserved: the per-player cache for
    fold k was built using only games in folds ≠ k, so the league mean
    derived from it inherits the same property.

    Args:
        player_cache: per-player cache DataFrame with columns ``vector``,
            ``asof_date``, ``asof_game_num``, ``fold_id``, ``schema_version``.

    Returns:
        DataFrame with columns ``asof_date``, ``asof_game_num``, ``fold_id``,
        ``schema_version``, ``vector``, ``n_players_in_mean``.
    """
    if len(player_cache) == 0:
        return pd.DataFrame(
            columns=[
                "asof_date", "asof_game_num", "fold_id",
                "schema_version", "vector", "n_players_in_mean",
            ]
        )

    schema_versions = player_cache["schema_version"].unique()
    if len(schema_versions) > 1:
        raise ValueError(
            f"player_cache mixes schema versions {schema_versions.tolist()}; "
            f"rebuild before aggregating"
        )
    schema_version = int(schema_versions[0])

    rows: list[dict] = []
    for (asof_date, asof_num, fold), group in player_cache.groupby(
        ["asof_date", "asof_game_num", "fold_id"], sort=False
    ):
        stacked = np.stack(
            [np.asarray(v, dtype=float) for v in group["vector"]]
        )
        # NaN-aware mean: ignore NaNs per cell. All-NaN cells stay NaN; that's
        # intentional, but ``nanmean`` warns about them — silence here.
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            league_vec = np.nanmean(stacked, axis=0)
        rows.append({
            "asof_date": asof_date,
            "asof_game_num": int(asof_num),
            "fold_id": int(fold),
            "schema_version": schema_version,
            "vector": league_vec.astype(np.float32).tolist(),
            "n_players_in_mean": int(len(group)),
        })
    return pd.DataFrame(rows)


def blend_with_league_mean(
    per_player_vec: np.ndarray,
    league_mean_vec: np.ndarray,
    profile_confidence: float,
) -> np.ndarray:
    """Blend a per-player profile vector with a league-mean fallback.

    Two-step rule:

    1. **NaN-fill from league mean.** Wherever the per-player vector has
       NaN (typical for never-thrown pitch types, empty windows, etc.),
       substitute the league-mean value. This handles debut players whose
       per-player vector is mostly NaN.
    2. **Confidence-weighted blend.** ``c · per_player + (1 - c) · league``,
       where ``c = profile_confidence`` is the per-player confidence (0
       for debut, 1 for fully-populated 1000-pitch window).

    Returns the blended vector, same shape as the inputs.
    """
    if per_player_vec.shape != league_mean_vec.shape:
        raise ValueError(
            f"shape mismatch: per_player {per_player_vec.shape} vs "
            f"league_mean {league_mean_vec.shape}"
        )
    c = float(profile_confidence)
    filled = np.where(np.isnan(per_player_vec), league_mean_vec, per_player_vec)
    return (c * filled + (1.0 - c) * league_mean_vec).astype(per_player_vec.dtype)
