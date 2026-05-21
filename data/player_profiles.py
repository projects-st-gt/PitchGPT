"""Player profiles with the strict no-leakage discipline from ADR 003.

Every windowed feature must end strictly before the AB being predicted, where
the ordering is the lexicographic ``(game_date, game_num)`` ordinal. Same-game
prior content is never in any trailing window — the autoregressive sequence
already feeds in-game prior pitches; the windowed features are explicitly
out-of-game.

This module exposes:

- ``before_asof`` — the leakage primitive. Every other windowed feature
  builds on it.

**Pitcher profile features:**

- ``pitcher_arsenal`` — last-N-pitches arsenal composition + per-type velo/spin.
- ``pitcher_zone_heatmap_by_type`` — 9-cell in-zone heatmap per pitch type
  (the 63-vector input to the model when there are 7 canonical types,
  under the v2 Statcast SIS scheme).
- ``pitcher_count_conditional_entropy`` — Shannon entropy of pitch-type
  distribution at each (balls, strikes) state.
- ``pitcher_last_n_days_xwoba`` — primary recent-form xwOBA-against,
  time-based (default 30 days). Works uniformly for starters and relievers.
- ``pitcher_last_n_starts_xwoba`` — start-based recent-form xwOBA-against
  (default N=3). Starter-specific; use for analyses like tipping detection
  (ADR 005).
- ``pitcher_days_since_last_appearance`` — gap (in days) since this
  pitcher's most recent prior appearance. Captures injury layoffs,
  off-season, IL stints. Pairs with the recent-form features so the model
  can contextualize a sparse window.
- ``pitcher_velo_stats`` — per-(pitcher, pitch_type) mean/std for velocity
  binning.

**Batter profile features:**

- ``batter_last_n_days_woba`` — recent-form wOBA (default 14 days).
- ``batter_zone_grids`` — 9-cell in-zone swing%, whiff%, xBA grids.
- ``batter_chase_rate_by_type`` — out-of-zone swing rate per pitch type.
- ``batter_outcome_rates`` — K%, BB%, hard-contact% over trailing PAs.

**League fallback:**

- ``compute_league_velo_means`` — fallback for pitchers below the
  ``min_pitches_for_pitcher`` threshold.

Tests live in ``tests/test_player_profiles.py``; the synthetic-marker test is
designed to fail loudly if any function leaks current-or-later content.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dataset import PITCH_TYPES  # canonical 7-type vocabulary

# ---------- swing / whiff classification (per Statcast `description`) ----------

SWING_DESCRIPTIONS: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "missed_bunt",
    "bunt_foul_tip",
})

# Whiffs are swings that miss the ball entirely. Foul tips count as contact and
# are excluded; this matches Baseball Savant's whiff definition.
WHIFF_DESCRIPTIONS: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
})

# In-zone feature cells under the v2 (Statcast SIS 14-zone) scheme. The
# in-zone is a 3×3 grid (9 cells, internal indices 0..8). The 4 OOZ quadrants
# occupy internal indices 9..12; ``OUT_OF_ZONE_CELL`` is the FIRST OOZ index,
# so "is this pitch OOZ?" is ``feature_zone >= OUT_OF_ZONE_CELL`` (NOT
# equality, as in v1 where OOZ was a single cell). The zone-heatmap / grid
# features only cover the 9 in-zone cells; OOZ chase behaviour is captured
# by ``batter_chase_rate_by_type``.
N_IN_ZONE_CELLS: int = 9
OUT_OF_ZONE_CELL: int = 9

# ---------- leakage primitive ----------


def before_asof(
    df: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    asof_game_num: int,
) -> pd.DataFrame:
    """Return rows with ``(game_date, game_num)`` strictly before the asof ordinal.

    The asof ordinal is ``(asof_game_date, asof_game_num)``. Rows are eligible
    for the trailing window when:

    - ``row.game_date < asof_game_date``, OR
    - ``row.game_date == asof_game_date AND row.game_num < asof_game_num``.

    Same-game (same date AND same game_num) and later content is excluded.
    Doubleheader G1 (same date, lower game_num than G2) is correctly included
    in G2's window.
    """
    asof_date = pd.Timestamp(asof_game_date)
    earlier_date = df["game_date"] < asof_date
    same_date_earlier_num = (df["game_date"] == asof_date) & (df["game_num"] < asof_game_num)
    return df[earlier_date | same_date_earlier_num]


# ---------- pitcher profiles ----------


def pitcher_arsenal(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict:
    """Trailing-window arsenal composition + per-type means.

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof`` and sorted by chronological order. Required
            columns: ``pitch_type_canonical``, ``release_speed``. Optional:
            ``release_spin_rate``, ``spin_axis``.
        window_pitches: how many trailing pitches to use.

    Returns:
        Dict with:

        - ``arsenal_pct``: ``{pitch_type: fraction}``
        - ``mean_velo_by_type``: ``{pitch_type: mean release_speed}``
        - ``mean_spin_by_type``: ``{pitch_type: mean release_spin_rate}`` if available
        - ``n_pitches``: trailing-window size
        - ``profile_confidence``: ``min(n / window_pitches, 1.0)``
    """
    if len(pitcher_pitches) == 0:
        return {
            "arsenal_pct": {},
            "mean_velo_by_type": {},
            "mean_spin_by_type": {},
            "n_pitches": 0,
            "profile_confidence": 0.0,
        }

    window = pitcher_pitches.tail(window_pitches)
    arsenal = window["pitch_type_canonical"].value_counts(normalize=True).to_dict()
    velo = window.groupby("pitch_type_canonical")["release_speed"].mean().to_dict()
    spin = (
        window.groupby("pitch_type_canonical")["release_spin_rate"].mean().to_dict()
        if "release_spin_rate" in window.columns
        else {}
    )
    return {
        "arsenal_pct": arsenal,
        "mean_velo_by_type": velo,
        "mean_spin_by_type": spin,
        "n_pitches": len(window),
        "profile_confidence": min(len(window) / window_pitches, 1.0),
    }


def pitcher_arsenal_by_count(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[tuple[int, int, str], float]:
    """Per-(balls, strikes, pitch_type) usage fraction for a pitcher (v6).

    For each (balls, strikes) cell with at least one observation in the trailing
    window, return the fraction of pitches with each canonical pitch type. The
    7 fractions within a cell sum to 1.0.

    Cells with zero observations produce no entries — the caller writes NaN into
    the cache slot, and the loader's 3-step league-mean fallback fires.

    Cells with observations but where a type was not thrown get 0.0 for that
    type (the pitcher *can* throw it, just didn't in this count).

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof`` and sorted chronologically. Required columns:
            ``pitch_type_canonical``, ``balls``, ``strikes``.
        window_pitches: how many trailing pitches to use.

    Returns:
        ``{(balls, strikes, pitch_type): fraction}`` for cells with at least
        one observation. Each (b, s) appears with all 7 pitch types or none.
    """
    required = {"pitch_type_canonical", "balls", "strikes"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if len(pitcher_pitches) == 0:
        return {}
    window = pitcher_pitches.tail(window_pitches)
    out: dict[tuple[int, int, str], float] = {}
    # observed=True: don't generate phantom empty groups for unseen categorical (b,s) combos.
    grouped = window.groupby(["balls", "strikes"], observed=True)
    for (b, s), group in grouped:
        counts = group["pitch_type_canonical"].value_counts()
        total = float(counts.sum())
        for pt in PITCH_TYPES:
            out[(int(b), int(s), pt)] = float(counts.get(pt, 0)) / total
    return out


def pitcher_arsenal_by_stand(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[tuple[str, str], float]:
    """Per-(batter stand, pitch_type) usage fraction for a pitcher (v6).

    For each batter handedness ('L' or 'R') the pitcher faced in the trailing
    window, return the fraction of pitches with each canonical pitch type.
    Stands with zero observations produce no entries (cache slot stays NaN →
    league-mean fallback).

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``stand``.
        window_pitches: how many trailing pitches to use.

    Returns:
        ``{(stand, pitch_type): fraction}`` for stands with at least one
        observation. Each stand appears with all 7 pitch types or none.
    """
    required = {"pitch_type_canonical", "stand"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if len(pitcher_pitches) == 0:
        return {}
    window = pitcher_pitches.tail(window_pitches)
    out: dict[tuple[str, str], float] = {}
    # observed=True: don't generate phantom empty groups for unseen categorical stands.
    for stand, group in window.groupby("stand", observed=True):
        counts = group["pitch_type_canonical"].value_counts()
        total = float(counts.sum())
        for pt in PITCH_TYPES:
            out[(str(stand), pt)] = float(counts.get(pt, 0)) / total
    return out


def pitcher_movement_by_type(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[str, dict[str, float]]:
    """Per-pitch-type mean horizontal/vertical break for a pitcher (v6).

    For each pitch type the pitcher threw in the trailing window with at least
    one non-NaN ``pfx_x``/``pfx_z`` value, return the (NaN-aware) mean
    horizontal and vertical break. Types with no observations or all-NaN
    movement values are omitted (cache slot stays NaN → league-mean fallback).

    Statcast publishes pfx_x (horizontal break, feet) and pfx_z (vertical
    break, feet) from 2015 onward with near-zero NaN rates on harmonized
    rows. Per the spec, these are the key CU vs SL disambiguator at a given
    arm slot.

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``pfx_x``, ``pfx_z``.
        window_pitches: how many trailing pitches to use.

    Returns:
        ``{pitch_type: {"pfx_x": mean, "pfx_z": mean}}`` for types with at
        least one non-NaN observation in the window.
    """
    required = {"pitch_type_canonical", "pfx_x", "pfx_z"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if len(pitcher_pitches) == 0:
        return {}
    window = pitcher_pitches.tail(window_pitches)
    out: dict[str, dict[str, float]] = {}
    # observed=True: don't generate phantom empty groups for unseen types.
    for pt, group in window.groupby("pitch_type_canonical", observed=True):
        valid = group.dropna(subset=["pfx_x", "pfx_z"])
        if len(valid) == 0:
            continue
        out[str(pt)] = {
            "pfx_x": float(valid["pfx_x"].mean()),
            "pfx_z": float(valid["pfx_z"].mean()),
        }
    return out


def pitcher_last_n_starts_xwoba(
    pitcher_pitches: pd.DataFrame,
    *,
    n_starts: int = 3,
) -> dict:
    """Mean xwOBA-against across this pitcher's last ``n_starts`` completed starts.

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required columns: ``game_pk``, ``game_date``,
            ``game_num``, ``estimated_woba_using_speedangle``.
        n_starts: how many recent starts to average over.

    Returns dict with:

    - ``mean_xwoba``: mean of non-null xwOBA across the recent starts (NaN if empty)
    - ``n_starts``: number of starts actually included (≤ ``n_starts``)
    """
    if len(pitcher_pitches) == 0:
        return {"mean_xwoba": float("nan"), "n_starts": 0}

    start_ordinals = (
        pitcher_pitches.groupby("game_pk")[["game_date", "game_num"]]
        .first()
        .sort_values(["game_date", "game_num"], ascending=False)
    )
    last_starts = start_ordinals.head(n_starts).index.tolist()
    relevant = pitcher_pitches[pitcher_pitches["game_pk"].isin(last_starts)]
    mean = relevant["estimated_woba_using_speedangle"].dropna().mean()
    return {
        "mean_xwoba": float(mean) if pd.notna(mean) else float("nan"),
        "n_starts": len(last_starts),
    }


def pitcher_zone_heatmap_by_type(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[str, np.ndarray]:
    """9-cell normalized heatmap per pitch type, in-zone only (v2 SIS scheme).

    For each canonical ``pitch_type_canonical`` value, returns a 9-vector
    of fractions over the 9 in-zone cells (internal indices 0..8; OOZ
    quadrants 9..12 are excluded). The vector sums to 1.0 over in-zone
    pitches of that type, or to 0 if the pitcher threw no in-zone pitches
    of that type in the window.

    Combined across the 7 canonical pitch types, this yields the 63-vector
    pitcher-location feature the model consumes (per the ``statcast-pipeline``
    skill).

    Args:
        pitcher_pitches: this pitcher's pitches, already filtered via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``feature_zone``.
        window_pitches: how many trailing pitches to use.
    """
    if len(pitcher_pitches) == 0:
        return {}
    required = {"pitch_type_canonical", "feature_zone"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    window = pitcher_pitches.tail(window_pitches)
    # In-zone is internal indices [0, N_IN_ZONE_CELLS); OOZ is [N_IN_ZONE_CELLS, ...).
    in_zone = window[window["feature_zone"] < N_IN_ZONE_CELLS]

    heatmaps: dict[str, np.ndarray] = {}
    for pt, group in in_zone.groupby("pitch_type_canonical", observed=True):
        counts = np.zeros(N_IN_ZONE_CELLS, dtype=float)
        for cell, n in group["feature_zone"].value_counts().items():
            if 0 <= int(cell) < N_IN_ZONE_CELLS:
                counts[int(cell)] = n
        total = counts.sum()
        heatmaps[str(pt)] = counts / total if total > 0 else counts
    return heatmaps


def pitcher_count_conditional_entropy(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[tuple[int, int], float]:
    """Shannon entropy (nats) of pitch-type distribution at each (balls, strikes).

    A pitcher who always throws fastball at 0-2 has entropy 0 in that cell.
    A pitcher who splits 50/50 between two pitch types has entropy ln(2) ≈ 0.693.
    Maximum entropy with 7 canonical types is ln(7) ≈ 1.946.

    Returns ``{(balls, strikes): entropy}``. Cells with no pitches in the
    window are omitted from the dict.

    Args:
        pitcher_pitches: this pitcher's pitches, already filtered via
            ``before_asof``. Required columns: ``balls``, ``strikes``,
            ``pitch_type_canonical``.
        window_pitches: how many trailing pitches to use.
    """
    if len(pitcher_pitches) == 0:
        return {}
    required = {"balls", "strikes", "pitch_type_canonical"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    window = pitcher_pitches.tail(window_pitches)
    out: dict[tuple[int, int], float] = {}
    for (b, s), group in window.groupby(["balls", "strikes"], observed=True):
        probs = group["pitch_type_canonical"].value_counts(normalize=True).to_numpy()
        # Shannon entropy in nats; ignore zero-probability terms.
        nz = probs[probs > 0]
        out[(int(b), int(s))] = float(-(nz * np.log(nz)).sum())
    return out


def pitcher_last_n_days_xwoba(
    pitcher_pitches: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    *,
    days: int = 30,
) -> dict:
    """Mean xwOBA-against over this pitcher's pitches in the trailing ``days`` days.

    Time-based window so the metric works uniformly for starters and relievers
    (a starter's 30 days is ~5–6 starts; a reliever's 30 days is ~20–25
    outings). This is the primary recent-form feature per ADR 003.

    The complementary ``pitcher_last_n_starts_xwoba`` is start-based and is
    appropriate for starter-specific analyses (e.g., tipping detection per
    ADR 005). For the general pitcher recent-form profile feature, prefer
    this function.

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required columns: ``game_date``,
            ``estimated_woba_using_speedangle``.
        asof_game_date: anchor date; window is
            ``[asof_game_date - days, asof_game_date)``.
        days: window length in days.

    Returns dict with:

    - ``mean_xwoba``: mean of non-null xwOBA-against in the window (NaN if empty)
    - ``n_pitches``: number of pitches in the window
    """
    if len(pitcher_pitches) == 0:
        return {"mean_xwoba": float("nan"), "n_pitches": 0}

    if "game_date" not in pitcher_pitches.columns:
        raise KeyError("pitcher_last_n_days_xwoba needs 'game_date' column")
    if "estimated_woba_using_speedangle" not in pitcher_pitches.columns:
        raise KeyError(
            "pitcher_last_n_days_xwoba needs 'estimated_woba_using_speedangle' column"
        )

    asof_date = pd.Timestamp(asof_game_date)
    cutoff = asof_date - pd.Timedelta(days=days)
    in_window = pitcher_pitches[pitcher_pitches["game_date"] >= cutoff]
    if len(in_window) == 0:
        return {"mean_xwoba": float("nan"), "n_pitches": 0}

    mean = in_window["estimated_woba_using_speedangle"].dropna().mean()
    return {
        "mean_xwoba": float(mean) if pd.notna(mean) else float("nan"),
        "n_pitches": int(len(in_window)),
    }


def pitcher_days_since_last_appearance(
    pitcher_pitches: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
) -> int | None:
    """Days between this pitcher's most recent prior appearance and ``asof_game_date``.

    Captures layoffs from injury, IL stints, or off-season. Lets the model
    contextualize a thin recent-form window — a NaN/sparse `mean_xwoba`
    reading paired with `days_since_last_appearance = 75` is meaningfully
    different from the same reading with `days_since_last_appearance = 6`.

    Returns ``None`` when the pitcher has no prior appearances in the data
    (e.g., MLB debut). Caller can encode that as a separate "no prior
    appearance" sentinel or fall back on the league-mean profile per the
    skill's rookies/low-PA guidance.

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required column: ``game_date``.
        asof_game_date: anchor date.
    """
    if len(pitcher_pitches) == 0:
        return None
    if "game_date" not in pitcher_pitches.columns:
        raise KeyError("pitcher_days_since_last_appearance needs 'game_date' column")

    asof_date = pd.Timestamp(asof_game_date)
    most_recent = pd.Timestamp(pitcher_pitches["game_date"].max())
    return int((asof_date - most_recent).days)


def window_freshness(
    pitches: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    *,
    window_pitches: int = 1000,
) -> dict:
    """How stale is the trailing window?

    `profile_confidence` measures *window fullness* (n_pitches / window_pitches);
    this function measures *window freshness*. Stale-but-full profiles are
    the silent-bug case: a player who hasn't pitched/PA'd recently still has
    1000 historical pitches, so confidence ≈ 1, but the data may be months old.

    Cross-season staleness is the canonical example: an April 1 at-bat for a
    starter draws on the previous October's data. The 30-day recent-form
    feature returns NaN there (no in-season pitches yet); these freshness
    signals tell the model the *long* profile is also stale.

    Args:
        pitches: this player's pitches *already filtered* via ``before_asof``.
            Required column: ``game_date``.
        asof_game_date: anchor date.
        window_pitches: same window size used to build the rest of the profile.

    Returns dict with:

    - ``span_days``: days between the oldest pitch in the window and
      ``asof_game_date``. NaN when the window is empty.
    - ``pct_current_season``: fraction of pitches in the window whose
      year equals the asof year. NaN when the window is empty.
    - ``n_pitches_in_window``: equals ``min(len(pitches), window_pitches)``.
    """
    if len(pitches) == 0:
        return {
            "span_days": float("nan"),
            "pct_current_season": float("nan"),
            "n_pitches_in_window": 0,
        }
    if "game_date" not in pitches.columns:
        raise KeyError("window_freshness needs 'game_date' column")

    window = pitches.tail(window_pitches)
    asof_date = pd.Timestamp(asof_game_date)
    oldest = pd.Timestamp(window["game_date"].min())
    span_days = int((asof_date - oldest).days)
    asof_year = asof_date.year
    pct_current = float(
        (pd.to_datetime(window["game_date"]).dt.year == asof_year).mean()
    )
    return {
        "span_days": span_days,
        "pct_current_season": pct_current,
        "n_pitches_in_window": int(len(window)),
    }


def pitcher_velo_stats(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> pd.DataFrame:
    """Per-(pitch_type) trailing mean/std/count of release_speed.

    Args:
        pitcher_pitches: this pitcher's pitches already filtered via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``release_speed``.

    Returns DataFrame with columns ``pitch_type_canonical``, ``mean``, ``std``, ``n``.
    """
    if len(pitcher_pitches) == 0:
        return pd.DataFrame(columns=["pitch_type_canonical", "mean", "std", "n"])

    window = pitcher_pitches.tail(window_pitches)
    return (
        window.groupby("pitch_type_canonical")["release_speed"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )


# ---------- batter profiles ----------


def batter_last_n_days_woba(
    batter_pas: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
    *,
    days: int = 14,
) -> dict:
    """Mean wOBA over the batter's PAs in the trailing ``days`` days.

    Args:
        batter_pas: this batter's PAs (one row per PA, not per pitch),
            already filtered via ``before_asof``. Must include ``game_date``
            and at least one of ``woba_value`` or
            ``estimated_woba_using_speedangle``.
        asof_game_date: anchor date; the trailing window is
            ``[asof_game_date - days, asof_game_date)``.
        days: window length in days.

    Returns dict with ``mean_woba`` and ``n_pas`` (NaN/0 when empty).
    """
    if len(batter_pas) == 0:
        return {"mean_woba": float("nan"), "n_pas": 0}

    asof_date = pd.Timestamp(asof_game_date)
    cutoff = asof_date - pd.Timedelta(days=days)
    in_window = batter_pas[batter_pas["game_date"] >= cutoff]
    if len(in_window) == 0:
        return {"mean_woba": float("nan"), "n_pas": 0}

    woba_col = next(
        (c for c in ("woba_value", "estimated_woba_using_speedangle") if c in in_window.columns),
        None,
    )
    if woba_col is None:
        raise KeyError(
            "batter_last_n_days_woba needs 'woba_value' or "
            "'estimated_woba_using_speedangle' column"
        )

    mean = in_window[woba_col].dropna().mean()
    return {
        "mean_woba": float(mean) if pd.notna(mean) else float("nan"),
        "n_pas": int(len(in_window)),
    }


def batter_zone_grids(
    pitches_seen: pd.DataFrame,
    *,
    xba_col: str = "estimated_ba_using_speedangle",
    window_pitches: int = 1000,
) -> dict[str, np.ndarray]:
    """9-cell swing%, whiff%, and xBA grids for a batter (v2 SIS scheme).

    For each of the 9 in-zone cells (internal indices 0..8), computes:

    - ``swing``: fraction of pitches in this cell that the batter swung at
    - ``whiff``: fraction of *swings* in this cell that missed
    - ``xba``: mean xBA on contact in this cell (NaN where no contact)

    The 4 OOZ quadrants (internal indices 9..12) are intentionally not in
    these grids — chase rate is its own feature (``batter_chase_rate_by_type``).

    Args:
        pitches_seen: pitches *this batter saw*, already filtered via
            ``before_asof``. Required columns: ``description``,
            ``feature_zone``; optional column: the column named by
            ``xba_col`` (Statcast's expected BA on contact).
        xba_col: column to read xBA-on-contact from.
        window_pitches: how many trailing pitches to use.

    Returns dict with keys ``"swing"``, ``"whiff"``, ``"xba"``, each a
    9-vector. NaN-filled if the cell had no relevant pitches.
    """
    swing = np.full(N_IN_ZONE_CELLS, np.nan, dtype=float)
    whiff = np.full(N_IN_ZONE_CELLS, np.nan, dtype=float)
    xba = np.full(N_IN_ZONE_CELLS, np.nan, dtype=float)

    if len(pitches_seen) == 0:
        return {"swing": swing, "whiff": whiff, "xba": xba}

    required = {"description", "feature_zone"}
    missing = required - set(pitches_seen.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    window = pitches_seen.tail(window_pitches).copy()
    window["_swung"] = window["description"].isin(SWING_DESCRIPTIONS)
    window["_whiffed"] = window["description"].isin(WHIFF_DESCRIPTIONS)

    in_zone = window[window["feature_zone"] < N_IN_ZONE_CELLS]
    for cell, group in in_zone.groupby("feature_zone", observed=True):
        c = int(cell)
        if not (0 <= c < N_IN_ZONE_CELLS):
            continue
        n_total = len(group)
        n_swung = int(group["_swung"].sum())
        if n_total > 0:
            swing[c] = n_swung / n_total
        if n_swung > 0:
            whiff[c] = int(group["_whiffed"].sum()) / n_swung
        if xba_col in group.columns:
            contact = group[group["_swung"] & ~group["_whiffed"]]
            if len(contact) > 0 and contact[xba_col].notna().any():
                xba[c] = float(contact[xba_col].dropna().mean())

    return {"swing": swing, "whiff": whiff, "xba": xba}


def batter_chase_rate_by_type(
    pitches_seen: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[str, float]:
    """Per-pitch-type fraction of out-of-zone pitches the batter swung at.

    Args:
        pitches_seen: pitches this batter saw, already filtered via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``feature_zone``, ``description``.
        window_pitches: trailing window.

    Returns ``{pitch_type: chase_rate}``. Pitch types with no out-of-zone
    pitches in the window are omitted.
    """
    if len(pitches_seen) == 0:
        return {}
    required = {"pitch_type_canonical", "feature_zone", "description"}
    missing = required - set(pitches_seen.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    window = pitches_seen.tail(window_pitches).copy()
    window["_swung"] = window["description"].isin(SWING_DESCRIPTIONS)
    # OOZ is the range [N_IN_ZONE_CELLS, ...) — 4 quadrant cells.
    out_of_zone = window[window["feature_zone"] >= N_IN_ZONE_CELLS]

    out: dict[str, float] = {}
    for pt, group in out_of_zone.groupby("pitch_type_canonical", observed=True):
        if len(group) == 0:
            continue
        out[str(pt)] = float(group["_swung"].mean())
    return out


def batter_whiff_and_swing_by_type(
    pitches_seen: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[str, dict[str, float]]:
    """Per-pitch-type swing rate and whiff-on-swing rate for a batter (ADR 013).

    Over the trailing window of pitches this batter *saw*, for each pitch type:

    - ``swing_rate``: fraction of type-X pitches the batter swung at
    - ``whiff_on_swing``: fraction of those swings that missed (a whiff is a
      kind of swing, so this is in [0, 1]); NaN if there were type-X pitches
      but no swings at them.

    Complements ``batter_zone_grids`` (whiff *by zone*, not by type) and
    ``batter_chase_rate_by_type`` (out-of-zone swings, not whiffs). "Can this
    batter hit a slider?" is exactly ``whiff_on_swing["SL"]``.

    Args:
        pitches_seen: pitches this batter saw, already filtered via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``description``.
        window_pitches: trailing window.

    Returns ``{pitch_type: {"swing_rate": float, "whiff_on_swing": float}}``;
    pitch types with no pitches in the window are omitted.
    """
    if len(pitches_seen) == 0:
        return {}
    required = {"pitch_type_canonical", "description"}
    missing = required - set(pitches_seen.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    window = pitches_seen.tail(window_pitches).copy()
    window["_swung"] = window["description"].isin(SWING_DESCRIPTIONS)
    window["_whiffed"] = window["description"].isin(WHIFF_DESCRIPTIONS)
    out: dict[str, dict[str, float]] = {}
    for pt, group in window.groupby("pitch_type_canonical", observed=True):
        n = len(group)
        if n == 0:
            continue
        n_swung = int(group["_swung"].sum())
        out[str(pt)] = {
            "swing_rate": n_swung / n,
            "whiff_on_swing": (
                int(group["_whiffed"].sum()) / n_swung if n_swung > 0 else float("nan")
            ),
        }
    return out


def pitcher_arm_slot_by_type(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[str, float]:
    """Per-pitch-type mean arm slot (``arm_angle`` in degrees) for a pitcher (ADR 014 / Sprint 0b).

    Statcast publishes ``arm_angle`` (release-side arm slot in degrees, with 0
    = horizontal sidearm, positive = over-the-top) from 2020 onward. Pre-2020
    games have NaN; this function silently drops NaN per-type and returns
    only types with at least one valid measurement. The caller pairs these
    with the existing per-type ``has_pitch`` flags so the model can tell
    "pitch type never thrown" from "thrown but slot is unknown."

    Why per-type and not a single mean: pitchers often differ slot by pitch
    type (sweepers come from lower; rising 4-seamers from higher). The within
    -family pitch disambiguation in PitchGPT's μ̂ is exactly the place where
    per-type slot helps. Also load-bearing for the tipping detector (ADR 005).

    Args:
        pitcher_pitches: this pitcher's pitches, already filtered via
            ``before_asof`` and sorted chronologically. Required columns:
            ``pitch_type_canonical``, ``arm_angle``.
        window_pitches: trailing window.

    Returns ``{pitch_type: mean_arm_angle_degrees}``. Types with all-NaN
    arm_angle in the window are omitted.
    """
    if len(pitcher_pitches) == 0:
        return {}
    required = {"pitch_type_canonical", "arm_angle"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    window = pitcher_pitches.tail(window_pitches)
    out: dict[str, float] = {}
    for pt, group in window.groupby("pitch_type_canonical", observed=True):
        valid = group["arm_angle"].dropna()
        if len(valid) == 0:
            continue
        out[str(pt)] = float(valid.mean())
    return out


def matchup_cumulative_pa_stats(
    matchup_pas: pd.DataFrame,
) -> dict[str, float]:
    """Cumulative PA-level stats for one (pitcher, batter) pair (ADR 014 / roadmap A3).

    Counts of K/BB/HR/PAs across all prior PAs in the matchup. The trailing-window
    discipline is enforced by the caller (``before_asof``); this just aggregates.

    Args:
        matchup_pas: terminal pitches for this (pitcher, batter) pair, already
            filtered via ``before_asof`` and to ``events`` not null. Required
            column: ``events``.

    Returns dict with ``n_pas``, ``cum_k``, ``cum_bb``, ``cum_hr``. Zeros if empty.
    """
    if len(matchup_pas) == 0:
        return {"n_pas": 0.0, "cum_k": 0.0, "cum_bb": 0.0, "cum_hr": 0.0}
    if "events" not in matchup_pas.columns:
        raise KeyError("matchup_cumulative_pa_stats needs 'events' column")
    events = matchup_pas["events"]
    return {
        "n_pas": float(len(matchup_pas)),
        "cum_k": float(events.isin(["strikeout", "strikeout_double_play"]).sum()),
        "cum_bb": float(events.isin(["walk", "intent_walk"]).sum()),
        "cum_hr": float(events.eq("home_run").sum()),
    }


def matchup_last_face(
    matchup_pas: pd.DataFrame,
    asof_game_date: pd.Timestamp | str,
) -> dict[str, float]:
    """xwOBA on the last PA + days since last PA, for one (pitcher, batter) pair.

    Args:
        matchup_pas: terminal pitches for this pair, already filtered via
            ``before_asof`` and to ``events`` not null. Required columns:
            ``game_date``, ``estimated_woba_using_speedangle``.
        asof_game_date: anchor date.

    Returns dict with:
      - ``last_face_xwoba``: xwOBA on the most recent prior PA (NaN if missing).
      - ``days_since_last_face``: days between the most recent prior PA and
        ``asof_game_date`` (NaN if no prior PA).
    """
    if len(matchup_pas) == 0:
        return {"last_face_xwoba": float("nan"), "days_since_last_face": float("nan")}
    required = {"game_date", "estimated_woba_using_speedangle"}
    missing = required - set(matchup_pas.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    asof_ts = pd.Timestamp(asof_game_date)
    last_row = matchup_pas.iloc[-1]
    last_xwoba = last_row["estimated_woba_using_speedangle"]
    last_date = pd.Timestamp(last_row["game_date"])
    return {
        "last_face_xwoba": float(last_xwoba) if pd.notna(last_xwoba) else float("nan"),
        "days_since_last_face": float((asof_ts - last_date).days),
    }


def batter_outcome_rates(
    batter_pas: pd.DataFrame,
) -> dict[str, float]:
    """K%, BB%, hard-contact% over the batter's trailing PAs.

    Args:
        batter_pas: this batter's PAs (one row per PA), already filtered via
            ``before_asof``. Required column: ``events``. Hard-contact rate
            requires ``launch_speed`` if available; otherwise NaN.

    Returns dict with keys ``"k_pct"``, ``"bb_pct"``, ``"hard_contact_pct"``,
    ``"n_pas"``. NaN where the relevant denominator is zero.
    """
    n = len(batter_pas)
    if n == 0:
        return {"k_pct": float("nan"), "bb_pct": float("nan"),
                "hard_contact_pct": float("nan"), "n_pas": 0}

    if "events" not in batter_pas.columns:
        raise KeyError("batter_outcome_rates needs 'events' column")
    events = batter_pas["events"]
    n_k = int(events.isin(["strikeout", "strikeout_double_play"]).sum())
    n_bb = int(events.isin(["walk", "intent_walk"]).sum())

    hard_contact_pct = float("nan")
    if "launch_speed" in batter_pas.columns:
        contact = batter_pas["launch_speed"].dropna()
        if len(contact) > 0:
            # Hard contact is the standard Statcast threshold of ≥95 mph exit velocity.
            hard_contact_pct = float((contact >= 95).mean())

    return {
        "k_pct": n_k / n,
        "bb_pct": n_bb / n,
        "hard_contact_pct": hard_contact_pct,
        "n_pas": n,
    }


# ---------- league fallbacks ----------


def compute_league_velo_means(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    """League-wide mean/std of ``release_speed`` per pitch type, on the supplied window.

    The caller is responsible for filtering ``pitches`` via ``before_asof`` so
    the league means do not leak content from the current or later games.

    Returns DataFrame with columns ``pitch_type_canonical``, ``mean``, ``std``, ``n``.
    """
    if len(pitches) == 0:
        return pd.DataFrame(columns=["pitch_type_canonical", "mean", "std", "n"])
    return (
        pitches.groupby("pitch_type_canonical")["release_speed"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
