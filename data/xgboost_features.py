"""Feature engineering for the XGBoost baseline per the ``eval-protocol`` skill.

Produces a numeric feature matrix from harmonized pitches joined with
game-level metadata. The skill's prescribed feature set:

    count, batter handedness, pitcher arsenal frequencies, runners, outs,
    leverage, score_diff, previous pitch type, previous pitch result, TTO,
    days_rest, ballpark, umpire one-hot.

This v1 implements the most-impactful subset:

- count: ``balls``, ``strikes``
- handedness: ``p_throws_L``, ``stand_L``
- game state: ``outs_when_up``, ``on_1b``, ``on_2b``, ``on_3b``, ``inning``
- previous: ``prev_pitch_id`` (with ``-1`` sentinel for AB-start)
- game metadata (from MLB Stats API scrape): ``hp_umpire_id``, ``temp_f``,
  ``roof_closed``, ``wind_speed_mph``, ``venue_id``
- days_rest: per-pitcher day-gap between previous appearance and current game
- identity: ``pitcher_id`` (left raw — model sees this as a categorical-ish
  numeric; pair with the target-encoded arsenal features below for
  meaningful structure), ``batter_id``
- pitcher arsenal target-encoding: 7 columns ``pt_FF_rate, ..., pt_FS_rate``
  computed per-pitcher on **training data only** with Dirichlet smoothing
  toward the league mean. This is the "pitcher arsenal frequencies"
  feature the skill calls for.

Deferred (v2): leverage_index, score_diff, time-through-order. These are
real but each requires non-trivial derivation; the v1 set captures the
biggest signals and gets a real number on the eval table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import N_PITCH_TYPES, PITCH_TYPES, PITCH_TYPE_TO_ID
from data.harmonization import harmonize_dataframe

DEFAULT_META_PATH = Path("data/game_metadata/games.ndjson")
DAYS_REST_NO_PRIOR_SENTINEL: int = 999  # large value for "no prior appearance"
PREV_PITCH_BEGIN: int = -1


def load_game_metadata(path: Path = DEFAULT_META_PATH) -> pd.DataFrame:
    """Read the NDJSON metadata file into a DataFrame keyed by ``game_pk``."""
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    md = pd.DataFrame(rows)
    if "game_pk" not in md.columns:
        raise KeyError("metadata file missing game_pk")
    md["game_pk"] = md["game_pk"].astype(np.int64)
    md = md.drop_duplicates("game_pk", keep="last")
    # Normalize types/columns we depend on.
    for col in ("hp_umpire_id", "fb_umpire_id", "sb_umpire_id", "tb_umpire_id",
                "venue_id", "temp_f", "wind_speed_mph"):
        if col in md.columns:
            md[col] = pd.to_numeric(md[col], errors="coerce")
    if "roof_closed" in md.columns:
        md["roof_closed"] = md["roof_closed"].astype(bool)
    return md


def compute_pitcher_arsenal_encoding(
    train_df: pd.DataFrame,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Per-pitcher arsenal frequencies, Dirichlet-smoothed toward league mean.

    Returns a DataFrame keyed by ``pitcher`` with columns
    ``pt_FF_rate, pt_SI_rate, ..., pt_FS_rate`` (one per canonical type),
    plus ``arsenal_n_pitches`` for diagnostic purposes.

    Computed on training data only (caller's responsibility to slice).
    Use the same encoding at val/test time to avoid target leakage.
    """
    if not {"pitcher", "pitch_type_canonical"}.issubset(train_df.columns):
        raise KeyError("compute_pitcher_arsenal_encoding needs 'pitcher' and 'pitch_type_canonical'")

    # League prior over canonical types.
    league_counts = (
        train_df["pitch_type_canonical"]
        .value_counts()
        .reindex(PITCH_TYPES, fill_value=0)
        .to_numpy(dtype=float)
    )
    league_total = league_counts.sum()
    league_prior = (
        league_counts / league_total if league_total > 0
        else np.full(N_PITCH_TYPES, 1.0 / N_PITCH_TYPES)
    )

    # Per-pitcher counts → smoothed rates.
    counts = (
        train_df.groupby("pitcher")["pitch_type_canonical"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=PITCH_TYPES, fill_value=0)
    )
    pitcher_n = counts.sum(axis=1)
    smoothed = (counts.to_numpy(dtype=float) + alpha * league_prior)
    smoothed /= (pitcher_n.to_numpy(dtype=float)[:, None] + alpha)

    out = pd.DataFrame(
        smoothed,
        index=counts.index,
        columns=[f"pt_{t}_rate" for t in PITCH_TYPES],
    )
    out["arsenal_n_pitches"] = pitcher_n
    out.index.name = "pitcher"
    return out.reset_index()


def _add_prev_pitch_in_ab(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``prev_pitch_id`` column: ID of pitch just before this one in the AB.

    First pitch of an AB gets ``PREV_PITCH_BEGIN = -1``. The DataFrame is
    sorted internally and the original row order is restored before return.
    """
    original_index = df.index
    sorted_df = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).copy()
    grouped = sorted_df.groupby(["game_pk", "at_bat_number"], observed=True)
    prev_type = grouped["pitch_type_canonical"].shift(1)
    prev_id = prev_type.map(PITCH_TYPE_TO_ID).fillna(PREV_PITCH_BEGIN).astype(int)
    sorted_df["prev_pitch_id"] = prev_id
    return sorted_df.loc[original_index]


def _add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``days_rest`` per pitcher: days since their previous appearance."""
    original_index = df.index
    sorted_df = df.sort_values(["pitcher", "game_date"]).copy()
    sorted_df["game_date"] = pd.to_datetime(sorted_df["game_date"])
    # Per-pitcher game-date diffs, evaluated at game-day granularity (not
    # per-pitch — every pitch in a game shares the same rest value).
    game_first = (
        sorted_df.drop_duplicates(["pitcher", "game_date"])
        .sort_values(["pitcher", "game_date"])
    )
    game_first["prev_game_date"] = (
        game_first.groupby("pitcher", observed=True)["game_date"].shift(1)
    )
    game_first["days_rest"] = (
        (game_first["game_date"] - game_first["prev_game_date"]).dt.days
        .fillna(DAYS_REST_NO_PRIOR_SENTINEL).astype(int)
    )
    rest_lookup = game_first.set_index(["pitcher", "game_date"])["days_rest"]
    sorted_df["days_rest"] = sorted_df.set_index(
        ["pitcher", "game_date"]
    ).index.map(rest_lookup)
    sorted_df["days_rest"] = sorted_df["days_rest"].fillna(
        DAYS_REST_NO_PRIOR_SENTINEL
    ).astype(int)
    return sorted_df.loc[original_index]


def build_xgboost_features(
    pitches: pd.DataFrame,
    metadata: pd.DataFrame,
    arsenal_encoding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct the numeric feature matrix for XGBoost.

    Required ``pitches`` columns: ``game_pk``, ``at_bat_number``,
    ``pitch_number``, ``game_date``, ``pitcher``, ``batter``, ``balls``,
    ``strikes``, ``outs_when_up``, ``on_1b``, ``on_2b``, ``on_3b``,
    ``inning``, ``p_throws``, ``stand``, ``pitch_type_canonical``.
    Should already be harmonized (call ``harmonize_dataframe`` first).

    Required ``metadata`` columns: ``game_pk``, ``hp_umpire_id``, ``temp_f``,
    ``roof_closed``, ``wind_speed_mph``, ``venue_id``.

    ``arsenal_encoding`` optional: if provided, joins per-pitcher arsenal
    rates (7 columns) onto the result. Build via
    ``compute_pitcher_arsenal_encoding`` on **training data only**.

    Returns a DataFrame with numeric features only (caller can drop labels
    or pass them separately).
    """
    df = pitches.copy()

    # Handedness as 0/1 (L = 1, else 0).
    df["p_throws_L"] = (df["p_throws"] == "L").astype(np.int8)
    df["stand_L"] = (df["stand"] == "L").astype(np.int8)

    # Runner state from non-null indicators.
    df["on_1b_bool"] = df["on_1b"].notna().astype(np.int8)
    df["on_2b_bool"] = df["on_2b"].notna().astype(np.int8)
    df["on_3b_bool"] = df["on_3b"].notna().astype(np.int8)

    # Within-AB previous pitch.
    df = _add_prev_pitch_in_ab(df)

    # Days rest.
    df = _add_days_rest(df)

    # Join game metadata.
    md_cols = ["game_pk", "hp_umpire_id", "temp_f", "roof_closed",
               "wind_speed_mph", "venue_id"]
    md_subset = metadata[md_cols].copy()
    md_subset["roof_closed"] = md_subset["roof_closed"].astype(np.int8)
    df = df.merge(md_subset, on="game_pk", how="left")

    # Optional pitcher arsenal target encoding (joined on pitcher).
    if arsenal_encoding is not None:
        df = df.merge(arsenal_encoding, on="pitcher", how="left")
        # Pitchers unseen in training → fill arsenal rates with NaN; XGBoost
        # handles NaN natively (treats as missing for split decisions).

    feature_cols = [
        "balls", "strikes",
        "p_throws_L", "stand_L",
        "outs_when_up",
        "on_1b_bool", "on_2b_bool", "on_3b_bool",
        "inning",
        "prev_pitch_id",
        "hp_umpire_id", "temp_f", "roof_closed", "wind_speed_mph", "venue_id",
        "days_rest",
        "pitcher", "batter",
    ]
    if arsenal_encoding is not None:
        feature_cols += [f"pt_{t}_rate" for t in PITCH_TYPES] + ["arsenal_n_pitches"]

    available = [c for c in feature_cols if c in df.columns]
    return df[available].copy()
