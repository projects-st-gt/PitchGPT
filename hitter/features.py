"""Per-pitch feature builder for the hitter/swing model.

Assembles the inputs each cascade node sees (see hitter/MODEL_DESIGN.md §5):
base pitch + count + platoon, short-range recent-pitch lags (the sequencing
memory — fed directly, no attention), and the batter-profile vector (the source
of hitter discrimination). Pitcher "stuff" can be joined the same way.

Leakage-safe: the batter profile is looked up as-of the game's
``(game_date, game_num)`` via the leakage-tested ProfileCache (same convention
as ``model/pitchgpt_dataset.py``), with ``as_of_fallback=True`` so future-dated
games get each player's latest known form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate strike zone in feet (plate_x is batter-symmetric; plate_z absolute).
# A coarse in/out-of-zone flag for the swing + called-strike nodes; the pipeline
# has a finer 25-zone scheme if more resolution is wanted later.
_ZONE_HALF_WIDTH = 0.83      # ~ half of a 17" plate + ball radius, in feet
_ZONE_BOT = 1.5
_ZONE_TOP = 3.5

_BASE_NUM_COLS = ["type_id", "plate_x", "plate_z", "release_speed", "balls",
                  "strikes", "pitch_number"]


def add_zone_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``in_zone`` (1/0) from plate_x/plate_z (coarse, fixed zone)."""
    out = df.copy()
    in_x = df["plate_x"].abs() <= _ZONE_HALF_WIDTH
    in_z = df["plate_z"].between(_ZONE_BOT, _ZONE_TOP)
    out["in_zone"] = (in_x & in_z).astype("int8")
    return out


def add_recent_pitch_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-range, within-AB sequencing features (lag-1).

    ``prev_type_id`` (0 = first pitch of the AB), ``prev_in_zone``, and
    ``n_prev_pitches`` (= pitch_number - 1). Within-AB only — never crosses the
    AB boundary, so no leakage of a later AB's content.
    """
    out = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    g = out.groupby(["game_pk", "at_bat_number"], sort=False)
    out["prev_type_id"] = g["type_id"].shift(1).fillna(0).astype("int16")
    if "in_zone" in out.columns:
        out["prev_in_zone"] = g["in_zone"].shift(1).fillna(-1).astype("int8")
    out["n_prev_pitches"] = (out["pitch_number"] - 1).clip(lower=0).astype("int16")

    # Deception lags (2026-06-11, target the whiff under-prediction): how this
    # pitch DIFFERS from the previous one is the mechanism of a swing-and-miss.
    # Conventions (mirrored EXACTLY by hitter.rollout.build_step_features):
    # first pitch / missing prev measurement -> velo_diff 0.0, loc_dist -1.0
    # (sentinel, same as prev_in_zone), same_type vs prev_type_id=0 -> 0.
    out["prev_velo_diff"] = (
        (out["release_speed"] - g["release_speed"].shift(1))
        .fillna(0.0).astype("float32"))
    dx = out["plate_x"] - g["plate_x"].shift(1)
    dz = out["plate_z"] - g["plate_z"].shift(1)
    out["prev_loc_dist"] = (
        np.sqrt(dx ** 2 + dz ** 2).fillna(-1.0).astype("float32"))
    out["same_type_prev"] = (
        (out["type_id"] == out["prev_type_id"]).astype("int8"))
    return out


def attach_batter_profile(
    df: pd.DataFrame,
    batter_cache,
    *,
    prefix: str = "b",
) -> pd.DataFrame:
    """Join the batter profile vector as ``{prefix}0..{prefix}{D-1}`` columns.

    One lookup per (batter, game_pk) — the profile is constant within a game —
    then merged back to pitches. Uses the same as-of convention as
    ``model/pitchgpt_dataset.py`` (asof = game's date + game_num) with
    ``as_of_fallback=True``.
    """
    keys = df[["batter", "game_pk", "game_date"]].copy()
    if "game_num" in df.columns:
        keys["game_num"] = df["game_num"]
    else:
        keys["game_num"] = 1
    uniq = keys.drop_duplicates(["batter", "game_pk"]).reset_index(drop=True)

    vecs = []
    for row in uniq.itertuples(index=False):
        res = batter_cache.lookup(
            int(row.batter), pd.Timestamp(row.game_date), int(row.game_num),
            as_of_fallback=True,
        )
        vecs.append(np.asarray(res["vector"], dtype=np.float32))
    mat = np.vstack(vecs)
    prof = pd.DataFrame(mat, columns=[f"{prefix}{i}" for i in range(mat.shape[1])])
    prof.insert(0, "game_pk", uniq["game_pk"].values)
    prof.insert(0, "batter", uniq["batter"].values)
    return df.merge(prof, on=["batter", "game_pk"], how="left")


def build_base_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Base + zone + lag features (no profile join). Pure pandas, fast, testable.

    Returns a frame with the original rows plus ``in_zone``, ``prev_type_id``,
    ``prev_in_zone``, ``n_prev_pitches``, and a one-hot of platoon
    (``same_hand`` = 1 if stand == p_throws).
    """
    out = add_zone_flag(pitches)
    out = add_recent_pitch_lags(out)
    out["same_hand"] = (out["stand"] == out["p_throws"]).astype("int8")
    return out


# Feature-column groups the trainer will select (profile cols added by
# attach_batter_profile at train time).
BASE_FEATURE_COLS = _BASE_NUM_COLS + ["in_zone", "prev_type_id", "prev_in_zone",
                                      "n_prev_pitches", "same_hand"]

#: Deception lags — opted into by the swing/whiff nodes only (NOT folded into
#: BASE_FEATURE_COLS, so called_strike/contact nodes keep their trained specs
#: and the xwOBA map stays valid without a contact_quality retrain).
DECEPTION_FEATURE_COLS = ["prev_velo_diff", "prev_loc_dist", "same_type_prev"]
