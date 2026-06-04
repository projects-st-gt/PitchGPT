"""Train the hitter/swing cascade (XGBoost) — see hitter/MODEL_DESIGN.md.

The cascade (per pitch, given the thrown pitch + state):

    S1  swing?            (all pitches)            -> XGBoost binary
    S1b called-strike?    (takes only)             -> XGBoost binary  (ball vs CS)
    S2  whiff?            (swings only)            -> XGBoost binary
    S2b fair?             (count-constant in v0)   -> foul_rate_by_count
    S3  contact quality   (balls in play)          -> XGBoost regression on
                                                      xwOBA-on-contact

Each ML node trains on its own conditional population (no dilution by rows it
never sees), with per-node isotonic calibration and monotonic constraints where
baseball priors are unambiguous. The S3 target is
``estimated_woba_using_speedangle`` (xwOBA-on-contact), joined from
``data/raw/`` because the augmented frames drop it (statcast-pipeline skill).

Public surface:
- ``add_cascade_labels`` / ``node_population`` — leakage-free label + population
  assembly (pure, tested on synthetic data).
- ``foul_rate_by_count`` — the v0 S2b constant.
- ``attach_xwoba_target`` / ``attach_pitcher_profile`` — the data joins.
- ``train_node`` / ``main`` — fit + calibrate + persist to checkpoints/hitter/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hitter import labels as L

# ---------------------------------------------------------------------------
# Pitcher "stuff" feature subset (MODEL_DESIGN.md §5)
# ---------------------------------------------------------------------------
# The hitter conditions on the ACTUAL thrown pitch (plate_x/z, velo, spin from
# the per-pitch row), so the pitcher's *location heatmap* (175 dims) and
# *per-count arsenal* (84 dims, = pitch-selection, PitchGPT's job) are redundant
# noise. We keep only stuff-QUALITY summary: arsenal mix, velo/spin/arm-slot/pfx
# movement per type, and recent form. This is what makes 98 mph != 91 mph.
_PT = ("FF", "SI", "FC", "SL", "CU", "CH", "FS")
PITCHER_STUFF_FEATURES: list[str] = (
    [f"arsenal_{p}" for p in _PT]
    + [f"has_pitch_{p}" for p in _PT]
    + [f"mean_velo_{p}" for p in _PT]
    + [f"mean_spin_{p}" for p in _PT]
    + [f"arm_slot_{p}" for p in _PT]
    + [f"mean_pfx_x_{p}" for p in _PT]
    + [f"mean_pfx_z_{p}" for p in _PT]
    + ["recent_30d_xwoba", "recent_30d_n_pitches",
       "days_since_last_appearance", "profile_confidence"]
)  # 7*7 + 4 = 53 dims


def attach_pitcher_profile(
    df: pd.DataFrame,
    pitcher_cache,
    *,
    prefix: str = "p",
) -> pd.DataFrame:
    """Join the compact pitcher *stuff* vector as ``{prefix}0..`` columns.

    Mirrors ``hitter.features.attach_batter_profile``: one as-of lookup per
    (pitcher, game_pk) (leakage-safe, ``as_of_fallback=True``), then the full
    profile vector is sliced down to ``PITCHER_STUFF_FEATURES`` before merge.
    """
    from data.profile_cache import PITCHER_FEATURE_INDEX

    idx = [PITCHER_FEATURE_INDEX[f] for f in PITCHER_STUFF_FEATURES]
    keys = df[["pitcher", "game_pk", "game_date"]].copy()
    keys["game_num"] = df["game_num"] if "game_num" in df.columns else 1
    uniq = keys.drop_duplicates(["pitcher", "game_pk"]).reset_index(drop=True)

    vecs = []
    for row in uniq.itertuples(index=False):
        res = pitcher_cache.lookup(
            int(row.pitcher), pd.Timestamp(row.game_date), int(row.game_num),
            as_of_fallback=True,
        )
        vecs.append(np.asarray(res["vector"], dtype=np.float32)[idx])
    mat = np.vstack(vecs)
    prof = pd.DataFrame(mat, columns=[f"{prefix}{i}" for i in range(mat.shape[1])])
    prof.insert(0, "game_pk", uniq["game_pk"].values)
    prof.insert(0, "pitcher", uniq["pitcher"].values)
    return df.merge(prof, on=["pitcher", "game_pk"], how="left")

# ---------------------------------------------------------------------------
# Cascade labels + conditional populations (pure, fast, synthetic-tested)
# ---------------------------------------------------------------------------

#: Each ML node and the conditional population it trains on.
ML_NODES = ("swing", "called_strike", "whiff", "contact_quality")


def add_cascade_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-node cascade labels, each NaN where the node doesn't apply.

    - ``swing``         : 1/0 on every pitch.
    - ``called_strike`` : among TAKES, 1 if called strike else 0 (ball). NaN on swings.
    - ``whiff``         : among SWINGS, 1 if whiff else 0 (contact). NaN on takes.
    - ``fair``          : among CONTACT (swing & not whiff), 1 if in play else 0
                          (foul). NaN otherwise.

    NaN-where-inapplicable is deliberate: ``node_population`` filters on it so a
    node never trains on a row outside its population.
    """
    out = df.copy()
    swing = L.is_swing(out["description"]).astype("float32")
    out["swing"] = swing

    is_swing = swing == 1
    # whiff: defined on swings only
    whiff = L.is_whiff_given_swing(out["description"]).astype("float32")
    out["whiff"] = np.where(is_swing, whiff, np.nan).astype("float32")

    # fair: defined on contact (swing & not whiff) only
    is_contact = is_swing & (whiff == 0)
    fair = L.is_fair_given_swing(out["description"]).astype("float32")
    out["fair"] = np.where(is_contact, fair, np.nan).astype("float32")

    # called_strike: defined on takes only
    called = (out["description"] == "called_strike").astype("float32")
    out["called_strike"] = np.where(~is_swing, called, np.nan).astype("float32")
    return out


def node_population(df: pd.DataFrame, node: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(rows, target)`` for ``node``'s conditional population.

    - ``swing``           : all rows; target = swing.
    - ``called_strike``   : takes (swing==0); target = called_strike.
    - ``whiff``           : swings (swing==1); target = whiff.
    - ``contact_quality`` : balls in play (fair==1); target = xwOBA-on-contact.
    """
    if node == "swing":
        mask = df["swing"].notna()
        return df[mask], df.loc[mask, "swing"]
    if node == "called_strike":
        mask = df["called_strike"].notna()
        return df[mask], df.loc[mask, "called_strike"]
    if node == "whiff":
        mask = df["whiff"].notna()
        return df[mask], df.loc[mask, "whiff"]
    if node == "contact_quality":
        mask = (df["fair"] == 1) & df["estimated_woba_using_speedangle"].notna()
        return df[mask], df.loc[mask, "estimated_woba_using_speedangle"]
    raise ValueError(f"unknown node {node!r}")


def foul_rate_by_count(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """v0 S2b: empirical foul rate among contact, by (balls, strikes).

    Among contact (``fair`` defined), foul = ``fair == 0``, in-play = ``fair == 1``.
    Returns ``{(balls, strikes): P(foul | contact)}``.
    """
    contact = df[df["fair"].notna()]
    rates: dict[tuple[int, int], float] = {}
    for (b, s), grp in contact.groupby(["balls", "strikes"]):
        n = len(grp)
        if n:
            rates[(int(b), int(s))] = float((grp["fair"] == 0).mean())
    return rates


def attach_xwoba_target(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Join ``estimated_woba_using_speedangle`` from a raw frame.

    Keyed on (game_pk, at_bat_number, pitch_number) — the augmented frames drop
    the field, so S3's regression target comes from ``data/raw/``.
    """
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    cols = keys + ["estimated_woba_using_speedangle"]
    return df.merge(raw[cols], on=keys, how="left")
