"""Per-pitch label extraction for the hitter/swing model.

Decomposes each pitch into the plate-appearance physics nodes from
docs/Hitter_Swing_Model.md, derived from Statcast ``description`` (+ ``events``
for the in-play outcome). Each node is only defined on the rows that reach it
(swing-node on all pitches; whiff-node on swings; contact-outcome on balls in
play) — callers filter by the ``*_applies`` masks.
"""
from __future__ import annotations

import pandas as pd

# Statcast `description` groupings.
_SWING_DESCRS = {
    "foul", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
_WHIFF_DESCRS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
_FAIR_CONTACT_DESCRS = {"hit_into_play"}  # foul/foul_tip are contact but not fair


def is_swing(description: pd.Series) -> pd.Series:
    """1 if the batter swung (any swing, incl. fouls/whiffs), else 0."""
    return description.isin(_SWING_DESCRS).astype("int8")


def is_whiff_given_swing(description: pd.Series) -> pd.Series:
    """Among swings: 1 if whiff (missed), 0 if contact (foul or fair)."""
    return description.isin(_WHIFF_DESCRS).astype("int8")


def is_fair_given_swing(description: pd.Series) -> pd.Series:
    """Among swings: 1 if fair contact (ball in play), 0 otherwise (foul/whiff)."""
    return description.isin(_FAIR_CONTACT_DESCRS).astype("int8")


# In-play (fair-contact) outcome classes, from `events`.
_CONTACT_OUTCOMES = ["1B", "2B", "3B", "HR", "in_play_out"]
_EVENT_TO_CONTACT = {
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
}


def contact_outcome(events: pd.Series) -> pd.Series:
    """For balls in play, map `events` to {1B,2B,3B,HR,in_play_out}.

    Anything that isn't a clean hit type (field_out, force_out, GIDP, sac_fly,
    field_error, fielders_choice, …) collapses to in_play_out — the pitcher
    controls launch quality, not whether it found a glove (xwOBA refinement is a
    later upgrade; see the brainstorm).
    """
    return events.map(_EVENT_TO_CONTACT).fillna("in_play_out")
