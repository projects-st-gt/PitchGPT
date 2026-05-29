"""Pre-game synthetic-AB builder for App B's matchup cells.

A matchup-card cell asks: *"what would Pitcher P naturally do against
Batter B at a reference state (0-0 count, no runners, 0 outs, mid-game)?"*
That question doesn't reference any real AB — there's no observed pitch
history to draw from. We construct a one-row DataFrame that:

- Pins the per-AB state the model needs (count, runners, outs,
  pitcher_throws, batter_stand, ballpark, umpire, catcher, inning, etc.).
- Carries the pitcher and batter IDs so the dataset can look up their
  profiles from the cache.
- Has PAD/zero values for the per-pitch factors at position 0 (type, zone,
  velo, spin_rate, result, spin_axis). Those are *overwritten* by
  ``g_compute(intervention_position=0, intervention_type=None)`` per
  Option C — the rollout samples pitch 0 from the model's last-context-
  token propensity output and proceeds from there.

This builds the input for ONE matchup cell. The matchup-card computer
calls it ``63×`` per game (one per (P, B) pair).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data.preprocess_pitchgpt import (
    HANDEDNESS_MAP,
    INNING_HALF_BOT,
    INNING_HALF_TOP,
    INNING_MAX,
    ROOF_CLOSED,
    ROOF_OPEN,
    SCORE_DIFF_CLIP,
    TEMP_BUCKET_EDGES,
)
from model.pitchgpt_dataset import CATEGORICAL_CTX_COLS, PITCH_FACTOR_COLS_INT


# ============================================================
# Reference context — the matchup card's "neutral starting state"
# ============================================================


@dataclass
class ReferenceContext:
    """Pinned values for the matchup-card cell's starting state.

    Defaults match the brainstorm doc D2 (marginal context): count 0-0, no
    runners, 0 outs, mid-game inning, tied score, neutral weather, day game.
    Override any field for a v2 realistic-context per-cell deep-dive.
    """

    count_balls: int = 0
    count_strikes: int = 0
    runners_on_1b: bool = False
    runners_on_2b: bool = False
    runners_on_3b: bool = False
    outs: int = 0
    pitcher_fatigue_bucket: int = 1  # ~0-9 pitches in (start of appearance)
    inning: int = 5                  # mid-game
    inning_half: str = "Top"         # convention: visiting team batting
    score_diff: int = 0              # tied
    days_rest: int = 4               # well-rested starter
    tto: int = 1                     # first PA between this matchup
    temp_f: float = 72.0
    roof_closed: bool = False


# ============================================================
# Build one synthetic AB
# ============================================================


def _count_state_id(balls: int, strikes: int) -> int:
    """12-state encoding: ``balls (0..3) * 3 + strikes (0..2)``.

    Mirrors :func:`data.preprocess_pitchgpt.compute_count_state` but for one
    scalar — the synthetic AB is a single row.
    """
    b = max(0, min(3, int(balls)))
    s = max(0, min(2, int(strikes)))
    return b * 3 + s


def _runners_state_id(on_1b: bool, on_2b: bool, on_3b: bool) -> int:
    return 4 * int(bool(on_1b)) + 2 * int(bool(on_2b)) + int(bool(on_3b))


def _inning_half_id(half: str) -> int:
    if half == "Top":
        return INNING_HALF_TOP
    if half == "Bot":
        return INNING_HALF_BOT
    return 0  # PAD


def _bucket_inning_one(inning: int) -> int:
    """0=missing; 1..12 verbatim; 13+ collapsed to 13."""
    if inning is None:
        return 0
    return max(1, min(INNING_MAX + 1, int(inning)))


def _bucket_score_diff_one(score_diff: int) -> int:
    sd = max(-SCORE_DIFF_CLIP, min(SCORE_DIFF_CLIP, int(score_diff)))
    return sd + SCORE_DIFF_CLIP  # shift to 0..10


def _bucket_temp_one(temp_f: Optional[float]) -> int:
    if temp_f is None:
        return 0  # missing
    bins_idx = np.digitize([float(temp_f)], TEMP_BUCKET_EDGES)[0]
    return int(bins_idx + 1)  # 1=<40, 2=40-49, ..., 6=80+


def _bucket_roof_one(closed: Optional[bool]) -> int:
    if closed is None:
        return 0  # PAD
    return ROOF_CLOSED if bool(closed) else ROOF_OPEN


def _bucket_days_rest_one(days: Optional[int]) -> int:
    if days is None:
        return 0
    d = max(0, int(days))
    return min(d, 7) + 1  # 0..6 → 1..7; 7+ → 8


def _handedness_id(throws_or_stand: str) -> int:
    return HANDEDNESS_MAP.get(throws_or_stand, 0)


def build_synthetic_ab(
    *,
    pitcher_id: int,
    batter_id: int,
    game_date: str,                            # "YYYY-MM-DD" — drives asof in profiles
    pitcher_throws: str,                       # "R" or "L"
    batter_stand: str,                         # "R" or "L"
    ballpark_id: int = 0,                      # already vocab-mapped; 0 = UNK
    umpire_id: int = 0,
    catcher_id: int = 0,
    context: Optional[ReferenceContext] = None,
    game_pk: int = -1,                         # sentinel; the rollout doesn't use it
    at_bat_number: int = 1,
) -> pd.DataFrame:
    """Construct a one-row DataFrame representing the matchup-card cell's
    starting state.

    The result has every column that the model's dataset requires; per-pitch
    factors at position 0 are PAD/zero (will be overwritten by the rollout).
    Per-AB context (count, runners, outs, inning, etc.) comes from
    ``context``.

    Pass the returned frame to
    ``g_compute(intervention_position=0, intervention_type=None)`` to roll
    out one cell at n_paths Monte Carlo paths.
    """
    if context is None:
        context = ReferenceContext()

    cs = _count_state_id(context.count_balls, context.count_strikes)
    rs = _runners_state_id(
        context.runners_on_1b, context.runners_on_2b, context.runners_on_3b
    )

    # Single row; every column the dataset requires (see REQUIRED_AUG_COLS).
    row: dict = {
        # Identity / metadata
        "game_pk": int(game_pk),
        "at_bat_number": int(at_bat_number),
        "pitch_number": 1,
        "game_date": pd.Timestamp(game_date),
        "pitcher": int(pitcher_id),
        "batter": int(batter_id),
        "description": "synthetic",
        "events": None,

        # Pitch-factor columns at position 0 — all PAD (will be sampled).
        PITCH_FACTOR_COLS_INT["type"]: 0,
        PITCH_FACTOR_COLS_INT["zone"]: 0,
        PITCH_FACTOR_COLS_INT["velo"]: 0,
        PITCH_FACTOR_COLS_INT["spin_rate"]: 0,
        PITCH_FACTOR_COLS_INT["result"]: 0,
        PITCH_FACTOR_COLS_INT["count"]: cs,
        PITCH_FACTOR_COLS_INT["runners"]: rs,
        PITCH_FACTOR_COLS_INT["outs"]: max(0, min(2, int(context.outs))),
        PITCH_FACTOR_COLS_INT["pos"]: 0,
        PITCH_FACTOR_COLS_INT["pitcher_fatigue"]: max(
            0, min(11, int(context.pitcher_fatigue_bucket))
        ),

        # Spin axis at position 0 — zero (will be overwritten by spin_axis_fill
        # default during the rollout).
        "spin_axis_sin": 0.0,
        "spin_axis_cos": 0.0,

        # Categorical context columns the dataset reads at the AB level.
        CATEGORICAL_CTX_COLS["p_throws"]: _handedness_id(pitcher_throws),
        CATEGORICAL_CTX_COLS["stand"]: _handedness_id(batter_stand),
        CATEGORICAL_CTX_COLS["ballpark"]: int(ballpark_id),
        CATEGORICAL_CTX_COLS["umpire"]: int(umpire_id),
        CATEGORICAL_CTX_COLS["catcher"]: int(catcher_id),
        CATEGORICAL_CTX_COLS["inning"]: _bucket_inning_one(context.inning),
        CATEGORICAL_CTX_COLS["score_diff"]: _bucket_score_diff_one(context.score_diff),
        CATEGORICAL_CTX_COLS["inning_half"]: _inning_half_id(context.inning_half),
        CATEGORICAL_CTX_COLS["days_rest"]: _bucket_days_rest_one(context.days_rest),
        CATEGORICAL_CTX_COLS["tto"]: max(1, min(4, int(context.tto))),
        CATEGORICAL_CTX_COLS["temp"]: _bucket_temp_one(context.temp_f),
        CATEGORICAL_CTX_COLS["roof"]: _bucket_roof_one(context.roof_closed),
    }
    return pd.DataFrame([row])
