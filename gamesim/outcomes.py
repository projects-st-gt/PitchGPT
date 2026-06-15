"""Map Statcast event strings to the 7-class outcome vocab used by matchup cards.

The 7 classes — K, BB, 1B, 2B, 3B, HR, out — are the same vocabulary as the
hitter cascade model and the matchup card cells. This module provides the
canonical mapping from raw Statcast ``events`` strings to those classes, plus
the 25-state base-out encoding constants used by the transition matrix.

Base-out state encoding: 24 cells = 8 base states × 3 out states, plus one
absorbing INNING_OVER state (index 24). Base state is a 3-bit int:
  bit 2 (value 4) = runner on 1st
  bit 1 (value 2) = runner on 2nd
  bit 0 (value 1) = runner on 3rd
Same encoding as ``data/run_value.py:base_state_from_columns``.
"""
from __future__ import annotations

OUTCOME_CLASSES = ("K", "BB", "1B", "2B", "3B", "HR", "out")

EVENT_TO_OUTCOME: dict[str, str] = {
    "strikeout": "K",
    "strikeout_double_play": "K",

    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "BB",
    "catcher_interf": "BB",

    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",

    "field_out": "out",
    "grounded_into_double_play": "out",
    "double_play": "out",
    "force_out": "out",
    "sac_fly": "out",
    "sac_fly_double_play": "out",
    "sac_bunt": "out",
    "sac_bunt_double_play": "out",
    "fielders_choice": "out",
    "fielders_choice_out": "out",
    "triple_play": "out",
}

DROPPED_EVENTS = {
    "field_error",
    "truncated_pa",
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "wild_pitch",
    "passed_ball",
    "balk",
    "other_advance",
    "runner_double_play",
    "ejection",
    "game_advisory",
}

# Base-out state constants
BASES_EMPTY = 0       # 000
ON_3B = 1             # 001
ON_2B = 2             # 010
ON_2B_3B = 3          # 011
ON_1B = 4             # 100
ON_1B_3B = 5          # 101
ON_1B_2B = 6          # 110
BASES_LOADED = 7      # 111

N_BASE_STATES = 8
N_OUT_STATES = 3
INNING_OVER = 24      # absorbing state index

BASE_STATE_LABELS = [
    "___", "__3", "_2_", "_23", "1__", "1_3", "12_", "123",
]


def base_out_index(base_state: int, outs: int) -> int:
    """Flat index into 25-state space: base_state * 3 + outs."""
    return base_state * 3 + outs


def event_to_outcome(event: str) -> str | None:
    """Map a Statcast event string to one of 7 outcome classes.

    Returns None for events that can't be mapped (errors, non-PA events).
    """
    return EVENT_TO_OUTCOME.get(event)
