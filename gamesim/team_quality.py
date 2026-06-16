"""Team quality, home-field advantage, and outcome calibration adjustments.

Batters with fewer than PA_THRESHOLD plate appearances don't have enough
data for a reliable profile. Their predicted distributions default toward
league average, which is too generous for bad teams and too harsh for good
ones. We adjust by shifting probability mass based on team offensive quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gamesim.park import TEAM_ABBR

TEAM_QUALITY_PATH = Path("data/run_value/team_quality.json")
BATTER_PROFILE_PATH = Path("data/profiles/batter_fold_0.parquet")

PA_THRESHOLD = 100
PA_VECTOR_INDEX = 41
MAX_SHIFT_FRAC = 0.25

# Home-field advantage: small shift applied to ALL batters (not just thin).
# MLB home teams win ~54% and score ~0.3 more runs/game. A 2% shift on the
# positive-outcome mass translates to roughly that magnitude. Applied in both
# directions: home batters get +HFA_SHIFT, away batters get -HFA_SHIFT.
HFA_SHIFT = 0.02

POSITIVE_OUTCOMES = {"1B", "2B", "3B", "BB", "HR"}
NEGATIVE_OUTCOMES = {"K", "out"}


def load_team_quality() -> dict[str, float]:
    if not TEAM_QUALITY_PATH.exists():
        return {}
    with open(TEAM_QUALITY_PATH) as f:
        data = json.load(f)
    return data.get("factors", {})


def load_thin_profile_ids(threshold: int = PA_THRESHOLD) -> set[int]:
    """Return batter IDs with fewer than `threshold` career PAs in the profile."""
    if not BATTER_PROFILE_PATH.exists():
        return set()
    bp = pd.read_parquet(BATTER_PROFILE_PATH)
    latest = bp.sort_values("asof_date").groupby("player_id").last().reset_index()
    thin = set()
    for _, row in latest.iterrows():
        vec = row["vector"]
        n_pas = vec[PA_VECTOR_INDEX] if not np.isnan(vec[PA_VECTOR_INDEX]) else 0
        if n_pas < threshold:
            thin.add(int(row["player_id"]))
    return thin


def resolve_team_factor(team_name: str, quality_table: dict[str, float]) -> float:
    if team_name in quality_table:
        return quality_table[team_name]
    abbr = TEAM_ABBR.get(team_name)
    if abbr and abbr in quality_table:
        return quality_table[abbr]
    return 1.0


def adjust_dist_for_team_quality(
    dist: dict[str, float],
    team_factor: float,
) -> dict[str, float]:
    """Shift a thin-profile batter's distribution based on team quality.

    Bad teams (factor < 1): move probability from hits/walks toward K/out.
    Good teams (factor > 1): mild shift the other way.
    """
    deviation = team_factor - 1.0
    if abs(deviation) < 0.01:
        return dist

    shift_frac = deviation * MAX_SHIFT_FRAC
    positive_mass = sum(dist.get(o, 0.0) for o in POSITIVE_OUTCOMES)
    negative_mass = sum(dist.get(o, 0.0) for o in NEGATIVE_OUTCOMES)
    shift_amount = abs(shift_frac) * positive_mass

    adjusted = dict(dist)
    if deviation < 0:
        for o in POSITIVE_OUTCOMES:
            if positive_mass > 0:
                adjusted[o] = dist.get(o, 0.0) * (1 - abs(shift_frac))
        for o in NEGATIVE_OUTCOMES:
            if negative_mass > 0:
                adjusted[o] = dist.get(o, 0.0) + shift_amount * (dist.get(o, 0.0) / negative_mass)
    else:
        for o in POSITIVE_OUTCOMES:
            if positive_mass > 0:
                adjusted[o] = dist.get(o, 0.0) + shift_amount * (dist.get(o, 0.0) / positive_mass)
        for o in NEGATIVE_OUTCOMES:
            if negative_mass > 0:
                adjusted[o] = dist.get(o, 0.0) * (1 - shift_frac)

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


def adjust_dist_for_hfa(
    dist: dict[str, float],
    is_home: bool,
) -> dict[str, float]:
    """Apply home-field advantage shift to a batter's outcome distribution.

    Home batters: shift probability toward positive outcomes (hits/walks).
    Away batters: shift probability toward negative outcomes (K/out).
    """
    shift_frac = HFA_SHIFT if is_home else -HFA_SHIFT
    positive_mass = sum(dist.get(o, 0.0) for o in POSITIVE_OUTCOMES)
    negative_mass = sum(dist.get(o, 0.0) for o in NEGATIVE_OUTCOMES)
    shift_amount = abs(shift_frac) * positive_mass

    adjusted = dict(dist)
    if shift_frac < 0:
        for o in POSITIVE_OUTCOMES:
            if positive_mass > 0:
                adjusted[o] = dist.get(o, 0.0) * (1 - abs(shift_frac))
        for o in NEGATIVE_OUTCOMES:
            if negative_mass > 0:
                adjusted[o] = dist.get(o, 0.0) + shift_amount * (dist.get(o, 0.0) / negative_mass)
    else:
        for o in POSITIVE_OUTCOMES:
            if positive_mass > 0:
                adjusted[o] = dist.get(o, 0.0) + shift_amount * (dist.get(o, 0.0) / positive_mass)
        for o in NEGATIVE_OUTCOMES:
            if negative_mass > 0:
                adjusted[o] = dist.get(o, 0.0) * (1 - shift_frac)

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


# ── Outcome recalibration ──

CALIBRATION_PATH = Path("data/run_value/outcome_calibration.json")


def load_calibration_factors() -> dict[str, float]:
    if not CALIBRATION_PATH.exists():
        return {}
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)
    return data.get("factors", {})


def recalibrate_dist(
    dist: dict[str, float],
    factors: dict[str, float],
) -> dict[str, float]:
    """Apply per-outcome calibration ratios and renormalize."""
    if not factors:
        return dist
    adjusted = {k: v * factors.get(k, 1.0) for k, v in dist.items()}
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted
