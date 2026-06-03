"""Per-game matchup-card computer for MCSim App B.

Ties together the primitives built earlier on this branch into one function:

    (game spec) → (matchup-card payload dict)

For every (pitcher, batter) cell in a game's grid — each of the home staff
against the away lineup, and each of the away staff against the home lineup —
we build a synthetic reference-state AB (:func:`mcsim.state.build_synthetic_ab`)
and roll it out in natural mode (:func:`causal.g_computation.g_compute` with
``intervention_position=0, intervention_type=None``). Each cell reports the
predicted run-value distribution (median + 5/95 band), the model's top-1
AB outcome, the natural pitch-type propensity, and a support/trust flag.

**These cells are PREDICTIVE model rollouts, NOT causal estimates.** There is
no intervention and no AIPW here — natural mode samples what the model expects
the pitcher to do. We borrow :class:`causal.positivity.PositivityGate` only
for its *trust state* (a "is this matchup in-support?" flag driven by how
peaked the model's pitch-type propensity is); we deliberately do NOT surface
the gate's rationale string, whose wording is causal. See the App B brainstorm
doc's "Honest constraints" and the ``causal-layer`` skill's language
discipline. UI copy built on this payload must use predictive language.

This module returns the payload; it does NOT persist. Persisting is the
caller's job via :func:`mcsim.storage.write_prediction` (Step 5, the CLI
runner).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from causal.g_computation import AB_OUTCOME_NAMES, g_compute
from causal.nuisance import NuisanceModels
from causal.positivity import PositivityGate
from data.dataset import PITCH_TYPES
from mcsim.state import ReferenceContext, build_synthetic_ab


# ============================================================
# Cell specs — small value objects for the grid axes
# ============================================================


@dataclass
class PitcherSpec:
    """One arm on a staff. ``throws`` is "R"/"L"; ``is_starter`` flags the
    probable starter (vs. a bullpen arm) for the payload + UI ordering."""

    id: int
    name: str
    throws: str
    is_starter: bool = False


@dataclass
class BatterSpec:
    """One hitter in a lineup. ``stand`` is "R"/"L" (switch hitters resolve to
    the side they'd bat from against the cell's pitcher upstream)."""

    id: int
    name: str
    stand: str


# ============================================================
# Projected slash line
# ============================================================


def _slash_line(outcome_dist: dict) -> tuple[float, float, float]:
    """Projected (OBP, SLG, OPS) from the AB-outcome distribution.

    These are *aggregate* (ratio) stats, computed from the pooled per-PA
    outcome probabilities — NOT per-path like run value (a single PA has no
    well-defined SLG, since its AB denominator is 0 or 1). Each path is one
    plate appearance, so the probabilities sum to one PA:

    - ``OBP = P(reach base) = P(1B)+P(2B)+P(3B)+P(HR)+P(BB)`` (denominator PA=1).
    - ``SLG = total_bases / AB`` where ``AB = 1 - P(BB)`` (K and outs are
      at-bats; walks are not).
    - ``OPS = OBP + SLG``.

    HBP and sacrifice flies are absent from the outcome vocabulary, so this is
    a close approximation of official OBP/OPS (HBP is ~1% of PAs). The model's
    outcome calibration governs the absolute level; relative ordering across
    cells is the immediately useful signal.
    """
    bb = outcome_dist["BB"]
    hits = outcome_dist["1B"] + outcome_dist["2B"] + outcome_dist["3B"] + outcome_dist["HR"]
    obp = hits + bb
    total_bases = (
        outcome_dist["1B"]
        + 2 * outcome_dist["2B"]
        + 3 * outcome_dist["3B"]
        + 4 * outcome_dist["HR"]
    )
    ab = 1.0 - bb  # K and outs are at-bats; walks are not
    slg = total_bases / ab if ab > 0 else 0.0
    return obp, slg, obp + slg


# ============================================================
# One cell
# ============================================================


def _compute_cell(
    nuisance: NuisanceModels,
    *,
    pitcher: PitcherSpec,
    batter: BatterSpec,
    game_date: str,
    game_pk: int,
    ballpark_id: int,
    umpire_id: int,
    catcher_id: int,
    n_paths: int,
    rng_seed: Optional[int],
    context: ReferenceContext,
    gate: PositivityGate,
) -> dict:
    """Roll out one (pitcher, batter) cell and pack its summary dict.

    RV percentiles come from the per-path ``run_value`` array (truncated paths
    are NaN and excluded). The trust flag is driven by the modal natural
    pitch-type propensity, read off the exposed
    ``RolloutResult.intervention_type_propensity``.
    """
    ab = build_synthetic_ab(
        pitcher_id=pitcher.id,
        batter_id=batter.id,
        game_date=game_date,
        pitcher_throws=pitcher.throws,
        batter_stand=batter.stand,
        ballpark_id=ballpark_id,
        umpire_id=umpire_id,
        catcher_id=catcher_id,
        context=context,
        game_pk=game_pk,
    )
    r = g_compute(
        nuisance,
        ab,
        intervention_position=0,
        intervention_type=None,  # natural mode — what the model expects, not do(·)
        n_paths=n_paths,
        rng_seed=rng_seed,
    )

    # RV distribution from the per-path values (truncated paths are NaN).
    finite = r.run_value[np.isfinite(r.run_value)]
    if finite.size > 0:
        p05, p50, p95 = (float(x) for x in np.percentile(finite, [5, 50, 95]))
    else:
        p05 = p50 = p95 = float("nan")

    # Top-1 AB outcome from the outcome distribution (K/BB/1B/.../out).
    top1_idx = int(np.argmax(r.ab_outcome_distribution))
    outcome_dist = {
        name: float(r.ab_outcome_distribution[i])
        for i, name in enumerate(AB_OUTCOME_NAMES)
    }

    # Projected slash line from the same outcome distribution (aggregate stats).
    obp, slg, ops = _slash_line(outcome_dist)

    # Natural pitch-type propensity → modal type + its π̂ → trust flag.
    pi_type = r.intervention_type_propensity
    modal_type_idx = int(np.argmax(pi_type))
    p_hat_top_type = float(pi_type[modal_type_idx])
    trust_state = gate.gate(p_hat_top_type).state.value  # "green"/"yellow"/"red"

    return {
        "batter_id": batter.id,
        "batter_name": batter.name,
        "predicted_rv_median": p50,
        "predicted_rv_p05": p05,
        "predicted_rv_p95": p95,
        "predicted_top1_outcome": AB_OUTCOME_NAMES[top1_idx],
        "predicted_outcome_dist": outcome_dist,
        "predicted_obp": obp,
        "predicted_slg": slg,
        "predicted_ops": ops,
        "modal_type": PITCH_TYPES[modal_type_idx],
        "p_hat_top_type": p_hat_top_type,
        "trust_state": trust_state,
        "n_paths": int(r.n_paths),
        "n_truncated": int(r.n_truncated),
    }


# ============================================================
# One game's full card
# ============================================================


def compute_matchup_card(
    nuisance: NuisanceModels,
    *,
    game_pk: int,
    game_date: str,                          # "YYYY-MM-DD"
    home_team: str,
    away_team: str,
    home_pitchers: list[PitcherSpec],        # starter + bullpen, ordered
    away_pitchers: list[PitcherSpec],
    home_lineup: list[BatterSpec],           # ordered 1-9
    away_lineup: list[BatterSpec],
    ballpark_id: int = 0,
    umpire_id: int = 0,
    catcher_home_id: int = 0,
    catcher_away_id: int = 0,
    n_paths: int = 1000,
    rng_seed: Optional[int] = None,
    context: Optional[ReferenceContext] = None,
) -> dict:
    """Compute one game's matchup card.

    Returns the per-game payload dict (the JSON shape from the App B brainstorm
    doc § Storage Schema). Does NOT persist — the caller writes it via
    :func:`mcsim.storage.write_prediction`.

    Grid: home staff vs. away lineup, and away staff vs. home lineup. The
    catcher behind the plate for a cell is the *pitcher's own* team's catcher
    (``catcher_home_id`` for home pitchers, ``catcher_away_id`` for away ones).

    ``rng_seed``: if given, each cell uses a distinct derived seed
    (``rng_seed + cell_index``) so cells are reproducible but not identically
    correlated. If ``None``, each cell draws fresh entropy.
    """
    if context is None:
        context = ReferenceContext()
    gate = PositivityGate()  # ADR-002 defaults: τ_refuse=0.01, τ_green=0.05

    rows: list[dict] = []
    cell_index = 0

    # (team, pitchers, opposing_lineup, their own catcher) for each half of the grid.
    half_grids = [
        (home_team, home_pitchers, away_lineup, catcher_home_id),
        (away_team, away_pitchers, home_lineup, catcher_away_id),
    ]
    for team, pitchers, lineup, catcher_id in half_grids:
        for pitcher in pitchers:
            cells = []
            for batter in lineup:
                seed = None if rng_seed is None else rng_seed + cell_index
                cells.append(
                    _compute_cell(
                        nuisance,
                        pitcher=pitcher,
                        batter=batter,
                        game_date=game_date,
                        game_pk=game_pk,
                        ballpark_id=ballpark_id,
                        umpire_id=umpire_id,
                        catcher_id=catcher_id,
                        n_paths=n_paths,
                        rng_seed=seed,
                        context=context,
                        gate=gate,
                    )
                )
                cell_index += 1
            rows.append({
                "pitcher_id": pitcher.id,
                "name": pitcher.name,
                "team": team,
                "throws": pitcher.throws,
                "is_starter": pitcher.is_starter,
                "cells": cells,
            })

    def _starter(pitchers: list[PitcherSpec]) -> Optional[dict]:
        for p in pitchers:
            if p.is_starter:
                return {"pitcher_id": p.id, "name": p.name}
        return None

    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "starter_home": _starter(home_pitchers),
        "starter_away": _starter(away_pitchers),
        "rows": rows,
        "n_cells": cell_index,
        "n_paths_per_cell": n_paths,
        "reference_context": {
            "count": f"{context.count_balls}-{context.count_strikes}",
            "runners": "empty" if not (
                context.runners_on_1b or context.runners_on_2b or context.runners_on_3b
            ) else "occupied",
            "outs": context.outs,
            "inning": context.inning,
        },
    }
