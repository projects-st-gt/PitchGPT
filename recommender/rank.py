"""Trust-region-restricted causal recommendation — ranking core.

The recommender is, in essence, a ranking loop around ``g_compute`` with a
positivity gate refusing candidates the data can't confidently support.
See ``docs/recommender_brainstorm.md`` for the six design decisions
(D1–D6) and the reuse map.

Pipeline (one call to :func:`rank_pitch_types`):

1. **One forward pass** on the AB to get π̂(type | h) for ALL 7 candidates
   at the intervention position. This shares the trunk cost across the
   candidate set — much cheaper than 7 separate forward passes.
2. **Gate each candidate** on its own π̂. RED → refused list; GREEN /
   YELLOW → in-support list.
3. **Roll out each in-support candidate** with ``g_compute`` to get
   ``mean_run_value`` + ``se_run_value`` (the rollout machinery already
   battle-tested by the /query endpoint).
4. **Rank** in-support candidates by ``mean_run_value`` ascending —
   lower run value = better outcome for the pitcher (Y is measured from
   the batter's perspective).
5. **Tossup detection** (D5): the top-ranked candidate's 95% CI is
   compared against subsequent candidates'; overlapping CIs flag a
   tossup so the UX can say "either is a defensible call."
6. **Optional baseline contrast** (D2): when the observed pitch at the
   intervention position is in the AB, a per-candidate
   ``effect_vs_observed`` annotation is computed (intervention −
   observed) along with the per-candidate E-value.

Decisions made (per the brainstorm doc):

- D1 type-only (zone deferred to v2)
- D2 rank by absolute ``mean_run_value``; ``effect_vs_observed`` as
  secondary annotation
- D3 refused list returned separately (epistemic-humility framing)
- D4 no arsenal pre-filter — positivity gate handles it
- D5 tossup flag on overlapping 95% CIs
- D6 sync execution; pre-computed cache for the demo deferred to a
  separate script
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from causal.g_computation import (
    DEFAULT_AB_RUN_VALUE,
    g_compute,
)
from causal.nuisance import NuisanceModels, build_single_ab_batch
from causal.positivity import (
    PositivityGate,
    TAU_GREEN,
    TAU_SINGLE_STEP,
    TrustState,
)
from causal.sensitivity import e_value_for_continuous_effect
from data.dataset import (
    MODEL_PITCH_TYPES_END_IDX,
    MODEL_PITCH_TYPES_START_IDX,
    MODEL_TYPE_ID,
    PITCH_TYPES,
)

OUTCOME_SD_PROXY = 0.30  # matches the /query handler's E-value rescaling

# 95% CI half-width on a Gaussian; matches downstream ``ci_level=0.95`` defaults
_CI_Z = 1.96


# ============================================================
# Result dataclasses
# ============================================================


@dataclass
class CandidateRanking:
    """One pitch-type candidate's full diagnostic row."""

    pitch_type: str                       # "FF", "SI", "FC", "SL", "CU", "CH", "FS"
    p_hat: float                          # π̂(type | history before intervention pitch)
    trust_state: str                      # "green" | "yellow" | "red"
    rationale: str                        # short human-readable gate explanation

    # Populated when the candidate cleared the gate (green/yellow). For refused
    # (red) candidates these are NaN — the rollout was not run.
    mean_run_value: float = float("nan")  # E[Y | do(type=a)]; lower = better for pitcher
    se_run_value: float = float("nan")
    ci_lower: float = float("nan")        # 95% CI
    ci_upper: float = float("nan")
    n_truncated: int = 0                  # rollout paths that didn't terminate

    # Optional baseline-contrast annotation. Populated when the observed pitch
    # at the intervention position is in the AB AND this candidate cleared the
    # gate. effect = mean_run_value(candidate) − mean_run_value(observed); a
    # *negative* effect means "this pitch would be better than what was thrown."
    effect_vs_observed: Optional[float] = None
    e_value_point: Optional[float] = None
    e_value_ci_limit: Optional[float] = None

    # Set by the ranking loop after sorting (D5). True if this candidate's
    # 95% CI overlaps with the #1 in-support candidate's 95% CI.
    is_tossup: bool = False

    # 0-based rank in the in-support list (None for refused candidates).
    rank: Optional[int] = None

    # Per-candidate AB-outcome distribution at the terminal pitch under
    # do(type=a) — surfaced so a manager-style UI can show P(K), P(HR), etc.
    ab_outcome_dist: Optional[dict[str, float]] = None

    def is_refused(self) -> bool:
        return self.trust_state == "red"


@dataclass
class RankedRecommendations:
    """Full recommender result for one (AB, intervention_position) query."""

    ranked: list[CandidateRanking] = field(default_factory=list)
    refused: list[CandidateRanking] = field(default_factory=list)
    intervention_position: int = 0
    observed_type_at_position: Optional[str] = None
    n_paths: int = 0
    timing_seconds: float = 0.0


# ============================================================
# Forward + propensity extraction
# ============================================================


def _propensity_at_intervention(
    nuisance: NuisanceModels,
    ab_pitches: pd.DataFrame,
    intervention_position: int,
) -> np.ndarray:
    """One forward pass; return π̂(type | history) at the intervention position.

    Mirrors :func:`inference.api._baseline_expected_distribution` so a
    candidate's gate decision uses the same propensity the live /query uses.

    The returned array is length ``N_PITCH_TYPES`` (7), aligned with
    ``data.dataset.PITCH_TYPES`` order. ``MODEL_PITCH_TYPES_START_IDX`` accounts
    for the PAD slot at logit index 0 — see CLAUDE.md's bug-prevention rule
    on the type-vocab convention.
    """
    batch = build_single_ab_batch(nuisance, ab_pitches, n_replicates=1)
    out = nuisance.forward(batch)
    k = intervention_position
    nc = nuisance.model.N_CONTEXT_TOKENS
    pi_type = (
        out.propensity_probs["type"][
            0, nc + (k - 1), MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX
        ]
        .numpy()
        .astype(np.float64)
    )
    # Defensive renorm — temperature scaling preserves the sum, but float drift.
    pi_type = pi_type / max(pi_type.sum(), 1e-12)
    return pi_type


# ============================================================
# Ranking
# ============================================================


def rank_pitch_types(
    nuisance: NuisanceModels,
    ab_pitches: pd.DataFrame,
    *,
    intervention_position: int,
    n_paths: int = 200,
    candidates: Optional[list[str]] = None,
    tau_refuse: float = TAU_SINGLE_STEP,
    tau_green: float = TAU_GREEN,
    rng_seed: Optional[int] = None,
    outcome_sd: float = OUTCOME_SD_PROXY,
    run_value_table: np.ndarray = DEFAULT_AB_RUN_VALUE,
) -> RankedRecommendations:
    """Rank pitch-type candidates by expected run value with positivity gating.

    Args:
        nuisance: loaded ``NuisanceModels`` (calibrated checkpoint).
        ab_pitches: one AB's pitches as a DataFrame (same contract as
            :func:`causal.g_computation.g_compute`).
        intervention_position: 0-indexed pitch position to intervene at
            (must be ≥ 1 — the model doesn't autoregressively predict pitch[0]
            from no history) and < ``len(ab_pitches)``.
        n_paths: Monte Carlo paths per candidate's ``g_compute`` call. 200 is
            the demo-cache default; tests use 30–60 for speed.
        candidates: subset of ``PITCH_TYPES`` to evaluate. Default: all 7. No
            arsenal pre-filter (D4) — the positivity gate is the gatekeeper.
        tau_refuse, tau_green: positivity thresholds (ADR-002 defaults).
        rng_seed: forwarded to ``g_compute`` for deterministic rollouts.
        outcome_sd: ``OUTCOME_SD_PROXY`` for E-value standardization.
        run_value_table: AB-outcome → run-value map.

    Returns:
        A :class:`RankedRecommendations` with:
        - ``ranked``: in-support candidates sorted by ``mean_run_value`` asc
          (lower = better outcome for pitcher); ``is_tossup`` flagged when a
          candidate's 95% CI overlaps the #1 candidate's 95% CI.
        - ``refused``: red-gate candidates, no rollout run.
    """
    t_start = time.perf_counter()
    if intervention_position < 1:
        raise ValueError(
            f"intervention_position must be ≥ 1 (autoregressive constraint); "
            f"got {intervention_position}"
        )
    if intervention_position >= len(ab_pitches):
        raise ValueError(
            f"intervention_position {intervention_position} ≥ AB length "
            f"{len(ab_pitches)}; pick an earlier position"
        )
    candidates = list(candidates) if candidates is not None else list(PITCH_TYPES)
    unknown = [c for c in candidates if c not in MODEL_TYPE_ID]
    if unknown:
        raise ValueError(f"unknown pitch types in candidates: {unknown}")

    # 1. One forward pass → π̂ over all 7 types at the intervention position.
    pi_type = _propensity_at_intervention(nuisance, ab_pitches, intervention_position)
    p_hat_by_type = {pt: float(pi_type[i]) for i, pt in enumerate(PITCH_TYPES)}

    # 2. Gate each candidate; partition into in-support (green/yellow) and refused.
    gate = PositivityGate(tau_refuse=tau_refuse, tau_green=tau_green)
    in_support: list[CandidateRanking] = []
    refused: list[CandidateRanking] = []
    for pt in candidates:
        decision = gate.gate(p_hat_by_type[pt])
        # TrustState enum → lowercase string for JSON-friendly output.
        state = decision.state.name.lower()
        row = CandidateRanking(
            pitch_type=pt,
            p_hat=p_hat_by_type[pt],
            trust_state=state,
            rationale=decision.rationale,
        )
        if decision.state == TrustState.RED:
            refused.append(row)
        else:
            in_support.append(row)

    # 3. Rollout each in-support candidate. The cost is roughly
    #    n_in_support × n_paths × max_steps model forwards; for n_paths=200
    #    and ~6 in-support candidates the wall-clock is ~60 s on CPU.
    observed_type: Optional[str] = None
    try:
        observed_row = ab_pitches.iloc[intervention_position]
        observed_type_id = int(observed_row["type_id"])
        observed_idx = observed_type_id - MODEL_PITCH_TYPES_START_IDX
        if 0 <= observed_idx < len(PITCH_TYPES):
            observed_type = PITCH_TYPES[observed_idx]
    except (KeyError, IndexError, ValueError):
        pass  # observed pitch unknown — skip the vs-observed annotation

    observed_rollout = None
    for row in in_support:
        rollout = g_compute(
            nuisance, ab_pitches,
            intervention_position=intervention_position,
            intervention_type=row.pitch_type,
            n_paths=n_paths,
            rng_seed=rng_seed,
            run_value_table=run_value_table,
        )
        row.mean_run_value = float(rollout.mean_run_value)
        row.se_run_value = float(rollout.se_run_value)
        row.ci_lower = row.mean_run_value - _CI_Z * row.se_run_value
        row.ci_upper = row.mean_run_value + _CI_Z * row.se_run_value
        row.n_truncated = int(rollout.n_truncated)
        row.ab_outcome_dist = {
            "K": float(rollout.ab_outcome_distribution[0]),
            "BB": float(rollout.ab_outcome_distribution[1]),
            "1B": float(rollout.ab_outcome_distribution[2]),
            "2B": float(rollout.ab_outcome_distribution[3]),
            "3B": float(rollout.ab_outcome_distribution[4]),
            "HR": float(rollout.ab_outcome_distribution[5]),
            "out": float(rollout.ab_outcome_distribution[6]),
        }
        if observed_type is not None and row.pitch_type == observed_type:
            observed_rollout = rollout

    # If observed is in-support too (the common case), reuse its rollout for
    # the per-candidate effect/E-value annotations (D2). If observed was
    # refused or not in the candidate set, the annotation is skipped.
    if observed_type is not None and observed_rollout is not None:
        obs_mean = float(observed_rollout.mean_run_value)
        obs_se = float(observed_rollout.se_run_value)
        for row in in_support:
            row.effect_vs_observed = row.mean_run_value - obs_mean
            # Pooled SE for the difference of two independent rollouts.
            se = math.sqrt(row.se_run_value ** 2 + obs_se ** 2)
            ev = e_value_for_continuous_effect(
                effect=row.effect_vs_observed,
                se=se,
                outcome_sd=outcome_sd,
            )
            row.e_value_point = float(ev.point_e_value)
            row.e_value_ci_limit = float(ev.ci_e_value)

    # 4. Sort ascending — lowest expected run value = best for pitcher.
    in_support.sort(key=lambda r: r.mean_run_value)
    for i, row in enumerate(in_support):
        row.rank = i

    # 5. Tossup detection (D5). For each non-#1 candidate, check whether its
    # 95% CI overlaps the #1 candidate's 95% CI; if so, flag both as tossup.
    if len(in_support) >= 2:
        top = in_support[0]
        for row in in_support[1:]:
            if row.ci_lower <= top.ci_upper:
                row.is_tossup = True
                top.is_tossup = True

    return RankedRecommendations(
        ranked=in_support,
        refused=refused,
        intervention_position=intervention_position,
        observed_type_at_position=observed_type,
        n_paths=n_paths,
        timing_seconds=time.perf_counter() - t_start,
    )
