"""AIPW (Augmented Inverse Propensity Weighted) estimator for pitch-level interventions.

The doubly-robust score for the effect of action ``a`` on AB-level run value::

    ψ̂(a) = (1/n) Σ_i { μ̂(a, H_i)  +  𝟙[A_i = a] / π̂(a | H_i) · (Y_i - μ̂(a, H_i)) }

where for unit i (a pitch in our slice):

- ``H_i`` = the AB context + observed history through pitch index ``k-1``.
- ``A_i`` = the *actually observed* pitch type at position ``k``.
- ``Y_i`` = the *observed* AB-level run value (AB-outcome → run-value lookup).
- ``μ̂(a, H_i)`` = the model's expected AB-level run value under intervention ``A_k = a``.
  Computed from the AB-outcome head's distribution at position ``k`` given
  ``intended_actions["type"][k] = a``.
- ``π̂(a | H_i)`` = the propensity head's probability of pitch type ``a`` at
  position ``k`` given the history.

The contrast ``τ̂(a, a') = ψ̂(a) − ψ̂(a')`` is what the demo / writeup reports
(e.g., "average effect of SL vs FF on 0-2 to RHB"). Variance follows the
influence-function estimator: ``Var(τ̂) = (1/n²) Σ_i (φ_i(a) - φ_i(a'))²``.

**Cross-fitting is mandatory** for any reported ψ̂ / τ̂ (CLAUDE.md hard rule #3).
This module computes per-unit terms with a SINGLE-fit nuisance (developer mode);
:mod:`causal.crossfit` does the cross-fit aggregation across K=5 fold models.

**MVP scope** (per the g_computation MVP):
- ``μ̂(a, H_i)`` uses the AB-outcome head at the intervention position dotted
  with ``DEFAULT_AB_RUN_VALUE`` — fast, no rollout needed for population AIPW.
- The full rigorous rollout (``g_compute``) is used for the demo's single-AB
  query path; AIPW uses the cheaper marginal because we're aggregating across
  thousands of units.
- Positivity-violating units (``π̂(a|H) < τ_refuse``) are EXCLUDED from the
  estimate, count surfaced in diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from causal.g_computation import (
    AB_OUTCOME_NAMES,
    DEFAULT_AB_RUN_VALUE,
    TYPE_ID_OFFSET,
)
from causal.nuisance import NuisanceModels, build_single_ab_batch
from causal.positivity import TAU_SINGLE_STEP
from data.dataset import (
    MODEL_PITCH_TYPES_END_IDX,
    MODEL_PITCH_TYPES_START_IDX,
    MODEL_TYPE_ID,
    N_PITCH_TYPES,
    PITCH_TYPE_TO_ID,
    PITCH_TYPES,
)


def _resolve_pitch_type(p: int | str) -> tuple[int, str]:
    if isinstance(p, str):
        if p not in PITCH_TYPE_TO_ID:
            raise ValueError(f"unknown pitch type {p!r}; must be one of {PITCH_TYPES}")
        return PITCH_TYPE_TO_ID[p], p
    p_id = int(p)
    if not 0 <= p_id < N_PITCH_TYPES:
        raise ValueError(f"pitch type id must be in [0, {N_PITCH_TYPES}); got {p_id}")
    return p_id, PITCH_TYPES[p_id]


# ============================================================
# Per-unit nuisance computation
# ============================================================


@dataclass
class AIPWPerUnit:
    """Per-unit AIPW score terms for every candidate intervention.

    Computed once per (AB, intervention_position) pair; the contrast step
    then picks two columns (a, a') and combines the per-unit terms.
    """

    unit_key: tuple                          # (game_pk, at_bat_number)
    intervention_position: int
    a_observed: int                          # 0-indexed type of the actually-thrown pitch at position k
    y_observed: float                        # observed AB-outcome → run value
    ab_outcome_observed: int                 # 0..6
    mu_hat_per_action: np.ndarray            # (N_PITCH_TYPES,) — μ̂(a, H) for each a
    pi_hat_per_action: np.ndarray            # (N_PITCH_TYPES,) — π̂(a | H) for each a

    @property
    def phi_per_action(self) -> np.ndarray:
        """φ_i(a) = μ̂(a, H_i) + 𝟙[A_i = a]/π̂(a|H_i) · (Y_i - μ̂(a, H_i))."""
        phi = self.mu_hat_per_action.copy()
        a_obs = self.a_observed
        if 0 <= a_obs < N_PITCH_TYPES:
            # Only the observed-action component contributes to the IPW correction.
            pi_obs = self.pi_hat_per_action[a_obs]
            if pi_obs > 0:
                phi[a_obs] = self.mu_hat_per_action[a_obs] + (1.0 / pi_obs) * (
                    self.y_observed - self.mu_hat_per_action[a_obs]
                )
        return phi


def _observed_ab_outcome(events_value: object) -> Optional[int]:
    """Map terminal-pitch ``events`` → 0..6 AB-outcome class (None if uncategorizable)."""
    from model.pitchgpt_dataset import classify_ab_outcome, AB_OUTCOME_IGNORE
    cls = classify_ab_outcome(events_value)
    return None if cls == AB_OUTCOME_IGNORE else int(cls)


def compute_aipw_per_unit(
    nuisance: NuisanceModels,
    ab_pitches: pd.DataFrame,
    *,
    intervention_position: int,
    run_value_table: np.ndarray = DEFAULT_AB_RUN_VALUE,
) -> AIPWPerUnit | None:
    """Compute per-unit AIPW terms for ONE AB at a fixed intervention position.

    The forward pass replicates the AB N_PITCH_TYPES times along the batch dim
    and varies ``intended_actions["type"][k]`` across replicates, so one forward
    pass gives μ̂(a, H_i) for every candidate intervention. π̂(a | H_i) is read
    from any replicate's propensity head at the position predicting pitch[k].

    Args:
        nuisance: loaded NuisanceModels.
        ab_pitches: one AB's rows in chronological order.
        intervention_position: pitch index k where we'd intervene. Must be ≥ 1.
        run_value_table: 7-element AB-outcome → expected runs mapping.

    Returns:
        AIPWPerUnit, or None if the AB is unusable (e.g., no terminal-pitch
        ``events``, position out of range, etc.).
    """
    if intervention_position < 1 or intervention_position >= len(ab_pitches):
        return None
    if run_value_table.shape != (7,):
        raise ValueError(f"run_value_table must be (7,); got {run_value_table.shape}")

    last = ab_pitches.iloc[-1]
    ab_outcome_obs = _observed_ab_outcome(last.get("events"))
    if ab_outcome_obs is None:
        return None
    y_obs = float(run_value_table[ab_outcome_obs])

    # Observed action at intervention position. type_id in the parquet is 1-indexed
    # (PAD=0, PITCH_TYPES = 1..7). The model's heads use 0-indexed (0..6).
    obs_type_id_1 = int(ab_pitches.iloc[intervention_position]["type_id"])
    if obs_type_id_1 == 0:
        return None  # PAD or unknown — skip
    a_observed = obs_type_id_1 - TYPE_ID_OFFSET  # to model's 0..6

    # Build batch with N=N_PITCH_TYPES replicates.
    batch = build_single_ab_batch(nuisance, ab_pitches, n_replicates=N_PITCH_TYPES)
    T = batch["pitch_factors"]["type"].shape[1]
    k = intervention_position

    # Vary BOTH the input pitch_factors["type"][k] AND intended_actions["type"][k]
    # across replicates. The trunk reads pitch_factors (its hidden state at
    # position k depends on what the type IS), and the result/AB-outcome head
    # reads the trunk hidden state + intended_actions. To get different μ̂(a, H_i)
    # values for different a, BOTH have to change. (The earlier version mutated
    # only intended_actions, which produced the identical-μ̂-across-actions bug.)
    new_pitch_type = batch["pitch_factors"]["type"].clone()
    new_intended_type = batch["intended_actions"]["type"].clone()
    for a in range(N_PITCH_TYPES):
        # Dataset's type_id is 1..7 (PAD=0); model's argmax indexes 0..7. Our
        # model-side a ∈ [0, 7) maps to dataset type_id = a + 1.
        new_pitch_type[a, k] = a + TYPE_ID_OFFSET
        new_intended_type[a, k] = a + TYPE_ID_OFFSET
    batch["pitch_factors"]["type"] = new_pitch_type
    batch["intended_actions"]["type"] = new_intended_type
    # Zone/velo/spin remain at observed values (same MVP simplification as
    # elsewhere — we intervene on type only).

    out = nuisance.forward(batch)

    # μ̂(a, H_i): AB-outcome distribution at position k for replicate a, dotted with run_value_table.
    ab_probs = out.ab_outcome_probs[:, k, :].numpy().astype(np.float64)  # (N_PITCH_TYPES, 7)
    mu_hat = (ab_probs * run_value_table[None, :]).sum(axis=1)  # (N_PITCH_TYPES,)

    # π̂(a | H_i): propensity at the position predicting pitch k = seq idx NC + (k-1).
    # The propensity head outputs a softmax over the 8-class TYPE vocab —
    # see ``data.dataset.MODEL_PITCH_TYPES_*`` constants and the
    # "Bug-prevention discipline" section in CLAUDE.md for the convention.
    NC = nuisance.model.N_CONTEXT_TOKENS
    pi_full = out.propensity_probs["type"][
        0, NC + (k - 1), MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX
    ].numpy().astype(np.float64)
    pi_full = pi_full / pi_full.sum().clip(min=1e-12)

    return AIPWPerUnit(
        unit_key=(int(ab_pitches.iloc[0]["game_pk"]), int(ab_pitches.iloc[0]["at_bat_number"])),
        intervention_position=k,
        a_observed=a_observed,
        y_observed=y_obs,
        ab_outcome_observed=ab_outcome_obs,
        mu_hat_per_action=mu_hat,
        pi_hat_per_action=pi_full,
    )


# ============================================================
# Aggregation into a contrast
# ============================================================


@dataclass
class AIPWResult:
    """Doubly-robust contrast estimate ``τ̂(a, a') = ψ̂(a) − ψ̂(a')``."""

    intervention_a: int
    intervention_a_prime: int
    intervention_a_name: str
    intervention_a_prime_name: str

    # Sample sizes
    n_units_input: int                      # total units considered
    n_units_kept: int                       # units passing positivity for both a and a'
    n_units_treated_a: int                  # of kept, how many had A_i == a
    n_units_treated_a_prime: int            # similarly for a'
    n_units_positivity_violation: int       # excluded due to π̂ < τ_refuse for a or a'

    # Estimates + SEs
    psi_a: float
    psi_a_prime: float
    tau: float
    se_psi_a: float
    se_psi_a_prime: float
    se_tau: float

    # 95% Wald CIs
    tau_ci_lower: float
    tau_ci_upper: float

    # Health checks
    mean_pi_a: float                        # mean π̂(a | H) across kept units
    mean_pi_a_prime: float

    # Reported metadata
    positivity_threshold: float

    def __repr__(self) -> str:
        return (
            f"AIPWResult({self.intervention_a_name} vs {self.intervention_a_prime_name}: "
            f"τ̂={self.tau:+.4f} ± {1.96 * self.se_tau:.4f}  "
            f"[95% CI {self.tau_ci_lower:+.4f}, {self.tau_ci_upper:+.4f}]  "
            f"n={self.n_units_kept}/{self.n_units_input} kept)"
        )


def aipw_contrast(
    per_unit: list[AIPWPerUnit],
    intervention_a: int | str,
    intervention_a_prime: int | str,
    *,
    positivity_threshold: float = TAU_SINGLE_STEP,
    ci_z: float = 1.96,
) -> AIPWResult:
    """Aggregate per-unit AIPW terms into ``τ̂(a, a')`` with influence-function SE.

    Units violating positivity for either action (``π̂(a|H) < τ`` or
    ``π̂(a'|H) < τ``) are EXCLUDED from the estimator. The number excluded is
    reported in the result's diagnostics — per ADR 002, refusal is a feature.
    """
    a_id, a_name = _resolve_pitch_type(intervention_a)
    ap_id, ap_name = _resolve_pitch_type(intervention_a_prime)

    if not per_unit:
        raise ValueError("per_unit must be non-empty")

    # Filter by positivity for both actions.
    n_input = len(per_unit)
    kept: list[AIPWPerUnit] = []
    n_violation = 0
    for u in per_unit:
        if (
            u.pi_hat_per_action[a_id] < positivity_threshold
            or u.pi_hat_per_action[ap_id] < positivity_threshold
        ):
            n_violation += 1
        else:
            kept.append(u)

    if len(kept) == 0:
        raise RuntimeError(
            f"all {n_input} units violated positivity for at least one of "
            f"{a_name}/{ap_name} (τ_refuse = {positivity_threshold}). "
            f"AIPW estimate undefined — pick a more common intervention pair "
            f"or use the model-rollout view instead."
        )

    phi_a = np.array([u.phi_per_action[a_id] for u in kept], dtype=np.float64)
    phi_ap = np.array([u.phi_per_action[ap_id] for u in kept], dtype=np.float64)
    phi_diff = phi_a - phi_ap

    n = len(kept)
    psi_a = float(phi_a.mean())
    psi_ap = float(phi_ap.mean())
    tau = psi_a - psi_ap

    # Influence-function-based variance (sample variance / n).
    var_psi_a = float(phi_a.var(ddof=1) / n)
    var_psi_ap = float(phi_ap.var(ddof=1) / n)
    var_tau = float(phi_diff.var(ddof=1) / n)

    se_psi_a = float(np.sqrt(var_psi_a))
    se_psi_ap = float(np.sqrt(var_psi_ap))
    se_tau = float(np.sqrt(var_tau))
    ci_lo = tau - ci_z * se_tau
    ci_hi = tau + ci_z * se_tau

    n_a = sum(1 for u in kept if u.a_observed == a_id)
    n_ap = sum(1 for u in kept if u.a_observed == ap_id)
    mean_pi_a = float(np.mean([u.pi_hat_per_action[a_id] for u in kept]))
    mean_pi_ap = float(np.mean([u.pi_hat_per_action[ap_id] for u in kept]))

    return AIPWResult(
        intervention_a=a_id,
        intervention_a_prime=ap_id,
        intervention_a_name=a_name,
        intervention_a_prime_name=ap_name,
        n_units_input=n_input,
        n_units_kept=n,
        n_units_treated_a=n_a,
        n_units_treated_a_prime=n_ap,
        n_units_positivity_violation=n_violation,
        psi_a=psi_a,
        psi_a_prime=psi_ap,
        tau=tau,
        se_psi_a=se_psi_a,
        se_psi_a_prime=se_psi_ap,
        se_tau=se_tau,
        tau_ci_lower=ci_lo,
        tau_ci_upper=ci_hi,
        mean_pi_a=mean_pi_a,
        mean_pi_a_prime=mean_pi_ap,
        positivity_threshold=positivity_threshold,
    )


# ============================================================
# Convenience: compute per-unit AIPW across a population slice
# ============================================================


def compute_aipw_population(
    nuisance: NuisanceModels,
    pitches: pd.DataFrame,
    *,
    intervention_position: int = 1,
    max_units: Optional[int] = None,
    run_value_table: np.ndarray = DEFAULT_AB_RUN_VALUE,
    verbose: bool = False,
) -> list[AIPWPerUnit]:
    """Iterate over ABs in ``pitches``, compute per-unit AIPW terms.

    Args:
        nuisance: NuisanceModels.
        pitches: pitch-level DataFrame (one row per pitch); must have the
            columns ``PitchGPTAtBatDataset`` requires.
        intervention_position: pitch index at which to compute the intervention
            counterfactuals. Fixed across all ABs (MVP). The natural extension
            is "intervene at the position where (count, runners, outs) matches
            a slice definition" — wire as a follow-up.
        max_units: cap on number of ABs processed (for fast iteration).
        run_value_table: AB-outcome → expected runs mapping (default uses
            unconditional means; (base, outs)-conditional is the upgrade path).
        verbose: log every 100 units.

    Returns:
        List of AIPWPerUnit (units that couldn't be processed are skipped).
    """
    df = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    groups = df.groupby(["game_pk", "at_bat_number"], sort=False)

    out: list[AIPWPerUnit] = []
    n_processed = 0
    for key, ab in groups:
        if max_units is not None and len(out) >= max_units:
            break
        if len(ab) < intervention_position + 1:
            continue  # AB shorter than the intervention position
        try:
            term = compute_aipw_per_unit(
                nuisance, ab.reset_index(drop=True),
                intervention_position=intervention_position,
                run_value_table=run_value_table,
            )
        except Exception as exc:  # be permissive; skip ABs that fail
            if verbose:
                print(f"  skipping {key}: {type(exc).__name__}: {exc}")
            continue
        if term is not None:
            out.append(term)
        n_processed += 1
        if verbose and n_processed % 100 == 0:
            print(f"  processed {n_processed} ABs, {len(out)} kept")

    return out
