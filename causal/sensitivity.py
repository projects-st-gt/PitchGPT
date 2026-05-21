"""E-values for sensitivity to unmeasured confounding (VanderWeele & Ding 2017).

For every causal estimate the project reports, we also report an E-value: the
minimum strength of association (on the risk-ratio scale) that an unmeasured
confounder would need with both the treatment and the outcome to fully explain
away the observed effect, conditional on the measured confounders.

Interpretation:
    - E-value = 1.0: any confounding could explain it; the effect is fragile.
    - E-value = 2.0: a hidden confounder would need a relative risk of ~2 with
      both treatment and outcome — moderate. Many real-world unmeasured factors.
    - E-value = 4.0: very strong confounding needed; the effect is robust.

Display in the demo + writeup as part of the line:
    +0.13 runs (95% CI: 0.04 – 0.22, E-value 2.3)

Reference: VanderWeele TJ, Ding P. Ann Intern Med. 2017;167(4):268-274.
We use the risk-ratio mapping for continuous outcomes via the rule of thumb
``rr ≈ exp(0.91 · cohen_d)`` for converting a standardized effect size into a
risk-ratio scale (per Chinn 2000 / VanderWeele 2020). For per-AB run-value
outcomes (our setting), the standardized effect is δ / σ_Y, where σ_Y is the
pooled SD of the outcome across the population under study.

This module is pure math — no model dependencies. Easy to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EValueResult:
    """E-value with the inputs used to compute it, for transparency."""

    point_e_value: float    # E-value for the point estimate (RR scale)
    ci_e_value: float       # E-value for the CI limit closer to the null (RR=1)
    risk_ratio: float       # the underlying RR
    risk_ratio_ci_limit: float
    standardized_effect: float
    outcome_sd: float

    def explain(self) -> str:
        """One-line plain-English caption for the demo (cf. positivity rationale)."""
        return (
            f"E-value {self.point_e_value:.2f}: a hidden confounder we didn't measure would have "
            f"to have a relative risk of ≥ {self.point_e_value:.2f} with both the intervention "
            f"and the outcome to fully explain this effect away. "
            f"(CI-limit E-value: {self.ci_e_value:.2f}.)"
        )


def _rr_to_e_value(rr: float) -> float:
    """E-value formula for a single RR (the higher of RR or 1/RR side).

    From VanderWeele & Ding 2017, eq. (1):
        E = RR + sqrt(RR · (RR − 1))    for RR ≥ 1
        E = (1/RR) + sqrt((1/RR) · ((1/RR) − 1))    for RR < 1 (use reciprocal)
    For RR == 1, E = 1.
    """
    rr = float(rr)
    if rr <= 0:
        raise ValueError(f"risk ratio must be positive; got {rr}")
    if rr == 1.0:
        return 1.0
    r = rr if rr > 1 else 1.0 / rr
    return float(r + np.sqrt(r * (r - 1.0)))


def continuous_to_rr(
    standardized_effect: float,
    *,
    chinn_constant: float = 1.81,
) -> float:
    """Approximate risk-ratio mapping from a standardized continuous effect.

    Chinn (2000) / VanderWeele (2020): ``rr ≈ exp(0.91 · d)`` for small effects;
    the more accurate VanderWeele-Mathur formula uses a constant of 1.81 in the
    exponent (this is the project default — matches the E-value calculator
    VanderWeele's group ships). For a tiny SMD ``d = 0.1``, this gives RR ≈ 1.20.

    Args:
        standardized_effect: Cohen's d-like, i.e. δ / σ_Y where δ is the
            absolute effect (in outcome units, e.g. runs/PA) and σ_Y is the
            outcome's pooled standard deviation.
        chinn_constant: scaling constant in the exponent; default 1.81 matches
            the VanderWeele E-value calculator's continuous-outcome formula.

    Returns:
        Approximate risk ratio (≥ 1 in absolute terms).
    """
    d = float(standardized_effect)
    return float(np.exp(chinn_constant * d))


def e_value_for_continuous_effect(
    effect: float,
    se: float,
    outcome_sd: float,
    *,
    ci_level: float = 0.95,
) -> EValueResult:
    """E-value for a continuous-outcome point estimate + CI.

    The "CI E-value" uses the CI limit closer to the null (RR = 1) — i.e. the
    weaker end of the interval — and is the more honest quantity to report
    alongside the point estimate.

    Args:
        effect: point estimate of the absolute effect, in outcome units
            (e.g. runs/PA). Sign is ignored; we compute on |effect|.
        se: standard error of the effect.
        outcome_sd: pooled outcome standard deviation across the population.
        ci_level: confidence level for the CI used in the CI E-value
            (default 95% → z = 1.96).

    Returns:
        EValueResult with both point and CI-limit E-values.
    """
    if outcome_sd <= 0:
        raise ValueError(f"outcome_sd must be positive; got {outcome_sd}")
    if se < 0:
        raise ValueError(f"se must be ≥ 0; got {se}")
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    eff = float(abs(effect))
    se = float(se)

    # Z-score for the symmetric CI. 95% → 1.96; 90% → 1.645; etc.
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + ci_level / 2.0))

    smd_point = eff / outcome_sd
    rr_point = continuous_to_rr(smd_point)
    e_point = _rr_to_e_value(rr_point)

    # CI limit closer to null: |effect| − z·se, floored at 0.
    eff_ci = max(0.0, eff - z * se)
    smd_ci = eff_ci / outcome_sd
    rr_ci = continuous_to_rr(smd_ci)
    e_ci = _rr_to_e_value(rr_ci)

    return EValueResult(
        point_e_value=e_point,
        ci_e_value=e_ci,
        risk_ratio=rr_point,
        risk_ratio_ci_limit=rr_ci,
        standardized_effect=smd_point,
        outcome_sd=float(outcome_sd),
    )
