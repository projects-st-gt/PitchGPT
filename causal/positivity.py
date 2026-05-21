"""Positivity gating + ESS tracking (per ADR 002).

The single load-bearing UX innovation in PitchGPT's demo: refusing to estimate
when the data can't support an estimate. Refusal is foregrounded, not hidden.

**Single-step bands** (per ADR 002 — locked 2026-05-09):

| π̂(a* | h)      | gauge   | demo behaviour                                       |
|-----------------|---------|------------------------------------------------------|
| > 0.05          | GREEN   | show point estimate + CI + E-value                   |
| 0.01 – 0.05     | YELLOW  | show estimate with prominent extrapolating banner    |
| < 0.01          | RED     | refuse: "outside the data's support"                 |

**Multi-step rollouts** (per ADR 002): accumulate per-step inverse propensities
into rollout weights, compute ESS at each step. If ESS drops below 30 at any
step k, the rollout has wandered out of support — surface "supported through
step k − 1, beyond that we don't know" instead of producing a number.

This module is pure math/decision logic — no model dependencies. Keep it that
way so the gate is trivial to unit-test.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

# Locked thresholds (ADR 002). Do not change without bumping the ADR.
TAU_SINGLE_STEP: float = 0.01    # type-dimension refuse floor (7-class vocab)
TAU_GREEN: float = 0.05          # type-dimension green floor
ESS_FLOOR_PER_STEP: int = 30

# Separate thresholds for the ZONE dimension. The zone factor has 26 cells
# (vs the type factor's 7), so the baseline marginal per cell is ~1/26 ≈ 0.04.
# Applying the type-grade thresholds to zone is structurally too strict — even
# common cells like middle-middle land around 5-6% π̂, and corner cells around
# 1-3%. Scaled by the cardinality ratio (~26/7 ≈ 3.7), we use ~3-4× more
# permissive thresholds for zone:
TAU_ZONE_REFUSE: float = 0.003
TAU_ZONE_GREEN: float = 0.02


class TrustState(str, enum.Enum):
    """Three-band trust state for the demo's positivity gauge."""

    GREEN = "green"   # confident causal claim
    YELLOW = "yellow"  # extrapolating; show with prominent uncertainty banner
    RED = "red"       # refused; no point estimate; show model rollout only


@dataclass
class GateDecision:
    """One single-step gating decision."""

    state: TrustState
    p_hat: float           # the model's propensity for the intervention
    threshold: float       # τ at the relevant band boundary
    rationale: str         # short human-readable explanation for the demo UI

    @property
    def allow_causal_claim(self) -> bool:
        """True iff a causal point estimate + CI may be reported."""
        return self.state is not TrustState.RED


class PositivityGate:
    """Single-step positivity gate per ADR 002.

    Use ``gate(p_hat)`` for the per-query traffic-light decision. The boundaries
    are fixed at τ_refuse = 0.01 and τ_green = 0.05 by ADR 002; subclass /
    re-instantiate with custom thresholds only after re-running the empirical
    re-calibration (ADR 002 mandates this after each training run).
    """

    def __init__(
        self,
        tau_refuse: float = TAU_SINGLE_STEP,
        tau_green: float = TAU_GREEN,
    ):
        if not 0 < tau_refuse < tau_green < 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 < tau_refuse ({tau_refuse}) "
                f"< tau_green ({tau_green}) < 1; got refuse={tau_refuse}, green={tau_green}"
            )
        self.tau_refuse = float(tau_refuse)
        self.tau_green = float(tau_green)

    def gate(self, p_hat: float) -> GateDecision:
        """Single-step positivity decision for π̂(a* | h) = ``p_hat``."""
        p = float(p_hat)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p_hat must be in [0,1], got {p}")
        if p < self.tau_refuse:
            return GateDecision(
                state=TrustState.RED,
                p_hat=p,
                threshold=self.tau_refuse,
                rationale=(
                    f"refused: π̂(intervention | state) = {p:.4f} < τ = {self.tau_refuse:.2f}; "
                    f"the intervention is too rare in similar situations for a reliable "
                    f"causal estimate. Showing model rollout instead — not a causal claim."
                ),
            )
        if p < self.tau_green:
            return GateDecision(
                state=TrustState.YELLOW,
                p_hat=p,
                threshold=self.tau_refuse,
                rationale=(
                    f"extrapolating: π̂(intervention | state) = {p:.4f}, "
                    f"between τ_refuse = {self.tau_refuse:.2f} and τ_green = {self.tau_green:.2f}. "
                    f"Estimate produced but CI is wide and unmeasured-confounder sensitivity is "
                    f"the dominant uncertainty."
                ),
            )
        return GateDecision(
            state=TrustState.GREEN,
            p_hat=p,
            threshold=self.tau_green,
            rationale=(
                f"in-support: π̂(intervention | state) = {p:.4f} ≥ τ_green = "
                f"{self.tau_green:.2f}; the intervention appears regularly enough in similar "
                f"situations for a causal estimate."
            ),
        )

    def __repr__(self) -> str:
        return (
            f"PositivityGate(tau_refuse={self.tau_refuse:.4f}, "
            f"tau_green={self.tau_green:.4f})"
        )


# ---------- ESS + multi-step rollout checks ----------


def ess(weights: np.ndarray) -> float:
    """Effective sample size of an importance-weighted Monte Carlo set.

    ``ESS = (Σ w_i)² / Σ w_i²``. Equals N when all weights are equal; collapses
    toward 1 when the weight distribution is dominated by a single sample. For
    multi-step rollouts, the weights at step k are ``Π_{t≤k} π̂(a_t | h_t)^{-1}``
    across the N rollout paths.

    Args:
        weights: 1-D array of non-negative importance weights for N samples.

    Returns:
        ESS as a float in [1, N].
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D; got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s == 0:
        return 0.0
    return float(s * s / np.square(w).sum())


@dataclass
class MultiStepCheck:
    """Result of multi-step ESS tracking for an N-path rollout."""

    truncated_at_step: int | None    # None if no truncation; else the step k where ESS dropped
    ess_per_step: list[float]        # ESS[k] for each step k
    n_paths: int                     # how many paths originally
    floor: int                       # the ESS floor used

    @property
    def is_supported(self) -> bool:
        """True iff the rollout never hit the ESS floor."""
        return self.truncated_at_step is None


def multi_step_check(
    log_weights_per_step: np.ndarray,
    floor: int = ESS_FLOOR_PER_STEP,
) -> MultiStepCheck:
    """Check rollout ESS at each step; return the first violating step (if any).

    Args:
        log_weights_per_step: shape ``(n_steps, n_paths)`` — per-step log
            importance weights for each of the N rollout paths. Using log-space
            avoids numerical blow-up of cumulative products.
        floor: ESS floor below which the rollout is declared "out of support."
            Default 30, per ADR 002.

    Returns:
        MultiStepCheck describing whether and where the rollout truncated.
    """
    arr = np.asarray(log_weights_per_step, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"log_weights_per_step must be 2-D (steps, paths); got {arr.shape}")
    n_steps, n_paths = arr.shape

    cum = np.cumsum(arr, axis=0)  # cumulative log-weight at each step
    ess_per_step: list[float] = []
    truncated_at: int | None = None
    for k in range(n_steps):
        # Normalize log-weights at step k by their max for numerical stability
        c = cum[k]
        c_max = c.max()
        w = np.exp(c - c_max)  # in [0, 1], shape (n_paths,)
        ess_k = ess(w)
        ess_per_step.append(ess_k)
        if truncated_at is None and ess_k < floor:
            truncated_at = k

    return MultiStepCheck(
        truncated_at_step=truncated_at,
        ess_per_step=ess_per_step,
        n_paths=n_paths,
        floor=int(floor),
    )
