"""Causal-inference layer.

PitchGPT becomes a causal model in this module. The trained transformer serves
as both the propensity model π̂(a | h) (its next-pitch head) and the conditional
outcome model μ̂(y | a, h) (its two-stage result head). Cross-fitting, positivity
gating, and sensitivity analysis turn it into a defensible estimator.

Module layout (per the ``causal-layer`` skill):

- ``nuisance``: wraps a trained PitchGPT checkpoint as (π̂, μ̂) callables.
- ``positivity``: τ=0.01 gating + ESS tracking + traffic-light states (ADR 002).
- ``g_computation``: single-AB Monte Carlo rollout under do(A_k = a*).
- ``aipw``: population doubly-robust estimator.
- ``crossfit``: K=5 stratified-by-season, blocked-by-game_pk fold construction (ADR 006).
- ``sensitivity``: E-values (VanderWeele & Ding 2017).

**Language discipline** (per the skill): use causal language only when the
machinery is actually running. Outside this module, prefer "model rollout" /
"alternative completion" over "counterfactual" / "if he had thrown".
"""

from causal.nuisance import NuisanceModels  # noqa: F401
from causal.positivity import (  # noqa: F401
    PositivityGate,
    TrustState,
    ess,
    multi_step_check,
)
from causal.sensitivity import e_value_for_continuous_effect  # noqa: F401
from causal.g_computation import RolloutResult, g_compute  # noqa: F401
from causal.aipw import (  # noqa: F401
    AIPWPerUnit,
    AIPWResult,
    aipw_contrast,
    compute_aipw_per_unit,
    compute_aipw_population,
)
from causal.crossfit import (  # noqa: F401
    CrossfitConfig,
    CrossfitNuisance,
    compute_crossfit_per_unit,
    crossfit_aipw_contrast,
    verify_fold_balance,
)
