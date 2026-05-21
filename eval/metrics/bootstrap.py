"""Bootstrap CIs with at-bat resampling.

Per the ``eval-protocol`` skill: every metric in the eval table has a 95%
bootstrap CI, and the resampling unit is **at-bats, not pitches** —
within an AB pitches are not independent (count carries over, the
catcher's call sequence carries over, etc.). Treating pitches as IID
under-states the variance.

Default ``n_bootstraps`` is 1000 per the skill. Tables without CIs are
rejected; ``bootstrap_metric`` is the canonical function to call.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_N_BOOTSTRAPS: int = 1000
DEFAULT_CI: float = 0.95
DEFAULT_SEED: int = 0


def bootstrap_metric(
    metric_fn: Callable[..., float],
    at_bat_ids: np.ndarray,
    *per_pitch_arrays: np.ndarray,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Compute (point estimate, lo, hi) for ``metric_fn`` via at-bat resampling.

    Each bootstrap iteration samples at-bats with replacement (so the
    independence unit is the AB), then concatenates all pitches in those
    sampled ABs and calls ``metric_fn(*per_pitch_arrays_sampled)``.

    Args:
        metric_fn: callable taking per-pitch positional arrays. Must
            return a scalar.
        at_bat_ids: shape ``(N,)`` integer array assigning each pitch to
            an at-bat. Pitches sharing an ``at_bat_id`` are kept together
            during resampling.
        *per_pitch_arrays: per-pitch arrays to pass to ``metric_fn``.
            All must have shape ``(N, ...)`` matching ``at_bat_ids``.
        n_bootstraps: number of resamples. 1000 by default per the skill.
        ci: target CI level. Default 95%.
        seed: rng seed for reproducibility.

    Returns:
        ``(point, lo, hi)`` where ``point`` is the metric on the full
        sample and ``(lo, hi)`` is the bootstrap CI.
    """
    if len(per_pitch_arrays) == 0:
        raise ValueError("bootstrap_metric needs at least one per-pitch array")
    n = len(at_bat_ids)
    if any(len(a) != n for a in per_pitch_arrays):
        raise ValueError("all per-pitch arrays must match at_bat_ids length")

    rng = np.random.default_rng(seed)
    point = float(metric_fn(*per_pitch_arrays))

    # Index from at-bat id to the pitch indices belonging to that AB.
    unique_abs, inverse = np.unique(at_bat_ids, return_inverse=True)
    n_abs = len(unique_abs)
    if n_abs == 0:
        return point, float("nan"), float("nan")
    ab_to_indices = [np.where(inverse == i)[0] for i in range(n_abs)]

    boots = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        sampled_ab_idx = rng.integers(0, n_abs, size=n_abs)
        idx = np.concatenate([ab_to_indices[i] for i in sampled_ab_idx])
        sampled = [arr[idx] for arr in per_pitch_arrays]
        boots[b] = float(metric_fn(*sampled))

    alpha = 1.0 - ci
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return point, lo, hi
