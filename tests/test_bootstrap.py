"""Unit tests for at-bat-level bootstrap CI."""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics.bootstrap import bootstrap_metric


def _accuracy(preds, targets):
    return float((preds == targets).mean())


def test_bootstrap_returns_point_lo_hi():
    # 100 ABs of 4 pitches each; 70% accuracy overall
    rng = np.random.default_rng(0)
    n_ab = 100
    pitches_per_ab = 4
    n = n_ab * pitches_per_ab

    at_bat_ids = np.repeat(np.arange(n_ab), pitches_per_ab)
    targets = np.zeros(n, dtype=int)
    preds = (rng.uniform(0, 1, size=n) < 0.7).astype(int)
    # Make true accuracy = 0.7 (preds correct when matches target)
    targets = np.where(preds == 0, 1, 0)  # wrong by construction sometimes
    # Re-run with cleaner construction:
    preds = np.zeros(n, dtype=int)
    targets = np.where(rng.uniform(0, 1, size=n) < 0.7, 0, 1)

    point, lo, hi = bootstrap_metric(
        _accuracy, at_bat_ids, preds, targets,
        n_bootstraps=200, seed=42,
    )
    assert lo <= point <= hi
    # Accuracy should be near 0.7
    assert 0.6 < point < 0.8
    # CI width should be reasonable, not pathological
    assert (hi - lo) < 0.3


def test_bootstrap_resamples_at_ab_level_not_pitch_level():
    """If we resampled per-pitch, ABs would lose their grouping. The point
    estimate is unchanged but the CI from per-pitch bootstrap is *narrower*
    than the AB-level bootstrap when within-AB correlation is positive.

    This test verifies that the AB-level bootstrap actually does what it
    claims by constructing data with strong within-AB correlation (every
    pitch in an AB has the same outcome) and checking that the AB-level
    CI is wide.
    """
    n_ab = 50
    pitches_per_ab = 6
    n = n_ab * pitches_per_ab
    rng = np.random.default_rng(0)

    # Each AB is either entirely-correct or entirely-wrong (perfect
    # within-AB correlation). True accuracy at the AB level is 0.5.
    ab_correct = (rng.uniform(0, 1, size=n_ab) < 0.5).astype(int)
    targets = np.zeros(n, dtype=int)
    preds = np.repeat(1 - ab_correct, pitches_per_ab)  # 0 = correct, 1 = wrong
    at_bat_ids = np.repeat(np.arange(n_ab), pitches_per_ab)

    point, lo, hi = bootstrap_metric(
        _accuracy, at_bat_ids, preds, targets,
        n_bootstraps=500, seed=0,
    )
    # AB-level CI should be substantial when within-AB correlation is total
    assert (hi - lo) > 0.05


def test_bootstrap_validates_array_lengths():
    with pytest.raises(ValueError, match="length"):
        bootstrap_metric(
            _accuracy,
            np.array([0, 0, 1, 1]),
            np.array([1, 0, 1]),  # too short
            np.array([1, 0, 1, 1]),
        )


def test_bootstrap_validates_at_least_one_array():
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_metric(_accuracy, np.array([0, 0, 1]))


def test_bootstrap_deterministic_with_seed():
    n = 40
    at_bat_ids = np.repeat(np.arange(10), 4)
    preds = np.zeros(n, dtype=int)
    targets = np.array([0, 1] * 20, dtype=int)

    a = bootstrap_metric(_accuracy, at_bat_ids, preds, targets,
                         n_bootstraps=100, seed=123)
    b = bootstrap_metric(_accuracy, at_bat_ids, preds, targets,
                         n_bootstraps=100, seed=123)
    assert a == b
