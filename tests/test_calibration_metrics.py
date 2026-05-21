"""Unit tests for eval/metrics/calibration.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eval.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_diagram,
    top_k_accuracy,
)


def _probs(rows):
    """Helper: list-of-lists → numpy array, also normalized per row."""
    arr = np.asarray(rows, dtype=float)
    return arr / arr.sum(axis=1, keepdims=True)


# ---------- top-k ----------


def test_top_k_perfect_predictor():
    probs = _probs([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    targets = np.array([0, 1, 0])
    assert top_k_accuracy(probs, targets, k=1) == 1.0


def test_top_k_random_for_uniform():
    n_classes = 4
    rng = np.random.default_rng(0)
    probs = np.full((10_000, n_classes), 1.0 / n_classes)
    targets = rng.integers(0, n_classes, size=10_000)
    # Uniform predictions: all classes tied; argmax picks class 0; expected
    # accuracy = 1/n_classes
    acc = top_k_accuracy(probs, targets, k=1)
    assert math.isclose(acc, 1 / n_classes, abs_tol=0.02)


def test_top_k_skips_ignore_index():
    probs = _probs([[0.9, 0.1], [0.1, 0.9]])
    targets = np.array([0, -100])
    # Only first sample counts; predicted correctly → accuracy 1.0
    assert top_k_accuracy(probs, targets, k=1) == 1.0


def test_top_k_returns_nan_when_all_ignored():
    probs = _probs([[0.9, 0.1]])
    targets = np.array([-100])
    assert math.isnan(top_k_accuracy(probs, targets, k=1))


def test_top_3_includes_third_choice():
    # Distinct probabilities so argpartition's top-3 is unambiguous.
    probs = _probs([
        [0.50, 0.30, 0.15, 0.05],   # top-3 = {0, 1, 2}; target 2 is 3rd → in top-3
        [0.40, 0.25, 0.20, 0.15],   # top-3 = {0, 1, 2}; target 1 is 2nd → in top-3
    ])
    targets = np.array([2, 1])
    assert top_k_accuracy(probs, targets, k=3) == 1.0


def test_top_3_excludes_target_below_top_3():
    """Target prob is 4th — should NOT count in top-3."""
    probs = _probs([
        [0.40, 0.25, 0.20, 0.15],   # top-3 = {0, 1, 2}; target 3 is 4th → NOT in top-3
    ])
    targets = np.array([3])
    assert top_k_accuracy(probs, targets, k=3) == 0.0


# ---------- ECE ----------


def test_ece_perfectly_calibrated_predictor():
    """If confidence equals accuracy in every bin, ECE should be near 0."""
    rng = np.random.default_rng(0)
    n = 5000
    probs = rng.uniform(0, 1, size=n)
    targets = (rng.uniform(0, 1, size=n) < probs).astype(int)
    full_probs = np.stack([1 - probs, probs], axis=1)
    ece = expected_calibration_error(full_probs, targets, n_bins=15)
    assert ece < 0.05  # near zero, allowing for finite-sample noise


def test_ece_overconfident_predictor():
    """A predictor that says 0.9 when the truth is 0.5 should have a large ECE."""
    n = 1000
    probs = np.full((n, 2), [0.1, 0.9])
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 2, size=n)  # 50/50 truth
    ece = expected_calibration_error(probs, targets, n_bins=15)
    # Confidence 0.9, accuracy ~0.5 → ECE near 0.4
    assert ece > 0.3


# ---------- reliability diagram ----------


def test_reliability_diagram_returns_n_bins_rows():
    rng = np.random.default_rng(0)
    n = 1000
    probs = rng.dirichlet(np.ones(3), size=n)
    targets = rng.integers(0, 3, size=n)
    df = reliability_diagram(probs, targets, n_bins=10)
    assert len(df) == 10
    assert {"bin_index", "avg_confidence", "avg_accuracy", "count"}.issubset(df.columns)


def test_reliability_diagram_empty_input():
    df = reliability_diagram(np.zeros((0, 3)), np.zeros((0,), dtype=int))
    assert len(df) == 0


# ---------- Brier ----------


def test_brier_zero_for_perfect_one_hot_match():
    probs = _probs([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = np.array([0, 1])
    assert brier_score(probs, targets) == 0.0


def test_brier_two_for_perfect_anti_match():
    """Probability 1.0 on the WRONG class → Brier = 2.0 per sample (max)."""
    probs = _probs([[1.0, 0.0]])
    targets = np.array([1])
    assert brier_score(probs, targets) == 2.0


# ---------- log-loss ----------


def test_log_loss_zero_for_perfect_prediction():
    probs = _probs([[0.99, 0.01]])
    targets = np.array([0])
    assert log_loss(probs, targets) < 0.02


def test_log_loss_clips_extreme_zeros():
    probs = _probs([[0.0, 1.0]])
    targets = np.array([0])  # predicted 0 prob for correct class
    # Without clipping this would be inf; clipped log gives a large finite value
    val = log_loss(probs, targets)
    assert math.isfinite(val)
    assert val > 10  # very bad, but finite
