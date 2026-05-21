"""Marginal frequency baseline per the ``eval-protocol`` skill.

Predicts the empirical pitch-type distribution from training data. Outputs
the same probability vector for every input — no conditioning at all.

This is the floor-for-sanity. If a "real" model can't beat this on top-1
accuracy, something is broken upstream. Per the skill, calibration is the
primary lens; expect the marginal baseline to be reasonably calibrated by
construction (it IS the marginal distribution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dataset import N_PITCH_TYPES, PITCH_TYPES, PITCH_TYPE_TO_ID


class MarginalBaseline:
    """Predicts the constant marginal distribution from training data."""

    def __init__(self):
        self.probs_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "MarginalBaseline":
        """Estimate the marginal pitch-type distribution from ``df``.

        Required column: ``pitch_type_canonical`` (post-harmonization). Pitch
        types not in the canonical 7 are dropped from the count.
        """
        if "pitch_type_canonical" not in df.columns:
            raise KeyError(
                "MarginalBaseline.fit requires a 'pitch_type_canonical' column"
            )
        counts = df["pitch_type_canonical"].value_counts(normalize=False)
        probs = np.zeros(N_PITCH_TYPES, dtype=np.float64)
        for pt, n in counts.items():
            if pt in PITCH_TYPE_TO_ID:
                probs[PITCH_TYPE_TO_ID[pt]] = float(n)
        total = probs.sum()
        if total == 0:
            # Degenerate: no in-vocab pitches. Fall back to uniform so
            # downstream code doesn't divide by zero.
            self.probs_ = np.full(N_PITCH_TYPES, 1.0 / N_PITCH_TYPES, dtype=np.float64)
        else:
            self.probs_ = probs / total
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return ``(len(df), N_PITCH_TYPES)`` of the constant marginal."""
        if self.probs_ is None:
            raise RuntimeError("MarginalBaseline must be fit before predict_proba")
        return np.tile(self.probs_, (len(df), 1))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the most-frequent pitch type for every row."""
        if self.probs_ is None:
            raise RuntimeError("MarginalBaseline must be fit before predict")
        return np.full(len(df), int(self.probs_.argmax()), dtype=np.int64)

    @property
    def most_common_type(self) -> str:
        """Canonical pitch-type string of the marginal mode (e.g., ``'FF'``)."""
        if self.probs_ is None:
            raise RuntimeError("MarginalBaseline must be fit before reading mode")
        return PITCH_TYPES[int(self.probs_.argmax())]
