"""Count-conditional frequency baseline per the ``eval-protocol`` skill.

For each of the 12 ``(balls, strikes)`` count states, predicts the
empirical pitch-type distribution given that count. The skill flags this
as "surprisingly hard to beat on top-1 accuracy" — pitchers' choices are
strongly count-driven, and the marginal-given-count is a strong floor.

This is the canonical "bag-of-words" / count-only baseline. PitchGPT's
top-1 accuracy advantage over this baseline is the value of conditioning
on identity, history, and zone-by-pitch detail beyond just the count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dataset import N_PITCH_TYPES, PITCH_TYPE_TO_ID
from eval.baselines.marginal import MarginalBaseline


class CountConditionalBaseline:
    """Per-(balls, strikes) empirical pitch-type distribution.

    On unseen count states (rare in MLB but possible in synthetic data),
    falls back to the training-set marginal.
    """

    def __init__(self):
        self.probs_: dict[tuple[int, int], np.ndarray] | None = None
        self.fallback_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "CountConditionalBaseline":
        """Estimate ``P(pitch_type | balls, strikes)`` from ``df``.

        Required columns: ``pitch_type_canonical``, ``balls``, ``strikes``.
        """
        required = {"balls", "strikes", "pitch_type_canonical"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"CountConditionalBaseline.fit missing columns: {sorted(missing)}"
            )

        probs_per_count: dict[tuple[int, int], np.ndarray] = {}
        for (b, s), group in df.groupby(["balls", "strikes"], observed=True):
            counts = group["pitch_type_canonical"].value_counts(normalize=False)
            probs = np.zeros(N_PITCH_TYPES, dtype=np.float64)
            for pt, n in counts.items():
                if pt in PITCH_TYPE_TO_ID:
                    probs[PITCH_TYPE_TO_ID[pt]] = float(n)
            total = probs.sum()
            if total == 0:
                # Degenerate cell — use marginal at predict time.
                continue
            probs_per_count[(int(b), int(s))] = probs / total
        self.probs_ = probs_per_count

        # Fallback for unseen (b, s) cells at predict time.
        self.fallback_ = MarginalBaseline().fit(df).probs_
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row ``(N_PITCH_TYPES,)`` distribution given the count."""
        if self.probs_ is None or self.fallback_ is None:
            raise RuntimeError("CountConditionalBaseline must be fit before predict_proba")
        n = len(df)
        out = np.empty((n, N_PITCH_TYPES), dtype=np.float64)
        balls = df["balls"].astype(int).to_numpy()
        strikes = df["strikes"].astype(int).to_numpy()
        for i in range(n):
            out[i] = self.probs_.get((int(balls[i]), int(strikes[i])), self.fallback_)
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Argmax of per-row predicted distribution."""
        return self.predict_proba(df).argmax(axis=1).astype(np.int64)
