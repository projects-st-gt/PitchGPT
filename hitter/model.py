"""HitterModel — load the persisted cascade and serve calibrated predictions.

Loads the per-node artifacts written by ``hitter.train.save_models`` and exposes
the cascade probabilities the count-tree composition (``hitter.compose``) needs:
per pitch, P(swing), P(called-strike | take), P(whiff | swing), the v0 foul
constant, and xwOBA-on-contact. Prediction is reconstructed from the saved
booster + isotonic calibrator via ``train.predict_from_artifacts`` — identical to
train time (round-trip-tested).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hitter.train import ML_NODES, predict_from_artifacts

#: Foul-rate fallback for a count never seen in training (P(foul | contact)).
#: ~0.35 is a reasonable MLB-wide foul-on-contact rate; only hit for absent counts.
_DEFAULT_FOUL_RATE = 0.35


class HitterModel:
    """Loaded hitter/swing cascade. Vectorized, CPU, stateless after load."""

    def __init__(self, model_dir: str = "checkpoints/hitter"):
        self._dir = Path(model_dir)
        if not self._dir.exists():
            raise FileNotFoundError(
                f"hitter model dir {self._dir} missing; run `python -m hitter.train`"
            )
        import joblib

        self._artifacts: dict[str, dict] = {}
        for node in ML_NODES:
            path = self._dir / f"{node}.joblib"
            if path.exists():
                self._artifacts[node] = joblib.load(path)
        if not self._artifacts:
            raise FileNotFoundError(f"no node artifacts found under {self._dir}")

        meta_path = self._dir / "meta.json"
        self._meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self._foul_rates: dict[tuple[int, int], float] = {}
        for key, rate in self._meta.get("foul_rate_by_count", {}).items():
            b, s = key.split(",")
            self._foul_rates[(int(b), int(s))] = float(rate)
        # global mean foul rate as the per-model fallback (better than a constant)
        self._foul_fallback = (
            float(np.mean(list(self._foul_rates.values())))
            if self._foul_rates else _DEFAULT_FOUL_RATE
        )

    @property
    def nodes(self) -> list[str]:
        return list(self._artifacts)

    @property
    def metrics(self) -> dict:
        return self._meta.get("nodes", {})

    def predict_node(self, node: str, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(node) (binary) or xwOBA (contact_quality) per row of X."""
        if node not in self._artifacts:
            raise KeyError(f"node {node!r} not loaded (have {self.nodes})")
        return predict_from_artifacts(self._artifacts[node], X)

    def foul_rate(self, balls: int, strikes: int) -> float:
        """v0 S2b: P(foul | contact) for this count, with mean-foul fallback."""
        return self._foul_rates.get((int(balls), int(strikes)), self._foul_fallback)

    def predict_cascade(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """All node predictions for X at once (the compose.py entry point).

        Returns ``{swing, called_strike, whiff, contact_quality}`` -> arrays.
        """
        return {node: self.predict_node(node, X) for node in self._artifacts}
