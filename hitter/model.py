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

from hitter.train import NODE_OBJECTIVE, predict_from_artifacts

#: All node names that may have a saved artifact (ML_NODES + the optional
#: contact_outcome multiclass head).
_LOADABLE_NODES = tuple(NODE_OBJECTIVE)


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
        for node in _LOADABLE_NODES:
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
        if not self._foul_rates:
            raise ValueError(
                f"{self._dir}/meta.json has no foul_rate_by_count; the model is "
                "incomplete — retrain (no fabricated fallback is used)."
            )
        # Fallback for a count absent from training = the empirical MEAN of the
        # real per-count foul rates (all 12 counts are populated on real data, so
        # this is a guard, not a fabricated value).
        self._foul_fallback = float(np.mean(list(self._foul_rates.values())))

        # Optional per-count logit shifts for the binary nodes, fitted on
        # 2024H1 by scripts.hitter.calibrate_cascade_counts (the cascade is
        # well calibrated globally but ~1pp off per count exactly where K's
        # are decided). Applied in predict_cascade when X carries
        # balls/strikes. Absent file -> no-op.
        self._count_cal: dict[str, dict[str, float]] = {}
        cal_path = self._dir / "count_calibration.json"
        if cal_path.exists():
            self._count_cal = json.loads(cal_path.read_text()).get("deltas", {})

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

    def predict_cascade(self, X: pd.DataFrame,
                        apply_count_calibration: bool = True
                        ) -> dict[str, np.ndarray]:
        """All node predictions for X at once (the compose.py entry point).

        Returns ``{swing, called_strike, whiff, contact_quality}`` -> arrays.
        Per-count logit shifts (count_calibration.json) are applied to the
        binary nodes when present and X carries balls/strikes.
        """
        out = {node: self.predict_node(node, X) for node in self._artifacts}
        if (apply_count_calibration and self._count_cal
                and "balls" in X.columns and "strikes" in X.columns):
            eps = 1e-6
            balls = X["balls"].to_numpy().astype(int)
            strikes = X["strikes"].to_numpy().astype(int)
            for node, deltas in self._count_cal.items():
                if node not in out or not deltas:
                    continue
                d = np.zeros(len(X))
                for key, val in deltas.items():
                    b, s = (int(v) for v in key.split(","))
                    d[(balls == b) & (strikes == s)] = float(val)
                p = np.clip(out[node].astype(np.float64), eps, 1 - eps)
                out[node] = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + d)))
        return out
