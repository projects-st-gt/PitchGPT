"""XGBoost baseline per the ``eval-protocol`` skill.

The skill calls this "the bar PitchGPT must clear to justify its existence."
Expected to land within 2-3 points of PitchGPT on top-1 accuracy. The
transformer's edge is calibration, conditional rollouts, and the
recommender — not raw top-1.

Multi-class classification over the 7 canonical pitch types
(``data.dataset.PITCH_TYPES``). Uses ``xgboost.XGBClassifier`` with
``objective='multi:softprob'``. Default hyperparameters are reasonable
for a v1 baseline; tune on val for the final eval-table number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from data.dataset import N_PITCH_TYPES, PITCH_TYPE_TO_ID


class XGBoostBaseline:
    """Multi-class XGBoost over the canonical 7 pitch types.

    Args:
        n_estimators: number of boosting rounds. 200 is reasonable for a
            v1; tune later.
        max_depth: tree depth. 6 is XGBoost's default.
        learning_rate: 0.1 is standard.
        n_jobs: parallel threads. -1 uses all cores.
        random_state: reproducibility.
        early_stopping_rounds: optional; if set and val data is provided
            via ``fit(eval_set=...)``, stops boosting when val log-loss
            doesn't improve for this many rounds.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
        early_stopping_rounds: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self._model: xgb.XGBClassifier | None = None
        self._feature_columns: list[str] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        eval_features: pd.DataFrame | None = None,
        eval_targets: np.ndarray | None = None,
    ) -> "XGBoostBaseline":
        """Train. ``features`` must be numeric (no string columns)."""
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")

        kwargs: dict = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "objective": "multi:softprob",
            "num_class": N_PITCH_TYPES,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "tree_method": "hist",  # fast for large data
        }
        if self.early_stopping_rounds is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = xgb.XGBClassifier(**kwargs)
        self._feature_columns = list(features.columns)

        eval_set = None
        if eval_features is not None and eval_targets is not None:
            eval_set = [(eval_features[self._feature_columns], eval_targets)]

        self._model.fit(features, targets, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("XGBoostBaseline must be fit before predict_proba")
        return self._model.predict_proba(features[self._feature_columns])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("XGBoostBaseline must be fit before predict")
        return self._model.predict(features[self._feature_columns]).astype(np.int64)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Per-feature importance for sanity-checking what XGBoost is using.

        XGBoost returns scores keyed by feature *name* (when fit with a
        DataFrame) or by ``f<index>`` (when fit with a numpy array). Handle
        both, mapping back to ``self._feature_columns``.
        """
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("must fit before feature_importance")
        booster = self._model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        out = pd.Series(0.0, index=self._feature_columns, dtype=float)
        for k, v in scores.items():
            if k in out.index:
                out[k] = float(v)
            elif k.startswith("f") and k[1:].isdigit():
                out.iloc[int(k[1:])] = float(v)
            # else: unknown key, skip
        return out.sort_values(ascending=False)
