"""Per-pitcher n-gram baseline with Dirichlet smoothing per the
``eval-protocol`` skill — the strongest non-neural baseline.

For each pitcher *p* and each context ``c = (balls, strikes, last_n_pitches_in_AB)``,
estimate ``P(pitch_type | p, c)``. Smooth toward the league-conditional
prior at the same ``c`` with a Dirichlet concentration parameter ``alpha``.

This bakes in the most powerful non-sequential signal in baseball:
**pitcher identity**. Two pitchers in the same 0-2 count throw very
different mixes (Sale's slider vs. Verlander's slider vs. Tarik Skubal's
splitter), and just knowing who's pitching is worth a big jump in top-1
accuracy over count-only baselines.

The first pitch of an AB has no prior pitch; we use the sentinel ``"BEGIN"``
for ``prev_k`` slots when those positions are before the AB started.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dataset import N_PITCH_TYPES, PITCH_TYPE_TO_ID

# Sentinel value for "no prior pitch yet" (pitch position before the AB started).
BEGIN: str = "BEGIN"


def _add_prev_pitches(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Add ``prev_1, ..., prev_n`` columns: prior pitch types within the same AB.

    For the first pitch of an AB, ``prev_*`` is ``BEGIN``. For pitch *i*,
    ``prev_1`` is pitch *i−1*'s canonical type, ``prev_2`` is pitch *i−2*'s,
    and so on. The shift is computed in pitch-number order within each AB,
    but the **returned DataFrame is in the same row order as the input** —
    callers that align their predictions to a parallel target array depend
    on order being preserved.
    """
    if n <= 0:
        return df
    original_index = df.index
    df_sorted = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).copy()
    grouped = df_sorted.groupby(["game_pk", "at_bat_number"], observed=True)
    for k in range(1, n + 1):
        df_sorted[f"prev_{k}"] = (
            grouped["pitch_type_canonical"].shift(k).fillna(BEGIN)
        )
    # Restore caller's original row order so predict_proba(df)[i] aligns
    # with the i-th row of the input ``df``.
    return df_sorted.loc[original_index]


class PitcherNgramBaseline:
    """Per-pitcher pitch-type distribution conditional on (count, last n pitches),
    smoothed toward the league-conditional prior via Dirichlet.

    Args:
        n: number of prior pitches to condition on within the AB. ``n=0``
            uses just the count.
        alpha: Dirichlet concentration. Higher = more smoothing toward the
            league-conditional prior.

    The skill says ``n ∈ {1, 2, 3}``; per-fold val tuning of ``alpha`` is
    a nice-to-have refinement (default ``10.0`` is a reasonable starting
    point that puts roughly 1 effective league-prior pitch per slot).
    """

    def __init__(self, n: int = 1, alpha: float = 10.0):
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.n = int(n)
        self.alpha = float(alpha)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PitcherNgramBaseline":
        required = {
            "pitcher", "balls", "strikes", "pitch_type_canonical",
            "game_pk", "at_bat_number", "pitch_number",
        }
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"PitcherNgramBaseline.fit missing columns: {sorted(missing)}"
            )

        df = _add_prev_pitches(df, self.n)
        # Drop rows with non-canonical pitch types so they don't contaminate counts.
        pitch_ids = df["pitch_type_canonical"].map(PITCH_TYPE_TO_ID)
        df = df[pitch_ids.notna()].copy()
        df["pitch_id"] = pitch_ids[pitch_ids.notna()].astype(int)

        ctx_cols = ["balls", "strikes"] + [f"prev_{k}" for k in range(1, self.n + 1)]

        # Per-context league counts.
        self.league_counts_: dict[tuple, np.ndarray] = {}
        for keys, count in (
            df.groupby(ctx_cols + ["pitch_id"], observed=True).size().items()
        ):
            ctx = tuple(keys[:-1])
            pid = int(keys[-1])
            if ctx not in self.league_counts_:
                self.league_counts_[ctx] = np.zeros(N_PITCH_TYPES, dtype=np.float64)
            self.league_counts_[ctx][pid] += count

        # Per-(pitcher, context) counts.
        self.pitcher_counts_: dict[tuple, np.ndarray] = {}
        for keys, count in (
            df.groupby(["pitcher"] + ctx_cols + ["pitch_id"], observed=True)
            .size()
            .items()
        ):
            pitcher = int(keys[0])
            ctx = tuple(keys[1:-1])
            pid = int(keys[-1])
            cache_key = (pitcher, ctx)
            if cache_key not in self.pitcher_counts_:
                self.pitcher_counts_[cache_key] = np.zeros(
                    N_PITCH_TYPES, dtype=np.float64
                )
            self.pitcher_counts_[cache_key][pid] += count

        # Marginal fallback for any context unseen at predict time.
        self.marginal_ = np.zeros(N_PITCH_TYPES, dtype=np.float64)
        for pid, count in df.groupby("pitch_id").size().items():
            self.marginal_[int(pid)] = count
        if self.marginal_.sum() > 0:
            self.marginal_ = self.marginal_ / self.marginal_.sum()
        else:
            self.marginal_ = np.full(N_PITCH_TYPES, 1.0 / N_PITCH_TYPES)

        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PitcherNgramBaseline must be fit before predict_proba")

        df = _add_prev_pitches(df, self.n)
        n_rows = len(df)
        out = np.empty((n_rows, N_PITCH_TYPES), dtype=np.float64)

        ctx_cols = ["balls", "strikes"] + [f"prev_{k}" for k in range(1, self.n + 1)]
        pitchers = df["pitcher"].astype(int).to_numpy()
        ctx_arrays = [df[c].to_numpy() for c in ctx_cols]

        for i in range(n_rows):
            ctx = tuple([int(ctx_arrays[0][i]), int(ctx_arrays[1][i])]
                        + [ctx_arrays[k][i] for k in range(2, len(ctx_arrays))])

            league_counts = self.league_counts_.get(ctx)
            if league_counts is None:
                out[i] = self.marginal_
                continue
            league_prior = league_counts / league_counts.sum()

            pitcher_counts = self.pitcher_counts_.get((int(pitchers[i]), ctx))
            if pitcher_counts is None:
                # Unseen (pitcher, ctx): use league prior at this context.
                out[i] = league_prior
                continue

            # Dirichlet posterior mean: (counts + alpha * prior) / (n + alpha).
            posterior = pitcher_counts + self.alpha * league_prior
            out[i] = posterior / posterior.sum()

        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df).argmax(axis=1).astype(np.int64)
