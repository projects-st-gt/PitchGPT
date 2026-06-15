"""Base-out transition matrix: the bridge from PA outcomes to runs scored.

Given a (base_state, outs) situation and a PA outcome (K/BB/1B/2B/3B/HR/out),
the matrix tells you the probability distribution over:
  - next (base_state, outs) or INNING_OVER
  - runs scored on this PA

Built empirically from real play-by-play data (2017–2023 training split).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.run_value import base_state_from_columns
from gamesim.outcomes import (
    INNING_OVER,
    N_BASE_STATES,
    N_OUT_STATES,
    OUTCOME_CLASSES,
    base_out_index,
    event_to_outcome,
)

TRANSITION_PATH = Path("data/run_value/base_out_transition.parquet")


def build_base_out_transition_matrix(
    raw_dir: Path = Path("data/raw"),
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Build the empirical base-out transition matrix from raw PBP data.

    For each PA, we record:
      - from_state: (base_state, outs) before the PA
      - outcome: 7-class (K/BB/1B/2B/3B/HR/out)
      - to_state: (base_state, outs) after the PA, or INNING_OVER
      - runs: runs scored on this PA

    Returns a long-format DataFrame with columns:
      from_base, from_outs, outcome, to_state_idx, runs, n, prob
    """
    if seasons is None:
        seasons = list(range(2017, 2024))

    all_records: list[dict] = []

    for season in seasons:
        season_dir = raw_dir / str(season)
        if not season_dir.exists():
            continue
        parquets = sorted(season_dir.glob("*.parquet"))
        for pq in parquets:
            df = pd.read_parquet(pq)
            records = _extract_transitions_from_game_data(df)
            all_records.extend(records)

    if not all_records:
        raise RuntimeError("No transition records found — check data/raw/ paths")

    raw_df = pd.DataFrame(all_records)

    counts = (
        raw_df.groupby(["from_base", "from_outs", "outcome", "to_state_idx", "runs"])
        .size()
        .reset_index(name="n")
    )

    totals = counts.groupby(["from_base", "from_outs", "outcome"])["n"].transform("sum")
    counts["prob"] = counts["n"] / totals

    return counts


def _extract_transitions_from_game_data(df: pd.DataFrame) -> list[dict]:
    """Extract per-PA transition records from a day's worth of pitch data."""
    pa_df = df[df["events"].notna()].copy()
    if pa_df.empty:
        return []

    pa_df["outcome"] = pa_df["events"].map(event_to_outcome)
    pa_df = pa_df[pa_df["outcome"].notna()].copy()
    if pa_df.empty:
        return []

    pa_df["base_state"] = base_state_from_columns(pa_df)

    half_id_cols = ["game_pk", "inning", "inning_topbot"]
    pa_df = pa_df.sort_values(half_id_cols + ["at_bat_number"]).reset_index(drop=True)

    pa_df["half_id"] = (
        pa_df["game_pk"].astype(str) + "_"
        + pa_df["inning"].astype(str) + "_"
        + pa_df["inning_topbot"]
    )

    score_col = "post_bat_score" if "post_bat_score" in pa_df.columns else "bat_score"
    pa_df["runs_on_pa"] = pa_df[score_col].astype(float) - pa_df["bat_score"].astype(float)
    pa_df["runs_on_pa"] = pa_df["runs_on_pa"].clip(lower=0).fillna(0).astype(int)

    records = []
    grouped = pa_df.groupby("half_id")
    for _, half in grouped:
        half = half.sort_values("at_bat_number").reset_index(drop=True)
        for i in range(len(half)):
            row = half.iloc[i]
            from_base = int(row["base_state"])
            from_outs = int(row["outs_when_up"])
            outcome = row["outcome"]
            runs = int(row["runs_on_pa"])

            if i + 1 < len(half):
                next_row = half.iloc[i + 1]
                to_base = int(next_row["base_state"])
                to_outs = int(next_row["outs_when_up"])
                to_state_idx = base_out_index(to_base, to_outs)
            else:
                to_state_idx = INNING_OVER

            records.append({
                "from_base": from_base,
                "from_outs": from_outs,
                "outcome": outcome,
                "to_state_idx": to_state_idx,
                "runs": runs,
            })

    return records


class BaseOutTransition:
    """Loaded transition matrix for fast sampling during game simulation."""

    def __init__(self, df: pd.DataFrame):
        self._lookup: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for (fb, fo, oc), grp in df.groupby(["from_base", "from_outs", "outcome"]):
            to_states = grp["to_state_idx"].values.astype(np.int32)
            runs = grp["runs"].values.astype(np.int32)
            probs = grp["prob"].values.astype(np.float64)
            probs = probs / probs.sum()
            self._lookup[(int(fb), int(fo), oc)] = (to_states, runs, probs)

    @classmethod
    def load(cls, path: Path = TRANSITION_PATH) -> "BaseOutTransition":
        df = pd.read_parquet(path)
        return cls(df)

    def sample(
        self,
        base_state: int,
        outs: int,
        outcome: str,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Sample a (to_state_idx, runs_scored) from the transition matrix.

        Returns (to_state_idx, runs) where to_state_idx is either a flat
        base_out_index (0..23) or INNING_OVER (24).

        Raises KeyError if the (base_state, outs, outcome) combination has
        no empirical data — per hard rule 1a, never silently default.
        """
        key = (base_state, outs, outcome)
        if key not in self._lookup:
            raise KeyError(
                f"No transition data for base_state={base_state}, outs={outs}, "
                f"outcome={outcome}"
            )
        to_states, runs, probs = self._lookup[key]
        idx = rng.choice(len(to_states), p=probs)
        return int(to_states[idx]), int(runs[idx])

    def has_key(self, base_state: int, outs: int, outcome: str) -> bool:
        return (base_state, outs, outcome) in self._lookup
