"""K-fold cross-fit assignment per ADR 006 + ADR 008.

ADR 006 specifies:

- K=5 folds
- Blocked by ``game_pk`` (every pitch in a given game lands in the same
  fold, so within-game state never leaks across folds)
- Stratified by season (each fold has roughly equal representation across
  training years)

ADR 008 makes the profile cache fold-aware: for each fold k, the cache for
fold k is built using only pitches from games whose ``game_pk`` is *not*
assigned to fold k.

The assignment is deterministic given inputs and seed. Save and load via
``save_folds`` / ``load_folds`` so the same split is used everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

K_DEFAULT: int = 5
SEED_DEFAULT: int = 42


def assign_folds(
    games: pd.DataFrame,
    k: int = K_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> pd.DataFrame:
    """Assign each game to one of ``k`` folds, stratified by season.

    Args:
        games: DataFrame with columns ``game_pk`` and ``season`` (int year).
            Duplicate (game_pk, season) rows are deduped.
        k: number of folds.
        seed: rng seed for the within-season shuffle. Same seed reproduces
            the same assignment.

    Returns:
        DataFrame with columns ``game_pk``, ``season``, ``fold_id``.
        ``fold_id`` is in ``[0, k)``.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if not {"game_pk", "season"}.issubset(games.columns):
        raise KeyError("assign_folds requires 'game_pk' and 'season' columns")

    rng = np.random.default_rng(seed)
    deduped = games[["game_pk", "season"]].drop_duplicates()
    parts: list[dict] = []
    for season, group in deduped.groupby("season"):
        pks = sorted(group["game_pk"].unique().tolist())
        rng.shuffle(pks)
        for i, pk in enumerate(pks):
            parts.append({"game_pk": int(pk), "season": int(season), "fold_id": i % k})
    out = pd.DataFrame(parts)
    return out.sort_values(["season", "game_pk"]).reset_index(drop=True)


def fold_balance_deviation(
    assignments: pd.DataFrame,
) -> float:
    """Maximum per-(fold, season) deviation from the per-season mean fold count.

    Returns the max absolute fractional deviation. ADR 006 calls for ≤ 2%;
    a value above that indicates the seed produced an unusually unbalanced
    split for some season.
    """
    counts = (
        assignments.groupby(["fold_id", "season"])
        .size()
        .unstack(fill_value=0)
    )
    if counts.size == 0:
        return 0.0
    season_means = counts.mean(axis=0)  # mean across folds, per season
    season_means_safe = season_means.replace(0, 1)
    deviations = (counts.subtract(season_means, axis=1)
                  .div(season_means_safe, axis=1)
                  .abs())
    return float(deviations.max().max())


def save_folds(assignments: pd.DataFrame, path: Path) -> None:
    """Persist fold assignments to parquet (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    assignments.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_folds(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


class FoldAssignments:
    """In-memory lookup for fold assignments.

    Use cases at cache-build time:

    - "What fold is this game in?"  →  ``fold_of(game_pk)``
    - "Which games to *exclude* when building cache for fold k?"
      →  ``games_in_fold(k)``  (those are the ones to drop)
    - "Which games to *use* when building cache for fold k?"
      →  ``games_excluding_fold(k)``
    """

    def __init__(self, assignments: pd.DataFrame):
        if not {"game_pk", "fold_id"}.issubset(assignments.columns):
            raise KeyError("FoldAssignments needs 'game_pk' and 'fold_id' columns")
        self._lookup: dict[int, int] = {
            int(pk): int(fid) for pk, fid in zip(
                assignments["game_pk"], assignments["fold_id"]
            )
        }
        self._k = int(assignments["fold_id"].max()) + 1

    @property
    def k(self) -> int:
        return self._k

    def fold_of(self, game_pk: int) -> int | None:
        return self._lookup.get(int(game_pk))

    def games_in_fold(self, fold_id: int) -> list[int]:
        return [pk for pk, fid in self._lookup.items() if fid == fold_id]

    def games_excluding_fold(self, fold_id: int) -> list[int]:
        return [pk for pk, fid in self._lookup.items() if fid != fold_id]

    def __len__(self) -> int:
        return len(self._lookup)

    @classmethod
    def from_path(cls, path: Path) -> "FoldAssignments":
        return cls(load_folds(path))
