"""K=5 cross-fit aggregation for AIPW (ADR 006).

Single-fit AIPW uses one nuisance model trained on ALL the data, then evaluates
on (a slice of) that same data — which is asymptotically biased (the model
"memorized" the units it's now scoring). Cross-fitting fixes this:

- Split the training data into K folds, blocked by ``game_pk``, stratified by
  season.
- Train K nuisance models, each on the K−1 folds excluding fold k.
- For each unit, evaluate using the model that DIDN'T see that unit's game
  during training.

CLAUDE.md hard rule #3 + ADR 006: K=5, blocked by ``game_pk``, stratified by
season. Single-fit AIPW numbers don't go in PRs or the writeup.

This module is the *aggregator*. It assumes the K nuisance checkpoints already
exist (one calibrated checkpoint per fold). Per-fold AIPW terms come from
:func:`causal.aipw.compute_aipw_per_unit`; here we concatenate them and feed
the combined list to :func:`causal.aipw.aipw_contrast`.

When called with missing checkpoints, raises with a clear message naming
which fold's checkpoint needs to be trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from causal.aipw import (
    AIPWPerUnit,
    AIPWResult,
    aipw_contrast,
    compute_aipw_per_unit,
)
from causal.g_computation import DEFAULT_AB_RUN_VALUE
from causal.nuisance import NuisanceModels
from causal.positivity import TAU_SINGLE_STEP
from data.folds import FoldAssignments

DEFAULT_FOLDS_PATH = Path("data/folds/fold_assignments.parquet")
DEFAULT_CHECKPOINT_PATTERN = "checkpoints/small-v{var}-fold{fold}/checkpoint_calibrated.pt"


@dataclass
class CrossfitConfig:
    """Configuration for a K=5 cross-fit AIPW run.

    Attributes:
        k: number of folds. Must match the fold-assignments parquet.
        checkpoint_paths: dict mapping ``fold_id`` -> Path to that fold's
            *calibrated* checkpoint.
        folds_path: path to the fold-assignments parquet (per ADR 006 / 008).
        positivity_threshold: τ for refusing positivity-violating units.
        intervention_position: pitch index at which to evaluate the
            counterfactual (MVP — single fixed position).
        run_value_table: 7-element AB-outcome → expected runs lookup.
    """

    k: int
    checkpoint_paths: dict[int, Path]
    folds_path: Path = DEFAULT_FOLDS_PATH
    positivity_threshold: float = TAU_SINGLE_STEP
    intervention_position: int = 1
    run_value_table: np.ndarray = None  # set in __post_init__

    def __post_init__(self):
        if self.run_value_table is None:
            self.run_value_table = DEFAULT_AB_RUN_VALUE
        if not set(self.checkpoint_paths.keys()) == set(range(self.k)):
            raise ValueError(
                f"checkpoint_paths must cover folds 0..{self.k - 1}; "
                f"got keys {sorted(self.checkpoint_paths.keys())}"
            )
        for fold_id, path in self.checkpoint_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"fold {fold_id} checkpoint not found at {path}. "
                    f"Train it before running cross-fit AIPW. Single-fit AIPW "
                    f"numbers are not reportable per CLAUDE.md hard rule #3."
                )
        if not Path(self.folds_path).exists():
            raise FileNotFoundError(
                f"fold assignments not found at {self.folds_path}; "
                f"run `make build-folds` to produce it"
            )


class CrossfitNuisance:
    """K-fold nuisance dispatcher.

    Lazily loads each fold's NuisanceModels on first access. Per-unit calls
    use ``nuisance_for_game(game_pk)`` which routes to the model trained
    *without* that game in its training data — the property that makes the
    estimator asymptotically unbiased.
    """

    def __init__(self, config: CrossfitConfig):
        self.config = config
        self.folds = FoldAssignments.from_path(config.folds_path)
        if self.folds.k != config.k:
            raise RuntimeError(
                f"fold assignments has k={self.folds.k}, but config has k={config.k}"
            )
        self._cache: dict[int, NuisanceModels] = {}

    def _load_fold(self, fold_id: int) -> NuisanceModels:
        if fold_id not in self._cache:
            ckpt = self.config.checkpoint_paths[fold_id]
            self._cache[fold_id] = NuisanceModels(ckpt)
        return self._cache[fold_id]

    def nuisance_for_game(self, game_pk: int) -> NuisanceModels:
        """Return the nuisance model NOT trained on this game's fold."""
        fold_id = self.folds.fold_of(int(game_pk))
        if fold_id is None:
            # Game wasn't assigned to any fold (post-train_end / unknown game).
            # Use fold 0 by default — any fold is "out of fold" for a post-train_end game.
            fold_id = 0
        return self._load_fold(int(fold_id))

    def unload_all(self) -> None:
        """Drop loaded models to free memory."""
        self._cache.clear()

    def __repr__(self) -> str:
        loaded = sorted(self._cache.keys())
        return f"CrossfitNuisance(k={self.config.k}, loaded_folds={loaded})"


# ============================================================
# Compute per-unit + aggregate
# ============================================================


def compute_crossfit_per_unit(
    crossfit: CrossfitNuisance,
    pitches: pd.DataFrame,
    *,
    intervention_position: Optional[int] = None,
    max_units_per_fold: Optional[int] = None,
    verbose: bool = False,
) -> list[AIPWPerUnit]:
    """Walk every AB in ``pitches``, route to the fold-appropriate model,
    compute per-unit AIPW terms, return the concatenated list.

    Per ADR 006: each unit's nuisance is the model trained *without* the
    fold that unit's game lives in. This avoids the "trained on the same
    data you're evaluating" bias.

    Args:
        crossfit: CrossfitNuisance dispatcher (already configured + folds loaded).
        pitches: pitch-level DataFrame.
        intervention_position: pitch index k for the counterfactual. Defaults
            to ``crossfit.config.intervention_position``.
        max_units_per_fold: cap units per fold (for fast iteration during dev).
        verbose: log progress.

    Returns:
        Concatenated AIPWPerUnit list across all folds.
    """
    if intervention_position is None:
        intervention_position = crossfit.config.intervention_position

    df = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    groups = df.groupby(["game_pk", "at_bat_number"], sort=False)

    out: list[AIPWPerUnit] = []
    per_fold_count: dict[int, int] = {f: 0 for f in range(crossfit.config.k)}

    for (game_pk, _at_bat), ab in groups:
        if len(ab) < intervention_position + 1:
            continue
        fold_id = crossfit.folds.fold_of(int(game_pk))
        if fold_id is None:
            continue  # skip games not assigned to any fold
        if max_units_per_fold is not None and per_fold_count[fold_id] >= max_units_per_fold:
            continue
        nuisance = crossfit.nuisance_for_game(int(game_pk))
        try:
            term = compute_aipw_per_unit(
                nuisance, ab.reset_index(drop=True),
                intervention_position=intervention_position,
                run_value_table=crossfit.config.run_value_table,
            )
        except Exception as exc:
            if verbose:
                print(f"  skip ({game_pk}, {_at_bat}): {type(exc).__name__}: {exc}")
            continue
        if term is not None:
            out.append(term)
            per_fold_count[fold_id] += 1
        if verbose and (len(out) % 100 == 0):
            print(f"  {len(out)} units kept across folds; per-fold: {per_fold_count}")

    if verbose:
        print(f"  done: {len(out)} total units, per-fold: {per_fold_count}")
    return out


def crossfit_aipw_contrast(
    crossfit: CrossfitNuisance,
    pitches: pd.DataFrame,
    intervention_a: int | str,
    intervention_a_prime: int | str,
    *,
    intervention_position: Optional[int] = None,
    max_units_per_fold: Optional[int] = None,
    verbose: bool = False,
) -> AIPWResult:
    """Full cross-fit AIPW contrast pipeline on ``pitches``.

    Walks every AB, routes to the fold-appropriate nuisance model, computes
    per-unit AIPW terms, aggregates into ``τ̂(a, a')`` with influence-function
    SE. Positivity-violating units are excluded with a clear count in
    diagnostics (per ADR 002).
    """
    per_unit = compute_crossfit_per_unit(
        crossfit, pitches,
        intervention_position=intervention_position,
        max_units_per_fold=max_units_per_fold,
        verbose=verbose,
    )
    if not per_unit:
        raise RuntimeError(
            "no units kept after fold dispatch — check that the pitches DataFrame "
            "contains games covered by the fold assignments and that the AB lengths "
            "are sufficient for the intervention position."
        )
    return aipw_contrast(
        per_unit, intervention_a, intervention_a_prime,
        positivity_threshold=crossfit.config.positivity_threshold,
    )


# ============================================================
# Verification (per ADR 006)
# ============================================================


def verify_fold_balance(
    folds: FoldAssignments,
    pitches: pd.DataFrame,
    *,
    tolerance_pct: float = 2.0,
) -> dict:
    """Verify each fold is within ``tolerance_pct`` of the mean per-season pitch share.

    Per ADR 006: "after splitting, log per-fold pitch count by season; if any
    fold deviates by more than ±2% of mean season representation, regenerate
    the split with a different seed."

    Args:
        folds: loaded FoldAssignments.
        pitches: pitch-level DataFrame with ``game_pk`` and ``game_date``.
        tolerance_pct: allowed deviation from per-season mean (default 2%).

    Returns:
        Dict with per-fold counts + a ``passes`` boolean.
    """
    if "game_date" not in pitches.columns:
        raise KeyError("verify_fold_balance needs 'game_date' on pitches")

    df = pitches.copy()
    df["season"] = pd.to_datetime(df["game_date"]).dt.year
    df["fold"] = df["game_pk"].astype(int).map(lambda pk: folds.fold_of(pk))
    df = df.dropna(subset=["fold"])
    df["fold"] = df["fold"].astype(int)

    counts = df.groupby(["season", "fold"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    fractions = counts.div(totals, axis=0)
    mean_fraction = 1.0 / folds.k

    deviations = (fractions - mean_fraction).abs() * 100.0  # in percentage points
    max_dev = float(deviations.values.max())
    passes = max_dev <= tolerance_pct

    return {
        "passes": passes,
        "max_deviation_pct": max_dev,
        "tolerance_pct": tolerance_pct,
        "per_season_fractions": fractions.to_dict(),
    }
