"""Compute and save K-fold assignments per ADR 006.

Walks ``data/raw/`` to discover all (game_pk, season) pairs, runs
``assign_folds`` (K=5, stratified by season, blocked by game_pk),
saves to ``data/folds/fold_assignments.parquet``, and prints the balance
summary.

Wired up via ``make build-folds``. Re-running is idempotent and produces
the same assignment given the same inputs and seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from data.folds import (
    K_DEFAULT,
    SEED_DEFAULT,
    assign_folds,
    fold_balance_deviation,
    save_folds,
)

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/folds/fold_assignments.parquet")
BALANCE_THRESHOLD = 0.10  # warn if any (fold, season) cell deviates >10% from mean


def _discover_games(raw_dir: Path) -> pd.DataFrame:
    print(f"Scanning {raw_dir} for unique (game_pk, season)...")
    parts: list[pd.DataFrame] = []
    parquets = sorted(raw_dir.rglob("*.parquet"))
    for p in parquets:
        try:
            df = pd.read_parquet(p, columns=["game_pk", "game_date"])
        except Exception as exc:
            print(f"  warn: could not read {p.name}: {exc!r}")
            continue
        df = df.dropna(subset=["game_pk", "game_date"])
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
        parts.append(df[["game_pk", "season"]].drop_duplicates())
    if not parts:
        return pd.DataFrame(columns=["game_pk", "season"])
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def main() -> None:
    games = _discover_games(RAW_DIR)
    if games.empty:
        print("No games discovered; did extraction complete?")
        sys.exit(1)
    print(f"Found {len(games):,} unique games across {games['season'].nunique()} seasons.")

    assignments = assign_folds(games, k=K_DEFAULT, seed=SEED_DEFAULT)

    print(f"\nFold sizes:")
    sizes = assignments.groupby("fold_id").size()
    for fold_id, count in sizes.items():
        print(f"  fold {fold_id}: {count:,} games")

    print(f"\nPer-(fold, season) breakdown:")
    counts = (
        assignments.groupby(["fold_id", "season"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    print(counts.to_string())

    deviation = fold_balance_deviation(assignments)
    print(f"\nMax (fold, season) deviation from mean: {deviation * 100:.2f}%")
    if deviation > BALANCE_THRESHOLD:
        print(f"WARN: deviation exceeds {BALANCE_THRESHOLD * 100:.0f}% — "
              f"consider a different seed or recomputing.")

    save_folds(assignments, OUT_PATH)
    print(f"\nSaved {len(assignments):,} fold assignments to {OUT_PATH}")


if __name__ == "__main__":
    main()
