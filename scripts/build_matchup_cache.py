"""Build the fold-aware matchup cache (ADR 013 Decision 2 / ADR 014).

For each fold k in {0..K-1}, produce
``data/profiles/matchup_fold_{k}.parquet`` containing one row per
(pitcher_id, batter_id, asof_date, asof_game_num) seen in the corpus,
with the flattened matchup vector
(:func:`data.profile_cache.build_matchup_profile_vector`) computed using
only pitches from games whose fold_id != k, then ``before_asof``-filtered
to strictly-prior games (same-game ABs are excluded — see
:func:`data.player_profiles.before_asof`).

The vector length is ``MATCHUP_VECTOR_LEN`` (21) — 7 pitch-mix features +
7 whiff-on-swing features + cumulative PA counts + last-face signals +
matchup_confidence. Empty matchups (pair never faced before this game) get
zero pitch-mix + NaN whiff + zero cumulative + zero confidence per the
flattener's spec.

Mirrors :mod:`scripts.build_profile_cache` — same corpus loader, same fold
exclusion pattern, same ``before_asof`` rule. The fast path uses
``np.searchsorted`` on a per-(pitcher, batter)-pair composite ``(date,
game_num)`` key, the same trick that makes the batter cache 3–5× faster.

Wired via ``make build-matchup-cache``. Requires fold assignments to exist
(``make build-folds`` first) and the raw parquets under ``data/raw/``.

Run:

    python -m scripts.build_matchup_cache               # all 5 folds
    python -m scripts.build_matchup_cache --folds 0     # one fold
    python -m scripts.build_matchup_cache --max-games 200  # smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.folds import FoldAssignments
from data.profile_cache import (
    MATCHUP_SCHEMA_VERSION,
    MATCHUP_VECTOR_LEN,
    build_matchup_profile_vector,
)

# Reuse the corpus loader (same columns, same harmonization + zone tagging)
# and the composite-key helpers (single source of truth for the searchsorted
# trick — see the datetime-unit note in ``build_profile_cache._composite_sort_key``).
from scripts.build_profile_cache import (
    _load_corpus,
    _composite_sort_key,
    _composite_asof_key,
    RAW_DIR,
    FOLDS_PATH,
    OUT_DIR,
)


def _discover_matchup_asof_keys(corpus: pd.DataFrame) -> pd.DataFrame:
    """Unique ``(pitcher, batter, asof_date, asof_game_num, game_pk)`` triples.

    One row per (pitcher, batter, game) the cache must serve. The matchup
    vector encodes "what these two know about each other *coming into* this
    game," so within-game ABs share a single cache key (``before_asof``
    excludes same-game content anyway).
    """
    keys = (
        corpus[["pitcher", "batter", "game_date", "game_num", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk", "pitcher", "batter"])
        .reset_index(drop=True)
    )
    keys = keys.rename(columns={"game_date": "asof_date", "game_num": "asof_game_num"})
    return keys


def build_matchup_cache_for_fold_fast(
    corpus: pd.DataFrame,
    asof_keys: pd.DataFrame,
    folds: FoldAssignments,
    fold_id: int,
    out_path: Path,
) -> int:
    """Write the matchup cache for one fold to ``out_path``. Returns row count.

    Mirrors ``build_batter_cache_for_fold_fast`` but keyed by
    ``(pitcher, batter)`` pair instead of ``batter`` alone. Each pair's
    pitches+PAs are sorted once at fold scope; per-asof cutoff is
    ``O(log N_pair)`` via ``np.searchsorted``.
    """
    excluded_pks = set(folds.games_in_fold(fold_id))
    fold_corpus = corpus[~corpus["game_pk"].isin(excluded_pks)]
    pas_corpus = fold_corpus[fold_corpus["events"].notna()]

    # Sort once globally by (pitcher, batter, date, game_num); each pair's
    # view is then a contiguous slice of the sorted DataFrame.
    sorted_pitches = (
        fold_corpus
        .sort_values(["pitcher", "batter", "game_date", "game_num"])
        .reset_index(drop=True)
    )
    sorted_pas = (
        pas_corpus
        .sort_values(["pitcher", "batter", "game_date", "game_num"])
        .reset_index(drop=True)
    )

    pitches_by_pair = sorted_pitches.groupby(["pitcher", "batter"], sort=False).indices
    pas_by_pair = sorted_pas.groupby(["pitcher", "batter"], sort=False).indices

    rows: list[dict] = []
    for (pitcher_id, batter_id), group_keys in asof_keys.groupby(["pitcher", "batter"], sort=False):
        pitcher_id = int(pitcher_id)
        batter_id = int(batter_id)
        pair = (pitcher_id, batter_id)

        if pair in pitches_by_pair:
            idx = pitches_by_pair[pair]
            pair_pitches = sorted_pitches.iloc[idx].reset_index(drop=True)
            p_keys = _composite_sort_key(
                pair_pitches["game_date"].to_numpy(),
                pair_pitches["game_num"].to_numpy(),
            )
        else:
            pair_pitches = pd.DataFrame(columns=sorted_pitches.columns)
            p_keys = np.array([], dtype=np.int64)

        if pair in pas_by_pair:
            idx = pas_by_pair[pair]
            pair_pas = sorted_pas.iloc[idx].reset_index(drop=True)
            pa_keys = _composite_sort_key(
                pair_pas["game_date"].to_numpy(),
                pair_pas["game_num"].to_numpy(),
            )
        else:
            pair_pas = pd.DataFrame(columns=sorted_pas.columns)
            pa_keys = np.array([], dtype=np.int64)

        for _, key in group_keys.iterrows():
            asof_date = pd.Timestamp(key["asof_date"])
            asof_num = int(key["asof_game_num"])
            ak = _composite_asof_key(asof_date, asof_num)

            cutoff_p = int(np.searchsorted(p_keys, ak, side="left"))
            cutoff_pa = int(np.searchsorted(pa_keys, ak, side="left"))

            filtered_pitches = pair_pitches.iloc[:cutoff_p]
            filtered_pas = pair_pas.iloc[:cutoff_pa]

            vector = build_matchup_profile_vector(
                filtered_pitches, filtered_pas, asof_date
            )
            rows.append({
                "pitcher_id": pitcher_id,
                "batter_id": batter_id,
                "asof_date": asof_date.date(),
                "asof_game_num": asof_num,
                "fold_id": fold_id,
                "schema_version": MATCHUP_SCHEMA_VERSION,
                "vector": vector.astype(np.float32).tolist(),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fold-aware matchup cache.")
    parser.add_argument(
        "--folds", default="0,1,2,3,4",
        help="Comma-separated fold IDs to build (default: all).",
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Limit asof keys to those from the first N distinct game_pks "
             "(useful for testing without paying the full-corpus runtime).",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
        help="Output directory; per-fold parquets will be written here.",
    )
    args = parser.parse_args()

    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    out_dir = Path(args.out_dir)

    print("Loading fold assignments...")
    folds = FoldAssignments.from_path(FOLDS_PATH)

    print("Loading corpus...")
    t0 = time.monotonic()
    corpus = _load_corpus(RAW_DIR)
    print(f"  loaded {len(corpus):,} pitches in {time.monotonic() - t0:.1f}s")

    print(f"\nBuilding matchup cache for folds={fold_ids}, "
          f"schema_version={MATCHUP_SCHEMA_VERSION}, vec_len={MATCHUP_VECTOR_LEN}")
    print(f"Output dir: {out_dir.resolve()}\n")

    asof_keys = _discover_matchup_asof_keys(corpus)
    if args.max_games is not None:
        keep_pks = list(asof_keys["game_pk"].drop_duplicates().head(args.max_games))
        asof_keys = asof_keys[asof_keys["game_pk"].isin(keep_pks)].reset_index(drop=True)
        print(f"  --max-games={args.max_games}: limited to "
              f"{len(asof_keys):,} (pitcher, batter, game) keys across "
              f"{len(keep_pks)} distinct games")
    else:
        print(f"  {len(asof_keys):,} unique (pitcher, batter, game) cache keys to build")

    for fold_id in fold_ids:
        out_path = out_dir / f"matchup_fold_{fold_id}.parquet"
        t0 = time.monotonic()
        n = build_matchup_cache_for_fold_fast(corpus, asof_keys, folds, fold_id, out_path)
        elapsed = time.monotonic() - t0
        print(f"  fold {fold_id}: wrote {n:,} entries to {out_path.name} "
              f"({elapsed:.1f}s)")
    print()


if __name__ == "__main__":
    main()
