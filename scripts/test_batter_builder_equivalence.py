"""Side-by-side test: slow vs fast batter cache builder.

Both builders should produce identical output vectors for the same input.
We compare on a small slice to verify correctness before running the fast
builder on the full corpus.

Run via ``uv run python -m scripts.test_batter_builder_equivalence``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.folds import FoldAssignments
from scripts.build_profile_cache import (
    _discover_asof_keys,
    _load_corpus,
    build_batter_cache_for_fold,
    build_batter_cache_for_fold_fast,
)

RAW_DIR = Path("data/raw")
FOLDS_PATH = Path("data/folds/fold_assignments.parquet")
TMP_OLD = Path("data/profiles_test_old/batter_fold_0.parquet")
TMP_NEW = Path("data/profiles_test_new/batter_fold_0.parquet")


def main() -> None:
    print("Loading corpus...")
    t0 = time.monotonic()
    corpus = _load_corpus(RAW_DIR)
    print(f"  loaded {len(corpus):,} pitches in {time.monotonic() - t0:.1f}s")

    folds = FoldAssignments.from_path(FOLDS_PATH)

    print("\nDiscovering all batter asof keys, then limiting to first 30 games...")
    asof_keys = _discover_asof_keys(corpus, player_col="batter")
    keep_pks = list(asof_keys["game_pk"].drop_duplicates().head(30))
    asof_keys = asof_keys[asof_keys["game_pk"].isin(keep_pks)].reset_index(drop=True)
    print(f"  {len(asof_keys):,} (batter, game) keys across {len(keep_pks)} games")

    fold_id = 0

    print(f"\n=== OLD builder (fold {fold_id}) ===")
    TMP_OLD.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_old = build_batter_cache_for_fold(corpus, asof_keys, folds, fold_id, TMP_OLD)
    old_elapsed = time.monotonic() - t0
    print(f"  wrote {n_old:,} entries in {old_elapsed:.2f}s")

    print(f"\n=== NEW (fast) builder (fold {fold_id}) ===")
    TMP_NEW.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_new = build_batter_cache_for_fold_fast(corpus, asof_keys, folds, fold_id, TMP_NEW)
    new_elapsed = time.monotonic() - t0
    print(f"  wrote {n_new:,} entries in {new_elapsed:.2f}s")

    print(f"\n=== Comparison ===")
    print(f"  Speedup: {old_elapsed / new_elapsed:.2f}x")

    old_df = pd.read_parquet(TMP_OLD).sort_values(
        ["player_id", "asof_date", "asof_game_num"]
    ).reset_index(drop=True)
    new_df = pd.read_parquet(TMP_NEW).sort_values(
        ["player_id", "asof_date", "asof_game_num"]
    ).reset_index(drop=True)

    if len(old_df) != len(new_df):
        print(f"  FAIL: row count mismatch (old={len(old_df)}, new={len(new_df)})")
        return

    # Sort both by key columns and compare row-wise
    keys_match = (
        (old_df["player_id"] == new_df["player_id"]).all()
        and (old_df["asof_date"] == new_df["asof_date"]).all()
        and (old_df["asof_game_num"] == new_df["asof_game_num"]).all()
    )
    if not keys_match:
        print(f"  FAIL: key columns differ")
        return

    # Compare vectors (NaN-aware)
    old_vecs = np.stack([np.asarray(v, dtype=np.float32) for v in old_df["vector"]])
    new_vecs = np.stack([np.asarray(v, dtype=np.float32) for v in new_df["vector"]])

    nan_mask_old = np.isnan(old_vecs)
    nan_mask_new = np.isnan(new_vecs)

    if not np.array_equal(nan_mask_old, nan_mask_new):
        diff_locations = np.where(nan_mask_old != nan_mask_new)
        n_diffs = len(diff_locations[0])
        print(f"  FAIL: NaN patterns differ at {n_diffs} cells")
        return

    # For non-NaN cells, check they're close
    finite_mask = ~nan_mask_old
    diffs = np.abs(old_vecs[finite_mask] - new_vecs[finite_mask])
    max_diff = diffs.max() if diffs.size else 0.0
    n_meaningful_diffs = int((diffs > 1e-5).sum())

    print(f"  Max abs diff (non-NaN): {max_diff:.2e}")
    print(f"  Cells differing > 1e-5: {n_meaningful_diffs:,}")

    if n_meaningful_diffs == 0:
        print("\n  PASS — fast builder matches slow builder bit-for-bit (within float tolerance)")
        print("  Safe to run fast builder on full corpus.")
    else:
        # Show a sample of disagreements
        flat_old = old_vecs[finite_mask]
        flat_new = new_vecs[finite_mask]
        idx = np.argsort(-diffs)[:5]
        print("\n  FAIL — sample disagreements:")
        for i in idx:
            print(f"    old={flat_old[i]:.6f}  new={flat_new[i]:.6f}  diff={diffs[i]:.6f}")


if __name__ == "__main__":
    main()
