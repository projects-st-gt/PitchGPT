"""Build the fold-aware profile cache (ADR 008) for pitchers and/or batters.

For each fold k in {0..K-1}, produce
``data/profiles/{role}_fold_{k}.parquet`` containing one row per
(player_id, asof_date, asof_game_num) seen in the corpus, with the
flattened profile vector computed using only pitches from games whose
fold_id != k (and, on top of that, the ``before_asof`` strict-temporal
filter).

Use ``--role pitcher|batter|both`` (default both) and ``--max-games N``
for a bounded test run. The full corpus build for both roles is several
hours wallclock.

Wired via ``make build-profile-cache``. Requires fold assignments to exist
(``make build-folds`` first).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.folds import FoldAssignments
from data.player_profiles import before_asof
from data.profile_cache import (
    BATTER_VECTOR_LEN,
    PITCHER_FEATURE_NAMES,
    PITCHER_VECTOR_LEN,
    PROFILE_SCHEMA_VERSION,
    build_batter_profile_vector,
    build_pitcher_profile_vector,
    compute_league_means,
)

RAW_DIR = Path("data/raw")
FOLDS_PATH = Path("data/folds/fold_assignments.parquet")
OUT_DIR = Path("data/profiles")

# Columns we need from the raw parquets to build either profile.
# Pitcher-specific: pitcher, release_speed, release_spin_rate.
# Batter-specific: batter, description, events, estimated_ba_using_speedangle,
#   woba_value, launch_speed.
# Shared: game_pk, game_date, pitch_type, balls, strikes, plate_x/z, sz_top/bot,
#   p_throws, estimated_woba_using_speedangle.
NEEDED_COLUMNS: list[str] = sorted(set([
    "game_pk", "game_date",
    "pitcher", "batter",
    "pitch_type",
    "release_speed", "release_spin_rate",
    "balls", "strikes",
    "plate_x", "plate_z", "sz_top", "sz_bot", "p_throws",
    "description", "events",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "woba_value", "launch_speed",
    "arm_angle",  # v4 / Sprint 0b: per-pitch-type arm slot in pitcher profile.
    "zone",  # v5 / 14-zone migration: Statcast SIS zone (1-9, 11-14) for assign_feature_zone_14.
    "pfx_x",  # v6: horizontal break (feet) for mean_pfx_x_{pt} pitcher profile feature.
    "pfx_z",  # v6: vertical break (feet) for mean_pfx_z_{pt} pitcher profile feature.
    "stand",  # v6: batter handedness (L/R) for arsenal_{pt}_vs{stand} pitcher profile feature.
]))


def _load_corpus(raw_dir: Path) -> pd.DataFrame:
    """Load the columns needed for pitcher profile building, harmonize and tag."""
    from data.harmonization import harmonize_dataframe
    from data.zones import assign_feature_zone_14, valid_zone_mask

    parquets = sorted(raw_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquets under {raw_dir}; run extraction first")

    # Sample one parquet to determine which of our wanted columns are present;
    # some may be missing in older years. Reading the first file fully (~few MB)
    # is cheap and lets all subsequent reads use column projection.
    import pyarrow.parquet as pq
    sample_schema = pq.read_schema(parquets[0])
    available_in_first = set(sample_schema.names)
    cols = [c for c in NEEDED_COLUMNS if c in available_in_first]
    missing = [c for c in NEEDED_COLUMNS if c not in available_in_first]
    if missing:
        print(f"  note: columns missing from sample parquet, skipping: {missing}")

    parts: list[pd.DataFrame] = []
    for p in parquets:
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception as exc:
            print(f"  warn: could not read {p.name}: {exc!r}")
            continue
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # Statcast parquets sometimes store game_date as string; ensure datetime
    # so before_asof comparisons work. Statcast also doesn't always populate
    # game_num; default to 1 (regular game).
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_num"] = 1

    df = harmonize_dataframe(df, drop_unmapped=False)
    # Drop NaN Statcast zone (book-keeping rows: automatic_ball / pitch-clock).
    if "zone" in df.columns:
        df = df.loc[df["zone"].notna()].copy()
    df = df.loc[valid_zone_mask(df)].copy()
    df["feature_zone"] = assign_feature_zone_14(df).astype("int16")
    df = df.dropna(subset=["pitch_type_canonical"])
    return df


def _discover_asof_keys(corpus: pd.DataFrame, player_col: str) -> pd.DataFrame:
    """Unique (player, game_date, game_num, game_pk) the cache must serve.

    ``player_col`` is ``"pitcher"`` or ``"batter"``.
    """
    keys = (
        corpus[[player_col, "game_date", "game_num", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk", player_col])
        .reset_index(drop=True)
    )
    keys = keys.rename(columns={"game_date": "asof_date", "game_num": "asof_game_num"})
    return keys


def build_pitcher_cache_for_fold(
    corpus: pd.DataFrame,
    asof_keys: pd.DataFrame,
    folds: FoldAssignments,
    fold_id: int,
    out_path: Path,
) -> int:
    """Write the pitcher cache for one fold to ``out_path``. Returns row count."""
    excluded_pks = set(folds.games_in_fold(fold_id))
    fold_corpus = corpus[~corpus["game_pk"].isin(excluded_pks)]
    pitcher_groups = fold_corpus.groupby("pitcher", sort=False)

    rows: list[dict] = []
    for _, key in asof_keys.iterrows():
        pitcher_id = int(key["pitcher"])
        asof_date = pd.Timestamp(key["asof_date"])
        asof_game_num = int(key["asof_game_num"])

        if pitcher_id in pitcher_groups.groups:
            this_pitcher_all = pitcher_groups.get_group(pitcher_id)
            filtered = before_asof(this_pitcher_all, asof_date, asof_game_num)
        else:
            filtered = pd.DataFrame(columns=fold_corpus.columns)

        vector = build_pitcher_profile_vector(filtered, asof_date)
        rows.append({
            "player_id": pitcher_id,
            "asof_date": asof_date.date(),
            "asof_game_num": asof_game_num,
            "fold_id": fold_id,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "vector": vector.astype(np.float32).tolist(),
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return len(rows)


_NS_PER_DAY: int = 86_400 * 10**9


def _composite_sort_key(
    dates: np.ndarray, nums: np.ndarray
) -> np.ndarray:
    """Encode ``(game_date, game_num)`` as a sortable int64.

    ``date_in_days * 100 + game_num`` is monotone in lex order as long as
    ``game_num < 100`` (always true in MLB — doubleheaders give ``game_num``
    of 1 or 2). This lets us use ``np.searchsorted`` for fast asof cutoff
    instead of a full boolean filter per key.

    **Datetime-unit discipline (pandas 3.x):** parquets can land here as
    ``datetime64[us]`` rather than ``[ns]``. ``astype("int64")`` then yields
    µs-since-epoch, and ``// _NS_PER_DAY`` is off by 1000×, producing nonsense
    cutoffs that look like "everything before asof" silently. The
    ``astype("datetime64[ns]")`` round-trip normalizes to ns first so the
    division gives days regardless of the input unit. (Matching pattern in
    :func:`_composite_asof_key` below.)
    """
    date_ord = (
        pd.to_datetime(dates)
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
        // _NS_PER_DAY
    )
    return date_ord * 100 + nums.astype("int64")


def _composite_asof_key(asof_date: pd.Timestamp, asof_num: int) -> int:
    # ``pd.Timestamp.value`` is always ns-since-epoch regardless of source dtype
    # (Timestamp internally normalizes), so this path is already unit-safe.
    asof_ord = int(asof_date.value // _NS_PER_DAY)
    return asof_ord * 100 + int(asof_num)


def build_batter_cache_for_fold_fast(
    corpus: pd.DataFrame,
    asof_keys: pd.DataFrame,
    folds: FoldAssignments,
    fold_id: int,
    out_path: Path,
) -> int:
    """Vectorized batter cache build.

    Same output as ``build_batter_cache_for_fold`` (verified by side-by-side
    test on a small slice), but **3–5× faster** in practice. The slow
    version filtered the corpus per (batter, asof) key — O(N) per key,
    repeated 450K times. This version sorts each batter's pitches once,
    then uses ``np.searchsorted`` on a composite ``(date, game_num)`` key
    for an O(log N) asof cutoff per key. The flattener cost
    (``build_batter_profile_vector``) is unchanged.

    Output schema and parquet format are identical to the slow version.
    """
    excluded_pks = set(folds.games_in_fold(fold_id))
    fold_corpus = corpus[~corpus["game_pk"].isin(excluded_pks)]
    pas_corpus = fold_corpus[fold_corpus["events"].notna()]

    # Sort once globally (per role+fold); each batter's view is then a
    # contiguous slice of the sorted DataFrame.
    sorted_pitches = (
        fold_corpus
        .sort_values(["batter", "game_date", "game_num"])
        .reset_index(drop=True)
    )
    sorted_pas = (
        pas_corpus
        .sort_values(["batter", "game_date", "game_num"])
        .reset_index(drop=True)
    )

    # Map batter_id → row indices in each sorted DataFrame.
    pitches_by_batter = sorted_pitches.groupby("batter", sort=False).indices
    pas_by_batter = sorted_pas.groupby("batter", sort=False).indices

    rows: list[dict] = []
    for batter_id, group_keys in asof_keys.groupby("batter", sort=False):
        batter_id = int(batter_id)

        if batter_id in pitches_by_batter:
            idx = pitches_by_batter[batter_id]
            batter_pitches = sorted_pitches.iloc[idx].reset_index(drop=True)
            p_keys = _composite_sort_key(
                batter_pitches["game_date"].to_numpy(),
                batter_pitches["game_num"].to_numpy(),
            )
        else:
            batter_pitches = pd.DataFrame(columns=sorted_pitches.columns)
            p_keys = np.array([], dtype=np.int64)

        if batter_id in pas_by_batter:
            idx = pas_by_batter[batter_id]
            batter_pas = sorted_pas.iloc[idx].reset_index(drop=True)
            pa_keys = _composite_sort_key(
                batter_pas["game_date"].to_numpy(),
                batter_pas["game_num"].to_numpy(),
            )
        else:
            batter_pas = pd.DataFrame(columns=sorted_pas.columns)
            pa_keys = np.array([], dtype=np.int64)

        for _, key in group_keys.iterrows():
            asof_date = pd.Timestamp(key["asof_date"])
            asof_num = int(key["asof_game_num"])
            ak = _composite_asof_key(asof_date, asof_num)

            cutoff_p = int(np.searchsorted(p_keys, ak, side="left"))
            cutoff_pa = int(np.searchsorted(pa_keys, ak, side="left"))

            filtered_pitches = batter_pitches.iloc[:cutoff_p]
            filtered_pas = batter_pas.iloc[:cutoff_pa]

            vector = build_batter_profile_vector(
                filtered_pitches, filtered_pas, asof_date
            )
            rows.append({
                "player_id": batter_id,
                "asof_date": asof_date.date(),
                "asof_game_num": asof_num,
                "fold_id": fold_id,
                "schema_version": PROFILE_SCHEMA_VERSION,
                "vector": vector.astype(np.float32).tolist(),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return len(rows)


def build_batter_cache_for_fold(
    corpus: pd.DataFrame,
    asof_keys: pd.DataFrame,
    folds: FoldAssignments,
    fold_id: int,
    out_path: Path,
) -> int:
    """Write the batter cache for one fold to ``out_path``. Returns row count.

    The batter flattener consumes both pitch-level data (for swing/whiff/xBA
    grids and chase rates) and PA-level data (for outcome rates and recent
    wOBA). PAs are derived as the subset of pitches with non-null ``events``
    (terminal pitches that mark the AB outcome).
    """
    excluded_pks = set(folds.games_in_fold(fold_id))
    fold_corpus = corpus[~corpus["game_pk"].isin(excluded_pks)]
    pitch_groups = fold_corpus.groupby("batter", sort=False)
    pas_corpus = fold_corpus[fold_corpus["events"].notna()]
    pa_groups = pas_corpus.groupby("batter", sort=False)

    rows: list[dict] = []
    empty_pitches = pd.DataFrame(columns=fold_corpus.columns)
    empty_pas = pd.DataFrame(columns=pas_corpus.columns)

    for _, key in asof_keys.iterrows():
        batter_id = int(key["batter"])
        asof_date = pd.Timestamp(key["asof_date"])
        asof_game_num = int(key["asof_game_num"])

        if batter_id in pitch_groups.groups:
            filtered_pitches = before_asof(
                pitch_groups.get_group(batter_id), asof_date, asof_game_num
            )
        else:
            filtered_pitches = empty_pitches

        if batter_id in pa_groups.groups:
            filtered_pas = before_asof(
                pa_groups.get_group(batter_id), asof_date, asof_game_num
            )
        else:
            filtered_pas = empty_pas

        vector = build_batter_profile_vector(filtered_pitches, filtered_pas, asof_date)
        rows.append({
            "player_id": batter_id,
            "asof_date": asof_date.date(),
            "asof_game_num": asof_game_num,
            "fold_id": fold_id,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "vector": vector.astype(np.float32).tolist(),
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fold-aware profile cache.")
    parser.add_argument(
        "--role", choices=["pitcher", "batter", "both"], default="both",
        help="Which role's cache to build (default: both).",
    )
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
        help="Output directory; per-(role, fold) parquets will be written here.",
    )
    args = parser.parse_args()

    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    out_dir = Path(args.out_dir)
    roles_to_build = ["pitcher", "batter"] if args.role == "both" else [args.role]

    print("Loading fold assignments...")
    folds = FoldAssignments.from_path(FOLDS_PATH)

    print("Loading corpus...")
    t0 = time.monotonic()
    corpus = _load_corpus(RAW_DIR)
    print(f"  loaded {len(corpus):,} pitches in {time.monotonic() - t0:.1f}s")

    print(f"\nBuilding profiles for roles={roles_to_build}, folds={fold_ids}, "
          f"schema_version={PROFILE_SCHEMA_VERSION}")
    print(f"Output dir: {out_dir.resolve()}\n")

    for role in roles_to_build:
        player_col = role  # "pitcher" or "batter" — also the column name in raw
        print(f"--- {role} ---")
        asof_keys = _discover_asof_keys(corpus, player_col=player_col)
        if args.max_games is not None:
            keep_pks = list(asof_keys["game_pk"].drop_duplicates().head(args.max_games))
            asof_keys = asof_keys[asof_keys["game_pk"].isin(keep_pks)].reset_index(drop=True)
            print(f"  --max-games={args.max_games}: limited to "
                  f"{len(asof_keys):,} ({role}, game) keys across "
                  f"{len(keep_pks)} distinct games")
        else:
            print(f"  {len(asof_keys):,} unique ({role}, game) cache keys to build")

        builder = (
            build_pitcher_cache_for_fold if role == "pitcher"
            else build_batter_cache_for_fold_fast
        )
        for fold_id in fold_ids:
            out_path = out_dir / f"{role}_fold_{fold_id}.parquet"
            t0 = time.monotonic()
            n = builder(corpus, asof_keys, folds, fold_id, out_path)
            elapsed = time.monotonic() - t0
            print(f"  fold {fold_id}: wrote {n:,} entries to {out_path.name} "
                  f"({elapsed:.1f}s)")

            # Aggregate to league mean. Per ADR 008 this inherits fold-awareness
            # from the per-player cache — same exclusion of fold_id's games.
            t0 = time.monotonic()
            player_cache = pd.read_parquet(out_path)
            league_df = compute_league_means(player_cache)
            league_path = out_dir / f"league_{role}_fold_{fold_id}.parquet"
            league_df.to_parquet(league_path, index=False)
            elapsed = time.monotonic() - t0
            print(f"  fold {fold_id}: wrote {len(league_df):,} league-mean entries "
                  f"to {league_path.name} ({elapsed:.1f}s)")
        print()


if __name__ == "__main__":
    main()
