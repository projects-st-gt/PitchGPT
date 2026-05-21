"""Apples-to-apples XGBoost with the full profile cache.

Same setup as ``run_simple_baselines.py``'s XGBoost block, BUT augments
each pitch's feature row with the 314-dim profile vector for its AB
(pitcher 223 + batter 91, looked up from the fold-0 ProfileCache and
broadcast per-pitch). This makes XGBoost see what the LSTM saw and
isolates the value of temporal modeling.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import (
    PITCH_TYPE_TO_ID,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from data.harmonization import harmonize_dataframe
from data.profile_cache import BATTER_VECTOR_LEN, PITCHER_VECTOR_LEN
from data.profile_cache_loader import ProfileCache
from data.xgboost_features import (
    build_xgboost_features,
    compute_pitcher_arsenal_encoding,
    load_game_metadata,
)
from eval.baselines.xgboost_baseline import XGBoostBaseline
from eval.metrics.bootstrap import bootstrap_metric
from eval.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    top_k_accuracy,
)

RAW_DIR = Path("data/raw")
NEEDED_COLUMNS = [
    "game_pk", "at_bat_number", "pitch_number",
    "game_date",
    "pitcher", "batter",
    "pitch_type",
    "balls", "strikes", "outs_when_up", "inning",
    "on_1b", "on_2b", "on_3b",
    "p_throws", "stand",
]


def _load_split(year_dirs, start, end):
    parts = []
    for yd in year_dirs:
        for p in sorted(yd.glob("*.parquet")):
            df = pd.read_parquet(p, columns=NEEDED_COLUMNS)
            df["game_date"] = pd.to_datetime(df["game_date"])
            df = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
            if len(df):
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=NEEDED_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _build_profile_matrix(
    df: pd.DataFrame,
    pcache: ProfileCache,
    bcache: ProfileCache,
) -> np.ndarray:
    """Return an (n_pitches, 314) matrix of per-AB profile broadcast per pitch.

    For each pitch in ``df``, looks up its AB's pitcher + batter profile
    vectors and concatenates. Vectors for the same AB are identical across
    pitches (the profile is per-AB, not per-pitch).
    """
    ab_info = (
        df.groupby(["game_pk", "at_bat_number"])
        .agg(
            pitcher=("pitcher", "first"),
            batter=("batter", "first"),
            game_date=("game_date", "first"),
        )
        .reset_index()
    )
    n_profile = PITCHER_VECTOR_LEN + BATTER_VECTOR_LEN
    print(f"  building profile cache for {len(ab_info):,} unique ABs...")
    t0 = time.monotonic()

    ab_profiles: dict[tuple[int, int], np.ndarray] = {}
    for _, row in ab_info.iterrows():
        pitcher_id = int(row["pitcher"])
        batter_id = int(row["batter"])
        asof_date = pd.Timestamp(row["game_date"])
        pvec = pcache.lookup(pitcher_id, asof_date, 1)["vector"]
        bvec = bcache.lookup(batter_id, asof_date, 1)["vector"]
        ab_profiles[
            (int(row["game_pk"]), int(row["at_bat_number"]))
        ] = np.concatenate([pvec, bvec]).astype(np.float32)
    print(f"  built profile dict in {time.monotonic() - t0:.1f}s")

    print(f"  broadcasting profile vectors across {len(df):,} pitches...")
    t0 = time.monotonic()
    out = np.empty((len(df), n_profile), dtype=np.float32)
    game_pks = df["game_pk"].astype(np.int64).to_numpy()
    ab_nums = df["at_bat_number"].astype(np.int64).to_numpy()
    for i in range(len(df)):
        out[i] = ab_profiles[(int(game_pks[i]), int(ab_nums[i]))]
    print(f"  broadcast in {time.monotonic() - t0:.1f}s")
    return out


def _evaluate(name, probs, targets, ab_ids):
    print(f"\n=== {name} ===")
    point, lo, hi = bootstrap_metric(
        lambda p, t: top_k_accuracy(p, t, k=1),
        ab_ids, probs, targets, n_bootstraps=300, seed=0,
    )
    print(f"  top-1 accuracy : {point:.4f}  [{lo:.4f}, {hi:.4f}]")
    point3, lo3, hi3 = bootstrap_metric(
        lambda p, t: top_k_accuracy(p, t, k=3),
        ab_ids, probs, targets, n_bootstraps=300, seed=0,
    )
    print(f"  top-3 accuracy : {point3:.4f}  [{lo3:.4f}, {hi3:.4f}]")
    ece = expected_calibration_error(probs, targets, n_bins=15)
    print(f"  ECE (15 bins)  : {ece:.4f}")
    brier = brier_score(probs, targets)
    print(f"  Brier score    : {brier:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument(
        "--subsample-train", type=int, default=None,
        help="Random subsample N training pitches (memory cap). "
             "Recommended ≤ 1_500_000 on 24 GB machines with 340 features.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    year_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())

    print("Loading training split...")
    t0 = time.monotonic()
    train = _load_split(year_dirs, pd.Timestamp("2017-01-01"), TRAIN_END)
    print(f"  loaded {len(train):,} pitches in {time.monotonic() - t0:.1f}s")
    train = harmonize_dataframe(train, drop_unmapped=True)
    print(f"  after harmonization: {len(train):,} pitches")

    if args.subsample_train is not None and len(train) > args.subsample_train:
        rng = np.random.default_rng(args.seed)
        # Subsample BY AB so we don't split mid-AB. Per-AB sample preserves
        # the sequence structure for the prev_pitch_id feature.
        unique_abs = (
            train["game_pk"].astype(np.int64) * 100
            + train["at_bat_number"].astype(np.int64)
        ).unique()
        # Estimate AB count needed for ~target pitch count
        avg_pitches_per_ab = len(train) / len(unique_abs)
        target_abs = int(args.subsample_train / avg_pitches_per_ab)
        chosen_abs = set(rng.choice(unique_abs, size=target_abs, replace=False).tolist())
        train_ab_ids = (
            train["game_pk"].astype(np.int64) * 100
            + train["at_bat_number"].astype(np.int64)
        )
        train = train[train_ab_ids.isin(chosen_abs)].reset_index(drop=True)
        print(f"  subsampled to {len(train):,} pitches "
              f"({len(chosen_abs):,} ABs) — memory cap")

    print("\nLoading val split...")
    t0 = time.monotonic()
    val = _load_split(year_dirs, VAL_START, VAL_END)
    print(f"  loaded {len(val):,} pitches in {time.monotonic() - t0:.1f}s")
    val = harmonize_dataframe(val, drop_unmapped=True)
    print(f"  after harmonization: {len(val):,} pitches")

    print("\nLoading metadata + arsenal encoding...")
    metadata = load_game_metadata()
    arsenal = compute_pitcher_arsenal_encoding(train, alpha=10.0)
    print(f"  metadata: {len(metadata):,} games; arsenal: {len(arsenal):,} pitchers")

    print("\nBuilding base XGBoost features (26 cols)...")
    t0 = time.monotonic()
    train_base = build_xgboost_features(train, metadata, arsenal_encoding=arsenal)
    val_base = build_xgboost_features(val, metadata, arsenal_encoding=arsenal)
    print(f"  built in {time.monotonic() - t0:.1f}s; train {train_base.shape}, val {val_base.shape}")

    print("\nLoading profile caches (fold 0)...")
    t0 = time.monotonic()
    pcache = ProfileCache(role="pitcher", fold_id=0)
    bcache = ProfileCache(role="batter", fold_id=0)
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    print("\nBuilding train profile matrix...")
    train_prof = _build_profile_matrix(train, pcache, bcache)
    print(f"  shape: {train_prof.shape}")

    print("\nBuilding val profile matrix...")
    val_prof = _build_profile_matrix(val, pcache, bcache)
    print(f"  shape: {val_prof.shape}")

    print("\nAssembling full feature DataFrames...")
    t0 = time.monotonic()
    n_profile = train_prof.shape[1]
    prof_cols = [f"prof_{i}" for i in range(n_profile)]
    train_prof_df = pd.DataFrame(train_prof, columns=prof_cols, index=train_base.index)
    val_prof_df = pd.DataFrame(val_prof, columns=prof_cols, index=val_base.index)
    train_full = pd.concat([train_base, train_prof_df], axis=1)
    val_full = pd.concat([val_base, val_prof_df], axis=1)
    print(f"  assembled in {time.monotonic() - t0:.1f}s; train_full {train_full.shape}")

    train_targets = train["pitch_type_canonical"].map(PITCH_TYPE_TO_ID).to_numpy(dtype=np.int64)
    val_targets = val["pitch_type_canonical"].map(PITCH_TYPE_TO_ID).to_numpy(dtype=np.int64)
    val_ab_ids = (
        val["game_pk"].astype(np.int64) * 100 + val["at_bat_number"].astype(np.int64)
    ).to_numpy()

    print(f"\nTraining XGBoost (n_estimators={args.n_estimators}, max_depth={args.max_depth}, n_jobs=1)...")
    t0 = time.monotonic()
    model = XGBoostBaseline(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.1,
        n_jobs=1,
    ).fit(train_full, train_targets)
    print(f"  trained in {time.monotonic() - t0:.1f}s")

    print("\nPredicting on val...")
    t0 = time.monotonic()
    probs = model.predict_proba(val_full)
    print(f"  predicted in {time.monotonic() - t0:.1f}s")

    _evaluate("XGBoost (with full profile)", probs, val_targets, val_ab_ids)

    print("\nTop-20 feature importances (gain):")
    print(model.feature_importance().head(20).to_string())


if __name__ == "__main__":
    main()
