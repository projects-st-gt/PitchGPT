"""Train + evaluate the profile-aware LSTM baseline.

Same XGBoost-style per-pitch features, plus per-AB profile vectors from
the profile cache (pitcher + batter, concatenated to 314 dims). Uses
fold-0 cache for both training and val — see ``lstm_with_profiles.py``
for the fold-aware caveat.

Trains on the temporal-train split (≤ 2023-12-31), evaluates on val
(2024-01-01 to 2024-07-15). Reports the same metrics as the XGBoost
runner so the eval table extends cleanly.
"""

from __future__ import annotations

import os

# OpenMP cap before any C-extension imports (same workaround as XGBoost).
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
from eval.baselines.lstm_with_profiles import LSTMBaselineWithProfiles
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


def _load_split(year_dirs, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
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


def _make_ab_profile_lookup(df: pd.DataFrame, pcache: ProfileCache,
                             bcache: ProfileCache):
    """Build a fast (ab_id) → 314-dim profile vector callable.

    The lookup is keyed by ``game_pk * 100 + at_bat_number``. It joins
    pitcher and batter profile vectors at each AB's game start date.
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
    ab_info["ab_id"] = (
        ab_info["game_pk"].astype(np.int64) * 100
        + ab_info["at_bat_number"].astype(np.int64)
    )
    info_dict = {
        int(row["ab_id"]): (
            int(row["pitcher"]),
            int(row["batter"]),
            pd.Timestamp(row["game_date"]),
        )
        for _, row in ab_info.iterrows()
    }

    def lookup(ab_id: int) -> np.ndarray:
        pitcher_id, batter_id, asof_date = info_dict[int(ab_id)]
        pvec = pcache.lookup(pitcher_id, asof_date, 1)["vector"]
        bvec = bcache.lookup(batter_id, asof_date, 1)["vector"]
        return np.concatenate([pvec, bvec]).astype(np.float32)

    return lookup


def _evaluate(name: str, probs, targets, ab_ids):
    print(f"\n=== {name} ===")
    point, lo, hi = bootstrap_metric(
        lambda p, t: top_k_accuracy(p, t, k=1),
        ab_ids, probs, targets,
        n_bootstraps=300, seed=0,
    )
    print(f"  top-1 accuracy : {point:.4f}  [{lo:.4f}, {hi:.4f}]")

    point3, lo3, hi3 = bootstrap_metric(
        lambda p, t: top_k_accuracy(p, t, k=3),
        ab_ids, probs, targets,
        n_bootstraps=300, seed=0,
    )
    print(f"  top-3 accuracy : {point3:.4f}  [{lo3:.4f}, {hi3:.4f}]")

    ece = expected_calibration_error(probs, targets, n_bins=15)
    print(f"  ECE (15 bins)  : {ece:.4f}")
    brier = brier_score(probs, targets)
    print(f"  Brier score    : {brier:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-games", type=int, default=None,
                        help="Limit training to first N games (for quick smoke tests).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--zero-profile", action="store_true",
                        help="zero the 314-dim profile vectors (diagnostic: isolates how much "
                             "of the LSTM's accuracy comes from the profile-in-h0 vs the per-pitch features)")
    parser.add_argument("--standardize-profile", action="store_true",
                        help="per-feature z-score the 314-dim profile before feeding it to h0/c0 "
                             "(otherwise raw spin-rate/velo magnitudes blow up h0 and saturate the LSTM gates)")
    args = parser.parse_args()

    year_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())

    print("Loading training split (≤ 2023-12-31)...")
    t0 = time.monotonic()
    train = _load_split(year_dirs, pd.Timestamp("2017-01-01"), TRAIN_END)
    print(f"  loaded {len(train):,} pitches in {time.monotonic() - t0:.1f}s")
    train = harmonize_dataframe(train, drop_unmapped=True)
    # CRITICAL: the raw Statcast parquets store pitches NON-chronologically
    # within an at-bat. _group_by_ab preserves row order, so without this sort
    # the LSTM would consume ABs in (mostly reverse) order — which turns the
    # `prev_pitch_id` feature into target leakage. Sort chronologically here.
    train = train.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    if args.max_train_games is not None:
        keep_pks = list(train["game_pk"].drop_duplicates().head(args.max_train_games))
        train = train[train["game_pk"].isin(keep_pks)].reset_index(drop=True)
        print(f"  --max-train-games={args.max_train_games}: limited to "
              f"{len(train):,} pitches")
    else:
        print(f"  after harmonization: {len(train):,} pitches")

    print("\nLoading val split (2024-01-01 to 2024-07-15)...")
    t0 = time.monotonic()
    val = _load_split(year_dirs, VAL_START, VAL_END)
    print(f"  loaded {len(val):,} pitches in {time.monotonic() - t0:.1f}s")
    val = harmonize_dataframe(val, drop_unmapped=True)
    val = val.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)  # see train sort above — anti-leak
    print(f"  after harmonization: {len(val):,} pitches")

    print("\nLoading metadata + arsenal encoding...")
    metadata = load_game_metadata()
    arsenal = compute_pitcher_arsenal_encoding(train, alpha=10.0)
    print(f"  metadata: {len(metadata):,} games; arsenal: {len(arsenal):,} pitchers")

    print("\nBuilding LSTM features (same as XGBoost)...")
    t0 = time.monotonic()
    train_feats = build_xgboost_features(train, metadata, arsenal_encoding=arsenal)
    val_feats = build_xgboost_features(val, metadata, arsenal_encoding=arsenal)
    print(f"  features in {time.monotonic() - t0:.1f}s; "
          f"{len(train_feats.columns)} columns")

    train_targets = train["pitch_type_canonical"].map(PITCH_TYPE_TO_ID).to_numpy(dtype=np.int64)
    val_targets = val["pitch_type_canonical"].map(PITCH_TYPE_TO_ID).to_numpy(dtype=np.int64)
    train_ab_ids = (
        train["game_pk"].astype(np.int64) * 100 + train["at_bat_number"].astype(np.int64)
    ).to_numpy()
    val_ab_ids = (
        val["game_pk"].astype(np.int64) * 100 + val["at_bat_number"].astype(np.int64)
    ).to_numpy()

    print("\nLoading profile caches (fold 0)...")
    t0 = time.monotonic()
    pcache = ProfileCache(role="pitcher", fold_id=0)
    bcache = ProfileCache(role="batter", fold_id=0)
    print(f"  loaded in {time.monotonic() - t0:.1f}s: {pcache!r}  {bcache!r}")

    n_profile = PITCHER_VECTOR_LEN + BATTER_VECTOR_LEN
    print(f"  n_profile = {n_profile} (={PITCHER_VECTOR_LEN} pitcher + {BATTER_VECTOR_LEN} batter)")

    print("\nBuilding profile lookup callables...")
    t0 = time.monotonic()
    train_lookup = _make_ab_profile_lookup(train, pcache, bcache)
    val_lookup = _make_ab_profile_lookup(val, pcache, bcache)
    print(f"  built in {time.monotonic() - t0:.1f}s")
    if args.zero_profile:
        _zero = np.zeros(n_profile, dtype=np.float32)
        train_lookup = lambda *a, **k: _zero
        val_lookup = lambda *a, **k: _zero
        print("  --zero-profile: profile vectors zeroed (h0/c0 ≈ learned constant; "
              "model sees only the per-pitch features)")
    elif args.standardize_profile:
        from model.pitchgpt_dataset import ProfileStandardizer
        _std = ProfileStandardizer()  # fit on training-period cache entries; leak-safe

        def _wrap(orig):
            def _f(ab_id):
                v = orig(ab_id)
                p = _std.apply(v[:PITCHER_VECTOR_LEN], "pitcher")
                b = _std.apply(v[PITCHER_VECTOR_LEN:], "batter")
                return np.nan_to_num(np.concatenate([p, b]), nan=0.0).astype(np.float32)
            return _f
        train_lookup, val_lookup = _wrap(train_lookup), _wrap(val_lookup)
        print("  --standardize-profile: profile vectors z-scored per feature before h0/c0")

    print(f"\nTraining LSTM (hidden={args.hidden_dim}, layers={args.num_layers}, "
          f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})...")
    t0 = time.monotonic()
    lstm = LSTMBaselineWithProfiles(
        n_profile=n_profile,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    ).fit(train_feats, train_targets, train_ab_ids, train_lookup)
    print(f"  trained in {time.monotonic() - t0:.1f}s")

    print("\nPredicting on val...")
    t0 = time.monotonic()
    probs = lstm.predict_proba(val_feats, val_ab_ids, val_lookup)
    print(f"  predicted in {time.monotonic() - t0:.1f}s")

    _evaluate("LSTM (with profiles)", probs, val_targets, val_ab_ids)


if __name__ == "__main__":
    main()
