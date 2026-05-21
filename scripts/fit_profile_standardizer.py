"""Refit the profile standardizer (``data/preprocess_artifacts/v1/profile_standardization.npz``).

Recomputes per-feature mean/std for the pitcher and batter profile vectors over
**training-period** (``asof_date <= TRAIN_END``) cache entries from fold 0, NaN-aware.
Constant features get std = 1.0 (so they standardize to ~0). Run this after rebuilding
the profile cache (e.g. after a schema/feature change such as ADR 013).

    python -m scripts.fit_profile_standardizer

The model's ``ProfileStandardizer`` (model/pitchgpt_dataset.py) loads the four arrays it
writes: ``pitcher_mean``, ``pitcher_std``, ``batter_mean``, ``batter_std``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import TRAIN_END

CACHE_DIR = Path("data/profiles")
OUT_PATH = Path("data/preprocess_artifacts/v1/profile_standardization.npz")


def _fit(role: str, fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / f"{role}_fold_{fold}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `make build-profile-cache` first")
    df = pd.read_parquet(path)
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    df = df[df["asof_date"] <= pd.Timestamp(TRAIN_END)]
    if df.empty:
        raise RuntimeError(f"no training-period entries in {path}")
    X = np.stack([np.asarray(v, dtype=np.float64) for v in df["vector"]])  # (N, D)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std).astype(np.float32)
    print(
        f"  {role}: fit over {len(df):,} training-period entries (fold {fold}); D={X.shape[1]}; "
        f"mean∈[{mean.min():.3g}, {mean.max():.3g}], std∈[{std.min():.3g}, {std.max():.3g}]"
    )
    return mean, std


def main() -> None:
    print(f"Refitting profile standardizer (TRAIN_END={TRAIN_END}) -> {OUT_PATH}")
    pm, ps = _fit("pitcher")
    bm, bs = _fit("batter")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, pitcher_mean=pm, pitcher_std=ps, batter_mean=bm, batter_std=bs)
    print(f"saved {OUT_PATH}: pitcher {pm.shape}, batter {bm.shape}")


if __name__ == "__main__":
    main()
