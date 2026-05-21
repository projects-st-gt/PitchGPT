"""In-memory loader for the profile cache.

Reads per-(role, fold) parquets into in-memory dicts and exposes a
``ProfileLookup``-shaped callable suitable for ``AtBatDataset``. Validates
the cache's schema version on load and refuses to serve a stale cache
(prevents the silent feature-slot misalignment failure mode).

Per ADR 008, each cache is fold-aware: training code that needs profiles
for an at-bat in fold k loads the cache for fold k, which was built with
games in fold k excluded. The loader's single-fold interface (one
``ProfileCache`` per fold) makes this explicit.

Usage:

    from data.profile_cache_loader import ProfileCache

    pc = ProfileCache(role="pitcher", fold_id=0,
                      profiles_dir=Path("data/profiles"))
    vec = pc.lookup(player_id=12345,
                    asof_date="2024-04-15",
                    asof_game_num=1)["vector"]

The lookup applies league-mean fallback for sparse rows and missing
players: per-player NaN slots are filled from the league mean, then
weighted with ``profile_confidence``.

Wiring into ``AtBatDataset`` (the ``ProfileLookup`` contract is just a
``(player_id, asof_date, asof_game_num) -> {"vector": ndarray}`` callable):

    from data.dataset import AtBatDataset
    from data.profile_cache_loader import ProfileCache

    pc_p = ProfileCache(role="pitcher", fold_id=k)
    pc_b = ProfileCache(role="batter",  fold_id=k)
    ds = AtBatDataset(
        pitches=tagged_pitches,
        pitcher_profile_lookup=pc_p.lookup,
        batter_profile_lookup=pc_b.lookup,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.profile_cache import (
    BATTER_FEATURE_INDEX,
    BATTER_VECTOR_LEN,
    PITCHER_FEATURE_INDEX,
    PITCHER_VECTOR_LEN,
    PROFILE_SCHEMA_VERSION,
    blend_with_league_mean,
)

DEFAULT_PROFILES_DIR = Path("data/profiles")


class ProfileCache:
    """Loaded per-(role, fold) profile cache with NaN-fill league fallback."""

    def __init__(
        self,
        role: str,
        fold_id: int,
        profiles_dir: Path | None = None,
    ):
        if role not in ("pitcher", "batter"):
            raise ValueError(f"role must be 'pitcher' or 'batter', got {role!r}")
        self.role = role
        self.fold_id = int(fold_id)
        self._profiles_dir = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
        self._vector_len = (
            PITCHER_VECTOR_LEN if role == "pitcher" else BATTER_VECTOR_LEN
        )
        self._confidence_idx = (
            PITCHER_FEATURE_INDEX["profile_confidence"]
            if role == "pitcher"
            else BATTER_FEATURE_INDEX["profile_confidence"]
        )

        self._player_lookup: dict[tuple[int, pd.Timestamp, int], np.ndarray] = {}
        self._league_lookup: dict[tuple[pd.Timestamp, int], np.ndarray] = {}
        self._load_player_cache()
        self._load_league_cache()

    # ---------- loading ----------

    def _load_player_cache(self) -> None:
        path = self._profiles_dir / f"{self.role}_fold_{self.fold_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"per-player cache missing at {path}; run "
                f"`make build-profile-cache` first"
            )
        df = pd.read_parquet(path)
        self._verify_schema(df, path)
        for _, row in df.iterrows():
            key = (
                int(row["player_id"]),
                pd.Timestamp(row["asof_date"]),
                int(row["asof_game_num"]),
            )
            self._player_lookup[key] = np.asarray(row["vector"], dtype=np.float32)

    def _load_league_cache(self) -> None:
        path = self._profiles_dir / f"league_{self.role}_fold_{self.fold_id}.parquet"
        if not path.exists():
            # League cache missing is a warning, not a hard failure: the loader
            # falls back to zeros for missing slots. Build script should have
            # produced this.
            return
        df = pd.read_parquet(path)
        self._verify_schema(df, path)
        for _, row in df.iterrows():
            key = (
                pd.Timestamp(row["asof_date"]),
                int(row["asof_game_num"]),
            )
            self._league_lookup[key] = np.asarray(row["vector"], dtype=np.float32)

    def _verify_schema(self, df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            # Empty cache file is allowed (e.g., a fold with no data, or a
            # test fixture). Schema can't be verified, but there's nothing
            # to serve, so the loader degrades gracefully to falling through
            # to league-only or zero-fallback.
            return
        versions = df["schema_version"].unique()
        if len(versions) != 1:
            raise RuntimeError(
                f"{path} mixes schema versions {versions.tolist()}; rebuild"
            )
        if int(versions[0]) != PROFILE_SCHEMA_VERSION:
            raise RuntimeError(
                f"{path} has schema_version {versions[0]}; loader expects "
                f"{PROFILE_SCHEMA_VERSION}. Rebuild via `make build-profile-cache`."
            )
        sample = np.asarray(df["vector"].iloc[0])
        if len(sample) != self._vector_len:
            raise RuntimeError(
                f"{path} vector length {len(sample)} != expected {self._vector_len}"
            )

    # ---------- lookup ----------

    def lookup(
        self,
        player_id: int,
        asof_date: pd.Timestamp | str,
        asof_game_num: int,
    ) -> dict:
        """Return the blended profile vector for this (player, asof) key.

        Three-step fallback chain:

        1. **Per-player exists.** Blend with league-mean via
           ``blend_with_league_mean``: NaN slots get league-mean values;
           non-NaN slots get confidence-weighted blend.
        2. **Per-player missing, league exists.** Use the league-mean vector
           outright (debut player at a known asof).
        3. **Both missing.** Zero vector. Source flagged so caller can tell.

        After steps 1-2, any remaining NaN in the result (which can happen
        when *both* the per-player slot and the league-mean slot are NaN —
        typical for the very first days of the corpus, when no player has
        a 30-day prior window) is replaced with 0. The model receives
        purely numeric input; sparse-data slots are still distinguishable
        from real zeros via paired count features (``n_pitches``,
        ``n_pas``, ``profile_confidence``, etc.).
        """
        asof_ts = pd.Timestamp(asof_date)
        league_key = (asof_ts, int(asof_game_num))
        league_vec = self._league_lookup.get(league_key)

        player_key = (int(player_id), asof_ts, int(asof_game_num))
        player_vec = self._player_lookup.get(player_key)

        if player_vec is not None:
            confidence = float(player_vec[self._confidence_idx])
            if league_vec is None:
                fallback = np.zeros(self._vector_len, dtype=np.float32)
                vec = np.where(np.isnan(player_vec), fallback, player_vec).astype(
                    np.float32
                )
            else:
                vec = blend_with_league_mean(player_vec, league_vec, confidence)
            vec = np.nan_to_num(vec, nan=0.0).astype(np.float32)
            return {"vector": vec, "source": "per_player_blended"}

        if league_vec is not None:
            vec = np.nan_to_num(league_vec.copy(), nan=0.0).astype(np.float32)
            return {"vector": vec, "source": "league_only"}

        return {
            "vector": np.zeros(self._vector_len, dtype=np.float32),
            "source": "zero_fallback",
        }

    def __len__(self) -> int:
        return len(self._player_lookup)

    def __repr__(self) -> str:
        return (
            f"ProfileCache(role={self.role!r}, fold_id={self.fold_id}, "
            f"n_player_entries={len(self._player_lookup):,}, "
            f"n_league_entries={len(self._league_lookup):,})"
        )
