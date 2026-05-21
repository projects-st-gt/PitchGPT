"""Player MLBAM-id → name lookup, backed by the Chadwick Bureau register.

The Chadwick Bureau publishes an open registry of MLB player IDs and names.
``pybaseball.chadwick_register()`` downloads + caches a ~5MB CSV with
``key_mlbam`` → ``name_first`` / ``name_last``. We thin this to a single
parquet cached under ``data/preprocess_artifacts/`` so subsequent boots are
near-instant.

Lazy-loaded at first call. Returns ``"First Last"``; falls back to the raw ID
string when the player isn't in the register (rare — typically only debutants
whose record hasn't propagated yet).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_PATH = Path("data/preprocess_artifacts/player_names.parquet")


class _NameCache:
    _df: Optional[pd.DataFrame] = None

    @classmethod
    def _build(cls) -> pd.DataFrame:
        """Build/refresh the player-names parquet from Chadwick. ~5MB download."""
        try:
            from pybaseball import chadwick_register
        except ImportError as e:
            raise ImportError(
                "pybaseball is required for player name lookup; "
                "see pyproject.toml dependencies"
            ) from e
        reg = chadwick_register()
        # Filter to MLB players (ones with an mlbam id and an MLB debut year).
        reg = reg.dropna(subset=["key_mlbam"]).copy()
        reg["key_mlbam"] = reg["key_mlbam"].astype(int)
        reg["name_first"] = reg["name_first"].fillna("")
        reg["name_last"] = reg["name_last"].fillna("")
        df = reg[["key_mlbam", "name_first", "name_last"]].drop_duplicates(
            "key_mlbam", keep="last"
        )
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)
        return df

    @classmethod
    def load(cls) -> pd.DataFrame:
        if cls._df is not None:
            return cls._df
        if CACHE_PATH.exists():
            cls._df = pd.read_parquet(CACHE_PATH)
        else:
            print(f"[player_names] building cache (one-time, ~5MB download)...")
            cls._df = cls._build()
            print(f"[player_names] cached {len(cls._df):,} players to {CACHE_PATH}")
        return cls._df

    @classmethod
    def lookup(cls, mlbam_id: int) -> Optional[str]:
        df = cls.load()
        row = df[df["key_mlbam"] == int(mlbam_id)]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        first = str(r["name_first"]).strip()
        last = str(r["name_last"]).strip()
        if first and last:
            return f"{first} {last}"
        if last:
            return last
        if first:
            return first
        return None


def name_for_mlbam(mlbam_id: Optional[int]) -> Optional[str]:
    """Public: get a player's name from their MLBAM id.

    Returns None if the lookup fails (debutants, malformed ids). Callers
    should fall back to displaying the raw id in that case.
    """
    if mlbam_id is None:
        return None
    try:
        return _NameCache.lookup(int(mlbam_id))
    except Exception as e:
        print(f"[player_names] lookup failed for {mlbam_id!r}: {e}")
        return None
