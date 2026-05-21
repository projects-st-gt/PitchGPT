"""Pitch type harmonization.

Statcast emits ~15 pitch_type codes; this module collapses them to 7
canonical types per the ``statcast-pipeline`` skill. Unmapped codes
(eephus, intentional ball, etc.) signal at-bats that should be dropped
during preprocessing — never silently coerced.

Update only ``PITCH_TYPE_MAP``; do not inline the mapping at call sites.
"""

from __future__ import annotations

import pandas as pd

PITCH_TYPE_MAP: dict[str, str] = {
    "FF": "FF", "FA": "FF",
    "SI": "SI", "FT": "SI",
    "FC": "FC", "CT": "FC",
    "SL": "SL", "ST": "SL", "SV": "SL",
    "CU": "CU", "KC": "CU", "CS": "CU", "KN": "CU",
    "CH": "CH", "FO": "CH",
    "FS": "FS",
}

CANONICAL_TYPES: list[str] = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]


def harmonize_pitch_type(code: str | None) -> str | None:
    """Map a raw Statcast pitch_type code to a canonical type, or None if unmapped."""
    if code is None:
        return None
    return PITCH_TYPE_MAP.get(code)


def harmonize_dataframe(df: pd.DataFrame, drop_unmapped: bool = True) -> pd.DataFrame:
    """Add ``pitch_type_canonical`` to ``df``.

    When ``drop_unmapped`` is True, drop every at-bat that contains any unmapped
    pitch (joined on ``(game_pk, at_bat_number)``). The whole AB is dropped — not
    just the offending pitch — because partial sequences corrupt the autoregressive
    training signal.
    """
    if "pitch_type" not in df.columns:
        raise KeyError("expected 'pitch_type' column in input DataFrame")

    df = df.copy()
    df["pitch_type_canonical"] = df["pitch_type"].map(PITCH_TYPE_MAP)

    if drop_unmapped and {"game_pk", "at_bat_number"}.issubset(df.columns):
        bad_ab = (
            df.loc[df["pitch_type_canonical"].isna(), ["game_pk", "at_bat_number"]]
            .drop_duplicates()
            .assign(_drop=True)
        )
        if len(bad_ab):
            merged = df.merge(bad_ab, on=["game_pk", "at_bat_number"], how="left")
            df = merged[merged["_drop"].isna()].drop(columns="_drop").copy()

    return df
