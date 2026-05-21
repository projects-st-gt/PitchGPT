"""Zone tagging.

Two zone schemes coexist in this project:

- **Action zone** (5 cells, ADR 001) — the action space for causal counterfactuals.
  Cells: ``up``, ``down``, ``arm-side``, ``glove-side``, ``out-of-zone``.
- **Feature zone v2** (13 cells, SIS / MLB-Statcast scheme) — what the model
  observes about location. 9 in-zone cells (3×3 grid) + 4 OOZ quadrants.
  Statcast publishes this directly in the raw ``zone`` column with labels
  1–9 + 11–14 (no zone 10); ``assign_feature_zone_14`` maps those external
  SIS labels to dense internal indices [0, 13).

The legacy ``Feature zone v1`` (26 cells = 5×5 grid + OOZ, computed from
``plate_x``/``plate_z``) remains as ``assign_feature_zone`` for back-compat;
v1 is sparse in practice (only the middle 3 of 5 x-columns get populated)
and is deprecated in favor of v2 plus Statcast's native zone column.

Both schemes consume the same raw inputs (``plate_x``, ``plate_z``,
``sz_top``, ``sz_bot``, plus ``p_throws`` for the action zone, plus the
Statcast ``zone`` column for v2). Use the action zone for causal
interventions; use the feature zone for model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MLB plate is 17 inches; half-width ≈ 0.83 ft. Statcast plate_x in feet.
PLATE_HALF_WIDTH_FT = 0.83

# Below this we treat sz_top/sz_bot as a measurement error and drop the row
# at dataset stage (see ADR 001 boundary handling).
MIN_ZONE_HEIGHT_FT = 1.0

ACTION_ZONES: list[str] = ["up", "down", "arm-side", "glove-side", "out-of-zone"]

# ---------- Feature zone v1 (legacy 26-class, deprecated) ----------
FEATURE_ZONE_OUT_OF_ZONE = 25  # cells 0-24 are the 5x5 in-zone grid

# ---------- Feature zone v2 (Statcast SIS 14-zone, 13 actual indices) ----------
# In-zone: 9 cells (3×3 grid, SIS labels 1-9, internal indices 0-8)
# OOZ: 4 quadrants (SIS labels 11-14, internal indices 9-12)
# 13 actual indices total. "14-zone" refers to the SIS label scheme; index 10
# (SIS label 10) never exists in Statcast data.
N_IN_ZONE_CELLS_14: int = 9
N_FEATURE_ZONES_14: int = 13
FEATURE_ZONE_OOZ_START_14: int = 9   # internal indices [9, 13) are OOZ
FEATURE_ZONE_OOZ_END_14: int = 13    # exclusive

# External (SIS) label → internal (model) dense index. Labels 1-9 are the
# in-zone 3×3 grid (top-left = SIS 1 = internal 0, reading rows top→bottom,
# left→right); labels 11-14 are the OOZ quadrants (UL, UR, LL, LR).
SIS_TO_INTERNAL: dict[int, int] = {
    1: 0, 2: 1, 3: 2,      # top row in-zone
    4: 3, 5: 4, 6: 5,      # middle row in-zone
    7: 6, 8: 7, 9: 8,      # bottom row in-zone
    11: 9,  12: 10,         # upper-left, upper-right OOZ
    13: 11, 14: 12,         # lower-left, lower-right OOZ
}
# Inverse for display/API: internal index → SIS label.
INTERNAL_TO_SIS: dict[int, int] = {v: k for k, v in SIS_TO_INTERNAL.items()}


def valid_zone_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows with a sane strike-zone height; caller drops the rest."""
    return (df["sz_top"] - df["sz_bot"]) >= MIN_ZONE_HEIGHT_FT


def assign_action_zone(df: pd.DataFrame) -> pd.Categorical:
    """5-zone categorical per ADR 001.

    Required columns: ``plate_x``, ``plate_z``, ``sz_top``, ``sz_bot``, ``p_throws``.
    """
    required = {"plate_x", "plate_z", "sz_top", "sz_bot", "p_throws"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    sz_height = df["sz_top"] - df["sz_bot"]
    z_norm = (df["plate_z"] - df["sz_bot"]) / sz_height
    x_norm = df["plate_x"] / PLATE_HALF_WIDTH_FT
    in_zone = z_norm.between(0, 1) & x_norm.abs().le(1)

    # arm_sign: arm-side is positive (plate_x · arm_sign). RHP arm-side = +x; LHP = -x.
    arm_sign = pd.Series(np.where(df["p_throws"] == "L", -1, 1), index=df.index)
    arm_side_signed = x_norm * arm_sign

    in_up = in_zone & (z_norm > 0.67)
    in_down = in_zone & (z_norm < 0.33)
    in_mid = in_zone & ~in_up & ~in_down  # 0.33 ≤ z_norm ≤ 0.67
    arm_side = in_mid & (arm_side_signed > 0)
    glove_side = in_mid & (arm_side_signed <= 0)

    out = pd.Series("out-of-zone", index=df.index, dtype="object")
    out.loc[in_up] = "up"
    out.loc[in_down] = "down"
    out.loc[arm_side] = "arm-side"
    out.loc[glove_side] = "glove-side"

    return pd.Categorical(out, categories=ACTION_ZONES, ordered=False)


def assign_feature_zone(df: pd.DataFrame) -> pd.Series:
    """26-class integer label for the model's location observation.

    Cells 0–24: 5 vertical bands × 5 horizontal columns inside the zone.
    Cell 25: out-of-zone. Bands are even quintiles of normalized height;
    columns clip ``plate_x`` to ``[-1.5, 1.5]`` ft and split into 5 columns.

    Required columns: ``plate_x``, ``plate_z``, ``sz_top``, ``sz_bot``.
    """
    required = {"plate_x", "plate_z", "sz_top", "sz_bot"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    sz_height = df["sz_top"] - df["sz_bot"]
    z_norm = (df["plate_z"] - df["sz_bot"]) / sz_height
    x_norm = df["plate_x"] / PLATE_HALF_WIDTH_FT
    in_zone = z_norm.between(0, 1) & x_norm.abs().le(1)

    # 5 z-bands, 0-4 from bottom to top.
    z_band = np.clip((z_norm * 5).astype(int), 0, 4)
    # 5 x-cols, 0-4 left to right; clip to [-1.5, 1.5] then map to [0, 5).
    x_clipped = np.clip(df["plate_x"], -1.5, 1.5)
    x_band = np.clip(((x_clipped + 1.5) / 3.0 * 5).astype(int), 0, 4)

    cell = pd.Series((5 * z_band + x_band).astype("int16"), index=df.index)
    cell.loc[~in_zone] = FEATURE_ZONE_OUT_OF_ZONE
    return cell.astype("int16")


def assign_feature_zone_14(df: pd.DataFrame) -> pd.Series:
    """13-class integer label for the model's location observation (Statcast SIS scheme).

    Maps Statcast's native ``zone`` column (SIS labels 1-9 in-zone + 11-14 OOZ
    quadrants, no zone 10) to dense internal indices [0, 13):

    - Internal 0..8: in-zone 3×3 grid (top-left = 0, reading rows top→bottom)
    - Internal 9..12: OOZ quadrants (upper-left, upper-right, lower-left,
      lower-right)

    The Statcast classifier uses the per-batter strike zone (``sz_top``,
    ``sz_bot``), so this is more umpire-accurate than the legacy v1 scheme
    that computed cells from a flat plate_x box.

    Required column: ``zone``. NaN entries (intentional walks since 2017
    record 4 ``automatic_ball`` rows per IBB with no pitch data, and
    pitch-clock violations since 2023 generate ``automatic_ball`` /
    ``automatic_strike`` rows) raise — caller must drop these upstream
    along with NaN ``pitch_type`` since they are book-keeping artifacts,
    not measured pitches.
    """
    if "zone" not in df.columns:
        raise KeyError("assign_feature_zone_14 requires the Statcast 'zone' column")

    raw = df["zone"]
    if raw.isna().any():
        n_na = int(raw.isna().sum())
        raise ValueError(
            f"{n_na} rows have NaN 'zone'; drop these upstream "
            f"(book-keeping rows: automatic_ball / intent_walk)"
        )

    raw_int = raw.astype("int16")
    valid = pd.Series(list(SIS_TO_INTERNAL.keys()), dtype="int16")
    bad_mask = ~raw_int.isin(valid)
    if bad_mask.any():
        bad_vals = sorted(raw_int[bad_mask].unique().tolist())
        raise ValueError(
            f"unexpected Statcast zone values {bad_vals}; "
            f"valid SIS labels are {sorted(SIS_TO_INTERNAL.keys())}"
        )

    return raw_int.map(SIS_TO_INTERNAL).astype("int16")
