"""Compute and save per-season RE24 + count_value tables from training years.

Per ADR 004, both tables are recomputed per season (run environments shift)
and use *training data only*. Outputs:

- ``data/run_value/re24_{year}.parquet`` — 24-cell RE24 table per training year
- ``data/run_value/count_value_{year}.parquet`` — 12-cell count_value table per year
- ``data/run_value/woba_to_runs.json`` — per-season per-pitch in-play slope
  (the multiplier ``in_play_run_value`` uses to convert xwOBA-on-contact
  into a run-value contribution; not Tango's per-PA wOBA scale)

Training years per the project temporal split: 2017–2023.

After this runs, prints the 2023 RE24 table inline so we can eyeball the
"bases empty, 0 outs ≈ 0.50" sanity check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.run_value import (
    compute_count_value,
    compute_in_play_woba_to_runs_slope,
    compute_re24,
)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/run_value")
TRAIN_YEARS = list(range(2017, 2024))  # 2017–2023 inclusive

NEEDED_COLUMNS = [
    "game_pk", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b",
    "outs_when_up", "bat_score", "post_bat_score",
    "balls", "strikes",
    "delta_run_exp", "estimated_woba_using_speedangle",
]


def _load_year(year: int) -> pd.DataFrame:
    year_dir = RAW_DIR / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"missing {year_dir}; did extraction complete?")
    parts: list[pd.DataFrame] = []
    for p in sorted(year_dir.glob("*.parquet")):
        parts.append(pd.read_parquet(p, columns=NEEDED_COLUMNS))
    if not parts:
        raise FileNotFoundError(f"no parquets under {year_dir}")
    return pd.concat(parts, ignore_index=True)


def _print_re24(table: pd.DataFrame, year: int) -> None:
    print(f"\nRE24[{year}] (expected runs from this state to end of half-inning):")
    print(f"  {'base_state':<12} {'outs=0':>8} {'outs=1':>8} {'outs=2':>8}")
    pivoted = table.pivot(index="base_state", columns="outs", values="expected_runs")
    for bs in sorted(table["base_state"].unique()):
        bits = format(int(bs), "03b")  # 1B 2B 3B as bits
        pretty = f"{bs} ({bits})"
        row = pivoted.loc[bs]
        cells = " ".join(f"{row.get(o, float('nan')):>8.3f}" for o in [0, 1, 2])
        print(f"  {pretty:<12} {cells}")


def _print_count_value(table: pd.DataFrame, year: int) -> None:
    print(f"\ncount_value[{year}] (mean delta_run_exp per (balls, strikes)):")
    print(f"  {'count':<8} {'value':>10} {'n':>10}")
    for _, row in table.sort_values(["balls", "strikes"]).iterrows():
        cnt = f"{int(row['balls'])}-{int(row['strikes'])}"
        print(f"  {cnt:<8} {row['count_value']:>+10.4f} {int(row['n']):>10,}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building run-value tables for training years {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}")
    print(f"Output: {OUT_DIR.resolve()}\n")

    woba_constants: dict[int, float] = {}

    for year in TRAIN_YEARS:
        print(f"--- {year} ---")
        df = _load_year(year)
        print(f"  loaded {len(df):,} pitches")

        re24 = compute_re24(df)
        re24_path = OUT_DIR / f"re24_{year}.parquet"
        re24.to_parquet(re24_path, index=False)
        print(f"  wrote {re24_path.name} ({len(re24)} cells)")

        cv = compute_count_value(df)
        cv_path = OUT_DIR / f"count_value_{year}.parquet"
        cv.to_parquet(cv_path, index=False)
        print(f"  wrote {cv_path.name} ({len(cv)} cells)")

        try:
            k = compute_in_play_woba_to_runs_slope(df)
            woba_constants[year] = k
            print(f"  in-play wOBA→runs slope: {k:.4f} (per-pitch contact slope)")
        except Exception as exc:
            print(f"  could not fit in-play wOBA→runs slope: {exc!r}")

    # Save the per-year per-pitch in-play slopes.
    woba_path = OUT_DIR / "woba_to_runs.json"
    with woba_path.open("w") as f:
        json.dump(
            {
                "in_play_slope_per_year": woba_constants,
                "note": (
                    "Per-pitch in-play contact slope used by in_play_run_value; "
                    "not Tango/FanGraphs's per-PA wOBA scale."
                ),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {woba_path}")

    # Sanity-check eyeball: print the 2023 RE24 table for visual verification.
    last_re24 = pd.read_parquet(OUT_DIR / f"re24_{TRAIN_YEARS[-1]}.parquet")
    last_cv = pd.read_parquet(OUT_DIR / f"count_value_{TRAIN_YEARS[-1]}.parquet")
    _print_re24(last_re24, TRAIN_YEARS[-1])
    _print_count_value(last_cv, TRAIN_YEARS[-1])

    # Quick anchor checks against published values
    print(f"\nAnchor checks ({TRAIN_YEARS[-1]}):")
    cell_000_0 = last_re24[(last_re24["base_state"] == 0) & (last_re24["outs"] == 0)]
    cell_111_0 = last_re24[(last_re24["base_state"] == 7) & (last_re24["outs"] == 0)]
    cell_000_2 = last_re24[(last_re24["base_state"] == 0) & (last_re24["outs"] == 2)]
    if len(cell_000_0):
        print(f"  bases empty, 0 outs:  {float(cell_000_0['expected_runs'].iloc[0]):.3f} "
              f"(expected ~0.50)")
    if len(cell_111_0):
        print(f"  bases loaded, 0 outs: {float(cell_111_0['expected_runs'].iloc[0]):.3f} "
              f"(expected ~2.30)")
    if len(cell_000_2):
        print(f"  bases empty, 2 outs:  {float(cell_000_2['expected_runs'].iloc[0]):.3f} "
              f"(expected ~0.10)")


if __name__ == "__main__":
    main()
