"""Data sanity checks per the ``statcast-pipeline`` skill.

Runs after extraction completes and (optionally) after the metadata scrape.
Each check prints a status line; warnings are non-blocking, failures cause
a non-zero exit. Wired up as ``make data-sanity``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from data.harmonization import PITCH_TYPE_MAP

RAW_DIR = Path("data/raw")
GAME_META_DIR = Path("data/game_metadata")

# Approximate published Statcast totals per year. Used as a sanity range,
# not as a hard reference. Within ±5% = pass; ±5–10% = warn; >10% = fail.
PUBLISHED_PITCH_COUNTS: dict[int, int] = {
    2017: 720_000,
    2018: 720_000,
    2019: 745_000,
    2020: 295_000,  # COVID 60-game season
    2021: 750_000,
    2022: 740_000,
    2023: 750_000,
    2024: 750_000,
    2025: 760_000,
    # 2026 is partial (in-season); no reference.
}


class Result:
    """Tracks pass/warn/fail across all checks."""

    def __init__(self):
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        print(f"  ok    {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL  {msg}")


def _year_dirs() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())


# ---------- 1. year partitions ----------


def check_year_partitions_present(r: Result) -> None:
    print("\n[1] Year partitions present")
    yds = _year_dirs()
    if not yds:
        r.fail(f"no year directories under {RAW_DIR}")
        return
    r.ok(f"{len(yds)} year dirs: {[p.name for p in yds]}")


# ---------- 2. pitch counts per year ----------


def check_pitch_counts_per_year(r: Result) -> None:
    print("\n[2] Pitch counts per year (vs. approximate published totals)")
    for yd in _year_dirs():
        year = int(yd.name)
        n = 0
        for p in yd.glob("*.parquet"):
            try:
                n += len(pd.read_parquet(p, columns=["pitch_type"]))
            except Exception as exc:
                r.warn(f"{year}: could not read {p.name}: {exc!r}")
        published = PUBLISHED_PITCH_COUNTS.get(year)
        if published is None:
            r.warn(f"{year}: {n:,} pitches (no published reference)")
            continue
        ratio = n / published if published else 0
        line = f"{year}: {n:>9,} pitches (published ~{published:,}; {ratio:.1%})"
        if 0.95 <= ratio <= 1.05:
            r.ok(line)
        elif 0.90 <= ratio <= 1.10:
            r.warn(line)
        else:
            r.fail(line)


# ---------- 3. NaN in critical columns ----------


def check_no_nan_in_critical_columns(r: Result) -> None:
    print("\n[3] No NaN in critical numeric columns (per-year sample)")
    cols = ["release_speed", "plate_x", "plate_z"]
    any_issue = False
    for yd in _year_dirs():
        files = sorted(yd.glob("*.parquet"))
        if not files:
            continue
        # Sample mid-season file
        sample = files[len(files) // 2]
        try:
            df = pd.read_parquet(sample, columns=cols)
        except Exception as exc:
            r.warn(f"{yd.name}: could not read sample {sample.name}: {exc!r}")
            continue
        for col in cols:
            null_pct = df[col].isna().mean() * 100
            if null_pct < 0.01:
                continue
            any_issue = True
            line = f"{yd.name}/{sample.name}: {col} null pct = {null_pct:.2f}%"
            if null_pct < 5:
                r.warn(line)
            else:
                r.fail(line)
    if not any_issue:
        r.ok("no NaN issues in sampled files")


# ---------- 4. pitch type harmonization coverage ----------


def check_pitch_type_coverage(r: Result, year: int = 2024) -> None:
    print(f"\n[4] Pitch type harmonization coverage ({year})")
    yd = RAW_DIR / str(year)
    if not yd.exists():
        r.warn(f"{year} not present; skipping")
        return
    counts: pd.Series = pd.Series(dtype=int)
    for p in yd.glob("*.parquet"):
        try:
            s = pd.read_parquet(p, columns=["pitch_type"])["pitch_type"].value_counts()
            counts = counts.add(s, fill_value=0)
        except Exception:
            pass
    counts = counts.astype(int).sort_values(ascending=False)
    if counts.sum() == 0:
        r.fail(f"no pitches found in {year}")
        return

    print(f"      Top 10 pitch types in {year}:")
    for pt, c in counts.head(10).items():
        in_map = "  yes  " if pt in PITCH_TYPE_MAP else "  NO   "
        print(f"        {pt:<4} [{in_map}] {c:>9,} ({c / counts.sum() * 100:>5.2f}%)")

    mapped = counts[counts.index.isin(PITCH_TYPE_MAP)].sum()
    coverage = mapped / counts.sum()
    line = f"harmonization covers {coverage * 100:.3f}% of {year} pitches"
    if coverage >= 0.999:
        r.ok(line)
    elif coverage >= 0.99:
        r.warn(line + " (below 99.9%)")
    else:
        r.fail(line + " (below 99%)")


# ---------- 5. held-out pitcher cohort ----------


def check_held_out_pitcher_cohort(r: Result, debut_year_min: int = 2024) -> None:
    print(f"\n[5] Held-out-pitcher cohort (debut ≥ {debut_year_min}-01-01)")
    pitcher_first_seen: dict[int, pd.Timestamp] = {}
    for yd in _year_dirs():
        for p in sorted(yd.glob("*.parquet")):
            try:
                df = pd.read_parquet(p, columns=["pitcher", "game_date"])
            except Exception:
                continue
            mins = df.groupby("pitcher")["game_date"].min()
            for pid, gd in mins.items():
                t = pd.Timestamp(gd)
                if pid not in pitcher_first_seen or t < pitcher_first_seen[pid]:
                    pitcher_first_seen[pid] = t

    cutoff = pd.Timestamp(f"{debut_year_min}-01-01")
    debut_recent = [pid for pid, t in pitcher_first_seen.items() if t >= cutoff]
    total = len(pitcher_first_seen)

    line = f"{len(debut_recent)} pitchers debuted ≥ {debut_year_min} (out of {total} total)"
    if len(debut_recent) >= 50:
        r.ok(line)
    elif len(debut_recent) > 0:
        r.warn(line + " — small held-out cohort")
    else:
        r.fail(line + " — held-out cohort is empty")


# ---------- 6. game metadata coverage ----------


def check_game_metadata_coverage(r: Result) -> None:
    print("\n[6] Game metadata coverage (umpire + weather)")
    ndjson = GAME_META_DIR / "games.ndjson"
    if not GAME_META_DIR.exists() or not ndjson.exists() or ndjson.stat().st_size == 0:
        r.warn("data/game_metadata/games.ndjson missing or empty — run `make extract-meta`")
        return

    raw_pks: set[int] = set()
    for parquet in RAW_DIR.rglob("*.parquet"):
        try:
            col = pd.read_parquet(parquet, columns=["game_pk"])
        except Exception:
            continue
        raw_pks.update(int(x) for x in col["game_pk"].dropna().unique())

    meta_pks: set[int] = set()
    with ndjson.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                meta_pks.add(int(row["game_pk"]))
            except Exception:
                continue

    missing = raw_pks - meta_pks
    if not raw_pks:
        r.warn("no raw game_pks found")
        return
    coverage = (len(raw_pks) - len(missing)) / len(raw_pks)
    line = (
        f"{coverage * 100:.2f}% coverage "
        f"({len(raw_pks):,} raw game_pks; {len(meta_pks):,} in metadata; "
        f"{len(missing):,} missing)"
    )
    if coverage >= 0.99:
        r.ok(line)
    elif coverage >= 0.95:
        r.warn(line)
    else:
        r.fail(line)


# ---------- main ----------


def main() -> None:
    r = Result()
    print("=== PitchGPT data sanity checks ===")
    check_year_partitions_present(r)
    check_pitch_counts_per_year(r)
    check_no_nan_in_critical_columns(r)
    check_pitch_type_coverage(r)
    check_held_out_pitcher_cohort(r)
    check_game_metadata_coverage(r)

    print("\n=== Summary ===")
    print(f"  Passed:   {len(r.passed)}")
    print(f"  Warnings: {len(r.warnings)}")
    print(f"  Failures: {len(r.failures)}")

    if r.failures:
        print("\n!! FAILURES present — do not proceed to training")
        sys.exit(1)
    if r.warnings:
        print("\n.  warnings present — review before proceeding")
        sys.exit(0)
    print("\n   all clean")
    sys.exit(0)


if __name__ == "__main__":
    main()
