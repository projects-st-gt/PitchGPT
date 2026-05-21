"""Show extraction status: checkpoint progress and per-year pitch counts.

Wired up via ``make status``. Will grow as later phases land (latest training
run, last eval) — for now it covers Phase 1 only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _print_extraction_status(raw_dir: Path) -> None:
    checkpoint = raw_dir / "_checkpoint.json"
    if not checkpoint.exists():
        print(f"Extraction: not started (no checkpoint at {checkpoint})")
        return
    state = json.loads(checkpoint.read_text())
    completed = state.get("completed_dates", [])
    if not completed:
        print("Extraction: started but no days complete yet.")
        return
    first = min(completed)
    last = max(completed)
    print(
        f"Extraction: {len(completed):,} days complete; "
        f"first={first}, last={last}"
    )


def _print_year_partitions(raw_dir: Path) -> None:
    if not raw_dir.exists():
        return
    year_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not year_dirs:
        print("\nNo year partitions yet.")
        return
    print("\nPer-year pitch counts:")
    for yd in year_dirs:
        files = sorted(yd.glob("*.parquet"))
        if not files:
            continue
        total = 0
        for f in files:
            try:
                total += len(pd.read_parquet(f, columns=["pitch_type"]))
            except Exception as exc:
                print(f"  ! could not read {f.name}: {exc!r}")
        print(f"  {yd.name}: {len(files):4d} day-files, {total:>10,} pitches")


def main() -> None:
    print("=== PitchGPT status ===\n")
    raw_dir = Path("data/raw")
    _print_extraction_status(raw_dir)
    _print_year_partitions(raw_dir)


if __name__ == "__main__":
    main()
