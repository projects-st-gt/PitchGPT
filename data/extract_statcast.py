"""Statcast extraction with checkpoint and resume.

Day-by-day pulls via pybaseball, written as ``{output}/{year}/{date}.parquet``.
Off-season days return empty; they are still recorded in the checkpoint so
resume does not retry them. Re-running picks up exactly where the last
successful day left off.

"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pybaseball
from tqdm import tqdm

DEFAULT_START = "2017-03-15"
DEFAULT_OUTPUT = Path("data/raw")
INITIAL_BACKOFF_SEC = 5
MAX_BACKOFF_SEC = 300

# Per-fetch hard timeout. pybaseball wraps `requests` with no default timeout,
# so a stalled connection to Savant can block forever. We enforce a wall-clock
# limit per day-fetch via a daemon-threaded wrapper.
PER_DAY_TIMEOUT_SEC = 90

# After this many consecutive timeouts on the same day, give up on it. The day
# is recorded as complete with zero pitches; revisit later if it matters.
MAX_TIMEOUTS_PER_DAY = 3


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"completed_dates": []}
    with path.open() as f:
        return json.load(f)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


def _fetch_day_unchecked(d_str: str, sink: dict) -> None:
    """Worker target: call pybaseball.statcast and stash the result/exception."""
    try:
        df = pybaseball.statcast(start_dt=d_str, end_dt=d_str)
        sink["df"] = df if df is not None else pd.DataFrame()
    except Exception as exc:
        sink["exc"] = exc


def fetch_day(d: date) -> pd.DataFrame:
    """Fetch one day of Statcast pitches with a hard wall-clock timeout and backoff.

    pybaseball's underlying ``requests`` calls have no default timeout, so a
    stalled response from Savant can hang the process indefinitely. We run
    each fetch on a daemon thread and join with a timeout; if it doesn't
    return in time we orphan the thread (it dies with the process) and retry.

    Returns an empty DataFrame after ``MAX_TIMEOUTS_PER_DAY`` consecutive
    timeouts on the same day so the overall extraction can advance. The day
    is logged loudly at the giving-up point.
    """
    d_str = d.isoformat()
    backoff = INITIAL_BACKOFF_SEC
    timeouts = 0
    while True:
        sink: dict = {}
        worker = threading.Thread(
            target=_fetch_day_unchecked,
            args=(d_str, sink),
            daemon=True,
            name=f"statcast-{d_str}",
        )
        worker.start()
        worker.join(timeout=PER_DAY_TIMEOUT_SEC)

        if worker.is_alive():
            timeouts += 1
            if timeouts >= MAX_TIMEOUTS_PER_DAY:
                tqdm.write(
                    f"[{d_str}] timed out {timeouts}x in a row "
                    f"(>{PER_DAY_TIMEOUT_SEC}s each); giving up — empty day recorded"
                )
                return pd.DataFrame()
            wait = min(backoff, MAX_BACKOFF_SEC)
            tqdm.write(
                f"[{d_str}] timeout #{timeouts} (>{PER_DAY_TIMEOUT_SEC}s); sleeping {wait}s"
            )
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)
            continue

        if "exc" in sink:
            wait = min(backoff, MAX_BACKOFF_SEC)
            tqdm.write(f"[{d_str}] fetch error: {sink['exc']!r}; sleeping {wait}s")
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)
            continue

        return sink["df"]


def write_day_parquet(df: pd.DataFrame, d: date, output_dir: Path) -> Path:
    year_dir = output_dir / str(d.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    out_path = year_dir / f"{d.isoformat()}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Statcast extraction with resume.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit(f"--end ({end}) must be >= --start ({start})")

    output_dir = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else output_dir / "_checkpoint.json"
    )

    state = load_checkpoint(checkpoint_path)
    completed = set(state["completed_dates"])
    all_dates = list(_daterange(start, end))
    todo = [d for d in all_dates if d.isoformat() not in completed]

    print(
        f"Extraction plan: {len(all_dates)} days requested, "
        f"{len(completed)} already complete, {len(todo)} to fetch.\n"
        f"Output: {output_dir.resolve()}\n"
        f"Checkpoint: {checkpoint_path.resolve()}"
    )

    pbar = tqdm(todo, desc="extracting", unit="day")
    for d in pbar:
        d_str = d.isoformat()
        df = fetch_day(d)
        n = 0 if df is None or df.empty else len(df)
        if n > 0:
            write_day_parquet(df, d, output_dir)
        state["completed_dates"].append(d_str)
        save_checkpoint(checkpoint_path, state)
        pbar.set_postfix({"date": d_str, "pitches": n})


if __name__ == "__main__":
    main()
