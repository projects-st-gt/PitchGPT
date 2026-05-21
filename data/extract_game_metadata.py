"""Per-game metadata fetch from MLB Stats API.

For each unique ``game_pk`` discovered in ``data/raw/``, hit the
``/api/v1.1/game/{game_pk}/feed/live`` endpoint and pull out the home-plate
umpire and weather. These are confounders required by ADR 003 that are
missing from the Statcast pull (umpire column is 100% null; temperature
is not present at all).

Output:

- ``data/game_metadata/games.ndjson``  one JSON object per fetched game,
  append-only so a crash loses at most a single in-flight write.
- ``data/game_metadata/_checkpoint.json``  set of completed game_pks; resume
  picks up exactly where the last successful fetch left off.

Run after ``make extract`` completes. Polite by default at 1 req/sec.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/game_metadata")
ENDPOINT = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
INITIAL_BACKOFF_SEC = 5
MAX_BACKOFF_SEC = 300
DEFAULT_RATE_LIMIT_SEC = 1.0
REQUEST_TIMEOUT_SEC = 30


def load_checkpoint(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    return {int(x) for x in data.get("completed_game_pks", [])}


def save_checkpoint(path: Path, completed: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({"completed_game_pks": sorted(completed)}, f)
    tmp.replace(path)


def discover_game_pks(raw_dir: Path) -> list[int]:
    """Return all distinct ``game_pk`` values across every parquet under ``raw_dir``."""
    if not raw_dir.exists():
        return []
    parquets = sorted(raw_dir.rglob("*.parquet"))
    if not parquets:
        return []
    pks: set[int] = set()
    for p in parquets:
        try:
            col = pd.read_parquet(p, columns=["game_pk"])
            pks.update(int(x) for x in col["game_pk"].dropna().unique())
        except Exception as exc:
            tqdm.write(f"warn: could not read {p}: {exc!r}")
    return sorted(pks)


def _safe_int(s: object) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _parse_wind(raw: str | None) -> tuple[int | None, str | None]:
    """Parse strings like ``'7 mph, In From CF'`` into ``(7, 'In From CF')``.

    Returns ``(None, None)`` for empty input. The API uses the literal string
    ``'None'`` for wind direction when speed is 0; that is normalized to Python ``None``.
    """
    if not raw:
        return None, None
    parts = raw.split(",", 1)
    speed: int | None = None
    direction: str | None = None
    if parts:
        toks = parts[0].strip().split()
        if toks:
            speed = _safe_int(toks[0])
    if len(parts) > 1:
        d = parts[1].strip()
        direction = None if d in ("", "None") else d
    return speed, direction


def parse_game_feed(game_pk: int, data: dict) -> dict:
    """Extract the fields ADR 003 requires from a feed/live JSON response."""
    gd = data.get("gameData") or {}
    ld = data.get("liveData") or {}
    weather = gd.get("weather") or {}
    boxscore = ld.get("boxscore") or {}
    officials = boxscore.get("officials") or []

    by_type: dict[str, dict] = {}
    for entry in officials:
        otype = (entry or {}).get("officialType") or ""
        off = (entry or {}).get("official") or {}
        by_type[otype] = off

    hp = by_type.get("Home Plate") or {}
    fb = by_type.get("First Base") or {}
    sb = by_type.get("Second Base") or {}
    tb = by_type.get("Third Base") or {}

    wind_speed, wind_dir = _parse_wind(weather.get("wind"))
    condition = weather.get("condition")
    roof_closed = bool(condition) and "roof closed" in condition.lower()

    datetime_obj = gd.get("datetime") or {}
    venue = gd.get("venue") or {}

    return {
        "game_pk": int(game_pk),
        "game_date": datetime_obj.get("officialDate"),
        "game_datetime": datetime_obj.get("dateTime"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "hp_umpire_id": hp.get("id"),
        "hp_umpire_name": hp.get("fullName"),
        "fb_umpire_id": fb.get("id"),
        "sb_umpire_id": sb.get("id"),
        "tb_umpire_id": tb.get("id"),
        "temp_f": _safe_int(weather.get("temp")),
        "weather_condition": condition,
        "roof_closed": roof_closed,
        "wind_speed_mph": wind_speed,
        "wind_direction": wind_dir,
    }


def fetch_one(game_pk: int) -> dict | None:
    """Fetch + parse one game, with exponential backoff on transient errors."""
    url = ENDPOINT.format(game_pk=game_pk)
    backoff = INITIAL_BACKOFF_SEC
    while True:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
            if resp.status_code == 404:
                tqdm.write(f"[{game_pk}] 404, skipping")
                return None
            resp.raise_for_status()
            return parse_game_feed(game_pk, resp.json())
        except Exception as exc:
            wait = min(backoff, MAX_BACKOFF_SEC)
            tqdm.write(f"[{game_pk}] {exc!r}; sleeping {wait}s")
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)


def append_ndjson(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch per-game metadata from MLB Stats API.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--rate-limit-sec",
        type=float,
        default=DEFAULT_RATE_LIMIT_SEC,
        help="Min seconds between requests (default 1.0).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of games this run (handy for testing).",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "_checkpoint.json"
    ndjson_path = output_dir / "games.ndjson"

    print(f"Discovering game_pks under {raw_dir}...")
    all_pks = discover_game_pks(raw_dir)
    if not all_pks:
        print("No game_pks found. Did Statcast extraction complete?")
        return

    completed = load_checkpoint(checkpoint_path)
    todo = [pk for pk in all_pks if pk not in completed]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"Found {len(all_pks):,} unique game_pks; "
        f"{len(completed):,} already fetched; "
        f"{len(todo):,} to fetch this run.\n"
        f"Output: {ndjson_path.resolve()}"
    )
    if not todo:
        return

    pbar = tqdm(todo, desc="game_meta", unit="game")
    for pk in pbar:
        row = fetch_one(pk)
        if row is not None:
            append_ndjson(ndjson_path, row)
        completed.add(pk)
        save_checkpoint(checkpoint_path, completed)
        pbar.set_postfix({"game_pk": pk})
        time.sleep(args.rate_limit_sec)


if __name__ == "__main__":
    main()
