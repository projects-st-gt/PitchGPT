"""Pre-compute Monte Carlo game simulations from stored matchup cards.

Reads matchup cards from SQLite, runs the full sim engine (TTO, park factors,
platoon-aware bullpen), and stores the aggregated results as score_prediction
rows in the same database.

Usage:
    uv run python -m scripts.mcsim.run_gamesim [--dates 2026-06-03,2026-06-04]
    uv run python -m scripts.mcsim.run_gamesim --all

Omit --dates to process all dates that have matchup cards.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from gamesim.bullpen import load_batter_stand_lookup
from gamesim.montecarlo import simulate_from_card
from gamesim.park import load_park_factors
from gamesim.transition import BaseOutTransition
from mcsim.storage import DEFAULT_DB_PATH, init_db, write_prediction


def _load_workload_table() -> dict[int, int]:
    path = Path("data/run_value/pitcher_workload.json")
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def _get_schedule_team_ids(date: str) -> dict[int, tuple[int, int]]:
    """Fetch the MLB schedule for ``date`` and return {game_pk: (home_team_id, away_team_id)}."""
    import urllib.request
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
        result = {}
        for day in data.get("dates", []):
            for g in day.get("games", []):
                gpk = g.get("gamePk")
                htid = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
                atid = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
                if gpk and htid and atid:
                    result[gpk] = (htid, atid)
        return result
    except Exception:
        return {}


def _get_rotation_ids(date: str, home_team_id: int, away_team_id: int,
                      home_starter_id: int | None, away_starter_id: int | None,
                      ) -> set[int]:
    """Identify rotation starters for both teams by checking who started
    games in the 6 days before ``date``. Returns pitcher IDs to exclude
    from the bullpen (does not include today's starters)."""
    from datetime import datetime, timedelta
    import urllib.request

    rotation: set[int] = set()
    d = datetime.strptime(date, "%Y-%m-%d")
    start = (d - timedelta(days=6)).strftime("%Y-%m-%d")
    today_starters = {home_starter_id, away_starter_id} - {None}

    for team_id in (home_team_id, away_team_id):
        try:
            url = (f"https://statsapi.mlb.com/api/v1/schedule"
                   f"?sportId=1&startDate={start}&endDate={date}"
                   f"&teamId={team_id}&hydrate=probablePitcher")
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.load(resp)
            for day in data.get("dates", []):
                for g in day.get("games", []):
                    for side in ("home", "away"):
                        team_data = g.get("teams", {}).get(side, {})
                        if team_data.get("team", {}).get("id") == team_id:
                            pp = team_data.get("probablePitcher", {})
                            pid = pp.get("id")
                            if pid and pid not in today_starters:
                                rotation.add(pid)
        except Exception:
            pass
    return rotation


def main() -> None:
    ap = argparse.ArgumentParser(description="Run game sims from stored matchup cards.")
    ap.add_argument("--dates", type=str, default=None,
                    help="Comma-separated YYYY-MM-DD dates (default: all)")
    ap.add_argument("--all", action="store_true", help="Process all dates with matchup cards")
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--force", action="store_true", help="Overwrite existing sim results")
    args = ap.parse_args()

    conn = init_db(args.db)

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
    else:
        rows = conn.execute(
            "SELECT DISTINCT prediction_date FROM predictions WHERE app='matchup_card' ORDER BY prediction_date"
        ).fetchall()
        dates = [r[0] for r in rows]

    if not dates:
        print("No matchup cards found.")
        return

    print(f"Loading resources...")
    transition = BaseOutTransition.load("data/run_value/base_out_transition.parquet")
    park_table = load_park_factors()
    stand_lookup = load_batter_stand_lookup()
    workload_table = _load_workload_table()
    print(f"  transition matrix: loaded")
    print(f"  park factors: {len(park_table)} parks")
    print(f"  batter stands: {len(stand_lookup)} batters")
    print(f"  workload table: {len(workload_table)} pitchers")

    total_games = 0
    total_skipped = 0

    for date in dates:
        cards = conn.execute(
            "SELECT game_pk, payload_json, model_ckpt_hash FROM predictions WHERE app='matchup_card' AND prediction_date=?",
            (date,),
        ).fetchall()

        if not cards:
            print(f"\n{date}: no matchup cards")
            continue

        schedule_lookup = _get_schedule_team_ids(date)
        print(f"\n{date}: {len(cards)} games (rotation lookup: {len(schedule_lookup)} games in schedule)")

        for i, card_row in enumerate(cards, 1):
            game_pk = card_row[0]
            payload = json.loads(card_row[1])
            ckpt_hash = card_row[2]

            if not args.force:
                existing = conn.execute(
                    "SELECT id FROM predictions WHERE game_pk=? AND prediction_date=? AND app='score_prediction'",
                    (game_pk, date),
                ).fetchone()
                if existing:
                    total_skipped += 1
                    continue

            home = payload.get("home_team", "?")
            away = payload.get("away_team", "?")

            home_starter_id = (payload.get("starter_home") or {}).get("pitcher_id")
            away_starter_id = (payload.get("starter_away") or {}).get("pitcher_id")

            rotation_ids = None
            if game_pk in schedule_lookup:
                home_tid, away_tid = schedule_lookup[game_pk]
                try:
                    rotation_ids = _get_rotation_ids(
                        date, home_tid, away_tid,
                        home_starter_id, away_starter_id)
                except Exception:
                    pass

            t0 = time.time()
            result = simulate_from_card(
                payload, transition,
                n_sims=args.n_sims,
                workload_table=workload_table,
                batter_stand_lookup=stand_lookup,
                park_factors_table=park_table,
                rotation_pitcher_ids=rotation_ids,
            )
            elapsed = time.time() - t0

            summary = result.summary()
            summary["home_starter_name"] = getattr(result, "home_starter_name", "Unknown")
            summary["away_starter_name"] = getattr(result, "away_starter_name", "Unknown")
            summary["home_starter_workload"] = getattr(result, "home_starter_workload", 24)
            summary["away_starter_workload"] = getattr(result, "away_starter_workload", 24)

            write_prediction(
                conn,
                game_pk=game_pk,
                prediction_date=date,
                app="score_prediction",
                payload=summary,
                model_ckpt_hash=ckpt_hash,
            )

            wp = summary["win_prob_home"]
            proj_h = summary["projected_score"]["home"]
            proj_a = summary["projected_score"]["away"]
            print(f"  [{i}/{len(cards)}] {away:18s} @ {home:18s}  "
                  f"WP(H)={wp:.3f}  proj {proj_a:.1f}-{proj_h:.1f}  "
                  f"({elapsed:.1f}s)")
            total_games += 1

    print(f"\nDone: {total_games} games simulated, {total_skipped} skipped (already exist)")


if __name__ == "__main__":
    main()
