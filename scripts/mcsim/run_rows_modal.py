"""Run matchup cards on Modal at PITCHER-ROW granularity (fast wall-clock).

Each pitcher's row is one Modal work unit (~13 cells), so ~600 units fan across
many T4 containers in parallel → wall-clock ~30-60 min instead of ~10h, with a
row landing every few seconds for visibility. Same cells / paths / numbers as the
per-game path — rows are independent and merged back into the game card payload
(identical structure to compute_matchup_card). Cost is flat (model loads once per
warm container).

    python -m scripts.mcsim.run_rows_modal --date 2026-06-04 --date 2026-06-05 --n-paths 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mcsim.mlb_api import get_active_roster, get_schedule
from mcsim.state import ReferenceContext
from mcsim.storage import (
    DEFAULT_DB_PATH, init_db, register_model_version, write_prediction,
)
from scripts.mcsim.run_matchup_cards import compute_ckpt_hash

LOCAL_CKPT = Path("checkpoints_modal/releases/tiny-v1c1-sax-cal-20260611.pt")


def _starter(pitchers):
    for p in pitchers:
        if p.is_starter:
            return {"pitcher_id": p.id, "name": p.name}
    return None


def _ref_context_dict(ctx: ReferenceContext) -> dict:
    return {
        "count": f"{ctx.count_balls}-{ctx.count_strikes}",
        "runners": "empty" if not (ctx.runners_on_1b or ctx.runners_on_2b
                                    or ctx.runners_on_3b) else "occupied",
        "outs": ctx.outs, "inning": ctx.inning,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Matchup cards on Modal, per-pitcher-row")
    ap.add_argument("--date", required=True, action="append", dest="dates")
    ap.add_argument("--n-paths", type=int, default=500)
    ap.add_argument("--rng-seed", type=int, default=1)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    from modal_app import app, row_remote

    # Build all row work-units + per-game metadata for the merge.
    games_meta: dict = {}
    tasks: list = []
    for date in args.dates:
        games = get_schedule(date)
        if args.max_games:
            games = games[: args.max_games]
        for g in games:
            home_p, home_h = get_active_roster(
                g.home_team_id, date, probable_pitcher_id=g.home_probable_pitcher_id)
            away_p, away_h = get_active_roster(
                g.away_team_id, date, probable_pitcher_id=g.away_probable_pitcher_id)
            games_meta[(date, g.game_pk)] = {
                "home_team": g.home_team, "away_team": g.away_team,
                "starter_home": _starter(home_p), "starter_away": _starter(away_p),
                "expected_rows": len(home_p) + len(away_p),
            }
            seed = args.rng_seed
            for half_pitchers, opp_lineup, team in (
                (home_p, away_h, g.home_team), (away_p, home_h, g.away_team)):
                for p in half_pitchers:
                    tasks.append({
                        "game_pk": g.game_pk, "date": date, "pitcher": p,
                        "lineup": opp_lineup, "team": team,
                        "n_paths": args.n_paths, "rng_seed": seed * 1000})
                    seed += 1
    print(f"{len(tasks)} pitcher-rows across {len(games_meta)} games "
          f"({len(args.dates)} dates) -> Modal T4, n_paths={args.n_paths}", flush=True)

    ckpt_hash = compute_ckpt_hash(LOCAL_CKPT) if LOCAL_CKPT.exists() else "modal-tiny-v1c1-sax"
    conn = init_db(args.db_path)
    register_model_version(conn, ckpt_hash=ckpt_hash, label="tiny-v1c1-sax+cascade")

    results: dict = {}                                  # (date, gpk) -> [row, ...]
    done_rows = 0
    t0 = time.time()
    with app.run():
        for r in row_remote.map(tasks):
            key = (r["date"], r["game_pk"])
            results.setdefault(key, []).append(r["row"])
            done_rows += 1
            ncells = len(r["row"]["cells"])
            print(f"  [{done_rows}/{len(tasks)} rows, {time.time()-t0:.0f}s] "
                  f"{r['date']} game {r['game_pk']} pitcher {r['row']['pitcher_id']} "
                  f"({ncells} cells)", flush=True)
            # write the game card as soon as all its rows are in
            meta = games_meta[key]
            if len(results[key]) == meta["expected_rows"]:
                rows = results[key]
                payload = {
                    "game_pk": r["game_pk"], "game_date": r["date"],
                    "home_team": meta["home_team"], "away_team": meta["away_team"],
                    "starter_home": meta["starter_home"], "starter_away": meta["starter_away"],
                    "rows": rows, "n_cells": sum(len(x["cells"]) for x in rows),
                    "n_paths_per_cell": args.n_paths,
                    "reference_context": _ref_context_dict(ReferenceContext()),
                }
                write_prediction(conn, game_pk=r["game_pk"], prediction_date=r["date"],
                                 app="matchup_card", payload=payload, model_ckpt_hash=ckpt_hash)
                print(f"  >>> GAME CARD written: {r['date']} "
                      f"[{meta['away_team']} @ {meta['home_team']}] "
                      f"{payload['n_cells']} cells", flush=True)
    print(f"done: {len(tasks)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
