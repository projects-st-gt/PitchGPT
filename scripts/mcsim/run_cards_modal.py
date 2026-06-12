"""Run pitchGPT+cascade matchup cards on Modal (GPU, one container per game).

Fans every game for a date out to ``modal_app.card_remote`` (L4 GPU), collects the
card payloads, and writes them to the local SQLite the demo reads. Much faster than
the local CPU rollout: GPU forwards + many games in parallel containers.

    python -m scripts.mcsim.run_cards_modal --date 2026-06-04 --n-paths 300

Requires: cascade artifacts + the release V2 fuel + profiles on the ``pitchgpt-data`` volume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcsim.mlb_api import get_schedule
from mcsim.storage import (
    DEFAULT_DB_PATH, init_db, register_model_version, write_prediction,
)
from scripts.mcsim.run_matchup_cards import compute_ckpt_hash

LOCAL_CKPT = Path("checkpoints_modal/releases/tiny-v1c1-sax-cal-20260611.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run matchup cards on Modal (GPU)")
    ap.add_argument("--date", required=True, action="append", dest="dates",
                    help="YYYY-MM-DD (repeatable — all dates run in ONE Modal app)")
    ap.add_argument("--n-paths", type=int, default=300)
    ap.add_argument("--rng-seed", type=int, default=1)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    from modal_app import app, card_remote

    # Build tasks for ALL dates and fan them through a SINGLE app.run() — two
    # concurrent app.run() of the same Modal app conflict and stop each other.
    tasks = []
    for date in args.dates:
        games = get_schedule(date)
        if args.max_games:
            games = games[: args.max_games]
        tasks += [{
            "game_pk": g.game_pk, "date": date,
            "home_team_id": g.home_team_id, "away_team_id": g.away_team_id,
            "home_team": g.home_team, "away_team": g.away_team,
            "home_probable_pitcher_id": g.home_probable_pitcher_id,
            "away_probable_pitcher_id": g.away_probable_pitcher_id,
            "n_paths": args.n_paths, "rng_seed": args.rng_seed,
        } for g in games]
    print(f"{len(tasks)} games across {len(args.dates)} date(s) -> Modal (L4, n_paths={args.n_paths})")

    ckpt_hash = compute_ckpt_hash(LOCAL_CKPT) if LOCAL_CKPT.exists() else "modal-tiny-v1c1-sax"
    conn = init_db(args.db_path)
    register_model_version(conn, ckpt_hash=ckpt_hash, label="tiny-v1c1-sax+cascade")

    n_written = 0
    with app.run():
        for card in card_remote.map(tasks):
            gpk = card.pop("_game_pk")
            away = card.pop("_away"); home = card.pop("_home"); cdate = card.pop("_date")
            write_prediction(conn, game_pk=gpk, prediction_date=cdate,
                             app="matchup_card", payload=card, model_ckpt_hash=ckpt_hash)
            n_written += 1
            print(f"  [{cdate}] [{away} @ {home}] game_pk={gpk}: {card.get('n_cells')} cells written")
    print(f"done: {n_written}/{len(tasks)} cards")


if __name__ == "__main__":
    main()
