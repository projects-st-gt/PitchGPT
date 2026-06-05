"""Run pitchGPT+cascade matchup cards on Modal (GPU, one container per game).

Fans every game for a date out to ``modal_app.card_remote`` (L4 GPU), collects the
card payloads, and writes them to the local SQLite the demo reads. Much faster than
the local CPU rollout: GPU forwards + many games in parallel containers.

    python -m scripts.mcsim.run_cards_modal --date 2026-06-04 --n-paths 300

Requires: cascade artifacts + small-v7 + profiles on the ``pitchgpt-data`` volume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcsim.mlb_api import get_schedule
from mcsim.storage import (
    DEFAULT_DB_PATH, init_db, register_model_version, write_prediction,
)
from scripts.mcsim.run_matchup_cards import compute_ckpt_hash

LOCAL_CKPT = Path("checkpoints_modal/small-fold0-v7/checkpoint_calibrated.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run matchup cards on Modal (GPU)")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--n-paths", type=int, default=300)
    ap.add_argument("--rng-seed", type=int, default=1)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    from modal_app import app, card_remote

    games = get_schedule(args.date)
    if args.max_games:
        games = games[: args.max_games]
    tasks = [{
        "game_pk": g.game_pk, "date": args.date,
        "home_team_id": g.home_team_id, "away_team_id": g.away_team_id,
        "home_team": g.home_team, "away_team": g.away_team,
        "home_probable_pitcher_id": g.home_probable_pitcher_id,
        "away_probable_pitcher_id": g.away_probable_pitcher_id,
        "n_paths": args.n_paths, "rng_seed": args.rng_seed,
    } for g in games]
    print(f"{len(tasks)} games for {args.date} -> Modal (L4, n_paths={args.n_paths})")

    ckpt_hash = compute_ckpt_hash(LOCAL_CKPT) if LOCAL_CKPT.exists() else "modal-small-v7"
    conn = init_db(args.db_path)
    register_model_version(conn, ckpt_hash=ckpt_hash, label="small-fold0-v7+cascade")

    n_written = 0
    with app.run():
        for card in card_remote.map(tasks):
            gpk = card.pop("_game_pk")
            away = card.pop("_away"); home = card.pop("_home")
            write_prediction(conn, game_pk=gpk, prediction_date=args.date,
                             app="matchup_card", payload=card, model_ckpt_hash=ckpt_hash)
            n_written += 1
            print(f"  [{away} @ {home}] game_pk={gpk}: {card.get('n_cells')} cells written")
    print(f"done: {n_written}/{len(tasks)} cards for {args.date}")


if __name__ == "__main__":
    main()
