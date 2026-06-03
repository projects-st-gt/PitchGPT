"""CLI for MCSim App B — fetch post-game ACTUALS and persist them.

For a given date, looks up the schedule, and for each game that is Final pulls
the real final score + per-PA matchup events from the MLB live feed and writes
them via :func:`mcsim.storage.write_actual` (two-pass COALESCE upsert, so this
can be re-run as more games finish without nulling earlier rows).

This is the counterpart to ``run_matchup_cards.py``: that one writes
predictions the night before; this one writes what actually happened, so the
demo can overlay them and the eval can pool real PAs into a calibration check.

Non-final games are skipped (logged), so the job is safe to run repeatedly
through a game day. Real MLB data only.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from mcsim.mlb_actuals import GameActuals, get_game_actuals
from mcsim.mlb_api import get_schedule
from mcsim.storage import DEFAULT_DB_PATH, init_db, write_actual


def fetch_and_store_actuals(
    conn,
    *,
    date: str,
    game_pks: "list[int] | None" = None,
) -> "list[GameActuals]":
    """Fetch actuals for each Final game on ``date`` and persist them.

    Returns the list of persisted GameActuals. Non-final games are skipped
    (logged). A failure on one game is logged and skipped so it can't abort
    the batch.
    """
    games = get_schedule(date)
    if game_pks:
        wanted = set(game_pks)
        games = [g for g in games if g.game_pk in wanted]

    stored: list[GameActuals] = []
    for g in games:
        try:
            a = get_game_actuals(g.game_pk)
            if not a.is_final:
                print(f"  skip game_pk={g.game_pk} ({g.away_team} @ {g.home_team}): "
                      f"not final ({a.status})")
                continue
            write_actual(
                conn,
                game_pk=a.game_pk,
                final_score_home=a.final_score_home,
                final_score_away=a.final_score_away,
                winner=a.winner,
                matchup_events=a.matchup_events,
            )
            stored.append(a)
            print(f"[{g.away_team} @ {g.home_team}] game_pk={g.game_pk}: "
                  f"{a.final_score_away}-{a.final_score_home} winner={a.winner} | "
                  f"{len(a.matchup_events)} PAs written")
        except Exception as e:  # one bad game must not abort the batch
            print(f"  SKIP game_pk={g.game_pk}: {type(e).__name__}: {e}")
            print(traceback.format_exc(), file=sys.stderr)
    return stored


def main() -> None:
    ap = argparse.ArgumentParser(description="MCSim App B post-game actuals fetcher")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--game-pk", type=int, action="append", dest="game_pks",
                    help="restrict to these game_pks (repeatable)")
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    conn = init_db(args.db_path)
    print(f"date={args.date}  db={args.db_path}")
    stored = fetch_and_store_actuals(conn, date=args.date, game_pks=args.game_pks)
    print(f"done: {len(stored)} final game(s) written for {args.date}")


if __name__ == "__main__":
    main()
