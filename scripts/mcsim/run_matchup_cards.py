"""CLI runner for MCSim App B — pull real games + rosters from the MLB Stats API,
compute one matchup card per game, and persist via mcsim.storage.

Grid: all rostered pitchers x all opposing position players (both halves). See
docs/superpowers/specs/2026-06-02-mcsim-appB-step5-live-runner-design.md.

Sequential v1 (no multiprocessing). Per-cell cost is ~linear in n_paths
(~40ms/path on CPU); an all-vs-all game is ~338 cells, so tune --n-paths and
--max-games for the run budget. Missing profiles do not crash — debut players
get a league-mean fallback (data/profile_cache_loader.py).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from causal.nuisance import NuisanceModels
from mcsim.matchup_card import compute_matchup_card
from mcsim.mlb_api import get_active_roster, get_schedule
from mcsim.storage import (
    DEFAULT_DB_PATH,
    init_db,
    register_model_version,
    write_prediction,
)

DEFAULT_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")


def compute_ckpt_hash(ckpt_path: Path, *, n_chars: int = 16) -> str:
    """Truncated sha256 of the checkpoint file bytes — provenance for which
    exact weights produced a prediction. No such helper existed elsewhere."""
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]


def run_matchup_cards(
    nuisance: NuisanceModels,
    conn,
    *,
    date: str,
    game_pks: "list[int] | None",
    n_paths: int,
    rng_seed: "int | None",
    model_ckpt_hash: str,
    dry_run: bool = False,
    max_games: "int | None" = None,
) -> list[dict]:
    """Fetch the schedule for ``date``, compute one all-vs-all matchup card per
    game, and (unless dry_run) persist it. Returns the list of card payloads.

    A failure on one game is logged and skipped so it can't abort the batch.
    """
    games = get_schedule(date)
    if game_pks:
        wanted = set(game_pks)
        games = [g for g in games if g.game_pk in wanted]
    if max_games is not None:
        games = games[:max_games]

    cards: list[dict] = []
    for g in games:
        try:
            home_p, home_h = get_active_roster(
                g.home_team_id, date, probable_pitcher_id=g.home_probable_pitcher_id)
            away_p, away_h = get_active_roster(
                g.away_team_id, date, probable_pitcher_id=g.away_probable_pitcher_id)
            card = compute_matchup_card(
                nuisance,
                game_pk=g.game_pk,
                game_date=date,
                home_team=g.home_team,
                away_team=g.away_team,
                home_pitchers=home_p,
                away_pitchers=away_p,
                home_lineup=home_h,
                away_lineup=away_h,
                n_paths=n_paths,
                rng_seed=rng_seed,
            )
            if not dry_run:
                write_prediction(
                    conn,
                    game_pk=g.game_pk,
                    prediction_date=date,
                    app="matchup_card",
                    payload=card,
                    model_ckpt_hash=model_ckpt_hash,
                )
            cards.append(card)
            print(f"[{g.away_team} @ {g.home_team}] game_pk={g.game_pk}: "
                  f"{card['n_cells']} cells"
                  f"{' (dry-run, not written)' if dry_run else ' written'}")
        except Exception as e:  # one bad game must not abort the batch
            print(f"  SKIP game_pk={g.game_pk}: {type(e).__name__}: {e}")
    return cards


def main() -> None:
    ap = argparse.ArgumentParser(description="MCSim App B matchup-card runner")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--game-pk", type=int, action="append", dest="game_pks",
                    help="restrict to these game_pks (repeatable)")
    ap.add_argument("--n-paths", type=int, default=250)
    ap.add_argument("--rng-seed", type=int, default=None)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nuisance = NuisanceModels(args.ckpt, device="cpu")
    ckpt_hash = compute_ckpt_hash(args.ckpt)
    conn = init_db(args.db_path)
    register_model_version(conn, ckpt_hash=ckpt_hash, label=args.ckpt.parent.name)
    print(f"ckpt={args.ckpt}  hash={ckpt_hash}  n_paths={args.n_paths}  db={args.db_path}")

    cards = run_matchup_cards(
        nuisance, conn,
        date=args.date,
        game_pks=args.game_pks,
        n_paths=args.n_paths,
        rng_seed=args.rng_seed,
        model_ckpt_hash=ckpt_hash,
        dry_run=args.dry_run,
        max_games=args.max_games,
    )
    print(f"done: {len(cards)} card(s) for {args.date}")


if __name__ == "__main__":
    main()
