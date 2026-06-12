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
import sys
import traceback
from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from causal.nuisance_v2 import load_nuisance_auto
from mcsim.matchup_card import compute_matchup_card
from mcsim.mlb_api import get_active_roster, get_schedule
from mcsim.storage import (
    DEFAULT_DB_PATH,
    init_db,
    register_model_version,
    write_prediction,
)

DEFAULT_CKPT = Path("checkpoints_modal/releases/tiny-v1c1-sax-cal-20260611.pt")


def compute_ckpt_hash(ckpt_path: Path, *, n_chars: int = 16) -> str:
    """Truncated sha256 of the checkpoint file bytes — provenance for which
    exact weights produced a prediction. No such helper existed elsewhere."""
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]


def _compute_one_game(nuisance, game, date: str, n_paths: int, rng_seed,
                      progress_every=None, outcome_model="head", hitter_ctx=None):
    """Fetch both rosters for one game and compute its all-vs-all card.

    Shared by the sequential and parallel paths so they produce identical
    card structure.
    """
    home_p, home_h = get_active_roster(
        game.home_team_id, date, probable_pitcher_id=game.home_probable_pitcher_id)
    away_p, away_h = get_active_roster(
        game.away_team_id, date, probable_pitcher_id=game.away_probable_pitcher_id)
    return compute_matchup_card(
        nuisance,
        game_pk=game.game_pk,
        game_date=date,
        home_team=game.home_team,
        away_team=game.away_team,
        home_pitchers=home_p,
        away_pitchers=away_p,
        home_lineup=home_h,
        away_lineup=away_h,
        n_paths=n_paths,
        rng_seed=rng_seed,
        progress_every=progress_every,
        outcome_model=outcome_model,
        hitter_ctx=hitter_ctx,
    )


# ---- parallel workers (game-level pool) -----------------------------------
# Per-cell thread scaling is flat (1 thread ~= 8 threads/cell), so we run one
# game per worker at torch threads=1 and let many games run concurrently. Each
# worker loads its OWN NuisanceModels once (models don't pickle across procs);
# spawn re-imports this module, so the worker fns must live at module level.

_WORKER_NUISANCE = None  # populated once per worker process by _init_worker
_WORKER_HITTER_CTX = None  # cascade context (hitter mode only)
_WORKER_OUTCOME_MODEL = "head"


def _init_worker(ckpt_path_str: str, outcome_model: str = "head",
                 hitter_dir: str = "checkpoints/hitter") -> None:
    import torch as _torch

    _torch.set_num_threads(1)
    global _WORKER_NUISANCE, _WORKER_HITTER_CTX, _WORKER_OUTCOME_MODEL
    _WORKER_NUISANCE = load_nuisance_auto(Path(ckpt_path_str), device="cpu")
    _WORKER_OUTCOME_MODEL = outcome_model
    if outcome_model == "hitter":
        from hitter.rollout import load_hitter_ctx
        _WORKER_HITTER_CTX = load_hitter_ctx(hitter_dir)


def _worker_compute_game(task):
    game, date, n_paths, rng_seed, progress_every = task
    try:
        card = _compute_one_game(
            _WORKER_NUISANCE, game, date, n_paths, rng_seed, progress_every=progress_every,
            outcome_model=_WORKER_OUTCOME_MODEL, hitter_ctx=_WORKER_HITTER_CTX)
        return (game.game_pk, game.away_team, game.home_team, card, None)
    except Exception:
        return (game.game_pk, game.away_team, game.home_team, None, traceback.format_exc())


def run_matchup_cards(
    nuisance: "NuisanceModels | None",
    conn,
    *,
    date: str,
    game_pks: "list[int] | None",
    n_paths: int,
    rng_seed: "int | None",
    model_ckpt_hash: str,
    dry_run: bool = False,
    max_games: "int | None" = None,
    n_workers: int = 1,
    ckpt_path: "Path | None" = None,
    progress_every: "int | None" = None,
    outcome_model: str = "head",
    hitter_dir: str = "checkpoints/hitter",
) -> list[dict]:
    """Fetch the schedule for ``date``, compute one all-vs-all matchup card per
    game, and (unless dry_run) persist it. Returns the list of card payloads.

    ``n_workers > 1`` runs games across a process pool (each worker loads its own
    model at torch threads=1; requires ``ckpt_path``). DB writes always happen in
    this (parent) process, so SQLite stays single-writer.

    A failure on one game is logged and skipped so it can't abort the batch.
    """
    games = get_schedule(date)
    if game_pks:
        wanted = set(game_pks)
        games = [g for g in games if g.game_pk in wanted]
    if max_games is not None:
        games = games[:max_games]

    def _persist_and_log(game_pk, away, home, card):
        if not dry_run:
            write_prediction(
                conn, game_pk=game_pk, prediction_date=date, app="matchup_card",
                payload=card, model_ckpt_hash=model_ckpt_hash,
            )
        print(f"[{away} @ {home}] game_pk={game_pk}: {card['n_cells']} cells"
              f"{' (dry-run, not written)' if dry_run else ' written'}")

    cards: list[dict] = []

    if n_workers and n_workers > 1:
        if ckpt_path is None:
            raise ValueError("n_workers > 1 requires ckpt_path (workers load their own model)")
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        tasks = [(g, date, n_paths, rng_seed, progress_every) for g in games]
        print(f"parallel: {len(tasks)} games across {n_workers} workers (1 torch thread each)"
              f"  outcome_model={outcome_model}")
        with ctx.Pool(processes=n_workers, initializer=_init_worker,
                      initargs=(str(ckpt_path), outcome_model, hitter_dir)) as pool:
            for game_pk, away, home, card, err in pool.imap_unordered(_worker_compute_game, tasks):
                if err is not None:
                    print(f"  SKIP game_pk={game_pk}: worker error (see stderr)")
                    print(err, file=sys.stderr)
                    continue
                _persist_and_log(game_pk, away, home, card)
                cards.append(card)
        return cards

    hitter_ctx = None
    if outcome_model == "hitter":
        from hitter.rollout import load_hitter_ctx
        hitter_ctx = load_hitter_ctx(hitter_dir)

    for g in games:
        try:
            card = _compute_one_game(nuisance, g, date, n_paths, rng_seed,
                                     progress_every=progress_every,
                                     outcome_model=outcome_model, hitter_ctx=hitter_ctx)
            _persist_and_log(g.game_pk, g.away_team, g.home_team, card)
            cards.append(card)
        except Exception as e:  # one bad game must not abort the batch
            # Log the full traceback to stderr: the broad except also catches
            # genuine bugs (a typo'd kwarg, a payload-key error), which would
            # otherwise hide behind a one-line SKIP and silently zero the batch.
            print(f"  SKIP game_pk={g.game_pk}: {type(e).__name__}: {e}")
            print(traceback.format_exc(), file=sys.stderr)
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
    ap.add_argument("--n-workers", type=int, default=1,
                    help="games to run in parallel (each worker loads its own model "
                         "at 1 torch thread; >1 needs the checkpoint on disk)")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print a per-game cell-progress line every N cells (0 to silence)")
    ap.add_argument("--outcome-model", choices=["head", "hitter"], default="head",
                    help="'head' = transformer outcome; 'hitter' = pitchGPT pitches + cascade outcomes")
    ap.add_argument("--hitter-dir", default="checkpoints/hitter")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt_hash = compute_ckpt_hash(args.ckpt)
    conn = init_db(args.db_path)
    register_model_version(conn, ckpt_hash=ckpt_hash, label=args.ckpt.parent.name)
    # In parallel mode each worker loads its own model — don't load one here.
    nuisance = None if args.n_workers > 1 else load_nuisance_auto(args.ckpt, device="cpu")
    print(f"ckpt={args.ckpt}  hash={ckpt_hash}  n_paths={args.n_paths}  "
          f"workers={args.n_workers}  db={args.db_path}")

    cards = run_matchup_cards(
        nuisance, conn,
        date=args.date,
        game_pks=args.game_pks,
        n_paths=args.n_paths,
        rng_seed=args.rng_seed,
        model_ckpt_hash=ckpt_hash,
        dry_run=args.dry_run,
        max_games=args.max_games,
        n_workers=args.n_workers,
        ckpt_path=args.ckpt,
        progress_every=(args.progress_every or None),
        outcome_model=args.outcome_model,
        hitter_dir=args.hitter_dir,
    )
    print(f"done: {len(cards)} card(s) for {args.date}")


if __name__ == "__main__":
    main()
