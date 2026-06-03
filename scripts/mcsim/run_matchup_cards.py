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
