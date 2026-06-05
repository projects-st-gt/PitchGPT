"""Modal entry point for PitchGPT training.

Wraps :func:`scripts.train_pitchgpt.train` so it can run on a Modal A100
with a persistent Volume holding the augmented parquets, profile cache, and
checkpoints. Designed so the same training function runs identically locally
(via ``scripts.train_pitchgpt``) and remotely (via this app).

Quick start (assumes ``modal token new`` has been run):

    # One-time: upload augmented data + profile cache to the Volume.
    python scripts/upload_to_modal.py

    # Smoke test — 100 steps on Tiny.
    modal run modal_app.py::train_remote --size sanity --max-steps 100 \\
        --max-pitches 250000 --batch-size 64

    # Phase B — Tiny v1, single fold, full training corpus.
    modal run modal_app.py::train_remote --size tiny --fold 0 --epochs 3

Checkpoints land on the Volume at ``/data/checkpoints/{run_name}/``. Pull
them locally with::

    modal volume get pitchgpt-data checkpoints ./checkpoints_modal
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

# ---- Container image ----

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "pandas>=2.2",
        "pyarrow>=15",
        "numpy>=1.26,<2.1",
        "tqdm>=4.66",
        "scikit-learn==1.8.0",
    )
    .add_local_python_source("model", "data", "scripts")
)

# ---- Persistent Volume (lazy-create on first put) ----
# Holds augmented parquets, profile cache, run-value tables, and checkpoints.
volume = modal.Volume.from_name("pitchgpt-data", create_if_missing=True)

app = modal.App("pitchgpt")


@app.function(
    image=image,
    gpu="L4",  # workload is data-loading-bound (num_workers=0, pandas __getitem__),
               # not GPU-bound — an L4 (24GB) matches A100 throughput here at ~⅓ the
               # cost. A 25M-param model on ~15-token sequences fits trivially.
    volumes={"/data": volume},
    timeout=60 * 60 * 24,  # 24h — Modal's max. ~0.8 steps/s (dataloader ceiling),
                           # so 6 epochs × ~4.7K steps ≈ 10h; `small` similar.
)
def train_remote(
    size: str = "tiny",
    fold_id: int = 0,
    epochs: int = 3,
    max_steps: Optional[int] = None,
    max_pitches: Optional[int] = None,
    batch_size: int = 256,
    warmup_steps: int = 2000,
    log_every: int = 50,
    eval_every: int = 1000,
    run_name: Optional[str] = None,
    seed: int = 42,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 1e-3,
    standardize_profiles: bool = True,
    arsenal_per_pitch: bool = True,  # ADR 009
    propensity_situational: bool = True,  # ADR 010
    concat_then_project: bool = False,  # ADR 011
    profile_film: bool = False,  # ADR 012
    zone_spatial_weight: float = 0.0,  # v5 14-zone EMD aux loss
    type_focal_gamma: float = 0.0,     # focal loss on type head
    type_class_weight_alpha: float = 0.0,  # inverse-freq class weighting
    type_conditioned_heads: bool = False,  # ADR 013
) -> dict:
    """Remote A100 training. Reads data from the mounted Volume, writes
    checkpoints back to the Volume. Returns the local-training summary dict.
    """
    from scripts.train_pitchgpt import train

    summary = train(
        augmented_dir=Path("/data/augmented"),
        profiles_dir=Path("/data/profiles"),
        ckpt_dir=Path("/data/checkpoints"),
        fold_id=fold_id,
        size=size,
        epochs=epochs,
        max_steps=max_steps,
        max_pitches=max_pitches,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        log_every=log_every,
        eval_every=eval_every,
        run_name=run_name,
        seed=seed,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        standardize_profiles=standardize_profiles,
        arsenal_per_pitch=arsenal_per_pitch,
        propensity_situational=propensity_situational,
        concat_then_project=concat_then_project,
        profile_film=profile_film,
        zone_spatial_weight=zone_spatial_weight,
        type_focal_gamma=type_focal_gamma,
        type_class_weight_alpha=type_class_weight_alpha,
        type_conditioned_heads=type_conditioned_heads,
    )

    # Make sure files we wrote to the Volume are flushed for the next call /
    # external readers (``modal volume get``).
    volume.commit()
    return summary


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def list_data() -> dict:
    """Inventory check — verify what's on the Volume after upload."""
    root = Path("/data")
    inventory = {}
    for sub in ("augmented", "profiles", "preprocess_artifacts", "checkpoints"):
        path = root / sub
        if not path.exists():
            inventory[sub] = "missing"
            continue
        files = list(path.rglob("*"))
        n_files = sum(1 for f in files if f.is_file())
        bytes_total = sum(f.stat().st_size for f in files if f.is_file())
        inventory[sub] = {
            "n_files": n_files,
            "size_mb": round(bytes_total / 1e6, 1),
        }
    return inventory


@app.local_entrypoint()
def main(
    size: str = "tiny",
    fold: int = 0,
    epochs: int = 3,
    max_steps: Optional[int] = None,
    max_pitches: Optional[int] = None,
    batch_size: int = 256,
    run_name: Optional[str] = None,
    early_stop_patience: int = 0,
    no_arsenal_per_pitch: bool = False,
    no_standardize_profiles: bool = False,
    no_propensity_situational: bool = False,
    concat_then_project: bool = False,  # ADR 011
    profile_film: bool = False,  # ADR 012
    zone_spatial_weight: float = 0.0,  # v5 14-zone EMD aux loss
    type_focal_gamma: float = 0.0,     # focal loss on type head (0 = CE)
    type_class_weight_alpha: float = 0.0,  # inverse-freq class weighting
    type_conditioned_heads: bool = False,  # ADR 013
):
    """Convenience entrypoint for `modal run modal_app.py`."""
    summary = train_remote.remote(
        size=size,
        fold_id=fold,
        epochs=epochs,
        max_steps=max_steps,
        max_pitches=max_pitches,
        batch_size=batch_size,
        run_name=run_name,
        early_stop_patience=early_stop_patience,
        arsenal_per_pitch=not no_arsenal_per_pitch,
        standardize_profiles=not no_standardize_profiles,
        propensity_situational=not no_propensity_situational,
        concat_then_project=concat_then_project,
        profile_film=profile_film,
        zone_spatial_weight=zone_spatial_weight,
        type_focal_gamma=type_focal_gamma,
        type_class_weight_alpha=type_class_weight_alpha,
        type_conditioned_heads=type_conditioned_heads,
    )
    print("\nRemote training complete:")
    import json
    print(json.dumps(summary, indent=2))


# ---- Matchup-card inference (pitchGPT + hitter cascade) on GPU --------------
# A separate image: cards need xgboost (cascade) + requests (MLB API) + the
# hitter/mcsim/causal packages. No augmented data needed — only model + profiles
# + cascade artifacts live on the volume.
card_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", "pandas>=2.2", "pyarrow>=15", "numpy>=1.26,<2.1",
                 "scikit-learn==1.8.0", "xgboost>=2.0", "joblib", "requests>=2.31",
                 "tqdm>=4.66")
    .add_local_python_source("model", "data", "scripts", "hitter", "mcsim", "causal")
)


@app.function(image=card_image, gpu="T4", volumes={"/data": volume},
              timeout=60 * 60 * 2)
def card_remote(task: dict) -> dict:
    """Compute one game's pitchGPT+cascade matchup card on a GPU. ``task`` carries
    the game fields + date + n_paths + rng_seed. Returns the card payload dict."""
    import os
    from pathlib import Path
    import torch
    from causal.nuisance import NuisanceModels
    from mcsim.matchup_card import compute_matchup_card
    from mcsim.mlb_api import get_active_roster
    from hitter.rollout import load_hitter_ctx

    # The volume mounts at /data, so the code's relative "data/profiles" (used by
    # NuisanceModels' own ProfileCache) resolves to /data/profiles when cwd="/".
    os.chdir("/")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nz = NuisanceModels(
        Path("/data/checkpoints/small-fold0-v7/checkpoint_calibrated.pt"), device=dev)
    ctx = load_hitter_ctx("/data/checkpoints/hitter", profiles_dir="/data/profiles")

    date = task["date"]
    home_p, home_h = get_active_roster(
        task["home_team_id"], date, probable_pitcher_id=task.get("home_probable_pitcher_id"))
    away_p, away_h = get_active_roster(
        task["away_team_id"], date, probable_pitcher_id=task.get("away_probable_pitcher_id"))
    card = compute_matchup_card(
        nz, game_pk=task["game_pk"], game_date=date,
        home_team=task["home_team"], away_team=task["away_team"],
        home_pitchers=home_p, away_pitchers=away_p,
        home_lineup=home_h, away_lineup=away_h,
        n_paths=task.get("n_paths", 300), rng_seed=task.get("rng_seed", 1),
        outcome_model="hitter", hitter_ctx=ctx,
        progress_every=task.get("progress_every", 50))
    card["_game_pk"] = task["game_pk"]
    card["_away"] = task["away_team"]
    card["_home"] = task["home_team"]
    card["_date"] = task["date"]
    return card


@app.function(image=card_image, gpu="T4", volumes={"/data": volume}, timeout=900)
def bench_cell(n_paths: int = 500, n_cells: int = 3) -> dict:
    """Time a single matchup cell at n_paths on T4 — to size the real per-cell cost
    + per-game ETA before committing to a full run."""
    import os, time
    os.chdir("/")
    from pathlib import Path
    import torch
    from causal.nuisance import NuisanceModels
    from mcsim.matchup_card import compute_matchup_card, PitcherSpec, BatterSpec
    from hitter.rollout import load_hitter_ctx

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    nz = NuisanceModels(
        Path("/data/checkpoints/small-fold0-v7/checkpoint_calibrated.pt"), device=dev)
    ctx = load_hitter_ctx("/data/checkpoints/hitter", profiles_dir="/data/profiles")
    load_s = time.time() - t
    bats = [BatterSpec(id=b, name=str(b), stand="R")
            for b in [592450, 665489, 660271][:n_cells]]
    t0 = time.time()
    card = compute_matchup_card(
        nz, game_pk=1, game_date="2024-08-01", home_team="A", away_team="B",
        home_pitchers=[PitcherSpec(id=640455, name="P", throws="L", is_starter=True)],
        away_pitchers=[], home_lineup=[], away_lineup=bats,
        n_paths=n_paths, rng_seed=1, outcome_model="hitter", hitter_ctx=ctx)
    dt = time.time() - t0
    return {"device": dev, "model+cascade_load_s": round(load_s, 1),
            "n_cells": n_cells, "total_s": round(dt, 1),
            "per_cell_s": round(dt / n_cells, 1),
            "est_per_game_min": round(dt / n_cells * 338 / 60, 1),
            "sample_ops": [round(c["predicted_ops"], 3)
                           for half in card["grid"] for cell in half.get("cells", [])
                           for c in [cell]][:3] if "grid" in card else "n/a"}


# ---- Finer-grained fan-out: one work unit per PITCHER ROW -------------------
# Same cells, same paths, same numbers — just sliced so Modal parallelizes ~600
# small units instead of 24 big games (wall-clock ~30-60 min vs ~10h). Model +
# cascade load ONCE per warm container (module globals persist across .map inputs)
# so cost stays flat.
_ROW_NZ = None
_ROW_CTX = None


# ---- Per-PA backtest fan-out: pitchGPT+cascade outcome dist for real PAs -----
# One Modal task = a CHUNK of held-out PAs. Each PA is rolled out at NEUTRAL 0-0
# context (apples-to-apples with the lookup count-tree) via _compute_cell. Model +
# cascade load ONCE per warm container (module globals). Capped at 10 containers to
# respect the account plan. Returns [{idx, dist}] so the local driver scores all
# three pitch sources on the identical PA sample.
_BT_NZ = None
_BT_CTX = None


@app.function(image=card_image, gpu="T4", volumes={"/data": volume},
              timeout=60 * 60, max_containers=10)
def backtest_remote(task: dict) -> list[dict]:
    """Roll out pitchGPT+cascade per-PA outcome dists for a chunk of real PAs.

    ``task = {"specs": [{idx, pitcher_id, batter_id, throws, stand, game_date,
    game_pk}, ...], "n_paths": int, "rng_seed": int}``. Returns
    ``[{"idx": int, "dist": {7-class outcome dist}}]``.
    """
    import os
    os.chdir("/")
    global _BT_NZ, _BT_CTX
    from pathlib import Path
    import torch
    from causal.nuisance import NuisanceModels
    from causal.positivity import PositivityGate
    from mcsim.matchup_card import _compute_cell, PitcherSpec, BatterSpec
    from mcsim.state import ReferenceContext
    from hitter.rollout import load_hitter_ctx

    if _BT_NZ is None:                       # load once per warm container
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _BT_NZ = NuisanceModels(
            Path("/data/checkpoints/small-fold0-v7/checkpoint_calibrated.pt"), device=dev)
        _BT_CTX = load_hitter_ctx("/data/checkpoints/hitter", profiles_dir="/data/profiles")

    gate = PositivityGate()
    context = ReferenceContext()             # neutral 0-0, empty bases
    n_paths = task["n_paths"]
    base_seed = task["rng_seed"]
    out = []
    for j, s in enumerate(task["specs"]):
        pitcher = PitcherSpec(id=int(s["pitcher_id"]), name=str(s["pitcher_id"]),
                              throws=s["throws"] if s["throws"] in ("R", "L") else "R")
        batter = BatterSpec(id=int(s["batter_id"]), name=str(s["batter_id"]),
                            stand=s["stand"] if s["stand"] in ("R", "L") else "R")
        cell = _compute_cell(
            _BT_NZ, pitcher=pitcher, batter=batter, game_date=s["game_date"],
            game_pk=int(s["game_pk"]), ballpark_id=0, umpire_id=0, catcher_id=0,
            n_paths=n_paths, rng_seed=base_seed + j, context=context, gate=gate,
            outcome_model="hitter", hitter_ctx=_BT_CTX)
        out.append({"idx": int(s["idx"]), "dist": cell["predicted_outcome_dist"]})
    return out


@app.function(image=card_image, gpu="T4", volumes={"/data": volume}, timeout=60 * 40)
def row_remote(task: dict) -> dict:
    """Compute ONE pitcher's row of cells (pitcher x opposing lineup). Returns the
    row dict + (game_pk, date) so the local driver merges rows into game cards."""
    import os
    os.chdir("/")
    global _ROW_NZ, _ROW_CTX
    from pathlib import Path
    import torch
    from causal.nuisance import NuisanceModels
    from causal.positivity import PositivityGate
    from mcsim.matchup_card import _compute_cell
    from mcsim.state import ReferenceContext
    from hitter.rollout import load_hitter_ctx

    if _ROW_NZ is None:                       # load once per warm container
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _ROW_NZ = NuisanceModels(
            Path("/data/checkpoints/small-fold0-v7/checkpoint_calibrated.pt"), device=dev)
        _ROW_CTX = load_hitter_ctx("/data/checkpoints/hitter", profiles_dir="/data/profiles")

    pitcher = task["pitcher"]; gate = PositivityGate(); context = ReferenceContext()
    cells = []
    for i, batter in enumerate(task["lineup"]):
        cells.append(_compute_cell(
            _ROW_NZ, pitcher=pitcher, batter=batter, game_date=task["date"],
            game_pk=task["game_pk"], ballpark_id=0, umpire_id=0, catcher_id=0,
            n_paths=task["n_paths"], rng_seed=task["rng_seed"] + i, context=context,
            gate=gate, outcome_model="hitter", hitter_ctx=_ROW_CTX))
    row = {"pitcher_id": pitcher.id, "name": pitcher.name, "team": task["team"],
           "throws": pitcher.throws, "is_starter": pitcher.is_starter, "cells": cells}
    return {"game_pk": task["game_pk"], "date": task["date"], "row": row}
