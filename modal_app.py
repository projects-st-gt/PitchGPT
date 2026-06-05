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
        "scikit-learn>=1.3",
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
                 "scikit-learn>=1.3", "xgboost>=2.0", "joblib", "requests>=2.31",
                 "tqdm>=4.66")
    .add_local_python_source("model", "data", "scripts", "hitter", "mcsim", "causal")
)


@app.function(image=card_image, gpu="L4", volumes={"/data": volume},
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
        outcome_model="hitter", hitter_ctx=ctx)
    card["_game_pk"] = task["game_pk"]
    card["_away"] = task["away_team"]
    card["_home"] = task["home_team"]
    return card
