---
name: modal-training
description: Use this skill whenever launching, monitoring, or debugging a Modal GPU training run — or any Modal fan-out job (matchup cards, backtests). Trigger on mentions of Modal, GPU training, remote training, modal run, modal volume, checkpoint pull, detach, L4, A100, fan-out, or anything that touches modal_app.py. Also trigger when a training job needs to be resumed, monitored for completion, or when pulling/calibrating a checkpoint after training.
---

# Modal GPU Training

PitchGPT training runs on Modal's GPU cloud. The same `train()` function runs
locally (via `scripts/train_pitchgpt.py`) and remotely (via `modal_app.py`).
Data lives on a persistent Modal Volume; code is baked into the container
image.

## The one rule you must never forget

**Always use `--detach` when launching training.**

```bash
# CORRECT — survives sleep, terminal close, session end
modal run --detach modal_app.py --size small --fold 0 --epochs 3 --run-name my-run

# WRONG — container dies when the local process dies
modal run modal_app.py --size small --fold 0 --epochs 3 --run-name my-run
```

`modal run` maintains a live connection. If the user's Mac sleeps, the
terminal closes, or the Claude session ends, the container is killed and no
checkpoint is saved. `--detach` disconnects the local process from the
container — the container runs independently and writes its checkpoint to the
Volume when done.

This burned us on 2026-06-05: a 10-hour training job died at step 700/14100
because the local `modal run` process was interrupted overnight. No checkpoint
was saved; the entire run was wasted.

## Architecture

```
local machine                           Modal cloud
--------------                          -----------
modal_app.py  ──modal run --detach──>   container (L4 GPU)
                                          ├── image: torch, pandas, etc.
                                          ├── code: model/, data/, scripts/
                                          │         (baked via add_local_python_source)
                                          └── /data (mounted Volume)
                                                ├── augmented/    (training data)
                                                ├── profiles/     (profile caches)
                                                ├── preprocess_artifacts/
                                                ├── run_value/    (RE24 tables)
                                                └── checkpoints/  (written by training)
```

**Code** is bundled into the container image by `add_local_python_source`.
Every `modal run` rebuilds the image with the current local source. To update
code on Modal, just re-run `modal run` — no separate upload needed.

**Data** lives on the persistent Volume `pitchgpt-data`. Upload once with
`python scripts/upload_to_modal.py`. Re-upload only after preprocessing
changes (new augmented parquets, rebuilt profiles, etc.).

## Complete workflow

### 1. One-time setup

```bash
modal token new                          # authenticate (once per machine)
python scripts/upload_to_modal.py        # upload ~1.3 GB of data (once, idempotent)
```

### 2. Smoke test (verify code + data before committing to hours of GPU time)

```bash
modal run --detach modal_app.py --size sanity --max-steps 50 \
  --max-pitches 200000 --run-name smoke-test
```

Check that all loss terms are finite, shapes are correct, and the log shows
training steps. Pull and inspect:

```bash
modal volume get pitchgpt-data checkpoints/smoke-test/log.jsonl /tmp/smoke.jsonl
tail -5 /tmp/smoke.jsonl | python3 -m json.tool
```

### 3. Launch real training

```bash
# v8 example (full flags):
modal run --detach modal_app.py \
  --size small --fold 0 --epochs 3 \
  --type-conditioned-heads \
  --location-mdn --autoregressive-exec-heads --no-ab-outcome-head \
  --result-loss-weight 0.3 \
  --type-focal-gamma 2.0 --type-class-weight-alpha 0.5 \
  --run-name small-fold0-v8

# v7 example (simpler):
modal run --detach modal_app.py \
  --size small --fold 0 --epochs 3 \
  --type-conditioned-heads \
  --run-name small-fold0-v7
```

Note the app ID printed at launch (e.g., `ap-FMtyau3UZ5eYKOgtntagK2`). You
can view it at `https://modal.com/apps/siddhartha-thakur/main/<app-id>`.

### 4. Monitor training

The log file flushes to the volume periodically. Pull it to check progress:

```bash
modal volume get pitchgpt-data checkpoints/<run-name>/log.jsonl /tmp/train.jsonl --force
tail -5 /tmp/train.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if d.get('event') == 'train_step':
        print(f\"step {d['step']:>6d} epoch {d['epoch']} loss={d['total']:.3f} type={d['type']:.3f} zone={d['zone']:.3f}\")
    elif d['event'] in ('final_eval', 'checkpoint_saved', 'best_checkpoint'):
        print(f\"** {d['event']} ** step={d.get('step','')} {d}\")
"
```

Check if the app is still running:

```bash
modal app list
```

An empty list means training finished (or crashed). Check for the checkpoint:

```bash
modal volume ls pitchgpt-data checkpoints/<run-name>/
```

A successful run produces `checkpoint.pt` (and optionally `checkpoint_best.pt`
if early stopping fired). A run that only has `log.jsonl` crashed before any
checkpoint was saved.

### 5. Pull checkpoint + calibrate

```bash
modal volume get pitchgpt-data checkpoints/<run-name> ./checkpoints_modal/<run-name>/

python -m scripts.calibrate_pitchgpt \
  --ckpt checkpoints_modal/<run-name>/checkpoint.pt \
  --mdn-check   # v8 only: MDN distributional check
```

Calibration fits per-head temperature scalars on the val split (2024H1) and
writes `checkpoint_calibrated.pt` alongside the original. The `--mdn-check`
flag (v8+) reports sampled-vs-real plate_x statistics (the fraction with
|plate_x|>1.1 should approach the real ~0.37).

### 6. Evaluate / gate

After calibration, run the standard eval or the backtest gate:

```bash
# Standard eval (type/zone top-1, ECE, held-out-pitcher cohort)
make eval CKPT=checkpoints_modal/<run-name>/checkpoint_calibrated.pt

# Backtest gate (v8 — must beat lookup 1.4435 / baseline 1.4631)
python -m scripts.hitter.run_backtest_modal \
  --n 800 --n-paths 300 --seed 0
```

## GPU selection

| GPU | VRAM | cost | when to use |
|-----|------|------|-------------|
| L4  | 24 GB | ~$0.60/h | default — 25M params fit trivially; dataloader is the bottleneck |
| A100 | 40-80 GB | ~$2-4/h | only if batch size needs >24 GB or you need bf16 tensor cores |
| T4  | 16 GB | ~$0.30/h | inference-only (matchup cards, backtests) |

The training workload is **dataloader-bound** (`num_workers=0`, pandas
`__getitem__`), not GPU-bound. An L4 matches A100 throughput at ~1/3 the cost.
Measured: ~1.1 steps/s for both `tiny` and `small` sizes on L4.

## Timing estimates

| size | params | steps/epoch | epochs | wall-clock (L4) |
|------|--------|-------------|--------|-----------------|
| tiny | ~6M | ~4,700 | 3 | ~4h |
| small | ~25M | ~4,700 | 3 | ~10-11h |

The first epoch is slightly slower (JIT, cache warmup). Steps/epoch =
ceil(n_train_AB / batch_size) = ceil(1,204,129 / 256) = 4,703.

## Fan-out jobs (matchup cards, backtests)

For inference at scale, `modal_app.py` provides fan-out functions that
distribute work across multiple containers:

- `card_remote` — one game's matchup card (T4 GPU)
- `row_remote` — one pitcher row of a matchup card (T4 GPU)
- `backtest_remote` — one chunk of per-PA backtest evaluations (T4 GPU)

These use `.map()` over task dicts. **Modal container cap = 10** (user's
plan). Size work accordingly (e.g., 800 PAs / chunk_size 25 = 32 tasks, which
runs in 4 waves of 8 at 10-container cap).

Fan-out jobs are typically short (minutes, not hours), so `--detach` is less
critical — but still recommended for anything over ~5 minutes.

## Gotchas and hard-won lessons

### The `--detach` rule (again, because it matters)
Without `--detach`, the training container dies when the local process dies.
Macs sleep. Terminals close. Sessions end. Always `--detach`.

### Code vs data
Code is baked into the image (`add_local_python_source`). Data is on the
Volume. If you change model code, just re-run `modal run` — the image
rebuilds automatically. If you change training data, re-upload with
`python scripts/upload_to_modal.py --only augmented`.

### Run-name collisions
Reusing a run name appends to the existing `log.jsonl` and overwrites the
checkpoint. If a previous run crashed, the log will have duplicate step
numbers from both runs. Consider adding a timestamp suffix for clarity, or
accept the noise (the checkpoint is always from the latest run).

### MPS on Apple Silicon
The AB-outcome head's gather operation miscompiles on MPS (Issue #2). CPU
training is far too slow for real runs. **All real training goes through
Modal.** Local runs are smoke-tests only (`--max-steps 50 --max-pitches
200000`).

### Volume flush lag
The training log (`log.jsonl`) flushes to the Volume periodically, not after
every step. When monitoring, the log may be several minutes behind the actual
training progress. The `volume.commit()` call at the end of `train_remote`
ensures the final checkpoint is flushed.

### Separate images for training vs inference
Training needs `torch + pandas + pyarrow + numpy + scikit-learn`. Inference
(matchup cards, backtests) also needs `xgboost + requests` for the cascade
and MLB API. These are separate `modal.Image` definitions in `modal_app.py`
so the training image stays lean.

### The upload is idempotent
`scripts/upload_to_modal.py` deduplicates by file hash. Re-running it after
no data changes is fast (seconds). Use `--only augmented` to upload just one
directory. Use `--force` after schema migrations.

## Files reference

| file | purpose |
|------|---------|
| `modal_app.py` | all Modal functions (train, card, backtest, bench) |
| `scripts/upload_to_modal.py` | one-time data upload to Volume |
| `scripts/train_pitchgpt.py` | the `train()` function (shared local/Modal) |
| `scripts/calibrate_pitchgpt.py` | post-training temperature scaling |
| `checkpoints_modal/` | local directory for pulled checkpoints |
