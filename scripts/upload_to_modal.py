"""One-time upload of PitchGPT data to the ``pitchgpt-data`` Modal Volume.

Run after ``data/preprocess_pitchgpt.py apply`` produces the augmented
parquets. Idempotent: re-running only uploads changed/new files (Modal's
volume.put deduplicates by file hash).

Uploads:
- ``data/augmented/``      (≈380 MB) — model-ready per-pitch parquets
- ``data/profiles/``       (≈870 MB) — per-fold profile cache
- ``data/preprocess_artifacts/`` — vocabs + spin-rate edges + velo stats
- ``data/run_value/``      — RE24 tables (used by Phase D's causal layer)

Total: ≈1.3 GB. At a typical 10 MB/s upstream this is ~2 min.

Usage:

    python scripts/upload_to_modal.py
    python scripts/upload_to_modal.py --dry-run    # print what would upload
    python scripts/upload_to_modal.py --only augmented profiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

VOLUME_NAME = "pitchgpt-data"

DEFAULT_PAIRS: list[tuple[Path, str]] = [
    (Path("data/augmented"), "augmented"),
    (Path("data/profiles"), "profiles"),
    (Path("data/preprocess_artifacts"), "preprocess_artifacts"),
    (Path("data/run_value"), "run_value"),
]


def file_inventory(local: Path) -> tuple[int, int]:
    files = [p for p in local.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> None:
    p = argparse.ArgumentParser(description="Upload PitchGPT data to Modal Volume")
    p.add_argument(
        "--only", nargs="+",
        help="Limit uploads to these top-level remote names (e.g. augmented profiles)",
    )
    p.add_argument("--dry-run", action="store_true", help="show plan only")
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files on the Volume (needed after schema migrations).",
    )
    args = p.parse_args()

    pairs = DEFAULT_PAIRS
    if args.only:
        pairs = [(lp, rp) for lp, rp in pairs if rp in set(args.only)]

    print(f"Target Volume: {VOLUME_NAME}")
    total_mb = 0.0
    for local, remote_name in pairs:
        if not local.exists():
            print(f"  SKIP   {local}  →  {remote_name}  (local path missing)")
            continue
        n_files, size_bytes = file_inventory(local)
        mb = size_bytes / 1e6
        total_mb += mb
        print(f"  STAGE  {local}  →  /{remote_name}   {n_files:>5} files, {mb:>7.1f} MB")
    print(f"  -----  total to upload: {total_mb:.1f} MB")

    if args.dry_run:
        print("\n--dry-run: no upload.")
        return

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=args.force) as batch:
        for local, remote_name in pairs:
            if not local.exists():
                continue
            print(f"  uploading {local} → /{remote_name} ...")
            batch.put_directory(local_path=local, remote_path=f"/{remote_name}")
    print("\nUpload complete. Verify with:\n    modal run modal_app.py::list_data")


if __name__ == "__main__":
    main()
