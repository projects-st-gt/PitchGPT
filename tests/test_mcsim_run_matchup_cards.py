"""Tests for the MCSim App B runner — checkpoint hashing + end-to-end orchestration."""
from __future__ import annotations

from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from scripts.mcsim import run_matchup_cards as runner


def test_compute_ckpt_hash_is_stable_and_truncated(tmp_path):
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"hello world")
    h1 = runner.compute_ckpt_hash(f)
    h2 = runner.compute_ckpt_hash(f)
    assert h1 == h2                      # deterministic
    assert len(h1) == 16                 # truncated
    # sha256("hello world") starts with b94d27b9934d3e08...
    assert h1 == "b94d27b9934d3e08"
