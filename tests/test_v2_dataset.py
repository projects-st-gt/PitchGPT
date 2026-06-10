"""Tests for the V2 at-bat dataset.

Tests 1-4 use a synthetic fixture (allowed in tests/ per CLAUDE.md rule 1a).
Test 5 loads a real augmented parquet to verify shapes on actual Statcast data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import PAD_ID
from model.v2.dataset import (
    CONTINUOUS_COLS,
    N_CONTINUOUS,
    V2AtBatDataset,
    collate_v2_at_bats,
)


# ============================================================
# Fixtures
# ============================================================

def _make_fake_pitch_row(
    game_pk: int,
    at_bat_number: int,
    pitch_number: int,
    type_id: int = 1,
    release_speed: float = 95.0,
    release_spin_rate: float = 2200.0,
    plate_x: float = 0.1,
    plate_z: float = 2.5,
    result_id: int = 2,
    count_state: int = 0,
    outs_state: int = 0,
    runners_state: int = 0,
    pitcher: int = 12345,
    batter: int = 67890,
    game_date: str = "2024-04-01",
    spin_axis_sin: float = 0.5,
    spin_axis_cos: float = -0.5,
) -> dict:
    return {
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "type_id": type_id,
        "release_speed": release_speed,
        "release_spin_rate": release_spin_rate,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "spin_axis_sin": spin_axis_sin,
        "spin_axis_cos": spin_axis_cos,
        "result_id": result_id,
        "count_state": count_state,
        "outs_state": outs_state,
        "runners_state": runners_state,
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
    }


def _make_3pitch_ab() -> pd.DataFrame:
    """A 3-pitch at-bat with known values."""
    rows = [
        _make_fake_pitch_row(1, 1, 1, type_id=1, release_speed=95.0,
                             release_spin_rate=2200, plate_x=0.1, plate_z=2.5,
                             result_id=2, count_state=0, outs_state=1,
                             runners_state=3),
        _make_fake_pitch_row(1, 1, 2, type_id=4, release_speed=85.0,
                             release_spin_rate=2600, plate_x=-0.3, plate_z=3.0,
                             result_id=1, count_state=1, outs_state=1,
                             runners_state=3),
        _make_fake_pitch_row(1, 1, 3, type_id=6, release_speed=87.0,
                             release_spin_rate=1800, plate_x=0.5, plate_z=1.8,
                             result_id=5, count_state=4, outs_state=1,
                             runners_state=3),
    ]
    return pd.DataFrame(rows)


def _make_2pitch_ab() -> pd.DataFrame:
    """A 2-pitch at-bat (different game_pk so it's a separate AB)."""
    rows = [
        _make_fake_pitch_row(2, 1, 1, type_id=3, release_speed=90.0,
                             release_spin_rate=2400, plate_x=0.0, plate_z=2.0,
                             result_id=1, count_state=0, outs_state=0,
                             runners_state=0),
        _make_fake_pitch_row(2, 1, 2, type_id=5, release_speed=80.0,
                             release_spin_rate=2700, plate_x=-0.5, plate_z=3.2,
                             result_id=6, count_state=3, outs_state=0,
                             runners_state=0),
    ]
    return pd.DataFrame(rows)


def _zero_profile_lookup(player_id, asof_date, asof_game_num):
    """Stub profile lookup returning zeros. Allowed in tests per rule 1a."""
    return {"vector": np.zeros(223, dtype=np.float32), "source": "test_stub"}


def _zero_batter_profile_lookup(player_id, asof_date, asof_game_num):
    return {"vector": np.zeros(91, dtype=np.float32), "source": "test_stub"}


@pytest.fixture
def dataset_3pitch():
    df = _make_3pitch_ab()
    return V2AtBatDataset(
        df,
        pitcher_profile_lookup=_zero_profile_lookup,
        batter_profile_lookup=_zero_batter_profile_lookup,
    )


@pytest.fixture
def dataset_mixed():
    """Two ABs: one 3-pitch, one 2-pitch (for collate tests)."""
    df = pd.concat([_make_3pitch_ab(), _make_2pitch_ab()], ignore_index=True)
    return V2AtBatDataset(
        df,
        pitcher_profile_lookup=_zero_profile_lookup,
        batter_profile_lookup=_zero_batter_profile_lookup,
    )


# ============================================================
# Test 1: Position 0 has type_id=0 (PAD) and continuous=zeros
# ============================================================

def test_position_0_is_start_token(dataset_3pitch):
    item = dataset_3pitch[0]
    assert item["type_ids"][0].item() == 0, (
        f"Position 0 type_id should be PAD(0), got {item['type_ids'][0].item()}"
    )
    assert torch.all(item["continuous"][0] == 0.0), (
        f"Position 0 continuous should be all zeros, got {item['continuous'][0]}"
    )
    assert item["result_ids"][0].item() == 0, (
        f"Position 0 result_id should be 0('none'), got {item['result_ids'][0].item()}"
    )
    assert item["pitch_number"][0].item() == 0, (
        f"Position 0 pitch_number should be 0, got {item['pitch_number'][0].item()}"
    )

    # Named numerical checks per CLAUDE.md discipline:
    # Position 0 count_state = 0 (start of AB), outs_state = 1, runners_state = 3
    assert item["count_state"][0].item() == 0, (
        f"Position 0 count_state = {item['count_state'][0].item()}, expected 0"
    )
    assert item["outs"][0].item() == 1, (
        f"Position 0 outs = {item['outs'][0].item()}, expected 1"
    )
    assert item["runners"][0].item() == 3, (
        f"Position 0 runners = {item['runners'][0].item()}, expected 3"
    )


# ============================================================
# Test 2: Targets are left-shifted (target.type[0] = type_ids[1])
# ============================================================

def test_targets_left_shifted(dataset_3pitch):
    item = dataset_3pitch[0]

    # The 3 real pitches have type_ids [1, 4, 6].
    # After prepend: type_ids = [0, 1, 4, 6]
    # Targets: type[0]=1, type[1]=4, type[2]=6, type[3]=-100 (PAD)
    assert item["targets"]["type"][0].item() == 1, (
        f"target.type[0] = {item['targets']['type'][0].item()}, "
        f"expected 1 (= type_ids[1] = pitch 1's type)"
    )
    assert item["targets"]["type"][1].item() == 4, (
        f"target.type[1] = {item['targets']['type'][1].item()}, expected 4"
    )
    assert item["targets"]["type"][2].item() == 6, (
        f"target.type[2] = {item['targets']['type'][2].item()}, expected 6"
    )

    # Continuous left-shift: target.continuous[0] should be pitch 1's values.
    # Pitch 1: release_speed=95.0, release_spin_rate=2200, plate_x=0.1, plate_z=2.5
    tc0 = item["targets"]["continuous"][0]
    assert tc0[0].item() == pytest.approx(95.0, abs=0.01), (
        f"target.continuous[0][release_speed] = {tc0[0].item()}, expected 95.0"
    )
    assert tc0[1].item() == pytest.approx(2200.0, abs=0.01), (
        f"target.continuous[0][release_spin_rate] = {tc0[1].item()}, expected 2200.0"
    )


# ============================================================
# Test 3: Last position target = -100 for type, NaN for continuous
# ============================================================

def test_last_position_target_is_pad(dataset_3pitch):
    item = dataset_3pitch[0]
    seq_len = len(item["type_ids"])
    assert seq_len == 4, f"Expected 3 pitches + 1 start = 4, got {seq_len}"

    last_idx = seq_len - 1
    assert item["targets"]["type"][last_idx].item() == PAD_ID, (
        f"target.type[{last_idx}] = {item['targets']['type'][last_idx].item()}, "
        f"expected PAD_ID ({PAD_ID})"
    )
    assert torch.all(torch.isnan(item["targets"]["continuous"][last_idx])), (
        f"target.continuous[{last_idx}] should be all NaN, "
        f"got {item['targets']['continuous'][last_idx]}"
    )


# ============================================================
# Test 4: Collate pads correctly with different-length ABs
# ============================================================

def test_collate_pads_correctly(dataset_mixed):
    # AB 0 = 3 pitches (seq_len 4), AB 1 = 2 pitches (seq_len 3)
    items = [dataset_mixed[0], dataset_mixed[1]]
    batch = collate_v2_at_bats(items)

    B, max_T = 2, 4  # max_T = 3+1 = 4 from the 3-pitch AB

    assert batch["type_ids"].shape == (B, max_T)
    assert batch["continuous"].shape == (B, max_T, N_CONTINUOUS)
    assert batch["padding_mask"].shape == (B, max_T)
    assert batch["targets"]["type"].shape == (B, max_T)
    assert batch["targets"]["continuous"].shape == (B, max_T, N_CONTINUOUS)

    # First AB (len 4): all positions real.
    assert batch["padding_mask"][0].sum().item() == 4

    # Second AB (len 3): 3 real + 1 padding.
    assert batch["padding_mask"][1].sum().item() == 3
    assert batch["padding_mask"][1, 3].item() is False

    # Padding values for the shorter AB at position 3:
    assert batch["type_ids"][1, 3].item() == 0, "type_ids padding should be 0"
    assert batch["result_ids"][1, 3].item() == 0, "result_ids padding should be 0"
    assert batch["targets"]["type"][1, 3].item() == PAD_ID, (
        f"target.type padding should be {PAD_ID}"
    )
    assert torch.all(torch.isnan(batch["targets"]["continuous"][1, 3])), (
        "target.continuous padding should be NaN"
    )

    # Profile shapes: just stacked.
    assert batch["pitcher_profile"].shape == (B, 223)
    assert batch["batter_profile"].shape == (B, 91)


# ============================================================
# Test 5: Real augmented parquet — load, build dataset, check shapes
# ============================================================

AUGMENTED_DIR = Path("data/augmented")


@pytest.mark.skipif(
    not (AUGMENTED_DIR / "2024").exists(),
    reason="No augmented data at data/augmented/2024",
)
def test_real_augmented_parquet():
    """Load one real augmented parquet, build dataset, verify shapes."""
    # Pick the first available parquet in 2024.
    year_dir = AUGMENTED_DIR / "2024"
    parquet_path = sorted(year_dir.glob("*.parquet"))[0]
    df = pd.read_parquet(parquet_path)

    ds = V2AtBatDataset(
        df,
        pitcher_profile_lookup=_zero_profile_lookup,
        batter_profile_lookup=_zero_batter_profile_lookup,
    )

    assert len(ds) > 0, f"Dataset should have at-bats, got {len(ds)}"

    item = ds[0]
    T_plus_1 = len(item["type_ids"])
    assert T_plus_1 >= 2, f"Sequence must be at least 2 (start + 1 pitch), got {T_plus_1}"

    # Shape checks.
    assert item["type_ids"].shape == (T_plus_1,)
    assert item["continuous"].shape == (T_plus_1, N_CONTINUOUS)
    assert item["result_ids"].shape == (T_plus_1,)
    assert item["count_state"].shape == (T_plus_1,)
    assert item["outs"].shape == (T_plus_1,)
    assert item["runners"].shape == (T_plus_1,)
    assert item["pitch_number"].shape == (T_plus_1,)
    assert item["padding_mask"].shape == (T_plus_1,)
    assert item["targets"]["type"].shape == (T_plus_1,)
    assert item["targets"]["continuous"].shape == (T_plus_1, N_CONTINUOUS)

    # Position 0 invariants.
    assert item["type_ids"][0].item() == 0
    assert torch.all(item["continuous"][0] == 0.0)
    assert item["result_ids"][0].item() == 0
    assert item["pitch_number"][0].item() == 0

    # Last target is PAD/-100 for type, NaN for continuous.
    assert item["targets"]["type"][-1].item() == PAD_ID
    assert torch.all(torch.isnan(item["targets"]["continuous"][-1]))

    # Left-shift check: target.type[0] should equal the first real pitch's type_id.
    first_real_type = item["type_ids"][1].item()
    assert item["targets"]["type"][0].item() == first_real_type, (
        f"target.type[0] = {item['targets']['type'][0].item()}, "
        f"expected {first_real_type} (first real pitch type_id)"
    )

    # Named numerical check: print the first AB's values for manual verification.
    print(f"\n--- Real parquet: {parquet_path.name} ---")
    print(f"  ABs in file: {len(ds)}")
    print(f"  First AB seq_len (T+1): {T_plus_1}")
    print(f"  type_ids: {item['type_ids'].tolist()}")
    print(f"  targets.type: {item['targets']['type'].tolist()}")
    print(f"  continuous[1] (first real pitch):"
          f" speed={item['continuous'][1, 0]:.1f} mph,"
          f" spin={item['continuous'][1, 1]:.0f} rpm,"
          f" plate_x={item['continuous'][1, 2]:.3f} ft,"
          f" plate_z={item['continuous'][1, 3]:.3f} ft")

    # Collate a small batch from the real data.
    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_v2_at_bats(items)
    B = len(items)
    max_T = max(len(it["type_ids"]) for it in items)
    assert batch["type_ids"].shape == (B, max_T)
    assert batch["continuous"].shape == (B, max_T, N_CONTINUOUS)
    print(f"  Collated batch: B={B}, max_T={max_T}")
