"""V2 at-bat dataset — emits items in PitchGPTV2's forward() format.

The key difference from V1: position 0 is a synthetic "before any pitch"
token.  The model predicts the first pitch at position 0, so the dataset
prepends a start position with type=PAD(0), continuous=zeros, result=0("none"),
and the real pre-AB game state (count, outs, runners).

Targets are left-shifted: the target at position t is the NEXT pitch's type
and continuous values. The last position gets PAD/-100/NaN targets (nothing
to predict after the final pitch).

Continuous values (release_speed, release_spin_rate, plate_x, plate_z) are
emitted raw — no normalization. The GMM head learns to predict in the
original units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset import PAD_ID

# Re-use the loading + splitting helpers from V1 so there's one truth for
# augmented-parquet discovery and the temporal split.
from model.pitchgpt_dataset import (
    load_augmented_pitches,
    split_augmented,
    ProfileStandardizer,
)

# The 4 continuous features in the order the model expects them.
CONTINUOUS_COLS = ["release_speed", "release_spin_rate", "plate_x", "plate_z"]
N_CONTINUOUS = len(CONTINUOUS_COLS)


class V2AtBatDataset(Dataset):
    """At-bat dataset for PitchGPTV2.

    Each item prepends a start position (type=0, continuous=zeros, result=0)
    before the real pitches, giving a sequence of length T+1 where T is the
    number of pitches in the at-bat.

    Args:
        pitches: augmented pitches DataFrame grouped into at-bats.
        pitcher_profile_lookup: ``(player_id, asof_date, game_num) -> {"vector": ndarray}``
        batter_profile_lookup: same for batter.
        profile_standardizer: optional z-score standardizer for profiles.
    """

    # Minimum columns we actually read from the augmented parquets.
    REQUIRED_COLS = frozenset({
        "game_pk", "at_bat_number", "pitch_number", "game_date",
        "pitcher", "batter",
        "type_id", "release_speed", "release_spin_rate", "plate_x", "plate_z",
        "result_id", "count_state", "runners_state", "outs_state",
    })

    def __init__(
        self,
        pitches: pd.DataFrame,
        *,
        pitcher_profile_lookup,
        batter_profile_lookup,
        profile_standardizer: Optional[ProfileStandardizer] = None,
    ):
        missing = self.REQUIRED_COLS - set(pitches.columns)
        if missing:
            raise KeyError(
                f"V2AtBatDataset missing columns: {sorted(missing)}; "
                f"did you run `data/preprocess_pitchgpt.py apply`?"
            )

        self._df = (
            pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
            .reset_index(drop=True)
        )
        groups = self._df.groupby(["game_pk", "at_bat_number"], sort=False)
        self._ab_keys = list(groups.groups.keys())
        self._ab_indices = [groups.indices[k] for k in self._ab_keys]

        self._pitcher_lookup = pitcher_profile_lookup
        self._batter_lookup = batter_profile_lookup
        self._standardizer = profile_standardizer

    def __len__(self) -> int:
        return len(self._ab_keys)

    def __getitem__(self, idx: int) -> dict:
        rows = self._df.iloc[self._ab_indices[idx]]
        T = len(rows)  # number of real pitches
        first = rows.iloc[0]

        # ---- Profiles ----
        asof_date = pd.Timestamp(first["game_date"])
        # game_num not in augmented parquets; default to 1.
        asof_game_num = int(first["game_num"]) if "game_num" in first.index else 1

        pitcher_vec = self._pitcher_lookup(
            int(first["pitcher"]), asof_date, asof_game_num
        )["vector"]
        batter_vec = self._batter_lookup(
            int(first["batter"]), asof_date, asof_game_num
        )["vector"]

        if self._standardizer is not None:
            pitcher_vec = self._standardizer.apply(pitcher_vec, "pitcher")
            batter_vec = self._standardizer.apply(batter_vec, "batter")

        # ---- Pitch-level sequences (length T, the real pitches) ----
        type_ids_raw = rows["type_id"].to_numpy(dtype=np.int64)    # 1..7
        result_ids_raw = rows["result_id"].to_numpy(dtype=np.int64)  # 1..7
        count_raw = rows["count_state"].to_numpy(dtype=np.int64)   # 0..11
        outs_raw = rows["outs_state"].to_numpy(dtype=np.int64)     # 0..2
        runners_raw = rows["runners_state"].to_numpy(dtype=np.int64)  # 0..7
        pitch_num_raw = rows["pitch_number"].to_numpy(dtype=np.int64)  # 1..T

        # Continuous: (T, 4). Replace NaN with 0.0.
        cont_raw = rows[CONTINUOUS_COLS].to_numpy(dtype=np.float64)
        cont_raw = np.nan_to_num(cont_raw, nan=0.0).astype(np.float32)

        # ---- Prepend start position (position 0 = "before any pitch") ----
        # type=PAD(0), continuous=zeros, result=0("none")
        # count/outs/runners = the real pre-AB state (same as pitch 1's state)
        seq_len = T + 1  # start position + T real pitches

        type_ids = np.zeros(seq_len, dtype=np.int64)
        type_ids[1:] = type_ids_raw

        continuous = np.zeros((seq_len, N_CONTINUOUS), dtype=np.float32)
        continuous[1:] = cont_raw

        result_ids = np.zeros(seq_len, dtype=np.int64)
        result_ids[1:] = result_ids_raw

        count_state = np.zeros(seq_len, dtype=np.int64)
        count_state[0] = int(count_raw[0])  # pre-AB count (same as pitch 1)
        count_state[1:] = count_raw

        outs = np.zeros(seq_len, dtype=np.int64)
        outs[0] = int(outs_raw[0])
        outs[1:] = outs_raw

        runners = np.zeros(seq_len, dtype=np.int64)
        runners[0] = int(runners_raw[0])
        runners[1:] = runners_raw

        pitch_number = np.zeros(seq_len, dtype=np.int64)
        pitch_number[0] = 0  # "no pitch yet"
        pitch_number[1:] = pitch_num_raw

        # ---- Targets: left-shifted ----
        # target at position t = pitch t+1's type + continuous
        # target at position seq_len-1 = no next pitch -> PAD/-100/NaN
        target_type = np.full(seq_len, PAD_ID, dtype=np.int64)
        target_type[:-1] = type_ids[1:]  # positions 0..(T-1) predict the next type

        target_continuous = np.full((seq_len, N_CONTINUOUS), float("nan"), dtype=np.float32)
        target_continuous[:-1] = continuous[1:]  # positions 0..(T-1) predict next continuous

        return {
            "pitcher_profile": torch.as_tensor(pitcher_vec, dtype=torch.float32),
            "batter_profile": torch.as_tensor(batter_vec, dtype=torch.float32),
            "type_ids": torch.as_tensor(type_ids, dtype=torch.long),
            "continuous": torch.as_tensor(continuous, dtype=torch.float32),
            "result_ids": torch.as_tensor(result_ids, dtype=torch.long),
            "count_state": torch.as_tensor(count_state, dtype=torch.long),
            "outs": torch.as_tensor(outs, dtype=torch.long),
            "runners": torch.as_tensor(runners, dtype=torch.long),
            "pitch_number": torch.as_tensor(pitch_number, dtype=torch.long),
            "padding_mask": torch.ones(seq_len, dtype=torch.bool),
            "targets": {
                "type": torch.as_tensor(target_type, dtype=torch.long),
                "continuous": torch.as_tensor(target_continuous, dtype=torch.float32),
            },
        }


# ============================================================
# Collate
# ============================================================


def collate_v2_at_bats(batch: list[dict]) -> dict:
    """Right-pad variable-length at-bats into a batch.

    Padding values:
        - type_ids, result_ids, count_state, outs, runners, pitch_number: 0
        - continuous: 0.0
        - padding_mask: False
        - targets.type: -100 (ignore_index)
        - targets.continuous: NaN
        - pitcher_profile, batter_profile: just stack
    """
    if not batch:
        raise ValueError("collate_v2_at_bats received an empty batch")

    B = len(batch)
    max_T = max(len(b["padding_mask"]) for b in batch)

    # Profiles: same length across items, just stack.
    pitcher_profile = torch.stack([b["pitcher_profile"] for b in batch])
    batter_profile = torch.stack([b["batter_profile"] for b in batch])

    def _pad_long(items: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        out = []
        for v in items:
            pad_n = max_T - len(v)
            if pad_n > 0:
                v = torch.cat([v, torch.full((pad_n,), pad_value, dtype=v.dtype)])
            out.append(v)
        return torch.stack(out)

    def _pad_float2d(items: list[torch.Tensor], pad_value: float) -> torch.Tensor:
        D = items[0].shape[-1]
        out = []
        for v in items:
            pad_n = max_T - v.shape[0]
            if pad_n > 0:
                v = torch.cat([v, torch.full((pad_n, D), pad_value, dtype=v.dtype)], dim=0)
            out.append(v)
        return torch.stack(out)

    type_ids = _pad_long([b["type_ids"] for b in batch], pad_value=0)
    result_ids = _pad_long([b["result_ids"] for b in batch], pad_value=0)
    count_state = _pad_long([b["count_state"] for b in batch], pad_value=0)
    outs = _pad_long([b["outs"] for b in batch], pad_value=0)
    runners = _pad_long([b["runners"] for b in batch], pad_value=0)
    pitch_number = _pad_long([b["pitch_number"] for b in batch], pad_value=0)
    continuous = _pad_float2d([b["continuous"] for b in batch], pad_value=0.0)

    # padding_mask: True for real, False for padding.
    padding_mask = torch.zeros(B, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        padding_mask[i, : len(b["padding_mask"])] = True

    # Targets
    target_type = _pad_long([b["targets"]["type"] for b in batch], pad_value=PAD_ID)
    target_continuous = _pad_float2d(
        [b["targets"]["continuous"] for b in batch], pad_value=float("nan")
    )

    return {
        "pitcher_profile": pitcher_profile,
        "batter_profile": batter_profile,
        "type_ids": type_ids,
        "continuous": continuous,
        "result_ids": result_ids,
        "count_state": count_state,
        "outs": outs,
        "runners": runners,
        "pitch_number": pitch_number,
        "padding_mask": padding_mask,
        "targets": {
            "type": target_type,
            "continuous": target_continuous,
        },
    }
