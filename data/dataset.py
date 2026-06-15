"""PyTorch dataset and collate for at-bat-level training.

Per the ``statcast-pipeline`` skill, each item is one at-bat returned as a
dict of tensors. The collate function packs a list of variable-length
at-bats into a batch, padding shorter ones to the longest in the batch and
producing a boolean ``padding_mask`` (True = real position, False = padding).

Two abstractions to keep the dataset testable:

- ``ProfileLookup`` is a callable interface — the dataset asks for a
  pitcher/batter profile vector given ``(player_id, asof_date, asof_game_num)``.
  Real implementations read from a pre-computed cache. Tests pass a mock.
- Tokenization vocabularies are explicit module-level constants
  (``PITCH_TYPES``, ``RESULT_CLASSES``, etc.) so the model and dataset agree
  on the embedding space without circular imports.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------- token vocabularies ----------

PITCH_TYPES: list[str] = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]
PITCH_TYPE_TO_ID: dict[str, int] = {p: i for i, p in enumerate(PITCH_TYPES)}
N_PITCH_TYPES: int = len(PITCH_TYPES)

# ----- Model-side type vocabulary convention -----
#
# The pitch-type FACTOR is stored in the parquet as integer ids 1..7 with PAD=0
# (see ``data.preprocess_pitchgpt.compute_type_id``). The model's propensity
# TYPE head emits a softmax over 8 classes indexed by these same ids — PAD at
# 0, PITCH_TYPES[0]=FF at MODEL index 1, ..., PITCH_TYPES[6]=FS at MODEL
# index 7. Code that slices model TYPE head outputs MUST account for this
# 1-indexing — three bugs in one session came from ``[:N_PITCH_TYPES]``
# slicing (which gets PAD + 6 of 7 types, missing FS).
#
# Asymmetry warning: the RESULT head emits only 7 logits (no PAD column);
# the dataset shifts result targets by -1 to bring them into [0, 7). Result
# head reads do NOT have this off-by-one trap.
#
# When in doubt, use ``MODEL_TYPE_ID["FF"]`` instead of writing ``1`` literally.
MODEL_TYPE_VOCAB_SIZE: int = N_PITCH_TYPES + 1   # 8 = PAD + 7 canonical
MODEL_TYPE_PAD_IDX: int = 0
MODEL_PITCH_TYPES_START_IDX: int = 1              # FF lives here (i.e., model index 1)
MODEL_PITCH_TYPES_END_IDX: int = MODEL_TYPE_VOCAB_SIZE  # exclusive end (= 8)

# Convenience: model-index for each named pitch type.
# ``PITCH_TYPES[i]`` lives at model index ``i + MODEL_PITCH_TYPES_START_IDX``.
MODEL_TYPE_ID: dict[str, int] = {
    pt: i + MODEL_PITCH_TYPES_START_IDX for i, pt in enumerate(PITCH_TYPES)
}

ACTION_ZONES: list[str] = ["up", "down", "arm-side", "glove-side", "out-of-zone"]
ACTION_ZONE_TO_ID: dict[str, int] = {z: i for i, z in enumerate(ACTION_ZONES)}
N_ACTION_ZONES: int = len(ACTION_ZONES)

# Feature zone IDs are already integers 0..25 from data.zones (cell 25 = OOZ).
N_FEATURE_ZONES: int = 26
N_VELO_BINS: int = 10  # deciles produced by data.preprocess.bin_velo_z_score

# 7-class result encoding from the statcast-pipeline skill.
RESULT_CLASSES: list[str] = [
    "ball", "called_strike", "swinging_strike", "foul",
    "in_play_out", "in_play_hit", "in_play_hr",
]
RESULT_TO_ID: dict[str, int] = {r: i for i, r in enumerate(RESULT_CLASSES)}
N_RESULTS: int = len(RESULT_CLASSES)

# Sentinel ID for unknown / unpadded categorical positions. Matches PyTorch's
# default ``ignore_index`` for cross-entropy loss so the loss skips these.
PAD_ID: int = -100

# Temporal split (non-negotiable). All bounds are inclusive.
# 2026+ data is intentionally outside every split: it is reserved for live-demo
# inference and ad-hoc validation. Adding 2026 to test would inflate the
# "generalization" claim with what is really an OOD slice on which the model
# was not trained or held out as a calibration target.
TRAIN_END: pd.Timestamp = pd.Timestamp("2023-12-31")
VAL_START: pd.Timestamp = pd.Timestamp("2024-01-01")
VAL_END: pd.Timestamp = pd.Timestamp("2024-07-15")
TEST_START: pd.Timestamp = pd.Timestamp("2024-07-16")
TEST_END: pd.Timestamp = pd.Timestamp("2025-12-31")

# Statcast description → result class
_BALL_DESCRIPTIONS = frozenset({
    "ball", "intent_ball", "blocked_ball", "hit_by_pitch",
    "pitchout",
})
_CALLED_STRIKE_DESCRIPTIONS = frozenset({"called_strike"})
_SWING_STRIKE_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
})
_FOUL_DESCRIPTIONS = frozenset({"foul", "foul_tip", "foul_bunt", "bunt_foul_tip"})
_IN_PLAY_DESCRIPTIONS = frozenset({"hit_into_play"})

_HIT_EVENTS = frozenset({"single", "double", "triple"})
_HR_EVENTS = frozenset({"home_run"})


def temporal_split_mask(pitches: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the train / val / test temporal split.

    Returns a dict with keys ``"train"``, ``"val"``, ``"test"``. Boundaries:

    - train: ``game_date <= 2023-12-31``
    - val:   ``2024-01-01 <= game_date <= 2024-07-15``
    - test:  ``2024-07-16 <= game_date <= 2025-12-31``

    2026+ dates are intentionally in NONE of these masks — they are reserved
    for live-demo inference and ad-hoc validation, not for model training or
    formal evaluation.

    Required column: ``game_date``. The split is defined on dates only;
    fold assignments (ADR 006) are an orthogonal concern within the
    training portion.
    """
    if "game_date" not in pitches.columns:
        raise KeyError("temporal_split_mask requires a 'game_date' column")
    dates = pd.to_datetime(pitches["game_date"])
    return {
        "train": dates <= TRAIN_END,
        "val": (dates >= VAL_START) & (dates <= VAL_END),
        "test": (dates >= TEST_START) & (dates <= TEST_END),
    }


def temporal_split(pitches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train / val / test slices per the project temporal split.

    Returns DataFrame views (not copies) sharing memory with the input.
    """
    masks = temporal_split_mask(pitches)
    return {k: pitches[m] for k, m in masks.items()}


def classify_result(description: object, events: object) -> int | None:
    """Map ``(description, events)`` to a 7-class result ID, or ``None`` if unmappable."""
    d = None if pd.isna(description) else description
    e = None if pd.isna(events) else events
    if d in _BALL_DESCRIPTIONS:
        return RESULT_TO_ID["ball"]
    if d in _CALLED_STRIKE_DESCRIPTIONS:
        return RESULT_TO_ID["called_strike"]
    if d in _SWING_STRIKE_DESCRIPTIONS:
        return RESULT_TO_ID["swinging_strike"]
    if d in _FOUL_DESCRIPTIONS:
        return RESULT_TO_ID["foul"]
    if d in _IN_PLAY_DESCRIPTIONS:
        if e in _HR_EVENTS:
            return RESULT_TO_ID["in_play_hr"]
        if e in _HIT_EVENTS:
            return RESULT_TO_ID["in_play_hit"]
        return RESULT_TO_ID["in_play_out"]
    return None


# ---------- profile lookup interface ----------

# Callable signature: (player_id, asof_date, asof_game_num) -> {"vector": ndarray}
# The "vector" key is the flattened profile features the model embeds. Other
# keys are allowed for diagnostics but not consumed by the dataset.
ProfileLookup = Callable[[int, pd.Timestamp, int], dict[str, np.ndarray]]


# ---------- AtBatDataset ----------


class AtBatDataset(Dataset):
    """Per-at-bat dataset returning the dict shape the model consumes.

    Each ``__getitem__(idx)`` returns one at-bat as:

    ```
    {
        "pitcher_profile": Tensor[d_p],
        "batter_profile":  Tensor[d_b],
        "context_tokens":  LongTensor[n_context],
        "pitch_factors":   {"type": LongTensor[T], "feature_zone": ..., ...},
        "result_factors":  LongTensor[T],
        "target_factors":  same shape as pitch_factors, shifted by 1; last
                           position is PAD_ID since there is no successor.
        "padding_mask":    BoolTensor[T] (all True per item; collate fills False
                           for padded positions in a batch).
    }
    ```

    Required ``pitches`` columns: ``game_pk``, ``at_bat_number``, ``pitch_number``,
    ``game_date``, ``pitcher``, ``batter``, ``pitch_type_canonical``,
    ``feature_zone``, ``action_zone``, ``description``, ``events``, ``balls``,
    ``strikes``, ``p_throws``, ``stand``, ``outs_when_up``, ``on_1b``, ``on_2b``,
    ``on_3b``. Optional: ``velo_bin``, ``game_num``.
    """

    REQUIRED_COLUMNS: frozenset[str] = frozenset({
        "game_pk", "at_bat_number", "pitch_number",
        "game_date", "pitcher", "batter",
        "pitch_type_canonical", "feature_zone", "action_zone",
        "description", "events", "balls", "strikes",
        "p_throws", "stand", "outs_when_up", "on_1b", "on_2b", "on_3b",
    })

    def __init__(
        self,
        pitches: pd.DataFrame,
        *,
        pitcher_profile_lookup: ProfileLookup,
        batter_profile_lookup: ProfileLookup,
        velo_bin_col: Optional[str] = "velo_bin",
    ):
        missing = self.REQUIRED_COLUMNS - set(pitches.columns)
        if missing:
            raise KeyError(f"AtBatDataset missing columns: {sorted(missing)}")

        self._df = (
            pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
            .reset_index(drop=True)
        )
        self._velo_bin_col = (
            velo_bin_col if velo_bin_col and velo_bin_col in self._df.columns else None
        )

        groups = self._df.groupby(["game_pk", "at_bat_number"], sort=False)
        self._ab_keys = list(groups.groups.keys())
        self._ab_indices = [groups.indices[k] for k in self._ab_keys]

        self._pitcher_lookup = pitcher_profile_lookup
        self._batter_lookup = batter_profile_lookup

    def __len__(self) -> int:
        return len(self._ab_keys)

    def __getitem__(self, idx: int) -> dict:
        rows = self._df.iloc[self._ab_indices[idx]]
        T = len(rows)
        first = rows.iloc[0]

        asof_date = pd.Timestamp(first["game_date"])
        asof_game_num = int(first["game_num"]) if "game_num" in first.index else 1

        pitcher_profile = self._pitcher_lookup(
            int(first["pitcher"]), asof_date, asof_game_num
        )["vector"]
        batter_profile = self._batter_lookup(
            int(first["batter"]), asof_date, asof_game_num
        )["vector"]

        type_ids = torch.tensor(
            [PITCH_TYPE_TO_ID.get(t, PAD_ID) for t in rows["pitch_type_canonical"]],
            dtype=torch.long,
        )
        feature_zone_ids = torch.tensor(
            rows["feature_zone"].astype(int).to_numpy(), dtype=torch.long
        )
        action_zone_ids = torch.tensor(
            [ACTION_ZONE_TO_ID.get(str(z), PAD_ID) for z in rows["action_zone"]],
            dtype=torch.long,
        )

        pitch_factors: dict[str, torch.Tensor] = {
            "type": type_ids,
            "feature_zone": feature_zone_ids,
            "action_zone": action_zone_ids,
        }
        if self._velo_bin_col is not None:
            pitch_factors["velo_bin"] = torch.tensor(
                [int(v) if pd.notna(v) else PAD_ID for v in rows[self._velo_bin_col]],
                dtype=torch.long,
            )

        result_ids = torch.tensor(
            [
                classify_result(d, e) if classify_result(d, e) is not None else PAD_ID
                for d, e in zip(rows["description"], rows["events"])
            ],
            dtype=torch.long,
        )

        # Targets are pitch factors shifted left by 1: position t predicts pitch
        # at t+1. The last position has no successor and gets PAD_ID, which the
        # model's loss should treat as ignore_index.
        target_factors: dict[str, torch.Tensor] = {}
        for k, v in pitch_factors.items():
            target_factors[k] = torch.cat([v[1:], torch.tensor([PAD_ID], dtype=v.dtype)])

        return {
            "pitcher_profile": torch.as_tensor(pitcher_profile, dtype=torch.float32),
            "batter_profile": torch.as_tensor(batter_profile, dtype=torch.float32),
            "context_tokens": _build_context_tokens(first),
            "pitch_factors": pitch_factors,
            "result_factors": result_ids,
            "target_factors": target_factors,
            "padding_mask": torch.ones(T, dtype=torch.bool),
        }


def _build_context_tokens(first_row: pd.Series) -> torch.Tensor:
    """Encode the AB-start situation as a small fixed-length LongTensor.

    Order: ``[p_throws, stand, outs, on_1b, on_2b, on_3b]``.
    Handedness: 0=R, 1=L. (Switch hitters resolve to the side they're batting
    in this PA, which is what ``stand`` already encodes.)
    """
    p_throws = 1 if first_row.get("p_throws") == "L" else 0
    stand = 1 if first_row.get("stand") == "L" else 0
    outs = int(first_row.get("outs_when_up", 0) or 0)
    on1 = int(pd.notna(first_row.get("on_1b")))
    on2 = int(pd.notna(first_row.get("on_2b")))
    on3 = int(pd.notna(first_row.get("on_3b")))
    return torch.tensor([p_throws, stand, outs, on1, on2, on3], dtype=torch.long)


# ---------- collate ----------


def collate_at_bats(batch: list[dict]) -> dict:
    """Pack a list of at-bat dicts into a batch.

    Variable-length per-pitch tensors are right-padded to the max length in
    the batch with ``PAD_ID``. The ``padding_mask`` is ``True`` for real
    positions and ``False`` for padding. Profiles and context tokens are
    fixed-shape per item, so they just stack.
    """
    if not batch:
        raise ValueError("collate_at_bats received an empty batch")

    batch_size = len(batch)
    max_T = max(len(b["padding_mask"]) for b in batch)

    pitcher_profile = torch.stack([b["pitcher_profile"] for b in batch])
    batter_profile = torch.stack([b["batter_profile"] for b in batch])
    context_tokens = torch.stack([b["context_tokens"] for b in batch])

    factor_keys = list(batch[0]["pitch_factors"].keys())
    pitch_factors: dict[str, torch.Tensor] = {}
    target_factors: dict[str, torch.Tensor] = {}
    for k in factor_keys:
        p_rows, t_rows = [], []
        for b in batch:
            v = b["pitch_factors"][k]
            t = b["target_factors"][k]
            pad_n = max_T - len(v)
            p_rows.append(torch.cat([v, torch.full((pad_n,), PAD_ID, dtype=v.dtype)]))
            t_rows.append(torch.cat([t, torch.full((pad_n,), PAD_ID, dtype=t.dtype)]))
        pitch_factors[k] = torch.stack(p_rows)
        target_factors[k] = torch.stack(t_rows)

    result_rows = []
    for b in batch:
        v = b["result_factors"]
        pad_n = max_T - len(v)
        result_rows.append(torch.cat([v, torch.full((pad_n,), PAD_ID, dtype=v.dtype)]))
    result_factors = torch.stack(result_rows)

    padding_mask = torch.zeros(batch_size, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        padding_mask[i, : len(b["padding_mask"])] = True

    return {
        "pitcher_profile": pitcher_profile,
        "batter_profile": batter_profile,
        "context_tokens": context_tokens,
        "pitch_factors": pitch_factors,
        "result_factors": result_factors,
        "target_factors": target_factors,
        "padding_mask": padding_mask,
    }
