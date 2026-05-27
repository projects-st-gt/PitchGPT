"""PitchGPT at-bat dataset — emits model-shaped batches from augmented parquets.

Sibling to ``data.dataset.AtBatDataset``; this one's output dict matches
:class:`model.pitchgpt.PitchGPT.forward`'s signature directly. Consumes the
augmented daily parquets produced by ``data.preprocess_pitchgpt`` and the
per-fold profile caches loaded via ``data.profile_cache_loader.ProfileCache``.

Per item returns the at-bat as:

```
{
    "pitcher_profile":      Tensor[pitcher_profile_dim],
    "batter_profile":       Tensor[batter_profile_dim],
    "categorical_context":  dict of 12 scalar LongTensors,
    "pitch_factors":        dict of 10 LongTensor[T] + spin_axis FloatTensor[T,2],
    "intended_actions":     dict of {type, zone, velo} LongTensor[T] +
                            spin_axis FloatTensor[T,2],
    "targets": {
        "propensity":    dict of LongTensor[T] (left-shifted; PAD_ID at end),
        "result":        LongTensor[T] (per-pitch result, PAD_ID where missing),
        "ab_outcome":    scalar LongTensor (terminal at-bat outcome class),
    },
    "padding_mask":         BoolTensor[T] (all True at item level),
}
```

``collate_pitchgpt_at_bats`` packs items into batched tensors with
right-padding, producing the structure :class:`PitchGPT` consumes after
unpacking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import Callable

from data.dataset import PAD_ID, PITCH_TYPES, ProfileLookup, temporal_split_mask

# ADR-013 Decision 2: matchup lookup is keyed on (pitcher_id, batter_id,
# asof_date, asof_game_num) rather than a single player. ``MatchupCache.lookup``
# (in data.profile_cache_loader) satisfies this signature.
MatchupLookup = Callable[[int, int, pd.Timestamp, int], dict[str, np.ndarray]]
from data.profile_cache import PITCHER_FEATURE_INDEX

DEFAULT_PROFILE_STD_PATH = Path("data/preprocess_artifacts/v1/profile_standardization.npz")

# Per-pitch arsenal feature (ADR 009): the 14-dim sub-vector of the pitcher
# profile that gets a dedicated per-pitch projection in the model — the 7
# `arsenal_{pt}` unconditional usage rates followed by the 7 `has_pitch_{pt}`
# binary flags. Pulled from the *raw* (pre-standardization) profile vector so
# the values are clean 0-1. Order: [arsenal_FF..arsenal_FS, has_pitch_FF..has_pitch_FS].
#
# Enumerate the exact 7+7 slot names by pitch type rather than prefix-matching
# `arsenal_` / `has_pitch_` — v6 added per-(type×count) and per-(type×stand)
# slots like `arsenal_FF_b0s0` and `arsenal_FF_vsL` that *also* start with
# `arsenal_`. A prefix match would pull those 84+14=98 slots in too, inflating
# the arsenal vector from 14 to 112 dims and crashing the arsenal_proj
# Linear(14, d_model) at forward time.
_ARSENAL_RATE_IDX = [PITCHER_FEATURE_INDEX[f"arsenal_{pt}"] for pt in PITCH_TYPES]
_HAS_PITCH_IDX = [PITCHER_FEATURE_INDEX[f"has_pitch_{pt}"] for pt in PITCH_TYPES]
ARSENAL_FEATURE_IDX: list[int] = _ARSENAL_RATE_IDX + _HAS_PITCH_IDX
N_ARSENAL_DIMS: int = len(ARSENAL_FEATURE_IDX)  # 14 — must match PitchGPTConfig.n_arsenal_dims


class ProfileStandardizer:
    """Per-feature z-score standardization for profile vectors.

    The pitcher (223-dim) and batter (91-dim) profile vectors mix features on
    wildly different scales — arsenal fractions (~0.3), velocities (~90 mph),
    spin rates (~2000 rpm), heatmap probabilities (~0.05), trailing-window
    counts (~100-1000). Fed raw into the context-token MLP, the high-magnitude
    features dominate the linear layer's output and gradient, making the
    low-magnitude-but-high-signal features (especially the arsenal composition,
    the strongest pitch-type predictor) effectively invisible to the model.

    Stats are fit on TRAINING-PERIOD profile cache entries only (no leakage)
    and stored in ``profile_standardization.npz``.
    """

    def __init__(self, path: Path = DEFAULT_PROFILE_STD_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"profile standardization stats not found at {path}; "
                f"run the fit step that produces them"
            )
        z = np.load(path)
        self.pitcher_mean = z["pitcher_mean"].astype(np.float32)
        self.pitcher_std = z["pitcher_std"].astype(np.float32)
        self.batter_mean = z["batter_mean"].astype(np.float32)
        self.batter_std = z["batter_std"].astype(np.float32)

    def apply(self, vec: np.ndarray, role: str) -> np.ndarray:
        if role == "pitcher":
            return ((vec - self.pitcher_mean) / self.pitcher_std).astype(np.float32)
        if role == "batter":
            return ((vec - self.batter_mean) / self.batter_std).astype(np.float32)
        raise ValueError(f"role must be 'pitcher' or 'batter', got {role!r}")

# ============================================================
# Augmented-parquet schema (must match data.preprocess_pitchgpt)
# ============================================================

PITCH_FACTOR_COLS_INT = {
    "type": "type_id",
    "zone": "feature_zone",
    "velo": "velo_bin",
    "spin_rate": "spin_rate_bin",
    "result": "result_id",
    "count": "count_state",
    "runners": "runners_state",
    "outs": "outs_state",
    "pos": "pos",
    "pitcher_fatigue": "pitcher_fatigue_bucket",
}

CATEGORICAL_CTX_COLS = {
    "p_throws": "p_throws_id",
    "stand": "stand_id",
    "ballpark": "ballpark_id",
    "umpire": "umpire_id",
    "catcher": "catcher_id",
    "inning": "inning_bucket",
    "score_diff": "score_diff_bucket",
    "inning_half": "inning_half",
    "days_rest": "days_rest_bucket",
    "tto": "tto_bucket",
    "temp": "temp_bucket",
    "roof": "roof_state",
}

# Extra categorical context introduced by ADR-013 Decision 2 (cross-AB context).
# Kept separate so that:
#   - the augmented-parquet schema check (REQUIRED_AUG_COLS) doesn't fail on
#     pre-v7p2 parquets that don't have ``tto_matchup_bucket``,
#   - the dataset only reads this column when a ``matchup_profile_lookup`` is
#     provided (i.e., cross-AB mode is explicitly enabled).
# ``load_augmented_pitches`` derives ``tto_matchup_bucket`` on-load from
# (game_pk, pitcher, batter, at_bat_number) — no re-augmentation needed.
CATEGORICAL_CTX_COLS_CROSS_AB = {
    "tto_matchup": "tto_matchup_bucket",
}

REQUIRED_AUG_COLS = (
    {"game_pk", "at_bat_number", "pitch_number", "game_date",
     "pitcher", "batter",
     "spin_axis_sin", "spin_axis_cos",
     "description", "events"}
    | set(PITCH_FACTOR_COLS_INT.values())
    | set(CATEGORICAL_CTX_COLS.values())
)


# ============================================================
# AB-outcome class mapping (7 classes per the architecture)
# ============================================================

AB_OUTCOME_K: int = 0
AB_OUTCOME_BB: int = 1
AB_OUTCOME_1B: int = 2
AB_OUTCOME_2B: int = 3
AB_OUTCOME_3B: int = 4
AB_OUTCOME_HR: int = 5
AB_OUTCOME_OUT: int = 6
AB_OUTCOME_IGNORE: int = PAD_ID  # for ABs without a clean terminal event

_K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
_BB_EVENTS = frozenset({"walk", "intent_walk", "hit_by_pitch"})
_1B_EVENTS = frozenset({"single"})
_2B_EVENTS = frozenset({"double"})
_3B_EVENTS = frozenset({"triple"})
_HR_EVENTS = frozenset({"home_run"})
_OUT_EVENTS = frozenset({
    "field_out", "force_out", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play",
    "grounded_into_double_play", "double_play", "triple_play",
    "fielders_choice", "fielders_choice_out",
    "field_error", "catcher_interf",
    "batter_interference", "fan_interference",
})


def classify_ab_outcome(event: object) -> int:
    """Map terminal-pitch ``events`` value to one of 7 AB-outcome classes."""
    if event is None or (isinstance(event, float) and np.isnan(event)):
        return AB_OUTCOME_IGNORE
    s = str(event)
    if s in _K_EVENTS:
        return AB_OUTCOME_K
    if s in _BB_EVENTS:
        return AB_OUTCOME_BB
    if s in _1B_EVENTS:
        return AB_OUTCOME_1B
    if s in _2B_EVENTS:
        return AB_OUTCOME_2B
    if s in _3B_EVENTS:
        return AB_OUTCOME_3B
    if s in _HR_EVENTS:
        return AB_OUTCOME_HR
    if s in _OUT_EVENTS:
        return AB_OUTCOME_OUT
    return AB_OUTCOME_IGNORE


# ============================================================
# Dataset
# ============================================================


class PitchGPTAtBatDataset(Dataset):
    """At-bat-level dataset that emits :class:`PitchGPT`-shaped items.

    Args:
        pitches: augmented pitches DataFrame (output of
            ``data.preprocess_pitchgpt.augment_day`` concatenated across days).
        pitcher_profile_lookup: ``ProfileLookup`` callable (typically a
            :class:`ProfileCache.lookup` method bound to the appropriate fold).
        batter_profile_lookup: same for batter.
    """

    def __init__(
        self,
        pitches: pd.DataFrame,
        *,
        pitcher_profile_lookup: ProfileLookup,
        batter_profile_lookup: ProfileLookup,
        profile_standardizer: Optional[ProfileStandardizer] = None,
        matchup_profile_lookup: Optional[MatchupLookup] = None,
    ):
        missing = REQUIRED_AUG_COLS - set(pitches.columns)
        if missing:
            raise KeyError(
                f"PitchGPTAtBatDataset missing augmented columns: {sorted(missing)}; "
                f"did you run `data/preprocess_pitchgpt.py apply`?"
            )

        # Cross-AB mode (ADR-013 Decision 2): when a matchup lookup is provided,
        # the dataset also emits ``matchup_profile`` per item and ``tto_matchup``
        # in ``categorical_context``. The on-load helper
        # :func:`load_augmented_pitches` derives ``tto_matchup_bucket`` from
        # existing columns; verify it landed before we promise to read it.
        self._matchup_lookup = matchup_profile_lookup
        if self._matchup_lookup is not None:
            need = set(CATEGORICAL_CTX_COLS_CROSS_AB.values())
            cross_ab_missing = need - set(pitches.columns)
            if cross_ab_missing:
                raise KeyError(
                    f"matchup_profile_lookup provided but augmented df missing "
                    f"{sorted(cross_ab_missing)}; load via "
                    f"`load_augmented_pitches` or call "
                    f"`data.preprocess_pitchgpt.compute_tto_matchup` on the df first."
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
        T = len(rows)
        first = rows.iloc[0]
        last = rows.iloc[-1]

        asof_date = pd.Timestamp(first["game_date"])
        asof_game_num = int(first["game_num"]) if "game_num" in first.index else 1

        pitcher_profile = self._pitcher_lookup(
            int(first["pitcher"]), asof_date, asof_game_num
        )["vector"]
        batter_profile = self._batter_lookup(
            int(first["batter"]), asof_date, asof_game_num
        )["vector"]
        # Per-pitch arsenal feature (ADR 009): pull the 14-dim arsenal+has-pitch
        # sub-vector from the RAW profile (before standardization), so the
        # model's Linear(14, d_model) sees clean 0-1 values.
        arsenal_feature = np.asarray(pitcher_profile, dtype=np.float32)[ARSENAL_FEATURE_IDX]
        if self._standardizer is not None:
            pitcher_profile = self._standardizer.apply(pitcher_profile, "pitcher")
            batter_profile = self._standardizer.apply(batter_profile, "batter")

        # Per-pitch integer factors. Each is a LongTensor[T].
        pitch_factors: dict[str, torch.Tensor] = {}
        for model_key, col in PITCH_FACTOR_COLS_INT.items():
            pitch_factors[model_key] = torch.as_tensor(
                rows[col].to_numpy(dtype=np.int64), dtype=torch.long
            )

        # Spin axis as (T, 2) float tensor: [sin, cos].
        spin_axis = torch.stack(
            [
                torch.as_tensor(rows["spin_axis_sin"].to_numpy(dtype=np.float32)),
                torch.as_tensor(rows["spin_axis_cos"].to_numpy(dtype=np.float32)),
            ],
            dim=-1,
        )
        pitch_factors["spin_axis"] = spin_axis

        # Categorical context: AB-level, but we read from the first row of the
        # AB (each AB has constant context within itself, per preprocessing).
        categorical_context: dict[str, torch.Tensor] = {}
        for model_key, col in CATEGORICAL_CTX_COLS.items():
            categorical_context[model_key] = torch.as_tensor(
                int(first[col]), dtype=torch.long
            )
        # ADR-013 Decision 2: cross-AB categoricals (tto_matchup) only when a
        # matchup lookup is wired. Keeps the dict shape pre-v7p2-compatible
        # when cross_ab_context is off.
        if self._matchup_lookup is not None:
            for model_key, col in CATEGORICAL_CTX_COLS_CROSS_AB.items():
                categorical_context[model_key] = torch.as_tensor(
                    int(first[col]), dtype=torch.long
                )

        # Intended actions (teacher-forced at training time = actual factors).
        intended_actions = {
            "type": pitch_factors["type"].clone(),
            "zone": pitch_factors["zone"].clone(),
            "velo": pitch_factors["velo"].clone(),
            "spin_axis": pitch_factors["spin_axis"].clone(),
        }

        # Propensity targets: left-shift each pitch factor by 1; last position
        # gets PAD_ID (no successor). Spin axis is treated as a regression
        # target via von-Mises parameterization downstream; here we only emit
        # the categorical-factor targets (the heads that use cross-entropy).
        propensity_targets: dict[str, torch.Tensor] = {}
        prop_keys = ("type", "zone", "velo", "spin_rate")
        for k in prop_keys:
            v = pitch_factors[k]
            propensity_targets[k] = torch.cat(
                [v[1:], torch.tensor([PAD_ID], dtype=v.dtype)]
            )

        # Result target: result_id - 1 → 0..6, PAD_ID where missing.
        result_target = pitch_factors["result"].clone() - 1
        result_target = result_target.where(
            pitch_factors["result"] != 0,
            torch.tensor(PAD_ID, dtype=result_target.dtype),
        )

        # AB-outcome target from terminal pitch's events.
        ab_outcome_target = torch.tensor(
            classify_ab_outcome(last.get("events")), dtype=torch.long
        )

        out = {
            "pitcher_profile": torch.as_tensor(pitcher_profile, dtype=torch.float32),
            "batter_profile": torch.as_tensor(batter_profile, dtype=torch.float32),
            "arsenal": torch.as_tensor(arsenal_feature, dtype=torch.float32),
            "categorical_context": categorical_context,
            "pitch_factors": pitch_factors,
            "intended_actions": intended_actions,
            "targets": {
                "propensity": propensity_targets,
                "result": result_target,
                "ab_outcome": ab_outcome_target,
            },
            "padding_mask": torch.ones(T, dtype=torch.bool),
        }
        # ADR-013 Decision 2 — emit the pitcher×batter matchup vector when
        # cross-AB mode is enabled. The collate stacks it into (B, matchup_dim).
        if self._matchup_lookup is not None:
            matchup_vec = self._matchup_lookup(
                int(first["pitcher"]),
                int(first["batter"]),
                asof_date,
                asof_game_num,
            )["vector"]
            out["matchup_profile"] = torch.as_tensor(matchup_vec, dtype=torch.float32)
        return out


# ============================================================
# Collate
# ============================================================


def collate_pitchgpt_at_bats(batch: list[dict]) -> dict:
    """Pack a list of model-shaped at-bat items into a batched dict.

    Variable-length per-pitch tensors are right-padded with ``PAD_ID``
    (for LongTensors) or zero (for the spin-axis sin/cos floats). The
    ``padding_mask`` is ``True`` for real positions and ``False`` for padding.
    """
    if not batch:
        raise ValueError("collate_pitchgpt_at_bats received an empty batch")

    B = len(batch)
    max_T = max(len(b["padding_mask"]) for b in batch)

    pitcher_profile = torch.stack([b["pitcher_profile"] for b in batch])
    batter_profile = torch.stack([b["batter_profile"] for b in batch])
    arsenal = torch.stack([b["arsenal"] for b in batch])  # (B, N_ARSENAL_DIMS) — ADR 009

    # Categorical context: scalar per key per item → stack to (B,).
    cat_keys = list(batch[0]["categorical_context"].keys())
    categorical_context = {
        k: torch.stack([b["categorical_context"][k] for b in batch])
        for k in cat_keys
    }

    int_factor_keys = [k for k in batch[0]["pitch_factors"].keys() if k != "spin_axis"]

    def _pad_long(rows_per_item: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        out_rows = []
        for v in rows_per_item:
            pad_n = max_T - len(v)
            if pad_n > 0:
                v = torch.cat([v, torch.full((pad_n,), pad_value, dtype=v.dtype)])
            out_rows.append(v)
        return torch.stack(out_rows)

    # Pad model-input factors with 0 (each factor's PAD/MISSING slot in the
    # corresponding embedding table). PAD_ID (-100) is a cross-entropy
    # ignore_index sentinel and is OUT OF VOCAB for an embedding lookup —
    # using it as input padding would crash F.embedding.
    pitch_factors: dict[str, torch.Tensor] = {}
    for k in int_factor_keys:
        pitch_factors[k] = _pad_long([b["pitch_factors"][k] for b in batch], pad_value=0)

    # Spin axis: (T, 2) per item → pad along T with zeros.
    spin_rows = []
    for b in batch:
        sa = b["pitch_factors"]["spin_axis"]
        pad_n = max_T - sa.shape[0]
        if pad_n > 0:
            sa = torch.cat([sa, torch.zeros(pad_n, 2, dtype=sa.dtype)], dim=0)
        spin_rows.append(sa)
    pitch_factors["spin_axis"] = torch.stack(spin_rows)

    # Intended actions mirror pitch factors at training time, but we re-pack
    # from the per-item dict so a caller can substitute counterfactual values
    # before collate if needed.
    intended_int_keys = [k for k in batch[0]["intended_actions"].keys() if k != "spin_axis"]
    intended_actions: dict[str, torch.Tensor] = {}
    for k in intended_int_keys:
        intended_actions[k] = _pad_long(
            [b["intended_actions"][k] for b in batch], pad_value=0
        )
    spin_rows = []
    for b in batch:
        sa = b["intended_actions"]["spin_axis"]
        pad_n = max_T - sa.shape[0]
        if pad_n > 0:
            sa = torch.cat([sa, torch.zeros(pad_n, 2, dtype=sa.dtype)], dim=0)
        spin_rows.append(sa)
    intended_actions["spin_axis"] = torch.stack(spin_rows)

    # Targets are passed to cross-entropy with ignore_index=PAD_ID; pad with
    # PAD_ID so the loss naturally skips padded positions.
    prop_keys = list(batch[0]["targets"]["propensity"].keys())
    propensity_targets = {
        k: _pad_long([b["targets"]["propensity"][k] for b in batch], pad_value=PAD_ID)
        for k in prop_keys
    }
    result_targets = _pad_long(
        [b["targets"]["result"] for b in batch], pad_value=PAD_ID
    )
    ab_outcome_targets = torch.stack([b["targets"]["ab_outcome"] for b in batch])

    padding_mask = torch.zeros(B, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        padding_mask[i, : len(b["padding_mask"])] = True

    out = {
        "pitcher_profile": pitcher_profile,
        "batter_profile": batter_profile,
        "arsenal": arsenal,
        "categorical_context": categorical_context,
        "pitch_factors": pitch_factors,
        "intended_actions": intended_actions,
        "targets": {
            "propensity": propensity_targets,
            "result": result_targets,
            "ab_outcome": ab_outcome_targets,
        },
        "padding_mask": padding_mask,
    }
    # ADR-013 Decision 2 — when items carry matchup_profile (cross-AB mode),
    # stack them. Absence is back-compat default.
    if "matchup_profile" in batch[0]:
        out["matchup_profile"] = torch.stack([b["matchup_profile"] for b in batch])
    return out


# ============================================================
# Convenience: load + split + build dataset
# ============================================================


def load_augmented_pitches(
    augmented_dir: Path,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Load augmented daily parquets across a year range into one DataFrame.

    Args:
        augmented_dir: root directory of augmented parquets
            (``data/augmented`` by default).
        years: optional list of years to include; if None, loads all.

    Side effect: derives ``tto_matchup_bucket`` (ADR-013 Decision 2) on-load
    via :func:`data.preprocess_pitchgpt.compute_tto_matchup`. It's a fast
    groupby cumcount on existing columns; doing it here avoids re-augmenting
    every parquet (the column is deterministic from ``game_pk``,
    ``at_bat_number``, ``pitcher``, ``batter``, all already present). The
    dataset only consumes this column when ``matchup_profile_lookup`` is
    wired, but precomputing it unconditionally keeps the loader idempotent
    and lets older callers ignore it.
    """
    parts = []
    if not augmented_dir.exists():
        raise FileNotFoundError(f"augmented dir not found: {augmented_dir}")
    year_dirs = sorted(p for p in augmented_dir.iterdir() if p.is_dir())
    for year_dir in year_dirs:
        try:
            year_int = int(year_dir.name)
        except ValueError:
            continue
        if years is not None and year_int not in years:
            continue
        for parquet in sorted(year_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(parquet))
    if not parts:
        raise RuntimeError(
            f"no augmented parquets found under {augmented_dir} for years={years}"
        )
    df = pd.concat(parts, ignore_index=True)
    # ADR-013 D2 — derive tto_matchup_bucket on-load (deterministic, cheap)
    if "tto_matchup_bucket" not in df.columns:
        need = {"game_pk", "at_bat_number", "pitcher", "batter"}
        if need.issubset(df.columns):
            from data.preprocess_pitchgpt import compute_tto_matchup
            df["tto_matchup_bucket"] = compute_tto_matchup(df)
    return df


def split_augmented(pitches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply the temporal split (CLAUDE.md) to a loaded augmented DataFrame."""
    masks = temporal_split_mask(pitches)
    return {k: pitches[m].reset_index(drop=True) for k, m in masks.items()}
