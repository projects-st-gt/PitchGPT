"""V2InputLayer — per-pitch token construction for PitchGPT v2.

Each pitch token is the sum of three contributions:

1. Type embedding — nn.Embedding(8, d_model), PAD at index 0
2. Continuous projection — a linear map from a 50-dim state vector:
       [velo, spin, plate_x, plate_z]  (4 floats)
     + result one-hot                  (8 = 7 results + "none")
     + count_state one-hot             (12)
     + outs one-hot                    (3)
     + runners one-hot                 (8)
     + pitch_number one-hot            (15)
     = 50 dims total
3. Positional embedding — nn.Embedding(15, d_model) keyed by pitch_number (0-based)

This is lighter than v9's factored-embedding approach: there are no zone/velo/
spin-rate categorical tables, no concat-then-project mixer. The continuous
pitcher properties go directly into a projection, which is simpler and
sufficient for base-v1c.

Weight init: all weights ~ N(0, cfg.init_std), all biases = 0.
Padding at type_emb index 0 is zeroed as part of standard init.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.v2.config import V2Config

# ---------------------------------------------------------------------------
# Dimension breakdown for the 50-dim state vector.
# Kept as module-level constants so callers can reference them without magic
# numbers. Changing one of these requires updating the V2Config accordingly.
# ---------------------------------------------------------------------------

_N_RESULT = 8            # 7 results + index 0 = "none" (position 0 in AB)
_N_COUNT = 12            # count states 0..11
_N_OUTS = 3              # 0, 1, 2 outs
_N_RUNNERS = 8           # runner occupancy bitmask 0..7
_N_PITCH_NUMBER = 15     # pitch 0..14 within the AB

_N_CATEGORICAL = _N_RESULT + _N_COUNT + _N_OUTS + _N_RUNNERS + _N_PITCH_NUMBER  # 46


def state_vec_dim(n_continuous: int) -> int:
    """State-vector width: continuous dims (cfg-driven; 4 for v1c checkpoints,
    6 from v1c.1 with the spin axis) + the fixed categorical one-hots."""
    return n_continuous + _N_CATEGORICAL


class V2InputLayer(nn.Module):
    """Build per-pitch tokens from type embedding + continuous projection + positional.

    Args (forward):
        type_ids     : (B, T) LongTensor 0..7 — pitch type, PAD = 0
        continuous   : (B, T, 4) float — [velo, spin, plate_x, plate_z]
        result_ids   : (B, T) LongTensor 0..7 — result of the current pitch
                       (0 = "none" for position 0, before any result is known)
        count_state  : (B, T) LongTensor 0..11 — pre-pitch count bucket
        outs         : (B, T) LongTensor 0..2
        runners      : (B, T) LongTensor 0..7 — runner occupancy bitmask
        pitch_number : (B, T) LongTensor 0..14 — pitch index within the AB

    Returns:
        (B, T, d_model) — one token per pitch

    Notes on the state vector layout:
        dims 0:4           — raw continuous floats
        dims 4:12          — result_id one-hot (8 classes)
        dims 12:24         — count_state one-hot (12 classes)
        dims 24:27         — outs one-hot (3 classes)
        dims 27:35         — runners one-hot (8 classes)
        dims 35:50         — pitch_number one-hot (15 classes)
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Type embedding: 8 vocab slots (PAD=0, FF=1 … FS=7)
        self.type_emb = nn.Embedding(cfg.n_pitch_types, d, padding_idx=0)

        # Projects the state vector → d_model (52 dims at n_continuous=6;
        # 50 for legacy 4-dim checkpoints — width comes from the config).
        self.continuous_proj = nn.Linear(state_vec_dim(cfg.n_continuous), d)

        # Positional embedding (pitch index within AB, 0-based, max 14)
        self.pos_emb = nn.Embedding(cfg.max_positions, d)

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=std)
        # Zero out the PAD row so PAD tokens contribute nothing.
        with torch.no_grad():
            self.type_emb.weight[0].zero_()
        nn.init.normal_(self.continuous_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.continuous_proj.bias)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Internal helper: build one-hot for a (B, T) integer tensor
    # ------------------------------------------------------------------

    @staticmethod
    def _one_hot(ids: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Convert (B, T) LongTensor to (B, T, n_classes) float one-hot.

        Clamps indices to [0, n_classes-1] to prevent CUDA scatter_ assertions
        on rare out-of-range data values.
        """
        B, T = ids.shape
        safe_ids = ids.clamp(0, n_classes - 1)
        oh = ids.new_zeros(B, T, n_classes, dtype=torch.float32)
        oh.scatter_(-1, safe_ids.unsqueeze(-1), 1.0)
        return oh

    def forward(
        self,
        type_ids: torch.Tensor,      # (B, T) LongTensor 0..7
        continuous: torch.Tensor,    # (B, T, 4)
        result_ids: torch.Tensor,    # (B, T) LongTensor 0..7
        count_state: torch.Tensor,   # (B, T) LongTensor 0..11
        outs: torch.Tensor,          # (B, T) LongTensor 0..2
        runners: torch.Tensor,       # (B, T) LongTensor 0..7
        pitch_number: torch.Tensor,  # (B, T) LongTensor 0..14
    ) -> torch.Tensor:               # (B, T, d_model)
        # 1. Type embedding (clamp to prevent OOB on rare edge-case data).
        te = self.type_emb(type_ids.clamp(0, self.cfg.n_pitch_types - 1))  # (B, T, d)

        # 2. Build the 50-dim state vector.
        #    Concatenate raw floats and one-hot encoded categoricals along dim=-1.
        result_oh  = self._one_hot(result_ids,   _N_RESULT)        # (B, T, 8)
        count_oh   = self._one_hot(count_state,  _N_COUNT)         # (B, T, 12)
        outs_oh    = self._one_hot(outs,         _N_OUTS)          # (B, T, 3)
        runners_oh = self._one_hot(runners,      _N_RUNNERS)       # (B, T, 8)
        pnum_oh    = self._one_hot(pitch_number, _N_PITCH_NUMBER)  # (B, T, 15)

        state_vec = torch.cat(
            [continuous, result_oh, count_oh, outs_oh, runners_oh, pnum_oh],
            dim=-1,
        )  # (B, T, 50)

        # 3. Project to d_model.
        cp = self.continuous_proj(state_vec)  # (B, T, d)

        # 4. Positional embedding.
        pe = self.pos_emb(pitch_number.clamp(0, self.cfg.max_positions - 1))  # (B, T, d)

        return te + cp + pe                   # (B, T, d)
