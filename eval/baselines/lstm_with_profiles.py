"""Profile-aware LSTM baseline per the ``eval-protocol`` skill.

This is the "apples-to-apples" LSTM the skill calls for: same inputs as
PitchGPT (per-pitch features + per-AB profile vectors from the cache),
LSTM instead of attention. Tests whether **attention specifically**
matters for sequence modeling here vs. recurrence.

Architecture:

- Per-pitch features: same flat numeric features XGBoost saw (count,
  handedness, prev pitch, etc.) → shape ``(B, T, F)``.
- Per-AB profile: 314-dim vector = 223 pitcher (from
  ``ProfileCache(role="pitcher")``) + 91 batter (from
  ``ProfileCache(role="batter")``) → shape ``(B, 314)``.
- The profile vector is projected via two ``Linear`` layers to LSTM-
  shaped ``(num_layers, B, hidden_dim)`` for the initial ``h0`` and
  ``c0``. This lets the LSTM start with per-AB context already loaded.
- LSTM rolls over the per-pitch sequence; head outputs per-position
  logits over the 7 pitch types.

Notes on fold-awareness: a strict cross-fit eval would build five
separate LSTMs, one per fold, each using the profile cache for its fold.
For a v1 baseline number we use **fold-0's profile cache** for both
training and val — fold 0 excludes 20% of training games' profile
contributions, an acceptable bias for an eval-table cell. A stricter
version is straightforward but deferred.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import N_PITCH_TYPES


def _pick_device(prefer: str | None = None) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _LSTMWithProfile(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_profile: int,
        hidden_dim: int,
        num_layers: int,
        n_classes: int = N_PITCH_TYPES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Project profile → initial (h0, c0). Each is (num_layers, B, hidden_dim).
        self.profile_to_h = nn.Linear(n_profile, num_layers * hidden_dim)
        self.profile_to_c = nn.Linear(n_profile, num_layers * hidden_dim)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        x_features: torch.Tensor,
        x_profile: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        # x_features: (B, T, F);  x_profile: (B, P);  lengths: (B,)
        B = x_features.size(0)
        h0 = self.profile_to_h(x_profile).view(B, self.num_layers, self.hidden_dim)
        c0 = self.profile_to_c(x_profile).view(B, self.num_layers, self.hidden_dim)
        h0 = h0.transpose(0, 1).contiguous()  # → (num_layers, B, hidden_dim)
        c0 = c0.transpose(0, 1).contiguous()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_features, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        out_packed, _ = self.lstm(packed, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out)


class _ABDataset(torch.utils.data.Dataset):
    """Per-AB samples: (per-pitch features, AB profile, targets)."""

    def __init__(
        self,
        features: torch.Tensor,     # (N, F)
        profiles: torch.Tensor,     # (N_AB, P) — one row per AB
        targets: torch.Tensor,      # (N,)
        ab_row_indices: list[np.ndarray],  # per-AB pitch row indices
    ):
        self.features = features
        self.profiles = profiles
        self.targets = targets
        self.ab_row_indices = ab_row_indices

    def __len__(self) -> int:
        return len(self.ab_row_indices)

    def __getitem__(self, idx: int):
        rows = self.ab_row_indices[idx]
        idx_t = torch.from_numpy(rows.astype(np.int64))
        return (
            self.features[idx_t],
            self.profiles[idx],
            self.targets[idx_t],
            len(rows),
        )


def _collate(batch):
    Xs, profs, ys, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    Xs_padded = nn.utils.rnn.pad_sequence(Xs, batch_first=True)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    profs_stacked = torch.stack(profs)
    return Xs_padded, profs_stacked, ys_padded, lengths


def _group_by_ab(at_bat_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (unique_ab_ids in encounter order, list of row-index arrays per AB)."""
    order = np.argsort(at_bat_ids, kind="stable")
    sorted_ids = at_bat_ids[order]
    breaks = np.where(np.diff(sorted_ids) != 0)[0] + 1
    groups = np.split(order, breaks)
    # Each group's first element gives the AB id (after we look it up via at_bat_ids)
    unique_ids = np.array([at_bat_ids[g[0]] for g in groups])
    return unique_ids, groups


class LSTMBaselineWithProfiles:
    """LSTM that consumes the profile cache the same way PitchGPT will.

    Args:
        n_profile: total profile vector length (pitcher + batter dims).
        hidden_dim, num_layers: LSTM size.
        epochs, batch_size, lr: training schedule.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``, or ``None`` for auto.
        seed: reproducibility.
    """

    def __init__(
        self,
        n_profile: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.n_profile = n_profile
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = _pick_device(device)
        self.seed = seed
        self.verbose = verbose
        self._net: _LSTMWithProfile | None = None
        self._feature_columns: list[str] | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    # ---------- standardize ----------

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = (X - self._feature_means) / np.where(
            self._feature_stds == 0, 1.0, self._feature_stds
        )
        return np.nan_to_num(X, nan=0.0).astype(np.float32)

    # ---------- fit ----------

    def fit(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        at_bat_ids: np.ndarray,
        ab_profile_lookup,  # callable(ab_id) -> np.ndarray of shape (n_profile,)
    ) -> "LSTMBaselineWithProfiles":
        if len(features) != len(targets) or len(features) != len(at_bat_ids):
            raise ValueError("features, targets, at_bat_ids must have matching lengths")
        torch.manual_seed(self.seed)

        self._feature_columns = list(features.columns)
        X_np = features.to_numpy(dtype=np.float32)
        with np.errstate(invalid="ignore"):
            self._feature_means = np.nanmean(X_np, axis=0).astype(np.float32)
            self._feature_stds = np.nanstd(X_np, axis=0).astype(np.float32)
        self._feature_means = np.nan_to_num(self._feature_means, nan=0.0)
        self._feature_stds = np.nan_to_num(self._feature_stds, nan=1.0)

        X_std = self._standardize(X_np)
        feats_tensor = torch.from_numpy(X_std)
        targets_tensor = torch.from_numpy(targets.astype(np.int64))

        unique_ab_ids, ab_groups = _group_by_ab(at_bat_ids)
        if self.verbose:
            print(f"  LSTM-profile dataset: {len(at_bat_ids):,} pitches in "
                  f"{len(ab_groups):,} ABs; device={self.device}")

        # Look up profiles for all ABs once
        profiles = np.zeros((len(unique_ab_ids), self.n_profile), dtype=np.float32)
        for i, ab_id in enumerate(unique_ab_ids):
            profiles[i] = ab_profile_lookup(int(ab_id))
        prof_tensor = torch.from_numpy(profiles)
        if self.verbose:
            print(f"  Loaded {len(unique_ab_ids):,} AB profiles, "
                  f"shape {profiles.shape}")

        ds = _ABDataset(feats_tensor, prof_tensor, targets_tensor, ab_groups)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=_collate, num_workers=0,
        )

        n_features = feats_tensor.shape[1]
        self._net = _LSTMWithProfile(
            n_features=n_features,
            n_profile=self.n_profile,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            t0 = time.monotonic()
            total_loss, total_correct, total_count = 0.0, 0, 0
            self._net.train()
            for X_batch, prof_batch, y_batch, lengths in loader:
                X_batch = X_batch.to(self.device)
                prof_batch = prof_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self._net(X_batch, prof_batch, lengths)
                loss = F.cross_entropy(
                    logits.reshape(-1, N_PITCH_TYPES),
                    y_batch.reshape(-1),
                    ignore_index=-100,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    valid_mask = y_batch != -100
                    preds = logits.argmax(dim=-1)
                    total_correct += int((preds[valid_mask] == y_batch[valid_mask]).sum())
                    total_count += int(valid_mask.sum())
                    total_loss += float(loss.item()) * int(valid_mask.sum())
            if self.verbose:
                avg_loss = total_loss / max(total_count, 1)
                acc = total_correct / max(total_count, 1)
                print(f"  epoch {epoch+1}/{self.epochs}: "
                      f"loss={avg_loss:.4f}, acc={acc:.4f}, "
                      f"{time.monotonic() - t0:.1f}s")
        return self

    # ---------- predict ----------

    @torch.no_grad()
    def predict_proba(
        self,
        features: pd.DataFrame,
        at_bat_ids: np.ndarray,
        ab_profile_lookup,
    ) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("must fit before predict_proba")
        if len(features) != len(at_bat_ids):
            raise ValueError("features and at_bat_ids must have matching lengths")

        self._net.eval()
        n = len(features)
        out = np.zeros((n, N_PITCH_TYPES), dtype=np.float32)

        X_np = features[self._feature_columns].to_numpy(dtype=np.float32)
        feats_tensor = torch.from_numpy(self._standardize(X_np))
        unique_ab_ids, ab_groups = _group_by_ab(at_bat_ids)

        for chunk_start in range(0, len(ab_groups), self.batch_size):
            chunk_groups = ab_groups[chunk_start:chunk_start + self.batch_size]
            chunk_ab_ids = unique_ab_ids[chunk_start:chunk_start + self.batch_size]
            X_list = [feats_tensor[torch.from_numpy(g.astype(np.int64))]
                      for g in chunk_groups]
            lengths = torch.tensor([len(g) for g in chunk_groups], dtype=torch.long)
            X_padded = nn.utils.rnn.pad_sequence(X_list, batch_first=True).to(self.device)
            profs = np.stack([ab_profile_lookup(int(ab_id)) for ab_id in chunk_ab_ids])
            prof_t = torch.from_numpy(profs).to(self.device)
            logits = self._net(X_padded, prof_t, lengths).cpu()
            probs = F.softmax(logits, dim=-1).numpy()
            for i, g in enumerate(chunk_groups):
                T = lengths[i].item()
                for pos in range(T):
                    out[g[pos]] = probs[i, pos]
        return out

    def predict(
        self,
        features: pd.DataFrame,
        at_bat_ids: np.ndarray,
        ab_profile_lookup,
    ) -> np.ndarray:
        return self.predict_proba(features, at_bat_ids, ab_profile_lookup).argmax(axis=1).astype(np.int64)
