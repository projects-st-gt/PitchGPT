"""LSTM baseline per the ``eval-protocol`` skill.

Tests whether **temporal modeling** (sequence over the AB) buys us anything
over the IID-feature XGBoost view. The skill specifies "same inputs as
PitchGPT" — once the full profile cache is built that comparison becomes
truly apples-to-apples. For now this lite version consumes the same flat
features XGBoost did, grouped into per-AB sequences, isolating the value
of the recurrent structure itself.

Architecture:
- Standardize numeric features on training data
- Group rows by ``(game_pk, at_bat_number)`` into variable-length sequences
- 1-layer LSTM, hidden 64 (small but tuned later)
- Linear head: hidden → 7 pitch types
- Cross-entropy with ``ignore_index=-100`` for padded positions
- Predicts at every position (autoregressive next-pitch given history-so-far)

Order preservation: predict_proba returns rows in **input** order, matching
the ``targets``/``at_bat_ids`` arrays the runner aligns against.
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
    """MPS on Apple Silicon if available; else CUDA; else CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int,
                 n_classes: int = N_PITCH_TYPES, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out)  # (B, T, C)


class _ABSequenceDataset(torch.utils.data.Dataset):
    """Holds (features, targets) tensors plus per-AB row index lists."""

    def __init__(
        self,
        features: torch.Tensor,    # (N, F)
        targets: torch.Tensor,     # (N,)
        ab_row_indices: list[np.ndarray],  # one entry per AB, original row indices
    ):
        self.features = features
        self.targets = targets
        self.ab_row_indices = ab_row_indices

    def __len__(self) -> int:
        return len(self.ab_row_indices)

    def __getitem__(self, idx: int):
        rows = self.ab_row_indices[idx]
        idx_t = torch.from_numpy(rows.astype(np.int64))
        return self.features[idx_t], self.targets[idx_t], len(rows)


def _collate_padded(batch):
    Xs, ys, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    Xs_padded = nn.utils.rnn.pad_sequence(Xs, batch_first=True)
    ys_padded = nn.utils.rnn.pad_sequence(
        ys, batch_first=True, padding_value=-100,
    )
    return Xs_padded, ys_padded, lengths


def _group_by_ab(at_bat_ids: np.ndarray) -> list[np.ndarray]:
    """For each unique AB id, return the array of original row indices."""
    order = np.argsort(at_bat_ids, kind="stable")
    sorted_ids = at_bat_ids[order]
    breaks = np.where(np.diff(sorted_ids) != 0)[0] + 1
    groups = np.split(order, breaks)
    return groups


class LSTMBaseline:
    """Sequence-modeling baseline: per-AB LSTM over flat per-pitch features.

    Args:
        hidden_dim, num_layers: LSTM size.
        epochs, batch_size, lr: training schedule.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``, or ``None`` for auto.
        seed: reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = _pick_device(device)
        self.seed = seed
        self.verbose = verbose
        self._net: _LSTMNet | None = None
        self._feature_columns: list[str] | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    # ---------- helpers ----------

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Apply training-fit standardization. Replaces NaN with 0 *after*
        standardization so the model sees a consistent "no info" value."""
        X = (X - self._feature_means) / np.where(
            self._feature_stds == 0, 1.0, self._feature_stds
        )
        return np.nan_to_num(X, nan=0.0).astype(np.float32)

    def _to_tensors(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = features[self._feature_columns].to_numpy(dtype=np.float32)
        X = self._standardize(X)
        feats_tensor = torch.from_numpy(X)
        targets_tensor = torch.from_numpy(targets.astype(np.int64))
        return feats_tensor, targets_tensor

    # ---------- fit ----------

    def fit(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        at_bat_ids: np.ndarray,
    ) -> "LSTMBaseline":
        if len(features) != len(targets) or len(features) != len(at_bat_ids):
            raise ValueError("features, targets, at_bat_ids must have matching lengths")
        torch.manual_seed(self.seed)

        self._feature_columns = list(features.columns)
        # Standardize using training-data mean/std (NaN-aware).
        X = features.to_numpy(dtype=np.float32)
        with np.errstate(invalid="ignore"):
            self._feature_means = np.nanmean(X, axis=0).astype(np.float32)
            self._feature_stds = np.nanstd(X, axis=0).astype(np.float32)
        # Replace any all-NaN columns with 0 mean / 1 std (won't happen in practice)
        self._feature_means = np.nan_to_num(self._feature_means, nan=0.0)
        self._feature_stds = np.nan_to_num(self._feature_stds, nan=1.0)

        feats_tensor, targets_tensor = self._to_tensors(features, targets)
        ab_groups = _group_by_ab(at_bat_ids)
        if self.verbose:
            print(f"  LSTM dataset: {len(at_bat_ids):,} pitches in "
                  f"{len(ab_groups):,} ABs; device={self.device}")

        ds = _ABSequenceDataset(feats_tensor, targets_tensor, ab_groups)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=_collate_padded, num_workers=0,
        )

        n_features = feats_tensor.shape[1]
        self._net = _LSTMNet(
            n_features, self.hidden_dim, self.num_layers
        ).to(self.device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            t0 = time.monotonic()
            total_loss, total_correct, total_count = 0.0, 0, 0
            self._net.train()
            for X_batch, y_batch, lengths in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self._net(X_batch, lengths)
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
    ) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("LSTMBaseline must be fit before predict_proba")
        if len(features) != len(at_bat_ids):
            raise ValueError("features and at_bat_ids must have matching lengths")

        self._net.eval()
        n = len(features)
        out = np.zeros((n, N_PITCH_TYPES), dtype=np.float32)

        # Tensors in input row order (no targets needed; pass zeros)
        feats_tensor, _ = self._to_tensors(features, np.zeros(n, dtype=np.int64))
        ab_groups = _group_by_ab(at_bat_ids)

        # Iterate ABs in batches of fixed size so we can use the LSTM's batched fwd.
        for chunk_start in range(0, len(ab_groups), self.batch_size):
            chunk = ab_groups[chunk_start:chunk_start + self.batch_size]
            tensors = [feats_tensor[torch.from_numpy(g.astype(np.int64))]
                       for g in chunk]
            lengths = torch.tensor([len(g) for g in chunk], dtype=torch.long)
            X_padded = nn.utils.rnn.pad_sequence(tensors, batch_first=True)
            X_padded = X_padded.to(self.device)
            logits = self._net(X_padded, lengths).cpu()  # (B, T, C)
            probs = F.softmax(logits, dim=-1).numpy()
            # Place back at original row indices
            for ab_idx, group in enumerate(chunk):
                T = lengths[ab_idx].item()
                for pos in range(T):
                    out[group[pos]] = probs[ab_idx, pos]
        return out

    def predict(
        self,
        features: pd.DataFrame,
        at_bat_ids: np.ndarray,
    ) -> np.ndarray:
        return self.predict_proba(features, at_bat_ids).argmax(axis=1).astype(np.int64)
