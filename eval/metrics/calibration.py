"""Calibration metrics for the eval table per the ``eval-protocol`` skill.

Every prediction model in the eval table reports all of:
top-1 / top-3 accuracy, ECE, reliability diagram, Brier, log-loss.

Conventions (from the skill):

- **ECE is the equal-mass version**: bin edges are quantiles of the
  predicted-probability distribution, not equal-width bins. Equal-mass
  bins reflect bin-importance weighting more honestly when most
  predictions cluster in a narrow confidence range.
- **Reliability diagrams use 15 quantile bins by default.**
- **``ignore_index`` defaults to ``-100``** (matching ``data.dataset.PAD_ID``)
  so padded positions in batches are skipped automatically.

Inputs are numpy arrays:

- ``probs``: shape ``(N, C)``, predicted probability per class.
- ``targets``: shape ``(N,)``, integer class IDs. Use ``-100`` for
  ignore-positions (padding etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_N_BINS: int = 15


def _filter_valid(
    probs: np.ndarray,
    targets: np.ndarray,
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = targets != ignore_index
    return probs[valid], targets[valid]


def top_k_accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    k: int = 1,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> float:
    """Mean fraction of samples whose target is in the top-k predicted classes."""
    p, t = _filter_valid(probs, targets, ignore_index)
    if len(p) == 0:
        return float("nan")
    if k == 1:
        return float((p.argmax(axis=1) == t).mean())
    if k >= p.shape[1]:
        return 1.0  # trivially perfect when k covers all classes
    # argpartition is O(N) instead of O(N log N) sort
    top_k = np.argpartition(-p, kth=k - 1, axis=1)[:, :k]
    return float((top_k == t[:, None]).any(axis=1).mean())


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> float:
    """Equal-mass ECE on max-probability calibration.

    Equal-mass bins place the same number of predictions in each bin
    (quantile cuts of the confidence distribution). The classical equal-
    width version under-weights high-confidence bins when most predictions
    cluster low; the equal-mass version is the convention this project
    uses (see ``eval-protocol`` skill).
    """
    p, t = _filter_valid(probs, targets, ignore_index)
    n = len(p)
    if n == 0:
        return float("nan")

    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == t).astype(float)

    quantile_edges = np.quantile(
        confidences, np.linspace(0.0, 1.0, n_bins + 1)
    )
    # Anchor outer edges and a tiny epsilon at the top so the last bin
    # includes confidence == 1.0.
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0 + 1e-9

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= quantile_edges[i]) & (
            confidences < quantile_edges[i + 1]
        )
        if not in_bin.any():
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> pd.DataFrame:
    """Per-bin (avg confidence, avg accuracy, count) for plotting.

    Returns a DataFrame with columns ``bin_index``, ``bin_lower``,
    ``bin_upper``, ``avg_confidence``, ``avg_accuracy``, ``count``. Empty
    bins are represented with ``count = 0`` and NaN aggregates.
    """
    p, t = _filter_valid(probs, targets, ignore_index)
    if len(p) == 0:
        # Return an empty-but-typed DataFrame so callers can still iterate.
        return pd.DataFrame(
            columns=[
                "bin_index", "bin_lower", "bin_upper",
                "avg_confidence", "avg_accuracy", "count",
            ]
        )

    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == t).astype(float)

    quantile_edges = np.quantile(
        confidences, np.linspace(0.0, 1.0, n_bins + 1)
    )
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0 + 1e-9

    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = quantile_edges[i], quantile_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences < hi)
        if in_bin.any():
            rows.append({
                "bin_index": i,
                "bin_lower": float(lo),
                "bin_upper": float(min(hi, 1.0)),
                "avg_confidence": float(confidences[in_bin].mean()),
                "avg_accuracy": float(correct[in_bin].mean()),
                "count": int(in_bin.sum()),
            })
        else:
            rows.append({
                "bin_index": i,
                "bin_lower": float(lo),
                "bin_upper": float(min(hi, 1.0)),
                "avg_confidence": float("nan"),
                "avg_accuracy": float("nan"),
                "count": 0,
            })
    return pd.DataFrame(rows)


def brier_score(
    probs: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> float:
    """Multi-class Brier score: mean of ``sum_c (p_c - 1{y == c})^2``.

    Lower is better. Bounded between 0 and 2 for one-hot targets.
    """
    p, t = _filter_valid(probs, targets, ignore_index)
    if len(p) == 0:
        return float("nan")
    n_classes = p.shape[1]
    one_hot = np.eye(n_classes, dtype=float)[t]
    return float(((p - one_hot) ** 2).sum(axis=1).mean())


def log_loss(
    probs: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> float:
    """Multi-class log-loss (cross-entropy)."""
    p, t = _filter_valid(probs, targets, ignore_index)
    if len(p) == 0:
        return float("nan")
    p_clipped = np.clip(p, eps, 1.0 - eps)
    target_probs = p_clipped[np.arange(len(t)), t]
    return float(-np.log(target_probs).mean())
