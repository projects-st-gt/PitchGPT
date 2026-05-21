"""Fit and evaluate the marginal + count-conditional baselines on real data.

Trains on the temporal-train split (≤ 2023-12-31), evaluates on the val
split (2024-01-01 to 2024-07-15). Reports top-1 / top-3 accuracy, ECE,
and Brier with at-bat-level bootstrap CIs per the ``eval-protocol`` skill.

This is the floor every later model has to beat (or be within calibration
range of, per the skill's framing).
"""

from __future__ import annotations

# Cap OpenMP threads BEFORE any C-extension imports — on macOS, mixing
# multiple OpenMP runtimes (numpy's libomp + xgboost's libgomp) at high
# parallelism produces ``pthread_mutex_init`` segfaults. 4 threads is a
# safe ceiling; XGBoost training is still fast enough.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import (
    TRAIN_END,
    VAL_END,
    VAL_START,
    PITCH_TYPE_TO_ID,
)
from data.harmonization import harmonize_dataframe
from eval.baselines.count_conditional import CountConditionalBaseline
from eval.baselines.marginal import MarginalBaseline
from eval.baselines.pitcher_ngram import PitcherNgramBaseline
from eval.baselines.xgboost_baseline import XGBoostBaseline
from data.xgboost_features import (
    build_xgboost_features,
    compute_pitcher_arsenal_encoding,
    load_game_metadata,
)
from eval.metrics.bootstrap import bootstrap_metric
from eval.metrics.calibration import (
    expected_calibration_error,
    top_k_accuracy,
    brier_score,
)

RAW_DIR = Path("data/raw")
NEEDED_COLUMNS = [
    "game_pk", "at_bat_number", "pitch_number",
    "game_date",
    "pitcher", "batter",
    "pitch_type",
    "balls", "strikes", "outs_when_up", "inning",
    "on_1b", "on_2b", "on_3b",
    "p_throws", "stand",
]


def _load_split(year_dirs, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for yd in year_dirs:
        for p in sorted(yd.glob("*.parquet")):
            df = pd.read_parquet(p, columns=NEEDED_COLUMNS)
            df["game_date"] = pd.to_datetime(df["game_date"])
            df = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
            if len(df):
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=NEEDED_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _evaluate(
    name: str,
    probs: np.ndarray,
    targets: np.ndarray,
    at_bat_ids: np.ndarray,
) -> dict:
    """Compute the eval-table cells for one baseline."""
    print(f"\n=== {name} ===")
    point, lo, hi = bootstrap_metric(
        lambda p, t: top_k_accuracy(p, t, k=1),
        at_bat_ids, probs, targets,
        n_bootstraps=300, seed=0,
    )
    print(f"  top-1 accuracy : {point:.4f}  [{lo:.4f}, {hi:.4f}]")

    point3, lo3, hi3 = bootstrap_metric(
        lambda p, t: top_k_accuracy(p, t, k=3),
        at_bat_ids, probs, targets,
        n_bootstraps=300, seed=0,
    )
    print(f"  top-3 accuracy : {point3:.4f}  [{lo3:.4f}, {hi3:.4f}]")

    ece = expected_calibration_error(probs, targets, n_bins=15)
    print(f"  ECE (15 bins)  : {ece:.4f}")

    brier = brier_score(probs, targets)
    print(f"  Brier score    : {brier:.4f}")

    return {
        "name": name,
        "top1": point, "top1_lo": lo, "top1_hi": hi,
        "top3": point3, "top3_lo": lo3, "top3_hi": hi3,
        "ece": ece,
        "brier": brier,
    }


def main() -> None:
    year_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    print("Loading training split (≤ 2023-12-31)...")
    t0 = time.monotonic()
    train = _load_split(year_dirs, pd.Timestamp("2017-01-01"), TRAIN_END)
    print(f"  loaded {len(train):,} pitches in {time.monotonic() - t0:.1f}s")
    train = harmonize_dataframe(train, drop_unmapped=True)
    print(f"  after harmonization: {len(train):,} pitches")

    print("\nLoading val split (2024-01-01 to 2024-07-15)...")
    t0 = time.monotonic()
    val = _load_split(year_dirs, VAL_START, VAL_END)
    print(f"  loaded {len(val):,} pitches in {time.monotonic() - t0:.1f}s")
    val = harmonize_dataframe(val, drop_unmapped=True)
    print(f"  after harmonization: {len(val):,} pitches")

    # Build per-pitch arrays for the eval split
    val_targets = val["pitch_type_canonical"].map(PITCH_TYPE_TO_ID).to_numpy(dtype=np.int64)
    val_ab_ids = (val["game_pk"].astype(np.int64) * 100 + val["at_bat_number"].astype(np.int64)).to_numpy()

    # ------- Fit + eval baselines -------
    print("\nFitting marginal baseline...")
    m = MarginalBaseline().fit(train)
    print(f"  most common pitch type: {m.most_common_type} "
          f"(P = {float(m.probs_.max()):.4f})")

    m_probs = m.predict_proba(val)
    m_results = _evaluate("Marginal", m_probs, val_targets, val_ab_ids)

    print("\nFitting count-conditional baseline...")
    cc = CountConditionalBaseline().fit(train)
    print(f"  trained on {len(cc.probs_)} count states")

    cc_probs = cc.predict_proba(val)
    cc_results = _evaluate("Count-conditional", cc_probs, val_targets, val_ab_ids)

    # Per-pitcher n-gram baselines for n in {0, 1}
    ngram_results = []
    for n in [0, 1]:
        print(f"\nFitting per-pitcher n-gram (n={n}, alpha=10.0)...")
        t0 = time.monotonic()
        nb = PitcherNgramBaseline(n=n, alpha=10.0).fit(train)
        print(f"  fit in {time.monotonic() - t0:.1f}s; "
              f"{len(nb.pitcher_counts_):,} (pitcher, ctx) cells, "
              f"{len(nb.league_counts_):,} ctx cells")

        t0 = time.monotonic()
        probs = nb.predict_proba(val)
        print(f"  predict in {time.monotonic() - t0:.1f}s")
        ngram_results.append(_evaluate(f"Pitcher n-gram (n={n})",
                                       probs, val_targets, val_ab_ids))

    # ----- XGBoost baseline (the strong bar) -----
    print("\nLoading game metadata for XGBoost...")
    t0 = time.monotonic()
    metadata = load_game_metadata()
    print(f"  loaded {len(metadata):,} games in {time.monotonic() - t0:.1f}s")

    print("\nComputing per-pitcher arsenal target encoding (training data only)...")
    t0 = time.monotonic()
    arsenal_enc = compute_pitcher_arsenal_encoding(train, alpha=10.0)
    print(f"  computed in {time.monotonic() - t0:.1f}s for "
          f"{len(arsenal_enc):,} pitchers")

    print("\nBuilding XGBoost features...")
    t0 = time.monotonic()
    train_feats = build_xgboost_features(train, metadata, arsenal_encoding=arsenal_enc)
    val_feats = build_xgboost_features(val, metadata, arsenal_encoding=arsenal_enc)
    print(f"  features built in {time.monotonic() - t0:.1f}s; "
          f"{len(train_feats.columns)} columns: {list(train_feats.columns)}")

    train_targets = train["pitch_type_canonical"].map(PITCH_TYPE_TO_ID).to_numpy(dtype=np.int64)

    # Single-threaded: macOS' multiple-OpenMP-runtimes issue (numpy's libomp
    # + xgboost's libgomp at high parallelism) segfaults with n_jobs > 1.
    # Single-threaded ~2 min for 4.7M rows; acceptable for a baseline.
    print("\nTraining XGBoost (200 trees, depth 6, single-threaded)...")
    t0 = time.monotonic()
    xgb_model = XGBoostBaseline(
        n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=1
    ).fit(train_feats, train_targets)
    print(f"  trained in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    xgb_probs = xgb_model.predict_proba(val_feats)
    print(f"  val prediction in {time.monotonic() - t0:.1f}s")

    xgb_results = _evaluate("XGBoost", xgb_probs, val_targets, val_ab_ids)

    print("\nXGBoost top-10 features by gain:")
    print(xgb_model.feature_importance().head(10).to_string())

    # ------- Summary table -------
    print("\n=== Summary (val split) ===")
    print(f"{'Baseline':<26} {'top-1':>7} {'95% CI':>22} {'top-3':>7} {'ECE':>7} {'Brier':>7}")
    for r in [m_results, cc_results] + ngram_results + [xgb_results]:
        ci = f"[{r['top1_lo']:.3f}, {r['top1_hi']:.3f}]"
        print(f"{r['name']:<26} {r['top1']:>7.4f} {ci:>22} "
              f"{r['top3']:>7.4f} {r['ece']:>7.4f} {r['brier']:>7.4f}")


if __name__ == "__main__":
    main()
