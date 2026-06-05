"""Per-PA backtest of the simulator's PITCH SOURCE.

The simulator's per-PA outcome distribution is produced by a pitch source feeding
the hitter cascade μ̂. Two pitch sources exist:

- **pitchGPT + cascade** — pitchGPT samples the pitch *sequence* autoregressively
  (sequence/context aware); the cascade decides each pitch's outcome. (MC rollout;
  run on GPU via ``modal_app.backtest_remote``.)
- **lookup + cascade** — a count-only empirical pitch mix feeds the cascade through
  the analytic count tree (:func:`hitter.compose.compose_pa`). No sequence.

This module grades both — plus a league-average constant baseline — against REAL
held-out 2024H2 at-bats with a proper score (per-PA log-loss, lower=better). All
three are scored on the SAME PA sample, SAME 7-class vocab
(:func:`hitter.eval.pa_outcome_class`), SAME scorer (:func:`hitter.eval.pa_logloss`),
SAME cascade + outcome map — so the only moving part between the two model paths is
the pitch source. That is the apples-to-apples comparison the project hinges on:
does sequence-aware pitch selection actually predict real outcomes better?

The pitchGPT MC rollout is the slow part and lives behind Modal; this module holds
the pure, locally-runnable pieces (sampling, baseline, the lookup path, scoring,
calibration-in-aggregate, bootstrap CIs). Real Statcast only; no placeholders.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from hitter.eval import pa_logloss, pa_outcome_class

PA_CLASSES = ["K", "BB", "1B", "2B", "3B", "HR", "out"]

# Real Statcast ``events`` values that terminate a batter's plate appearance.
# Restricting to this allowlist keeps base-running-only events (caught_stealing,
# pickoff, wild_pitch, …) — which also populate ``events`` — from being miscounted
# as PA outcomes. Everything here is then mapped by ``pa_outcome_class`` (which may
# still return None for the HBP/sac/interference exclusions).
PA_ENDING_EVENTS = {
    "single", "double", "triple", "home_run",
    "walk", "intent_walk",
    "strikeout", "strikeout_double_play",
    "field_out", "grounded_into_double_play", "double_play",
    "force_out", "fielders_choice", "fielders_choice_out",
    "field_error", "sac_fly", "sac_bunt", "sac_fly_double_play",
    "sac_bunt_double_play", "hit_by_pitch", "catcher_interf",
    "triple_play", "truncated_pa",
}

# Columns the backtest needs from the raw per-pitch parquet.
_PA_COLS = ["pitcher", "batter", "p_throws", "stand", "events",
            "game_date", "game_pk"]


def _year_files(raw_dir: str, start: str, end: str) -> list[str]:
    """Per-day parquet paths whose YEAR overlaps the [start, end] window."""
    y0, y1 = int(start[:4]), int(end[:4])
    files: list[str] = []
    for y in range(y0, y1 + 1):
        files += sorted(glob.glob(str(Path(raw_dir) / str(y) / "*.parquet")))
    return files


def load_terminal_pas(raw_dir: str = "data/raw", *, start: str, end: str,
                      verbose: bool = True) -> pd.DataFrame:
    """All scorable terminal PAs in [start, end] (inclusive game_date bounds).

    Returns one row per PA with columns: ``pitcher, batter, p_throws, stand,
    game_date, game_pk, outcome`` (outcome in :data:`PA_CLASSES`). PAs whose event
    is in the cascade's exclusion set (HBP/sac/interference) are dropped — they are
    not in the model's outcome vocabulary, so scoring them would be ill-defined.
    """
    frames = []
    dropped_nonpa = 0
    for f in _year_files(raw_dir, start, end):
        df = pd.read_parquet(f, columns=_PA_COLS)
        df = df[df["events"].notna()]
        if df.empty:
            continue
        in_allow = df["events"].isin(PA_ENDING_EVENTS)
        dropped_nonpa += int((~in_allow).sum())
        df = df[in_allow]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no raw parquets under {raw_dir} for {start}..{end}")
    pas = pd.concat(frames, ignore_index=True)
    pas["game_date"] = pd.to_datetime(pas["game_date"])
    pas = pas[(pas["game_date"] >= pd.Timestamp(start))
              & (pas["game_date"] <= pd.Timestamp(end))]
    pas["outcome"] = pas["events"].map(pa_outcome_class)
    n_excluded = int(pas["outcome"].isna().sum())
    pas = pas[pas["outcome"].notna()].reset_index(drop=True)
    if verbose:
        print(f"[backtest] {start}..{end}: {len(pas)} scorable PAs "
              f"(excluded {n_excluded} HBP/sac/CI; dropped {dropped_nonpa} non-PA events)")
        print("[backtest] outcome marginal:",
              {k: round(v, 4) for k, v in
               pas["outcome"].value_counts(normalize=True).reindex(PA_CLASSES).items()})
    return pas[["pitcher", "batter", "p_throws", "stand",
                "game_date", "game_pk", "outcome"]]


def league_baseline_dist(train_pas: pd.DataFrame) -> dict[str, float]:
    """League-average marginal 7-class distribution from TRAIN-split PAs.

    A leakage-safe constant predictor — the same dist for every PA. This is the
    'guess the base rate' baseline the model must beat.
    """
    counts = train_pas["outcome"].value_counts().reindex(PA_CLASSES).fillna(0.0)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("empty train_pas — cannot form league baseline")
    return {k: float(counts[k] / total) for k in PA_CLASSES}


def sample_pas(pas: pd.DataFrame, n: int, *, seed: int,
               max_per_matchup: int = 3) -> pd.DataFrame:
    """Uniform random sample of ``n`` PAs (unbiased for pooled log-loss).

    ``max_per_matchup`` caps how many PAs any single (pitcher, batter) pair may
    contribute, so a handful of frequent matchups can't dominate the estimate.
    """
    rng = np.random.default_rng(seed)
    shuffled = pas.iloc[rng.permutation(len(pas))]
    if max_per_matchup:
        shuffled = (shuffled.groupby(["pitcher", "batter"], sort=False)
                    .head(max_per_matchup))
    if len(shuffled) < n:
        return shuffled.reset_index(drop=True)
    return shuffled.iloc[:n].reset_index(drop=True)


def lookup_dist_for_matchup(ctx: dict, pitcher_pitches: pd.DataFrame,
                            batter_id: int, *, outcome_mode: str = "xwoba"
                            ) -> dict[str, float] | None:
    """One (pitcher, batter) per-PA outcome dist via the LOOKUP pitch source.

    ``ctx`` is :func:`hitter.rollout.load_hitter_ctx`'s bundle (hitter model
    ``hm``, outcome map ``xfn``, profile caches ``bc``/``pc``). ``pitcher_pitches``
    are that pitcher's real raw pitch rows (the count-conditioned empirical mix).
    Returns the normalized 7-class terminal distribution, or ``None`` if the
    count-tree solve is degenerate (a pitcher whose features get fully dropped
    yields a zero-weight count → ``w/w.sum()`` blows up to NaN/inf). A single NaN
    dist would poison the pooled log-loss, so we signal it instead of fabricating
    a number — the caller falls back to the league baseline for that PA.
    """
    from hitter.compose import compose_pa
    from hitter.eval import build_empirical_pitch_provider

    provider = build_empirical_pitch_provider(
        pitcher_pitches, batter_id, ctx["bc"], ctx["pc"])
    # A degenerate matchup makes w/w.sum() blow up inside compose; the resulting
    # NaN/inf is caught by the finite-check below, so silence the matmul warnings.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        metrics = compose_pa(ctx["hm"], provider, ctx["xfn"],
                             outcome_mode=outcome_mode)
    term = metrics["terminal"]
    vals = np.array([term[k] for k in PA_CLASSES], dtype=float)
    s = float(vals.sum())
    if not np.all(np.isfinite(vals)) or s <= 0:
        return None
    return {k: float(vals[i] / s) for i, k in enumerate(PA_CLASSES)}


def aggregate_calibration(pred_dists: list[dict], actuals: list[str]
                          ) -> pd.DataFrame:
    """Calibration-in-aggregate: mean predicted P(class) vs real frequency.

    The cheap fairness check (A): if a path's mean predicted P(HR) is far above the
    realized HR rate, its *level* is inflated even if its ordering is fine.
    """
    n = len(actuals)
    mean_pred = {k: float(np.mean([d.get(k, 0.0) for d in pred_dists]))
                 for k in PA_CLASSES}
    real_freq = {k: float(sum(a == k for a in actuals) / n) for k in PA_CLASSES}
    return pd.DataFrame({
        "class": PA_CLASSES,
        "mean_pred": [mean_pred[k] for k in PA_CLASSES],
        "real_freq": [real_freq[k] for k in PA_CLASSES],
        "pred_minus_real": [mean_pred[k] - real_freq[k] for k in PA_CLASSES],
    })


def bootstrap_logloss_ci(pred_dists: list[dict], actuals: list[str], *,
                         n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> dict[str, float]:
    """Point estimate + bootstrap CI for pooled per-PA log-loss."""
    rng = np.random.default_rng(seed)
    eps = 1e-9
    nll = np.array([-np.log(max(d.get(a, 0.0), eps))
                    for d, a in zip(pred_dists, actuals)])
    n = len(nll)
    boots = np.array([nll[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"logloss": float(nll.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def score_path(pred_dists: list[dict], actuals: list[str]) -> float:
    """Thin alias over :func:`hitter.eval.pa_logloss` for symmetry."""
    return pa_logloss(pred_dists, actuals)
