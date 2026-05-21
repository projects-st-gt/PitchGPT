"""PitchGPT preprocessing — derive model-ready features per pitch.

Consumes ``data/raw/{year}/{date}.parquet`` daily files plus
``data/game_metadata/games.ndjson``. Writes augmented daily parquets to
``data/augmented/{year}/{date}.parquet`` containing every derived feature
PitchGPT's :class:`model.embeddings.FactorEmbeddings` and
:class:`model.embeddings.ContextTokens` expect.

The pipeline has two stages:

1. **Fit** (training-only). Walks every daily parquet whose ``game_date`` is
   on/before ``TRAIN_END`` (2023-12-31, per CLAUDE.md temporal split) and
   computes:

   - spin-rate quantile edges (8 bins via training-data deciles)
   - per-pitch-type league mean/std of ``release_speed`` (for the type-relative
     velo z-score → 10-bin deciled feature)
   - vocabularies for ballpark (``venue_id``), umpire (``hp_umpire_id``),
     catcher (``fielder_2``); each vocab reserves ``0=PAD`` and ``1=UNK``.

   Artifacts land in ``data/preprocess_artifacts/v{N}/``.

2. **Apply** (every day, all years). Loads the artifacts, joins game
   metadata, derives all model-shaped features, and writes the augmented
   parquet.

Designed to be idempotent: skips a day whose augmented file already exists
and was written under the current ``SCHEMA_VERSION``.

ADR 003 Amendment 1 (2026-05-10) drives the categorical-context shape: we
emit ``inning_bucket``, ``score_diff_bucket``, ``inning_half`` instead of a
derived leverage feature, and ``pitcher_fatigue_bucket`` as a per-pitch
factor.

Phase B note: ``velo_bin`` is computed via per-pitch-type LEAGUE-wide z-score
deciles (training-only). This is a methodological compromise vs the
``pitchgpt-model`` skill's prescription (per-pitcher trailing-window
deciles), accepted for Phase B to avoid a full profile-aware velo-stats
pipeline. Promote to per-pitcher trailing windows before causal claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.dataset import (
    PAD_ID,
    PITCH_TYPE_TO_ID,
    TRAIN_END,
    classify_result,
)
from data.preprocess import VELO_DECILE_CUTS, bin_velo_z_score, harmonize_and_tag

SCHEMA_VERSION: int = 2  # v2 (2026-05-14): 14-zone migration via assign_feature_zone_14

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_META_PATH = Path("data/game_metadata/games.ndjson")
DEFAULT_AUGMENTED_DIR = Path("data/augmented")
DEFAULT_ARTIFACT_DIR = Path("data/preprocess_artifacts")

# Vocab caps (must match PitchGPTConfig defaults in model/config.py).
N_BALLPARKS: int = 64
N_UMPIRES: int = 256
N_CATCHERS: int = 384

N_SPIN_RATE_BINS: int = 9  # 8 quantile bins + PAD=0
N_VELO_BINS: int = 11  # 10 deciles (1..10) + PAD=0

# Score-diff: clip to [-5, +5], shift to [0, 10]; vocab 11, no PAD slot.
SCORE_DIFF_CLIP: int = 5

# Inning: 1..12 verbatim, 13+ collapsed to 13, missing → 0. Vocab 14.
INNING_MAX: int = 12

# Pitcher in-game fatigue buckets: 0-9, 10-19, ..., 90-99, 100+, PAD=0 → vocab 12.
FATIGUE_BUCKET_WIDTH: int = 10
FATIGUE_MAX_BUCKET: int = 10  # 0-9, ..., 90-99 → 1..10; 100+ → 11; PAD=0


# --- Categorical encoding for handedness and roof ---

HANDEDNESS_MAP = {"R": 1, "L": 2}  # PAD/UNK = 0
ROOF_OPEN: int = 1
ROOF_CLOSED: int = 2
INNING_HALF_TOP: int = 1
INNING_HALF_BOT: int = 2

# Temperature buckets (Fahrenheit):
# 0 = missing, 1 = <40, 2 = 40-49, 3 = 50-59, 4 = 60-69, 5 = 70-79, 6 = 80+.
TEMP_BUCKET_EDGES = [40, 50, 60, 70, 80]


# ============================================================
# Game-metadata loading
# ============================================================


def load_game_metadata(path: Path = DEFAULT_META_PATH) -> pd.DataFrame:
    """Read the NDJSON game metadata into a DataFrame keyed by ``game_pk``.

    Returns columns: ``game_pk, venue_id, hp_umpire_id, temp_f, roof_closed``.
    Other columns from the NDJSON are dropped to keep joins cheap.
    """
    if not path.exists():
        raise FileNotFoundError(f"game metadata not found at {path}")
    rows = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "game_pk": int(obj["game_pk"]),
                "venue_id": obj.get("venue_id"),
                "hp_umpire_id": obj.get("hp_umpire_id"),
                "temp_f": obj.get("temp_f"),
                "roof_closed": obj.get("roof_closed"),
            })
    return pd.DataFrame(rows)


# ============================================================
# Vocab building
# ============================================================


def build_vocab(values: pd.Series, max_size: int) -> dict[int, int]:
    """Build a frequency-ranked vocab. PAD=0, UNK=1, real ids 2..max_size-1.

    Returns a dict mapping the raw value (int) to its assigned id.
    Anything not in the dict at apply-time maps to UNK=1, and NaN maps to PAD=0.
    """
    if max_size < 3:
        raise ValueError(f"max_size must be >= 3, got {max_size}")
    counts = values.dropna().astype(int).value_counts()
    keep = counts.iloc[: max_size - 2].index.tolist()
    return {int(v): i + 2 for i, v in enumerate(keep)}


def apply_vocab(values: pd.Series, vocab: dict[int, int]) -> pd.Series:
    """Map values via vocab; UNK=1 for unknown non-NaN; PAD=0 for NaN."""
    out = pd.Series(1, index=values.index, dtype="int32")
    out.loc[values.isna()] = 0
    notna = values.notna()
    if notna.any():
        mapped = values[notna].astype(int).map(vocab)
        # Where the value was present in vocab, take the mapped id; else UNK.
        out.loc[notna] = mapped.fillna(1).astype("int32").values
    return out


# ============================================================
# Per-pitch derivations (vectorized)
# ============================================================


def compute_count_state(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    """12-state encoding: ``balls (0..3) * 3 + strikes (0..2)``."""
    b = balls.fillna(0).astype(int).clip(0, 3)
    s = strikes.fillna(0).astype(int).clip(0, 2)
    return (b * 3 + s).astype("int8")


def compute_runners_state(
    on_1b: pd.Series, on_2b: pd.Series, on_3b: pd.Series
) -> pd.Series:
    """8-state runner encoding: ``4*r1 + 2*r2 + r3`` where each is 1 if occupied.

    Statcast emits the runner's player id when occupied and NaN when empty.
    """
    r1 = on_1b.notna().astype("int8")
    r2 = on_2b.notna().astype("int8")
    r3 = on_3b.notna().astype("int8")
    return (4 * r1 + 2 * r2 + r3).astype("int8")


def compute_outs_state(outs_when_up: pd.Series) -> pd.Series:
    """0/1/2 outs. Anything else clipped."""
    return outs_when_up.fillna(0).astype(int).clip(0, 2).astype("int8")


def compute_pos(pitch_number: pd.Series, max_pos: int = 14) -> pd.Series:
    """Pitch index within the AB, 0-indexed and clipped to [0, max_pos]."""
    pos = (pitch_number.fillna(1).astype(int) - 1).clip(0, max_pos)
    return pos.astype("int8")


def bucket_inning(inning: pd.Series) -> pd.Series:
    """0=missing; 1..12 verbatim; 13+ collapsed to 13. Vocab 14."""
    out = pd.Series(0, index=inning.index, dtype="int16")
    valid = inning.notna()
    if valid.any():
        clipped = inning[valid].astype(int).clip(1, INNING_MAX + 1)
        out.loc[valid] = clipped.astype("int16").values
    return out


def bucket_score_diff(score_diff: pd.Series) -> pd.Series:
    """Clip to [-5, +5] and shift to [0, 10]. Vocab 11 (no PAD).

    Score diff is always defined when the AB is happening (no genuine NaN),
    but if a row is malformed we default to 5 (i.e. tied).
    """
    sd = score_diff.fillna(0).astype(int).clip(-SCORE_DIFF_CLIP, SCORE_DIFF_CLIP)
    return (sd + SCORE_DIFF_CLIP).astype("int8")


def bucket_inning_half(half: pd.Series) -> pd.Series:
    """'Top' → 1, 'Bot' → 2, else → 0. Vocab 3."""
    mapped = half.map({"Top": INNING_HALF_TOP, "Bot": INNING_HALF_BOT})
    return mapped.fillna(0).astype("int8")


def bucket_temp(temp_f: pd.Series) -> pd.Series:
    """7 buckets. 0 = missing, 1 = <40, 2 = 40-49, ..., 6 = 80+. Vocab 7."""
    out = pd.Series(0, index=temp_f.index, dtype="int8")
    valid = temp_f.notna()
    if valid.any():
        vals = temp_f[valid].astype(float).to_numpy()
        # np.digitize: returns index i such that edges[i-1] <= val < edges[i].
        # With our edges [40,50,60,70,80], bucket id = digitize result + 1
        # so missing handled separately, <40 → 1, 40-49 → 2, ..., 80+ → 6.
        bins = np.digitize(vals, TEMP_BUCKET_EDGES) + 1
        out.loc[valid] = bins.astype("int8")
    return out


def bucket_roof(roof_closed: pd.Series) -> pd.Series:
    """Bool: True → CLOSED (2), False → OPEN (1), NaN → PAD (0)."""
    out = pd.Series(0, index=roof_closed.index, dtype="int8")
    out.loc[roof_closed == True] = ROOF_CLOSED  # noqa: E712 — pandas bool compare
    out.loc[roof_closed == False] = ROOF_OPEN  # noqa: E712
    return out


def bucket_days_rest(days: pd.Series) -> pd.Series:
    """0..6 → 1..7; 7+ → 8; NaN → 0. Vocab 9."""
    out = pd.Series(0, index=days.index, dtype="int8")
    valid = days.notna()
    if valid.any():
        vals = days[valid].astype(int).clip(lower=0)
        bucketed = vals.where(vals < 7, 7) + 1
        out.loc[valid] = bucketed.astype("int8")
    return out


def bucket_pitcher_fatigue(cumulative_pitch_count: pd.Series) -> pd.Series:
    """Per-pitch cumulative count → bucket. Vocab 12.

    0..9 → 1, 10..19 → 2, ..., 90..99 → 10, 100+ → 11. PAD=0 reserved.
    """
    bucket = (cumulative_pitch_count // FATIGUE_BUCKET_WIDTH) + 1
    return bucket.clip(1, FATIGUE_MAX_BUCKET + 1).astype("int8")


def fit_spin_rate_edges(spin_rate: pd.Series, n_bins: int = 8) -> list[float]:
    """Quantile edges that split valid spin_rates into ``n_bins`` equal-mass bins.

    Returns ``n_bins - 1`` interior edges (the outer edges are -inf/+inf in
    practice). PAD (0) is reserved for NaN spin_rate at apply time.
    """
    valid = spin_rate.dropna().astype(float)
    if len(valid) == 0:
        raise ValueError("no valid spin_rate values to fit deciles")
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [float(x) for x in valid.quantile(qs).tolist()]


def bin_spin_rate(spin_rate: pd.Series, edges: list[float]) -> pd.Series:
    """Apply spin_rate quantile edges. NaN → 0 (PAD). Real values → 1..n_bins."""
    out = pd.Series(0, index=spin_rate.index, dtype="int8")
    valid = spin_rate.notna()
    if valid.any():
        bin_edges = [-np.inf, *edges, np.inf]
        binned = pd.cut(
            spin_rate[valid].astype(float),
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )
        out.loc[valid] = (binned.astype("int8") + 1).values
    return out


def fit_league_velo_stats(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch-type mean and std of ``release_speed``.

    Returns DataFrame indexed by ``pitch_type_canonical`` with columns
    ``mean``, ``std``.
    """
    if "release_speed" not in pitches.columns:
        raise KeyError("fit_league_velo_stats needs 'release_speed'")
    stats = (
        pitches.dropna(subset=["release_speed", "pitch_type_canonical"])
        .groupby("pitch_type_canonical")["release_speed"]
        .agg(["mean", "std"])
    )
    # Replace zero std with a small floor to avoid division-by-zero downstream.
    stats["std"] = stats["std"].where(stats["std"] > 1e-3, 1.0)
    return stats


def compute_velo_bin(
    pitches: pd.DataFrame, league_velo_stats: pd.DataFrame
) -> pd.Series:
    """Type-relative LEAGUE deciles. NaN → 0 (PAD), real values → 1..10.

    Phase B compromise vs. the per-pitcher trailing window prescribed in the
    ``pitchgpt-model`` skill. See module docstring.
    """
    out = pd.Series(0, index=pitches.index, dtype="int8")
    valid = pitches["release_speed"].notna() & pitches["pitch_type_canonical"].notna()
    if not valid.any():
        return out
    joined = pitches.loc[valid].merge(
        league_velo_stats.rename(columns={"mean": "_mean", "std": "_std"}),
        left_on="pitch_type_canonical",
        right_index=True,
        how="left",
    )
    z = (joined["release_speed"] - joined["_mean"]) / joined["_std"]
    # bin_velo_z_score returns Int8 0..9 (with NaN preserved). Add 1 to make
    # 1..10 and reserve 0 for PAD/NaN.
    bins = bin_velo_z_score(z).astype("Int8") + 1
    out.loc[valid] = bins.fillna(0).astype("int8").values
    return out


def compute_spin_axis_sincos(spin_axis_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert spin axis in degrees → (sin, cos). NaN → (0, 0)."""
    rad = np.deg2rad(spin_axis_deg.fillna(0).astype(float))
    sin_v = np.sin(rad).astype("float32")
    cos_v = np.cos(rad).astype("float32")
    nan_mask = spin_axis_deg.isna().to_numpy()
    sin_v[nan_mask] = 0.0
    cos_v[nan_mask] = 0.0
    return (
        pd.Series(sin_v, index=spin_axis_deg.index),
        pd.Series(cos_v, index=spin_axis_deg.index),
    )


def compute_tto_bucket(pitches: pd.DataFrame) -> pd.Series:
    """Times-through-order: which PA of the game is this for the batter?

    1, 2, 3, 4+ → 1..4; PAD=0. Computed per (``game_pk``, ``batter``) by
    counting unique ``at_bat_number`` values in encounter order.
    """
    required = {"game_pk", "at_bat_number", "batter"}
    if not required.issubset(pitches.columns):
        raise KeyError(f"compute_tto_bucket needs {sorted(required)}")
    ab_starts = (
        pitches[["game_pk", "at_bat_number", "batter"]]
        .drop_duplicates(["game_pk", "at_bat_number"])
        .sort_values(["game_pk", "at_bat_number"])
        .reset_index(drop=True)
    )
    ab_starts["pa_idx_in_game"] = (
        ab_starts.groupby(["game_pk", "batter"]).cumcount() + 1
    )
    ab_starts["tto"] = ab_starts["pa_idx_in_game"].clip(1, 4).astype("int8")
    merged = pitches.merge(
        ab_starts[["game_pk", "at_bat_number", "tto"]],
        on=["game_pk", "at_bat_number"],
        how="left",
    )
    return merged["tto"].fillna(0).astype("int8").to_numpy()


def compute_pitcher_fatigue(pitches: pd.DataFrame) -> pd.Series:
    """Cumulative pitch count for (game_pk, pitcher) BEFORE the current pitch.

    Sorted by (at_bat_number, pitch_number) within (game_pk, pitcher) — the
    autoregressive within-game ordering. Returns the cumcount as an int.
    """
    required = {"game_pk", "pitcher", "at_bat_number", "pitch_number"}
    if not required.issubset(pitches.columns):
        raise KeyError(f"compute_pitcher_fatigue needs {sorted(required)}")
    sorted_idx = pitches.sort_values(
        ["game_pk", "pitcher", "at_bat_number", "pitch_number"]
    ).index
    counts = (
        pitches.loc[sorted_idx]
        .groupby(["game_pk", "pitcher"])
        .cumcount()
    )
    # Realign to the original order so downstream callers can append directly.
    return counts.reindex(pitches.index).fillna(0).astype(int)


# ============================================================
# Pitch-type and result IDs
# ============================================================


def compute_type_id(pitch_type_canonical: pd.Series) -> pd.Series:
    """Map canonical pitch type string → integer id (1..7), PAD=0."""
    # PITCH_TYPE_TO_ID has ids 0..6; we shift to 1..7 to reserve 0 for PAD.
    mapped = pitch_type_canonical.map(
        lambda t: PITCH_TYPE_TO_ID.get(t, -1) + 1 if t in PITCH_TYPE_TO_ID else 0
    )
    return mapped.fillna(0).astype("int8")


def compute_result_id(description: pd.Series, events: pd.Series) -> pd.Series:
    """Map (description, events) → 7-class result id (1..7), PAD=0."""
    out = []
    for d, e in zip(description, events):
        r = classify_result(d, e)
        out.append(0 if r is None else r + 1)  # shift to 1..7
    return pd.Series(out, index=description.index, dtype="int8")


# ============================================================
# Fit stage — walk training days, build artifacts
# ============================================================


def _iter_raw_days(raw_dir: Path) -> Iterable[Path]:
    """Yield daily parquet paths sorted by date."""
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for day in sorted(year_dir.glob("*.parquet")):
            yield day


def fit_artifacts(
    raw_dir: Path = DEFAULT_RAW_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    schema_version: int = SCHEMA_VERSION,
    train_end: pd.Timestamp = TRAIN_END,
) -> Path:
    """Walk training-period daily parquets and write fitted artifacts.

    Artifacts written to ``{artifact_dir}/v{schema_version}/``:

    - ``spin_rate_edges.json`` — list of 7 quantile edges
    - ``league_velo_stats.parquet`` — per-canonical-pitch-type mean/std
    - ``ballpark_vocab.parquet`` — venue_id → id
    - ``umpire_vocab.parquet`` — hp_umpire_id → id
    - ``catcher_vocab.parquet`` — fielder_2 → id
    - ``manifest.json`` — schema version, train_end, fit counts

    Returns the path to the artifact directory.
    """
    out = artifact_dir / f"v{schema_version}"
    out.mkdir(parents=True, exist_ok=True)

    game_meta = load_game_metadata().set_index("game_pk")

    spin_rates: list[pd.Series] = []
    velo_stats_partials: list[pd.DataFrame] = []
    venue_counts: dict[int, int] = {}
    umpire_counts: dict[int, int] = {}
    catcher_counts: dict[int, int] = {}
    n_training_days = 0
    n_training_pitches = 0

    days = [p for p in _iter_raw_days(raw_dir)]
    for day in tqdm(days, desc="fit"):
        date_str = day.stem  # e.g. 2023-04-15
        try:
            dt = pd.Timestamp(date_str)
        except (ValueError, TypeError):
            continue
        if dt > train_end:
            continue
        df = pd.read_parquet(day)
        # Harmonize so we can use ``pitch_type_canonical`` cleanly.
        df = harmonize_and_tag(df)
        n_training_days += 1
        n_training_pitches += len(df)

        spin_rates.append(df["release_spin_rate"].dropna())

        local_stats = (
            df.dropna(subset=["release_speed", "pitch_type_canonical"])
            .groupby("pitch_type_canonical")["release_speed"]
            .agg(["sum", "count"])
        )
        velo_stats_partials.append(local_stats)

        # Join metadata to get venue/umpire counts.
        meta_local = df[["game_pk", "fielder_2"]].drop_duplicates("game_pk")
        meta_local = meta_local.join(game_meta, on="game_pk", how="left")
        for v in meta_local["venue_id"].dropna():
            venue_counts[int(v)] = venue_counts.get(int(v), 0) + 1
        for u in meta_local["hp_umpire_id"].dropna():
            umpire_counts[int(u)] = umpire_counts.get(int(u), 0) + 1
        # Catcher counts: weight by pitches (not games) so the top-K are the
        # most-seen catchers.
        for c in df["fielder_2"].dropna().astype(int):
            catcher_counts[c] = catcher_counts.get(c, 0) + 1

    if n_training_pitches == 0:
        raise RuntimeError("no training-period pitches found; check raw_dir and train_end")

    spin_all = pd.concat(spin_rates, ignore_index=True)
    edges = fit_spin_rate_edges(spin_all, n_bins=8)
    (out / "spin_rate_edges.json").write_text(
        json.dumps({"edges": edges, "n_bins": 8})
    )

    # Combine partial velo sums to a single mean/std table.
    combined = pd.concat(velo_stats_partials)
    pooled = (
        combined.reset_index()
        .groupby("pitch_type_canonical")
        .agg(sum=("sum", "sum"), count=("count", "sum"))
    )
    pooled["mean"] = pooled["sum"] / pooled["count"]
    # We need std too; recompute it in a second pass since variance from
    # streaming sums needs E[X^2]. Simpler: collect a sample of release_speed.
    # Approach: re-walk and gather per-type std via Welford or sample-based.
    velo_sq_sums: dict[str, float] = {pt: 0.0 for pt in pooled.index}
    velo_n: dict[str, int] = {pt: 0 for pt in pooled.index}
    for day in tqdm(days, desc="fit velo-std"):
        date_str = day.stem
        try:
            dt = pd.Timestamp(date_str)
        except (ValueError, TypeError):
            continue
        if dt > train_end:
            continue
        df = pd.read_parquet(day)
        df = harmonize_and_tag(df)
        valid = df.dropna(subset=["release_speed", "pitch_type_canonical"])
        for pt, grp in valid.groupby("pitch_type_canonical"):
            velo_sq_sums[pt] = velo_sq_sums.get(pt, 0.0) + float((grp["release_speed"] ** 2).sum())
            velo_n[pt] = velo_n.get(pt, 0) + len(grp)

    stds = []
    means = []
    types = []
    for pt in pooled.index:
        n = velo_n[pt]
        mean = pooled.loc[pt, "mean"]
        sq_mean = velo_sq_sums[pt] / n
        var = max(sq_mean - mean * mean, 1e-6)
        stds.append(float(np.sqrt(var)))
        means.append(float(mean))
        types.append(pt)
    league_velo_df = pd.DataFrame(
        {"pitch_type_canonical": types, "mean": means, "std": stds}
    ).set_index("pitch_type_canonical")
    league_velo_df.to_parquet(out / "league_velo_stats.parquet")

    ballpark_vocab = build_vocab(
        pd.Series(list(venue_counts.keys())), max_size=N_BALLPARKS
    )
    # Order by training-frequency, like build_vocab expects.
    ballpark_top = sorted(venue_counts.items(), key=lambda kv: -kv[1])[: N_BALLPARKS - 2]
    ballpark_vocab = {int(k): i + 2 for i, (k, _) in enumerate(ballpark_top)}

    umpire_top = sorted(umpire_counts.items(), key=lambda kv: -kv[1])[: N_UMPIRES - 2]
    umpire_vocab = {int(k): i + 2 for i, (k, _) in enumerate(umpire_top)}

    catcher_top = sorted(catcher_counts.items(), key=lambda kv: -kv[1])[: N_CATCHERS - 2]
    catcher_vocab = {int(k): i + 2 for i, (k, _) in enumerate(catcher_top)}

    pd.DataFrame(
        {"raw_id": list(ballpark_vocab.keys()), "vocab_id": list(ballpark_vocab.values())}
    ).to_parquet(out / "ballpark_vocab.parquet", index=False)
    pd.DataFrame(
        {"raw_id": list(umpire_vocab.keys()), "vocab_id": list(umpire_vocab.values())}
    ).to_parquet(out / "umpire_vocab.parquet", index=False)
    pd.DataFrame(
        {"raw_id": list(catcher_vocab.keys()), "vocab_id": list(catcher_vocab.values())}
    ).to_parquet(out / "catcher_vocab.parquet", index=False)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "train_end": str(train_end.date()),
                "n_training_days": n_training_days,
                "n_training_pitches": int(n_training_pitches),
                "n_ballpark_vocab": len(ballpark_vocab),
                "n_umpire_vocab": len(umpire_vocab),
                "n_catcher_vocab": len(catcher_vocab),
                "spin_rate_n_bins": 8,
            },
            indent=2,
        )
    )
    return out


# ============================================================
# Apply stage — augment one day
# ============================================================


def _load_vocab_parquet(path: Path) -> dict[int, int]:
    df = pd.read_parquet(path)
    return {int(r): int(v) for r, v in zip(df["raw_id"], df["vocab_id"])}


def load_artifacts(artifact_dir: Path) -> dict:
    """Load fitted artifacts from ``{artifact_dir}/v{N}/``."""
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}; run `fit_artifacts` first"
        )
    manifest = json.loads(manifest_path.read_text())
    edges_obj = json.loads((artifact_dir / "spin_rate_edges.json").read_text())
    league_velo = pd.read_parquet(artifact_dir / "league_velo_stats.parquet")
    return {
        "manifest": manifest,
        "spin_rate_edges": edges_obj["edges"],
        "league_velo_stats": league_velo,
        "ballpark_vocab": _load_vocab_parquet(artifact_dir / "ballpark_vocab.parquet"),
        "umpire_vocab": _load_vocab_parquet(artifact_dir / "umpire_vocab.parquet"),
        "catcher_vocab": _load_vocab_parquet(artifact_dir / "catcher_vocab.parquet"),
    }


# Columns retained in the augmented parquet (raw + derived). Anything not in
# this set is dropped to keep file size sane.
RAW_PASSTHROUGH = [
    "game_pk", "at_bat_number", "pitch_number",
    "game_date", "pitcher", "batter",
    "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "p_throws", "stand",
    "release_speed", "release_spin_rate", "spin_axis",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "inning", "inning_topbot",
    "bat_score", "fld_score",
    "description", "events",
    "pitcher_days_since_prev_game",
    "fielder_2",  # catcher id for downstream
    "pitch_type", "pitch_type_canonical",
    "action_zone", "feature_zone",
]


def augment_day(
    daily_parquet_path: Path,
    artifacts: dict,
    game_meta_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Apply every derivation to one day's parquet. Returns the augmented df."""
    df = pd.read_parquet(daily_parquet_path)
    df = harmonize_and_tag(df)
    if df.empty:
        return df

    # Join game metadata (venue_id, hp_umpire_id, temp_f, roof_closed).
    df = df.merge(game_meta_lookup, on="game_pk", how="left")

    # Vocab ids for context categoricals.
    df["ballpark_id"] = apply_vocab(df["venue_id"], artifacts["ballpark_vocab"])
    df["umpire_id"] = apply_vocab(df["hp_umpire_id"], artifacts["umpire_vocab"])
    df["catcher_id"] = apply_vocab(df["fielder_2"], artifacts["catcher_vocab"])

    # Handedness ids (R/L → 1/2, PAD=0).
    df["p_throws_id"] = df["p_throws"].map(HANDEDNESS_MAP).fillna(0).astype("int8")
    df["stand_id"] = df["stand"].map(HANDEDNESS_MAP).fillna(0).astype("int8")

    # Per-pitch derivations.
    df["type_id"] = compute_type_id(df["pitch_type_canonical"])
    df["count_state"] = compute_count_state(df["balls"], df["strikes"])
    df["runners_state"] = compute_runners_state(df["on_1b"], df["on_2b"], df["on_3b"])
    df["outs_state"] = compute_outs_state(df["outs_when_up"])
    df["pos"] = compute_pos(df["pitch_number"])
    df["spin_rate_bin"] = bin_spin_rate(df["release_spin_rate"], artifacts["spin_rate_edges"])
    sin_v, cos_v = compute_spin_axis_sincos(df["spin_axis"])
    df["spin_axis_sin"] = sin_v.to_numpy()
    df["spin_axis_cos"] = cos_v.to_numpy()
    df["velo_bin"] = compute_velo_bin(df, artifacts["league_velo_stats"])
    df["result_id"] = compute_result_id(df["description"], df["events"])

    # Context categoricals derived per-AB (constant within an AB).
    df["inning_bucket"] = bucket_inning(df["inning"])
    df["score_diff_bucket"] = bucket_score_diff(df["bat_score"] - df["fld_score"])
    df["inning_half"] = bucket_inning_half(df["inning_topbot"])
    df["days_rest_bucket"] = bucket_days_rest(df["pitcher_days_since_prev_game"])
    df["temp_bucket"] = bucket_temp(df["temp_f"])
    df["roof_state"] = bucket_roof(df["roof_closed"])

    # Per-game derived (TTO and pitcher fatigue need cumulative counts).
    df["tto_bucket"] = compute_tto_bucket(df)
    cum_pitches = compute_pitcher_fatigue(df)
    df["pitcher_fatigue_bucket"] = bucket_pitcher_fatigue(cum_pitches)

    # Schema marker (helps idempotent re-runs).
    df["pitchgpt_schema_version"] = SCHEMA_VERSION

    # Drop raw confounder columns we no longer need at the model layer.
    keep = list(set(RAW_PASSTHROUGH + [
        "ballpark_id", "umpire_id", "catcher_id",
        "p_throws_id", "stand_id",
        "type_id", "count_state", "runners_state", "outs_state", "pos",
        "spin_rate_bin", "spin_axis_sin", "spin_axis_cos",
        "velo_bin", "result_id",
        "inning_bucket", "score_diff_bucket", "inning_half",
        "days_rest_bucket", "temp_bucket", "roof_state",
        "tto_bucket", "pitcher_fatigue_bucket",
        "pitchgpt_schema_version",
    ]) & set(df.columns))
    return df[keep]


def apply_to_corpus(
    raw_dir: Path = DEFAULT_RAW_DIR,
    augmented_dir: Path = DEFAULT_AUGMENTED_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    schema_version: int = SCHEMA_VERSION,
    overwrite: bool = False,
) -> None:
    """Apply the augmentation to every daily parquet and write outputs."""
    artifacts = load_artifacts(artifact_dir / f"v{schema_version}")
    game_meta = load_game_metadata()
    days = list(_iter_raw_days(raw_dir))
    for day in tqdm(days, desc="apply"):
        rel = day.relative_to(raw_dir)
        out_path = augmented_dir / rel
        if out_path.exists() and not overwrite:
            # Quick schema check.
            try:
                existing = pd.read_parquet(out_path, columns=["pitchgpt_schema_version"])
                if (
                    len(existing) > 0
                    and int(existing["pitchgpt_schema_version"].iloc[0]) == schema_version
                ):
                    continue
            except Exception:
                pass  # corrupted → re-write below
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            aug = augment_day(day, artifacts, game_meta)
        except Exception as exc:
            tqdm.write(f"warn: failed to augment {day}: {exc!r}")
            continue
        if aug.empty:
            continue
        tmp = out_path.with_suffix(".tmp.parquet")
        aug.to_parquet(tmp, index=False)
        tmp.replace(out_path)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    p = argparse.ArgumentParser(description="PitchGPT preprocessing pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="fit artifacts on training-period data")
    p_fit.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p_fit.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)

    p_apply = sub.add_parser("apply", help="apply augmentation to corpus")
    p_apply.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p_apply.add_argument("--augmented-dir", type=Path, default=DEFAULT_AUGMENTED_DIR)
    p_apply.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p_apply.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    if args.cmd == "fit":
        out = fit_artifacts(raw_dir=args.raw_dir, artifact_dir=args.artifact_dir)
        print(f"fit artifacts → {out}")
    elif args.cmd == "apply":
        apply_to_corpus(
            raw_dir=args.raw_dir,
            augmented_dir=args.augmented_dir,
            artifact_dir=args.artifact_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
