"""Train the hitter/swing cascade (XGBoost) — see hitter/MODEL_DESIGN.md.

The cascade (per pitch, given the thrown pitch + state):

    S1  swing?            (all pitches)            -> XGBoost binary
    S1b called-strike?    (takes only)             -> XGBoost binary  (ball vs CS)
    S2  whiff?            (swings only)            -> XGBoost binary
    S2b fair?             (count-constant in v0)   -> foul_rate_by_count
    S3  contact quality   (balls in play)          -> XGBoost regression on
                                                      xwOBA-on-contact

Each ML node trains on its own conditional population (no dilution by rows it
never sees), with per-node isotonic calibration and monotonic constraints where
baseball priors are unambiguous. The S3 target is
``estimated_woba_using_speedangle`` (xwOBA-on-contact), joined from
``data/raw/`` because the augmented frames drop it (statcast-pipeline skill).

Public surface:
- ``add_cascade_labels`` / ``node_population`` — leakage-free label + population
  assembly (pure, tested on synthetic data).
- ``foul_rate_by_count`` — the v0 S2b constant.
- ``attach_xwoba_target`` / ``attach_pitcher_profile`` — the data joins.
- ``train_node`` / ``main`` — fit + calibrate + persist to checkpoints/hitter/.
"""
from __future__ import annotations

import os
# XGBoost 3.x QuantileDMatrix construction has a nondeterministic libomp race on
# macOS that SEGFAULTS under multithreading (flaky — same op crashes ~intermittently
# at any data size). Force single-threaded OpenMP before xgboost is imported; the
# trainer also passes n_jobs=1. One-time training cost; correctness > speed.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from hitter import labels as L
from hitter.features import (
    BASE_FEATURE_COLS,
    attach_batter_profile,
    build_base_features,
)

# ---------------------------------------------------------------------------
# Pitcher "stuff" feature subset (MODEL_DESIGN.md §5)
# ---------------------------------------------------------------------------
# The hitter conditions on the ACTUAL thrown pitch (plate_x/z, velo, spin from
# the per-pitch row), so the pitcher's *location heatmap* (175 dims) and
# *per-count arsenal* (84 dims, = pitch-selection, PitchGPT's job) are redundant
# noise. We keep only stuff-QUALITY summary: arsenal mix, velo/spin/arm-slot/pfx
# movement per type, and recent form. This is what makes 98 mph != 91 mph.
_PT = ("FF", "SI", "FC", "SL", "CU", "CH", "FS")
PITCHER_STUFF_FEATURES: list[str] = (
    [f"arsenal_{p}" for p in _PT]
    + [f"has_pitch_{p}" for p in _PT]
    + [f"mean_velo_{p}" for p in _PT]
    + [f"mean_spin_{p}" for p in _PT]
    + [f"arm_slot_{p}" for p in _PT]
    + [f"mean_pfx_x_{p}" for p in _PT]
    + [f"mean_pfx_z_{p}" for p in _PT]
    + ["recent_30d_xwoba", "recent_30d_n_pitches",
       "days_since_last_appearance", "profile_confidence"]
)  # 7*7 + 4 = 53 dims


def attach_pitcher_profile(
    df: pd.DataFrame,
    pitcher_cache,
    *,
    prefix: str = "p",
) -> pd.DataFrame:
    """Join the compact pitcher *stuff* vector as ``{prefix}0..`` columns.

    Mirrors ``hitter.features.attach_batter_profile``: one as-of lookup per
    (pitcher, game_pk) (leakage-safe, ``as_of_fallback=True``), then the full
    profile vector is sliced down to ``PITCHER_STUFF_FEATURES`` before merge.
    """
    from data.profile_cache import PITCHER_FEATURE_INDEX

    idx = [PITCHER_FEATURE_INDEX[f] for f in PITCHER_STUFF_FEATURES]
    keys = df[["pitcher", "game_pk", "game_date"]].copy()
    keys["game_num"] = df["game_num"] if "game_num" in df.columns else 1
    uniq = keys.drop_duplicates(["pitcher", "game_pk"]).reset_index(drop=True)

    vecs = []
    for row in uniq.itertuples(index=False):
        res = pitcher_cache.lookup(
            int(row.pitcher), pd.Timestamp(row.game_date), int(row.game_num),
            as_of_fallback=True,
        )
        vecs.append(np.asarray(res["vector"], dtype=np.float32)[idx])
    mat = np.vstack(vecs)
    prof = pd.DataFrame(mat, columns=[f"{prefix}{i}" for i in range(mat.shape[1])])
    prof.insert(0, "game_pk", uniq["game_pk"].values)
    prof.insert(0, "pitcher", uniq["pitcher"].values)
    return df.merge(prof, on=["pitcher", "game_pk"], how="left")

# ---------------------------------------------------------------------------
# Cascade labels + conditional populations (pure, fast, synthetic-tested)
# ---------------------------------------------------------------------------

#: Each ML node and the conditional population it trains on.
ML_NODES = ("swing", "called_strike", "whiff", "contact_quality")


def add_cascade_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-node cascade labels, each NaN where the node doesn't apply.

    - ``swing``         : 1/0 on every pitch.
    - ``called_strike`` : among TAKES, 1 if called strike else 0 (ball). NaN on swings.
    - ``whiff``         : among SWINGS, 1 if whiff else 0 (contact). NaN on takes.
    - ``fair``          : among CONTACT (swing & not whiff), 1 if in play else 0
                          (foul). NaN otherwise.

    NaN-where-inapplicable is deliberate: ``node_population`` filters on it so a
    node never trains on a row outside its population.
    """
    out = df.copy()
    swing = L.is_swing(out["description"]).astype("float32")
    out["swing"] = swing

    is_swing = swing == 1
    # whiff: defined on swings only
    whiff = L.is_whiff_given_swing(out["description"]).astype("float32")
    out["whiff"] = np.where(is_swing, whiff, np.nan).astype("float32")

    # fair: defined on contact (swing & not whiff) only
    is_contact = is_swing & (whiff == 0)
    fair = L.is_fair_given_swing(out["description"]).astype("float32")
    out["fair"] = np.where(is_contact, fair, np.nan).astype("float32")

    # called_strike: defined on takes only
    called = (out["description"] == "called_strike").astype("float32")
    out["called_strike"] = np.where(~is_swing, called, np.nan).astype("float32")
    return out


def node_population(df: pd.DataFrame, node: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(rows, target)`` for ``node``'s conditional population.

    - ``swing``           : all rows; target = swing.
    - ``called_strike``   : takes (swing==0); target = called_strike.
    - ``whiff``           : swings (swing==1); target = whiff.
    - ``contact_quality`` : balls in play (fair==1); target = xwOBA-on-contact.
    """
    if node == "swing":
        mask = df["swing"].notna()
        return df[mask], df.loc[mask, "swing"]
    if node == "called_strike":
        mask = df["called_strike"].notna()
        return df[mask], df.loc[mask, "called_strike"]
    if node == "whiff":
        mask = df["whiff"].notna()
        return df[mask], df.loc[mask, "whiff"]
    if node == "contact_quality":
        mask = (df["fair"] == 1) & df["estimated_woba_using_speedangle"].notna()
        return df[mask], df.loc[mask, "estimated_woba_using_speedangle"]
    raise ValueError(f"unknown node {node!r}")


def foul_rate_by_count(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """v0 S2b: empirical foul rate among contact, by (balls, strikes).

    Among contact (``fair`` defined), foul = ``fair == 0``, in-play = ``fair == 1``.
    Returns ``{(balls, strikes): P(foul | contact)}``.
    """
    contact = df[df["fair"].notna()]
    rates: dict[tuple[int, int], float] = {}
    for (b, s), grp in contact.groupby(["balls", "strikes"]):
        n = len(grp)
        if n:
            rates[(int(b), int(s))] = float((grp["fair"] == 0).mean())
    return rates


# ---------------------------------------------------------------------------
# Feature spec per node (MODEL_DESIGN.md §4-5): common stuff + node-specific
# ---------------------------------------------------------------------------
_BATTER_COLS = [f"b{i}" for i in range(57)]          # BATTER_VECTOR_LEN
_PITCHER_COLS = [f"p{i}" for i in range(len(PITCHER_STUFF_FEATURES))]
_MOVEMENT_COLS = ["release_spin_rate", "spin_axis_sin", "spin_axis_cos"]
_COMMON_FEATURES = BASE_FEATURE_COLS + _MOVEMENT_COLS + ["outs_when_up"] \
    + _BATTER_COLS + _PITCHER_COLS

#: Per-node feature list. S1b (called_strike) is mostly location/umpire/framing —
#: minimal batter signal; S3 (contact_quality) adds park/roof/temp (Coors carry).
NODE_FEATURES: dict[str, list[str]] = {
    "swing": _COMMON_FEATURES,
    "whiff": _COMMON_FEATURES,
    "called_strike": (BASE_FEATURE_COLS + ["umpire_id", "catcher_id"]
                      + _BATTER_COLS),
    "contact_quality": (_COMMON_FEATURES
                        + ["ballpark_id", "roof_state", "temp_bucket"]),
}
#: High-cardinality id columns handled via XGBoost native categorical support.
NODE_CATEGORICAL: dict[str, list[str]] = {
    "swing": [],
    "whiff": [],
    "called_strike": ["umpire_id", "catcher_id"],
    "contact_quality": ["ballpark_id", "roof_state", "temp_bucket"],
}
#: Monotone priors kept minimal + clean (forcing wrong ones hurts calibration).
#: whiff never decreases with velo (chase-and-miss) — the one unambiguous prior.
NODE_MONOTONE: dict[str, dict[str, int]] = {
    "swing": {},
    "whiff": {"release_speed": 1},
    "called_strike": {},
    "contact_quality": {},
}
NODE_OBJECTIVE = {
    "swing": "binary", "whiff": "binary", "called_strike": "binary",
    "contact_quality": "regression",
}


def build_inference_features(
    pitches: pd.DataFrame,
    batter_cache,
    pitcher_cache,
) -> pd.DataFrame:
    """Assemble per-pitch FEATURES for prediction (no labels, no xwOBA target).

    base/lag features + batter & pitcher profile joins — the same feature columns
    the boosters were trained on (categorical ctx like umpire/park ride along from
    the augmented rows). Used by the composition/matchup path.
    """
    df = build_base_features(pitches)
    df = attach_batter_profile(df, batter_cache, prefix="b")
    df = attach_pitcher_profile(df, pitcher_cache, prefix="p")
    return df


def build_training_frame(
    pitches: pd.DataFrame,
    raw: pd.DataFrame,
    batter_cache,
    pitcher_cache,
) -> pd.DataFrame:
    """Assemble the full per-pitch frame: cascade labels + base/lag features +
    batter & pitcher profile joins + the S3 xwOBA target.

    The single place where labels, features, and the two leakage-safe profile
    joins meet. Downstream ``node_population`` slices this per node.
    """
    df = add_cascade_labels(pitches)
    df = attach_xwoba_target(df, raw)
    df = build_base_features(df)
    df = attach_batter_profile(df, batter_cache, prefix="b")
    df = attach_pitcher_profile(df, pitcher_cache, prefix="p")
    return df


def prepare_node_features(
    X: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
    cat_dtypes: dict,
) -> pd.DataFrame:
    """Coerce a feature frame to the dtypes XGBoost needs (shared by train+infer).

    Non-categorical -> float32 (XGBoost 3.x SEGFAULTS on pandas nullable
    Int64/Float64, which the augmented parquet uses; NaN preserved -> missing).
    Categorical -> the train-fitted CategoricalDtype, with values unseen in train
    nulled out first (-> NaN -> missing), since XGBoost native categorical
    requires inference categories ⊆ train categories.
    """
    X = X[feature_names].copy()
    for c in feature_names:
        if c in categorical:
            s = X[c].where(X[c].isin(cat_dtypes[c].categories))
            X[c] = s.astype(cat_dtypes[c])
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def to_xgb_matrix(
    X: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
    cat_dtypes: dict,
) -> np.ndarray:
    """Build a dense float32 matrix XGBoost can train/predict on WITHOUT segfault.

    XGBoost 3.x's pandas *columnar* adapter segfaults at scale on macOS (the
    `MakeEncColumnarBatch` path). Feeding a dense numpy array sidesteps it.
    Categorical columns are encoded as their train-fitted integer codes (missing/
    unseen -> NaN); pair with ``feature_types_for`` so XGBoost treats them as
    categorical, not ordinal.
    """
    P = prepare_node_features(X, feature_names, categorical, cat_dtypes)
    M = np.empty((len(P), len(feature_names)), dtype=np.float32)
    cat_set = set(categorical)
    for j, c in enumerate(feature_names):
        if c in cat_set:
            codes = P[c].cat.codes.to_numpy().astype(np.float32)
            codes[codes < 0] = np.nan          # NaN/unseen code -1 -> missing
            M[:, j] = codes
        else:
            M[:, j] = P[c].to_numpy(dtype=np.float32)
    return M


def feature_types_for(feature_names: list[str], categorical: list[str]) -> list[str]:
    """XGBoost ``feature_types`` list: 'c' for categorical columns, 'q' otherwise."""
    cat_set = set(categorical)
    return ["c" if c in cat_set else "q" for c in feature_names]


def predict_from_artifacts(artifacts: dict, X: pd.DataFrame) -> np.ndarray:
    """Reproduce a node's calibrated prediction from its persisted artifacts.

    ``artifacts`` is the dict ``save_models`` writes (booster, calibrator,
    feature_names, categorical, cat_dtypes, objective). Binary -> isotonic-
    calibrated P(positive); regression -> raw value.
    """
    M = to_xgb_matrix(
        X, artifacts["feature_names"], artifacts["categorical"],
        artifacts["cat_dtypes"],
    )
    booster = artifacts["booster"]
    if artifacts["objective"] == "binary":
        raw = booster.predict_proba(M)[:, 1]
        return artifacts["calibrator"].transform(raw)
    return booster.predict(M)


def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass ECE of a binary positive-class probability.

    Bin predictions ``p`` into ``n_bins`` equal-mass quantile bins; ECE is the
    mass-weighted mean ``|mean_p - frac_positive|`` per bin. Equal-mass matches
    the project convention (eval/metrics/calibration.py); binary here because the
    cascade nodes emit one calibrated probability that gets multiplied into the
    per-PA outcome — so that single number must be honest.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    ece = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def train_node(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    objective: str,
    feature_names: list[str],
    monotone: dict[str, int] | None = None,
    categorical: list[str] | None = None,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    seed: int = 0,
    n_jobs: int | None = None,
) -> dict:
    """Fit one cascade node (XGBoost) + per-node isotonic calibration.

    ``objective``: ``"binary"`` (swing/whiff/called-strike) or ``"regression"``
    (contact-quality xwOBA). Binary nodes get isotonic calibration fit on the val
    set — the cascade multiplies these probabilities, so each must be calibrated,
    not just discriminative. Monotone constraints inject baseball priors
    (chase↑ out-of-zone, whiff↑ velo, xwOBA↑ middle) and clean up the SHAP plots.

    Returns ``{booster, calibrator, predict, metrics, feature_names, categorical}``
    where ``predict(X)`` returns calibrated probabilities (binary) or values (reg).

    NOTE: ECE is reported on the val set the isotonic map was fit on, so it is
    mildly optimistic; eval.py recomputes calibration on the held-out test set.
    """
    import os

    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

    # Single-threaded by default: the macOS libomp race in QuantileDMatrix
    # construction segfaults nondeterministically under any parallelism. n_jobs=1
    # is the only race-free config (verified stable across repeated runs).
    if n_jobs is None:
        n_jobs = 1

    categorical = categorical or []
    monotone = monotone or {}
    cons = tuple(monotone.get(f, 0) for f in feature_names)

    # Fit each categorical column's category set on TRAIN only. XGBoost native
    # categorical requires eval/predict categories ⊆ train categories, so val
    # ids unseen in train (2023-train vs 2024-val umpires/parks) must map to a
    # shared dtype where unseen -> NaN (treated as missing), not a new category.
    cat_dtypes = {
        c: pd.CategoricalDtype(categories=pd.Index(X_train[c].dropna().unique()))
        for c in categorical
    }

    def _mat(X: pd.DataFrame) -> np.ndarray:
        return to_xgb_matrix(X, feature_names, categorical, cat_dtypes)

    Xtr, Xva = _mat(X_train), _mat(X_val)
    ftypes = feature_types_for(feature_names, categorical)

    common = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", enable_categorical=bool(categorical),
        feature_types=ftypes, monotone_constraints=cons, random_state=seed,
        early_stopping_rounds=30, n_jobs=n_jobs,
    )

    metrics: dict[str, float] = {}
    if objective == "binary":
        model = xgb.XGBClassifier(objective="binary:logistic",
                                  eval_metric="logloss", **common)
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        raw_va = model.predict_proba(Xva)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_va, y_val.to_numpy())
        cal_va = iso.transform(raw_va)
        metrics["auc"] = float(roc_auc_score(y_val, cal_va))
        metrics["logloss"] = float(log_loss(y_val, np.clip(cal_va, 1e-7, 1 - 1e-7)))
        metrics["ece"] = binary_ece(cal_va, y_val.to_numpy())
        metrics["n_train"] = int(len(y_train))
        metrics["base_rate"] = float(y_train.mean())

        def predict(X: pd.DataFrame) -> np.ndarray:
            r = model.predict_proba(_mat(X))[:, 1]
            return iso.transform(r)

        return {"booster": model, "calibrator": iso, "predict": predict,
                "metrics": metrics, "feature_names": feature_names,
                "categorical": categorical, "cat_dtypes": cat_dtypes,
                "objective": objective}

    if objective == "regression":
        model = xgb.XGBRegressor(objective="reg:squarederror",
                                 eval_metric="rmse", **common)
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        pred_va = model.predict(Xva)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_val, pred_va)))
        metrics["pearson"] = float(np.corrcoef(pred_va, y_val.to_numpy())[0, 1])
        metrics["n_train"] = int(len(y_train))
        metrics["mean_target"] = float(y_train.mean())

        def predict(X: pd.DataFrame) -> np.ndarray:
            return model.predict(_mat(X))

        return {"booster": model, "calibrator": None, "predict": predict,
                "metrics": metrics, "feature_names": feature_names,
                "categorical": categorical, "cat_dtypes": cat_dtypes,
                "objective": objective}

    raise ValueError(f"objective must be 'binary' or 'regression', got {objective!r}")


def attach_xwoba_target(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Join ``estimated_woba_using_speedangle`` from a raw frame.

    Keyed on (game_pk, at_bat_number, pitch_number) — the augmented frames drop
    the field, so S3's regression target comes from ``data/raw/``.
    """
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    cols = keys + ["estimated_woba_using_speedangle"]
    return df.merge(raw[cols], on=keys, how="left")


# ---------------------------------------------------------------------------
# Data loading + orchestration
# ---------------------------------------------------------------------------

def load_pitch_frame(
    start: str,
    end: str,
    *,
    augmented_dir: str = "data/augmented",
    raw_dir: str = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load augmented pitches in [start, end] (inclusive, by date) + the matching
    raw frame (for the xwOBA target). Returns ``(augmented, raw)``.

    Per-day parquet layout ``{dir}/{year}/{date}.parquet`` (statcast-pipeline).
    """
    aug_files, raw_files = [], []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for f in sorted(glob.glob(f"{augmented_dir}/{year}/*.parquet")):
            d = Path(f).stem
            if start <= d <= end:
                aug_files.append(f)
        for f in sorted(glob.glob(f"{raw_dir}/{year}/*.parquet")):
            d = Path(f).stem
            if start <= d <= end:
                raw_files.append(f)
    if not aug_files:
        raise FileNotFoundError(f"no augmented parquet in [{start}, {end}]")
    aug = pd.concat((pd.read_parquet(f) for f in aug_files), ignore_index=True)
    raw_cols = ["game_pk", "at_bat_number", "pitch_number",
                "estimated_woba_using_speedangle"]
    raw = pd.concat(
        (pd.read_parquet(f, columns=raw_cols) for f in raw_files),
        ignore_index=True,
    )
    return aug, raw


def train_all_nodes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    nodes: tuple[str, ...] = ML_NODES,
    **node_kwargs,
) -> dict:
    """Train every ML node on its conditional population + the S2b foul constant.

    ``train_df`` / ``val_df`` are full feature frames from ``build_training_frame``.
    Returns ``{"nodes": {node: result}, "foul_rate_by_count": {...}}``.
    """
    results: dict[str, dict] = {}
    for node in nodes:
        Xtr, ytr = node_population(train_df, node)
        Xva, yva = node_population(val_df, node)
        res = train_node(
            Xtr, ytr, Xva, yva,
            objective=NODE_OBJECTIVE[node],
            feature_names=NODE_FEATURES[node],
            categorical=NODE_CATEGORICAL[node],
            monotone=NODE_MONOTONE[node],
            **node_kwargs,
        )
        results[node] = res
        m = res["metrics"]
        if NODE_OBJECTIVE[node] == "binary":
            print(f"  [{node:15s}] n={m['n_train']:>8,} base={m['base_rate']:.3f} "
                  f"AUC={m['auc']:.3f} logloss={m['logloss']:.3f} ECE={m['ece']:.3f}")
        else:
            print(f"  [{node:15s}] n={m['n_train']:>8,} mean={m['mean_target']:.3f} "
                  f"RMSE={m['rmse']:.3f} r={m['pearson']:.3f}")
    return {"nodes": results, "foul_rate_by_count": foul_rate_by_count(train_df)}


def save_models(bundle: dict, out_dir: str = "checkpoints/hitter") -> None:
    """Persist each node (booster + calibrator + feature spec) + foul rates.

    The ``predict`` closure is dropped (not picklable); ``HitterModel`` rebuilds
    it from the booster + calibrator at load time.
    """
    import json
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"nodes": {}, "foul_rate_by_count": {
        f"{b},{s}": r for (b, s), r in bundle["foul_rate_by_count"].items()}}
    for node, res in bundle["nodes"].items():
        joblib.dump(
            {k: res[k] for k in
             ("booster", "calibrator", "feature_names", "categorical",
              "cat_dtypes", "objective")},
            out / f"{node}.joblib",
        )
        meta["nodes"][node] = {"objective": res["objective"],
                               "metrics": res["metrics"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {len(bundle['nodes'])} nodes + foul rates -> {out}/")


def main(
    *,
    train_start: str = "2017-01-01",
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2024-07-15",
    fold_id: int = 0,
    out_dir: str = "checkpoints/hitter",
    **node_kwargs,
) -> dict:
    """End-to-end: load -> build features -> train all nodes -> persist.

    Temporal split per the hard rule (train ≤2023, val 2024H1). Profiles use a
    fixed fold's cache; the trailing-window discipline makes the join leakage-safe
    regardless of fold (it's not a cross-fit nuisance here).
    """
    from data.profile_cache_loader import ProfileCache

    bcache = ProfileCache(role="batter", fold_id=fold_id)
    pcache = ProfileCache(role="pitcher", fold_id=fold_id)

    print(f"loading train [{train_start}..{train_end}] + val [{val_start}..{val_end}]")
    tr_aug, tr_raw = load_pitch_frame(train_start, train_end)
    va_aug, va_raw = load_pitch_frame(val_start, val_end)
    print(f"  train pitches={len(tr_aug):,}  val pitches={len(va_aug):,}")

    print("building feature frames (labels + base + profile joins)...")
    train_df = build_training_frame(tr_aug, tr_raw, bcache, pcache)
    val_df = build_training_frame(va_aug, va_raw, bcache, pcache)

    print("training nodes:")
    bundle = train_all_nodes(train_df, val_df, **node_kwargs)
    save_models(bundle, out_dir)
    return bundle


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the hitter/swing cascade")
    ap.add_argument("--train-start", default="2017-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--val-start", default="2024-01-01")
    ap.add_argument("--val-end", default="2024-07-15")
    ap.add_argument("--fold-id", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints/hitter")
    args = ap.parse_args()
    main(train_start=args.train_start, train_end=args.train_end,
         val_start=args.val_start, val_end=args.val_end,
         fold_id=args.fold_id, out_dir=args.out_dir)
