"""FastAPI app for the PitchGPT counterfactual demo.

Endpoints:

- ``GET  /health`` — liveness + readiness (which heavy artifacts are loaded).
- ``GET  /at-bats`` — list available ABs for the demo's picker.
- ``POST /query``   — the counterfactual query (see :class:`schemas.QueryRequest`).

Heavy artifacts (NuisanceModels + val parquets) are lazy-loaded on first /query
or first /at-bats. The first request after boot is slow (~5-10 sec); subsequent
requests are dominated by the Monte Carlo rollout (~10 sec for N=200).

Run locally::

    uvicorn inference.api:app --reload --port 8000

The frontend (Sprint 2) posts JSON to ``http://localhost:8000/query`` and
renders ``QueryResponse``.
"""
from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from causal.g_computation import (
    AB_OUTCOME_NAMES,
    DEFAULT_AB_RUN_VALUE,
    g_compute,
)
from causal.nuisance import NuisanceModels, build_single_ab_batch
from causal.positivity import (
    GateDecision,
    PositivityGate,
    TAU_SINGLE_STEP,
    TAU_GREEN,
    TAU_ZONE_REFUSE,
    TAU_ZONE_GREEN,
    TrustState,
)
from causal.sensitivity import e_value_for_continuous_effect
from data.dataset import (
    MODEL_PITCH_TYPES_END_IDX,
    MODEL_PITCH_TYPES_START_IDX,
    MODEL_TYPE_ID,
    PITCH_TYPES,
    RESULT_CLASSES,
)
from inference.schemas import (
    ABContextResponse,
    AtBatListResponse,
    AtBatSummary,
    CounterfactualResult,
    ExpectedDistribution,
    GameListResponse,
    GameSummary,
    HealthResponse,
    ObservedAB,
    PitcherListResponse,
    PitcherProfileResponse,
    PitcherSummary,
    PitchSummary,
    PositionInfo,
    QueryRequest,
    QueryResponse,
    RefusalInfo,
)
from inference.player_names import _NameCache, name_for_mlbam
from data.profile_cache import (
    BATTER_FEATURE_INDEX,
    COUNT_STATES,
    N_IN_ZONE_CELLS,
    PITCHER_FEATURE_INDEX,
)
from model.pitchgpt_dataset import classify_ab_outcome, AB_OUTCOME_IGNORE

# ============================================================
# App + state
# ============================================================

DEFAULT_CHECKPOINT = Path("checkpoints_modal/tiny-fold0-v6/checkpoint_calibrated.pt")
DEFAULT_AUGMENTED_DIR = Path("data/augmented")
DEFAULT_VAL_GLOB = "2024/2024-*.parquet"  # MVP: serve from val 2024 H1

# Hardcoded for MVP — proper version reads observed AB-outcome run values from
# the (base, outs)-conditional RE24 lookup. See module docstring in
# ``causal.g_computation``.
RUN_VALUE_TABLE = DEFAULT_AB_RUN_VALUE
OUTCOME_SD_PROXY = 0.30  # for the E-value's Cohen-d-like rescaling

# Off-model flag (A.3 rollout viewer): a pitch is "off-model" when the model
# gave it less than this probability. A fixed floor — NOT an entropy-relative
# rule. An earlier surprisal-vs-entropy rule over-fired: in a peaked pitch-type
# distribution it flagged the model's #2 and #3 picks as surprising, which they
# are not. The fixed floor matches the positivity gate's fixed-τ philosophy.
OFF_MODEL_PROB_FLOOR = 0.10


class AppState:
    """Lazy-loaded heavy artifacts (model + val data + game-teams lookup)."""

    nuisance: Optional[NuisanceModels] = None
    val_pitches: Optional[pd.DataFrame] = None
    val_ab_keys: Optional[list[tuple[int, int]]] = None
    game_teams: Optional[dict[int, tuple[str, str]]] = None  # game_pk → (home, away)

    @classmethod
    def get_nuisance(cls) -> NuisanceModels:
        if cls.nuisance is None:
            print(f"[api] loading nuisance from {DEFAULT_CHECKPOINT}...")
            cls.nuisance = NuisanceModels(DEFAULT_CHECKPOINT, device="cpu")
            print(f"[api] loaded: {cls.nuisance}")
        return cls.nuisance

    @classmethod
    def get_game_teams(cls) -> dict[int, tuple[str, str]]:
        """Lazy game_pk → (home_team, away_team) lookup, cached on disk.

        Augmented parquets don't carry team names; raw 2024 parquets do.
        We scan once across the val date range, write a tiny cache, and load
        from there on subsequent boots.
        """
        if cls.game_teams is not None:
            return cls.game_teams
        cache_path = Path("data/preprocess_artifacts/game_teams_2024.parquet")
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            print(f"[api] building game_teams cache from raw 2024 parquets...")
            raw_files = sorted(glob.glob("data/raw/2024/*.parquet"))
            parts: list[pd.DataFrame] = []
            for f in raw_files:
                try:
                    parts.append(pd.read_parquet(f, columns=["game_pk", "home_team", "away_team"]))
                except Exception:
                    continue
            df = pd.concat(parts, ignore_index=True).drop_duplicates("game_pk").reset_index(drop=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            print(f"[api] cached {len(df):,} games' team-info to {cache_path}")
        cls.game_teams = {
            int(r.game_pk): (str(r.home_team), str(r.away_team))
            for r in df.itertuples(index=False)
        }
        return cls.game_teams

    @classmethod
    def get_val(cls) -> pd.DataFrame:
        if cls.val_pitches is None:
            print(f"[api] loading val pitches from {DEFAULT_AUGMENTED_DIR}/{DEFAULT_VAL_GLOB}...")
            files = sorted(glob.glob(str(DEFAULT_AUGMENTED_DIR / DEFAULT_VAL_GLOB)))
            # Filter to val period (≤ VAL_END = 2024-07-15) just in case.
            files = [f for f in files if Path(f).stem <= "2024-07-15"]
            if not files:
                raise RuntimeError(
                    f"no val parquets under {DEFAULT_AUGMENTED_DIR}/{DEFAULT_VAL_GLOB} ≤ 2024-07-15"
                )
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
            cls.val_pitches = df
            cls.val_ab_keys = list(df.groupby(["game_pk", "at_bat_number"]).groups.keys())
            print(f"[api] loaded {len(df):,} pitches, {len(cls.val_ab_keys):,} ABs")
        return cls.val_pitches


app = FastAPI(
    title="PitchGPT counterfactual demo API",
    description=(
        "Wraps the causal layer (g-computation + positivity + sensitivity) "
        "behind one HTTP endpoint. See CLAUDE.md for the project's hard rules; "
        "in particular: refusal under positivity violation IS a feature."
    ),
    version="0.1.0",
)
# Allow the Vite dev server to hit us. Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helpers
# ============================================================


def _format_count(balls: int, strikes: int) -> str:
    return f"{int(balls)}-{int(strikes)}"


def _ab_outcome_label(events_value: object) -> Optional[str]:
    cls = classify_ab_outcome(events_value)
    if cls == AB_OUTCOME_IGNORE:
        return None
    return AB_OUTCOME_NAMES[int(cls)]


def _build_observed_ab(ab: pd.DataFrame) -> ObservedAB:
    pitches: list[PitchSummary] = []
    for i, row in ab.reset_index(drop=True).iterrows():
        type_id_1based = int(row["type_id"])
        pitch_type = PITCH_TYPES[type_id_1based - 1] if type_id_1based > 0 else "PAD"
        pitches.append(PitchSummary(
            pitch_index=int(i),
            type=pitch_type,
            count_before=_format_count(int(row["balls"]), int(row["strikes"])),
            description=str(row["description"]) if pd.notna(row["description"]) else "",
        ))
    last = ab.iloc[-1]
    return ObservedAB(
        game_pk=int(ab.iloc[0]["game_pk"]),
        at_bat_number=int(ab.iloc[0]["at_bat_number"]),
        n_pitches=len(ab),
        pitches=pitches,
        terminal_event=str(last["events"]) if pd.notna(last.get("events")) else None,
        ab_outcome_class=_ab_outcome_label(last.get("events")),
    )


def _baseline_expected_distribution(
    nuisance: NuisanceModels, ab: pd.DataFrame, intervention_position: int
) -> tuple[ExpectedDistribution, np.ndarray, np.ndarray, float]:
    """One forward pass for the baseline (no intervention).

    Returns:
        - ExpectedDistribution: π̂ at position k and the AB-outcome's expected runs.
        - type_probs: (N_PITCH_TYPES,) numpy — for the gate check.
        - zone_probs: (13,) numpy — for joint positivity if a zone intervention is set.
        - baseline_run_value: scalar mean run value under the natural model.
    """
    batch = build_single_ab_batch(nuisance, ab, n_replicates=1)
    out = nuisance.forward(batch)
    k = intervention_position

    # π̂ at the position predicting pitch k.
    NC = nuisance.model.N_CONTEXT_TOKENS
    pi_type = out.propensity_probs["type"][
        0, NC + (k - 1), MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX
    ].numpy().astype(np.float64)
    pi_type = pi_type / pi_type.sum().clip(min=1e-12)

    # Zone propensity at the same sequence position. The zone head's vocab is 13
    # under v5 (SIS 14-zone: 9 in-zone + 4 OOZ quadrants) — no PAD offset, sliced
    # directly. Internal indices: 0..8 = in-zone 3x3 (top-left to bottom-right),
    # 9..12 = OOZ quadrants (UL, UR, LL, LR per the SIS_TO_INTERNAL map).
    pi_zone = out.propensity_probs["zone"][0, NC + (k - 1), :].numpy().astype(np.float64)
    pi_zone = pi_zone / pi_zone.sum().clip(min=1e-12)

    # μ̂ baseline: AB-outcome head at the last observed position, dotted with run values.
    ab_probs = out.ab_outcome_probs[0, -1, :].numpy().astype(np.float64)
    baseline_run_value = float((ab_probs * RUN_VALUE_TABLE).sum())

    return ExpectedDistribution(
        pitch_type_probs={pt: float(pi_type[i]) for i, pt in enumerate(PITCH_TYPES)},
        expected_ab_run_value=baseline_run_value,
    ), pi_type, pi_zone, baseline_run_value


def _outcome_dist_to_dict(arr: np.ndarray) -> dict[str, float]:
    return {AB_OUTCOME_NAMES[i]: float(arr[i]) for i in range(len(AB_OUTCOME_NAMES))}


def _resolve_pitcher_asof(
    pitcher_id: int, asof_date: Optional[str]
) -> tuple[pd.Timestamp, int]:
    """Find the most recent cache entry on/before ``asof_date`` for this pitcher.

    The profile cache is keyed by (player_id, asof_date, asof_game_num); for a
    "browse this pitcher on date D" query the caller usually doesn't know
    game_num. We scan this pitcher's keys, pick the latest entry whose
    asof_date ≤ requested date, breaking ties by the largest game_num.

    If ``asof_date`` is omitted, returns the absolute latest entry for the player.
    Raises HTTPException(404) if no entry exists for this pitcher.
    """
    nuisance = AppState.get_nuisance()
    cache = nuisance.pitcher_cache
    keys = [k for k in cache._player_lookup.keys() if k[0] == int(pitcher_id)]
    if not keys:
        raise HTTPException(status_code=404, detail=f"pitcher {pitcher_id} has no profile entries")
    if asof_date is None:
        keys.sort(key=lambda k: (k[1], k[2]))
        return keys[-1][1], keys[-1][2]
    ts = pd.Timestamp(asof_date)
    eligible = [k for k in keys if k[1] <= ts]
    if not eligible:
        raise HTTPException(
            status_code=404,
            detail=(
                f"pitcher {pitcher_id} has no profile entries on/before {asof_date}; "
                f"earliest entry is {min(k[1] for k in keys).date()}"
            ),
        )
    eligible.sort(key=lambda k: (k[1], k[2]))
    return eligible[-1][1], eligible[-1][2]


# ============================================================
# Endpoints
# ============================================================


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        nuisance_checkpoint=str(DEFAULT_CHECKPOINT),
        nuisance_loaded=AppState.nuisance is not None,
        val_pitches_loaded=AppState.val_pitches is not None,
        val_at_bats_count=len(AppState.val_ab_keys) if AppState.val_ab_keys else None,
    )


@app.get("/games", response_model=GameListResponse)
def list_games(limit: int = 200) -> GameListResponse:
    """List distinct games available in val, with team info + AB counts.

    Used by the demo's two-step picker (pick a game first, then an AB within it).
    """
    df = AppState.get_val()
    teams_lookup = AppState.get_game_teams()

    games_grouped = (
        df.groupby("game_pk")
        .agg(game_date=("game_date", "first"), n_at_bats=("at_bat_number", "nunique"))
        .reset_index()
        .sort_values("game_date")
    )
    items: list[GameSummary] = []
    for _, row in games_grouped.head(limit).iterrows():
        game_pk = int(row["game_pk"])
        teams = teams_lookup.get(game_pk)
        items.append(GameSummary(
            game_pk=game_pk,
            game_date=str(row["game_date"])[:10],
            home_team=teams[0] if teams else None,
            away_team=teams[1] if teams else None,
            n_at_bats=int(row["n_at_bats"]),
        ))
    return GameListResponse(items=items, total=len(games_grouped))


@app.get("/at-bats", response_model=AtBatListResponse)
def list_at_bats(
    limit: int = 50,
    min_pitches: int = 3,
    max_pitches: int = 10,
    game_pk: Optional[int] = None,
) -> AtBatListResponse:
    """List ABs for the demo's picker.

    Args:
        limit: max ABs returned.
        min_pitches / max_pitches: filter by AB length (defaults to 3-10 pitches).
        game_pk: if set, restricts to ABs in this game (for two-step picker).
    """
    df = AppState.get_val()
    if game_pk is not None:
        df = df[df["game_pk"] == int(game_pk)]
    items: list[AtBatSummary] = []
    counts = df.groupby(["game_pk", "at_bat_number"]).size()
    valid_keys = counts[(counts >= min_pitches) & (counts <= max_pitches)].index.tolist()
    for key in valid_keys[:limit]:
        ab = df[(df["game_pk"] == key[0]) & (df["at_bat_number"] == key[1])]
        first = ab.iloc[0]
        last = ab.iloc[-1]
        # Batter handedness — look up the "stand_id" categorical (encoded R=1, L=2)
        # or fall back to the raw "stand" string if present.
        stand_raw: Optional[str] = None
        if "stand" in first.index and pd.notna(first["stand"]):
            stand_raw = str(first["stand"])
        elif "stand_id" in first.index:
            sid = int(first["stand_id"])
            stand_raw = "R" if sid == 1 else ("L" if sid == 2 else None)
        throws_raw: Optional[str] = None
        if "p_throws" in first.index and pd.notna(first["p_throws"]):
            throws_raw = str(first["p_throws"])
        elif "p_throws_id" in first.index:
            pid = int(first["p_throws_id"])
            throws_raw = "R" if pid == 1 else ("L" if pid == 2 else None)

        pitcher_id = int(first["pitcher"])
        batter_id = int(first["batter"])
        items.append(AtBatSummary(
            game_pk=int(key[0]),
            at_bat_number=int(key[1]),
            game_date=str(first["game_date"])[:10],
            pitcher_id=pitcher_id,
            pitcher_name=name_for_mlbam(pitcher_id),
            batter_id=batter_id,
            batter_name=name_for_mlbam(batter_id),
            batter_stand=stand_raw,
            pitcher_throws=throws_raw,
            n_pitches=len(ab),
            pitch_types=[PITCH_TYPES[int(t) - 1] if int(t) > 0 else "PAD" for t in ab["type_id"]],
            pitch_zones=[int(z) for z in ab["feature_zone"]],
            terminal_event=str(last["events"]) if pd.notna(last.get("events")) else None,
        ))
    return AtBatListResponse(items=items, total=len(valid_keys))


@app.get("/ab-context", response_model=ABContextResponse)
def ab_context(game_pk: int, at_bat_number: int) -> ABContextResponse:
    """All the per-AB context the frontend needs to be reactive.

    Returns one forward pass through the model giving per-position expected
    distributions, plus the pitcher's arsenal and the batter's zone weakness
    heatmap (from the profile caches). Frontend calls this once per AB and
    reads from it as the user clicks around — no /query rollout needed for
    these views.
    """
    nuisance = AppState.get_nuisance()
    df = AppState.get_val()
    ab = df[(df["game_pk"] == game_pk) & (df["at_bat_number"] == at_bat_number)] \
        .sort_values("pitch_number").reset_index(drop=True)
    if len(ab) == 0:
        raise HTTPException(status_code=404, detail=f"AB not found: ({game_pk}, {at_bat_number})")

    pitcher_id = int(ab.iloc[0]["pitcher"])
    batter_id = int(ab.iloc[0]["batter"])
    pitcher_throws = str(ab.iloc[0].get("p_throws") or "")
    batter_stand = str(ab.iloc[0].get("stand") or "")

    # Forward through the model ONCE, on the full observed AB.
    batch = build_single_ab_batch(nuisance, ab, n_replicates=1)
    out = nuisance.forward(batch)
    NC = nuisance.model.N_CONTEXT_TOKENS

    # Per-position expected distributions. Position k's expected pitch comes
    # from propensity at seq_idx = NC + (k - 1), which predicts pitch k given
    # history through pitch k-1. For k=0 (the first pitch), there's no
    # "history through pitch -1" — set to None.
    positions: list[PositionInfo] = []
    for k in range(len(ab)):
        row = ab.iloc[k]
        type_id_1 = int(row["type_id"])
        pt = PITCH_TYPES[type_id_1 - 1] if type_id_1 > 0 else "PAD"

        expected_type_probs: Optional[dict[str, float]] = None
        expected_zone_probs: Optional[list[float]] = None
        expected_run_value: Optional[float] = None
        model_confidence: Optional[float] = None
        actual_pitch_surprisal: Optional[float] = None
        position_entropy: Optional[float] = None
        is_surprising: Optional[bool] = None

        if k >= 1:
            seq_idx = NC + (k - 1)
            pi_type = out.propensity_probs["type"][
                0, seq_idx, MODEL_PITCH_TYPES_START_IDX:MODEL_PITCH_TYPES_END_IDX
            ].numpy().astype(np.float64)
            pi_type = pi_type / pi_type.sum().clip(min=1e-12)
            expected_type_probs = {p: float(pi_type[i]) for i, p in enumerate(PITCH_TYPES)}

            pi_zone = out.propensity_probs["zone"][0, seq_idx, :].numpy().astype(np.float64)
            pi_zone = pi_zone / pi_zone.sum().clip(min=1e-12)
            expected_zone_probs = [float(z) for z in pi_zone]

            # AB-outcome head's expected run value if the AB ended at pitch k
            ab_probs = out.ab_outcome_probs[0, k, :].numpy().astype(np.float64)
            expected_run_value = float((ab_probs * RUN_VALUE_TABLE).sum())

            # Confidence / surprisal / entropy over the pitch-type distribution.
            # Surprisal and entropy (bits) are kept as displayed context. The
            # off-model flag itself uses the fixed OFF_MODEL_PROB_FLOOR.
            model_confidence = float(pi_type.max())
            position_entropy = float(-(pi_type * np.log2(np.clip(pi_type, 1e-12, None))).sum())
            if pt in PITCH_TYPES:
                p_actual = float(pi_type[PITCH_TYPES.index(pt)])
                actual_pitch_surprisal = float(-np.log2(max(p_actual, 1e-12)))
                is_surprising = p_actual < OFF_MODEL_PROB_FLOOR

        # Result-head prediction for this pitch — available for ALL k. The
        # result head reads the pitch's own intended action; no history needed.
        result_probs_k = out.result_probs[0, k, :].numpy().astype(np.float64)
        result_probs_k = result_probs_k / result_probs_k.sum().clip(min=1e-12)
        expected_result_probs = {
            RESULT_CLASSES[i]: float(result_probs_k[i]) for i in range(len(RESULT_CLASSES))
        }
        # Observed result class. result_id in the parquet is 1-indexed
        # (PAD=0, RESULT_CLASSES at 1..7); shift by -1 to name it.
        result_id_1 = int(row["result_id"]) if pd.notna(row.get("result_id")) else 0
        actual_result = (
            RESULT_CLASSES[result_id_1 - 1] if 1 <= result_id_1 <= len(RESULT_CLASSES) else None
        )

        def _coord(col: str) -> Optional[float]:
            v = row.get(col)
            return float(v) if pd.notna(v) else None

        positions.append(PositionInfo(
            position=k,
            pitch_type=pt,
            feature_zone=int(row["feature_zone"]),
            count_before=_format_count(int(row["balls"]), int(row["strikes"])),
            description=str(row["description"]) if pd.notna(row["description"]) else "",
            expected_pitch_type_probs=expected_type_probs,
            expected_zone_probs=expected_zone_probs,
            expected_ab_run_value=expected_run_value,
            expected_result_probs=expected_result_probs,
            actual_result=actual_result,
            model_confidence=model_confidence,
            actual_pitch_surprisal=actual_pitch_surprisal,
            position_entropy=position_entropy,
            is_surprising=is_surprising,
            plate_x=_coord("plate_x"),
            plate_z=_coord("plate_z"),
            sz_top=_coord("sz_top"),
            sz_bot=_coord("sz_bot"),
        ))

    # Pitcher arsenal (from his profile cache vector — sliced to the schema's
    # arsenal slots).
    pitcher_profile_raw = nuisance.pitcher_cache.lookup(
        pitcher_id, pd.Timestamp(ab.iloc[0]["game_date"]),
        int(ab.iloc[0].get("game_num", 1)),
    )["vector"]
    pitcher_arsenal_pct: dict[str, float] = {}
    pitcher_has_pitch: dict[str, bool] = {}
    for pt in PITCH_TYPES:
        # Profile slot names: `arsenal_{pt}` and `has_pitch_{pt}` (data/profile_cache.py)
        ars_idx = PITCHER_FEATURE_INDEX.get(f"arsenal_{pt}")
        has_idx = PITCHER_FEATURE_INDEX.get(f"has_pitch_{pt}")
        pitcher_arsenal_pct[pt] = float(pitcher_profile_raw[ars_idx]) if ars_idx is not None else 0.0
        pitcher_has_pitch[pt] = bool(pitcher_profile_raw[has_idx] > 0.5) if has_idx is not None else False

    # Batter zone heatmap (9-cell in-zone whiff% + swing%, SIS 14-zone).
    batter_profile_raw = nuisance.batter_cache.lookup(
        batter_id, pd.Timestamp(ab.iloc[0]["game_date"]),
        int(ab.iloc[0].get("game_num", 1)),
    )["vector"]
    batter_whiff_grid: list[Optional[float]] = []
    batter_swing_grid: list[Optional[float]] = []
    for i in range(N_IN_ZONE_CELLS):
        w_idx = BATTER_FEATURE_INDEX.get(f"whiff_z{i}")
        s_idx = BATTER_FEATURE_INDEX.get(f"swing_z{i}")
        w = float(batter_profile_raw[w_idx]) if w_idx is not None else float("nan")
        s = float(batter_profile_raw[s_idx]) if s_idx is not None else float("nan")
        batter_whiff_grid.append(None if (w != w) else w)  # NaN check
        batter_swing_grid.append(None if (s != s) else s)

    last = ab.iloc[-1]
    first_row = ab.iloc[0]

    # Scoreboard at the start of the AB.
    games_lookup = AppState.get_game_teams()
    teams = games_lookup.get(game_pk)
    home_team = teams[0] if teams else None
    away_team = teams[1] if teams else None

    inning_topbot = str(first_row.get("inning_topbot") or "")
    inning_half = "Top" if inning_topbot.lower().startswith("t") else (
        "Bot" if inning_topbot.lower().startswith("b") else None
    )
    # The batting team is the AWAY team in the top of the inning, HOME in the bottom.
    batting_team = away_team if inning_half == "Top" else home_team
    # bat_score / fld_score are at start of pitch — for the AB's start, use the first pitch's values.
    bat_score = int(first_row["bat_score"]) if pd.notna(first_row.get("bat_score")) else None
    fld_score = int(first_row["fld_score"]) if pd.notna(first_row.get("fld_score")) else None
    if inning_half == "Top":
        away_score, home_score = bat_score, fld_score
    elif inning_half == "Bot":
        away_score, home_score = fld_score, bat_score
    else:
        away_score, home_score = None, None

    return ABContextResponse(
        game_pk=game_pk,
        at_bat_number=at_bat_number,
        game_date=str(first_row["game_date"])[:10],
        pitcher_id=pitcher_id,
        pitcher_name=name_for_mlbam(pitcher_id),
        batter_id=batter_id,
        batter_name=name_for_mlbam(batter_id),
        pitcher_throws=pitcher_throws or None,
        batter_stand=batter_stand or None,
        n_pitches=len(ab),
        terminal_event=str(last["events"]) if pd.notna(last.get("events")) else None,
        ab_outcome_class=_ab_outcome_label(last.get("events")),
        pitcher_arsenal_pct=pitcher_arsenal_pct,
        pitcher_has_pitch=pitcher_has_pitch,
        batter_whiff_grid=batter_whiff_grid,
        batter_swing_grid=batter_swing_grid,
        positions=positions,
        home_team=home_team,
        away_team=away_team,
        inning=int(first_row["inning"]) if pd.notna(first_row.get("inning")) else None,
        inning_half=inning_half,
        home_score=home_score,
        away_score=away_score,
        outs_before_ab=int(first_row["outs_when_up"]) if pd.notna(first_row.get("outs_when_up")) else None,
        # on_1b/2b/3b store the runner's MLBAM id (or pd.NA if no runner).
        # Bool-ing pd.NA raises; check notna() first, then truthy.
        runner_on_1b=bool(pd.notna(first_row.get("on_1b")) and first_row.get("on_1b")),
        runner_on_2b=bool(pd.notna(first_row.get("on_2b")) and first_row.get("on_2b")),
        runner_on_3b=bool(pd.notna(first_row.get("on_3b")) and first_row.get("on_3b")),
        batting_team=batting_team,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t_start = time.perf_counter()

    if req.intervention_type not in MODEL_TYPE_ID:
        raise HTTPException(
            status_code=400,
            detail=f"intervention_type must be one of {PITCH_TYPES}, got {req.intervention_type!r}",
        )
    nuisance = AppState.get_nuisance()
    df = AppState.get_val()
    ab = df[
        (df["game_pk"] == req.game_pk) & (df["at_bat_number"] == req.at_bat_number)
    ].sort_values("pitch_number").reset_index(drop=True)
    if len(ab) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"AB not found: game_pk={req.game_pk}, at_bat_number={req.at_bat_number}",
        )
    if req.intervention_position >= len(ab):
        raise HTTPException(
            status_code=400,
            detail=(
                f"intervention_position {req.intervention_position} ≥ AB length {len(ab)}; "
                f"pick an earlier position (must be in [1, {len(ab)})"
            ),
        )

    observed = _build_observed_ab(ab)
    expected, pi_type_arr, pi_zone_arr, baseline_run_value = _baseline_expected_distribution(
        nuisance, ab, req.intervention_position
    )

    # Positivity gate — when zone is set, we require BOTH π̂(type) AND π̂(zone)
    # to independently clear their respective τ. Type and zone get DIFFERENT
    # thresholds because they have different cardinalities (type=7 cells,
    # zone=13 cells under v5 SIS 14-zone). With fewer cells, per-cell marginals
    # are larger (~7% vs the old ~3-4%), so the existing zone-dimension
    # thresholds may admit more queries than under v1 — re-tune after retrain
    # if the gauge looks too permissive.
    # The overall trust state is the WORSE of the two dimensions' states.
    p_hat_type = float(pi_type_arr[MODEL_TYPE_ID[req.intervention_type] - MODEL_PITCH_TYPES_START_IDX])
    type_gate = PositivityGate(tau_refuse=TAU_SINGLE_STEP, tau_green=TAU_GREEN)
    type_decision = type_gate.gate(p_hat_type)
    if req.intervention_zone is not None:
        p_hat_zone = float(pi_zone_arr[int(req.intervention_zone)])
        zone_gate = PositivityGate(tau_refuse=TAU_ZONE_REFUSE, tau_green=TAU_ZONE_GREEN)
        zone_decision = zone_gate.gate(p_hat_zone)
        # Worst-of-two as the binding state. Order: green > yellow > red.
        priority = {TrustState.GREEN: 2, TrustState.YELLOW: 1, TrustState.RED: 0}
        if priority[type_decision.state] <= priority[zone_decision.state]:
            decision = type_decision
        else:
            decision = zone_decision
        # Compose a more informative rationale that names both dimensions.
        decision = GateDecision(
            state=decision.state,
            p_hat=min(p_hat_type, p_hat_zone),
            threshold=decision.threshold,
            rationale=(
                f"type π̂({req.intervention_type})={p_hat_type:.4f} "
                f"(type τ_refuse={TAU_SINGLE_STEP}, τ_green={TAU_GREEN}); "
                f"zone π̂(cell {req.intervention_zone})={p_hat_zone:.4f} "
                f"(zone τ_refuse={TAU_ZONE_REFUSE}, τ_green={TAU_ZONE_GREEN}). "
                f"Binding dimension is the worse of the two."
            ),
        )
        p_hat = min(p_hat_type, p_hat_zone)
    else:
        p_hat_zone = None
        p_hat = p_hat_type
        decision = type_decision

    # Baseline rollout (intervention = observed type, optional observed zone at
    # position k) — gives us a like-for-like baseline AB-length contrast.
    observed_type_at_k = PITCH_TYPES[int(ab.iloc[req.intervention_position]["type_id"]) - 1]
    observed_zone_at_k = (
        int(ab.iloc[req.intervention_position]["feature_zone"])
        if req.intervention_zone is not None
        else None
    )
    baseline_rollout = g_compute(
        nuisance, ab,
        intervention_position=req.intervention_position,
        intervention_type=observed_type_at_k,
        intervention_zone=observed_zone_at_k,
        n_paths=req.n_paths, rng_seed=42,
    )

    # Intervention rollout.
    cf_rollout = g_compute(
        nuisance, ab,
        intervention_position=req.intervention_position,
        intervention_type=req.intervention_type,
        intervention_zone=req.intervention_zone,
        n_paths=req.n_paths, rng_seed=42,
    )

    # Always compute the counterfactual. The rollout itself doesn't need
    # positivity to produce numbers — it's just sampling from the trained
    # model under do(A_k = a*). The trust_state controls how to LABEL the
    # output (high-support / moderate / low-support), not whether to compute it.
    effect = cf_rollout.mean_run_value - baseline_rollout.mean_run_value
    se = float(np.sqrt(cf_rollout.se_run_value**2 + baseline_rollout.se_run_value**2))
    e_res = e_value_for_continuous_effect(effect, se, outcome_sd=OUTCOME_SD_PROXY)
    support_level = (
        "high" if decision.state is TrustState.GREEN
        else "moderate" if decision.state is TrustState.YELLOW
        else "low"
    )
    counterfactual = CounterfactualResult(
        is_causal_claim=(decision.state is TrustState.GREEN),
        support_level=support_level,
        effect_runs=effect,
        ci_lower=effect - 1.96 * se,
        ci_upper=effect + 1.96 * se,
        e_value_point=e_res.point_e_value,
        e_value_ci_limit=e_res.ci_e_value,
        intervention_mean_run_value=cf_rollout.mean_run_value,
        intervention_mean_ab_length=cf_rollout.mean_ab_length,
        intervention_outcome_distribution=_outcome_dist_to_dict(cf_rollout.ab_outcome_distribution),
        baseline_mean_run_value=baseline_rollout.mean_run_value,
        baseline_mean_ab_length=baseline_rollout.mean_ab_length,
        baseline_outcome_distribution=_outcome_dist_to_dict(baseline_rollout.ab_outcome_distribution),
        n_paths=cf_rollout.n_paths,
        n_truncated_paths=cf_rollout.n_truncated,
    )
    # The refusal block is retained but now empty — kept as Optional[None] for
    # API back-compat; the demo no longer renders it as a separate panel.
    refusal: Optional[RefusalInfo] = None

    return QueryResponse(
        trust_state=decision.state.value,
        p_hat_intervention=p_hat,
        p_hat_type=p_hat_type,
        p_hat_zone=p_hat_zone,
        intervention_type=req.intervention_type,
        intervention_zone=req.intervention_zone,
        intervention_position=req.intervention_position,
        rationale=decision.rationale,
        observed_ab=observed,
        expected_distribution=expected,
        counterfactual=counterfactual,
        refusal=refusal,
        timing_seconds=round(time.perf_counter() - t_start, 2),
    )


# ============================================================
# Tab 1 — Pitcher Profile Inspector
# ============================================================


@app.get("/pitchers", response_model=PitcherListResponse)
def list_pitchers(
    q: Optional[str] = None,
    limit: int = 25,
) -> PitcherListResponse:
    """List pitchers present in the profile cache, optionally name-filtered.

    The list is the set of distinct ``player_id``s in the loaded pitcher
    profile cache (fold 0), enriched with Chadwick names. The ``q`` parameter
    does a case-insensitive substring match against "First Last".
    """
    nuisance = AppState.get_nuisance()
    cache = nuisance.pitcher_cache

    # Aggregate per pitcher: latest_asof_date, n_entries.
    per_player: dict[int, tuple[pd.Timestamp, int]] = {}
    for (pid, ts, _gnum) in cache._player_lookup.keys():
        if pid not in per_player or ts > per_player[pid][0]:
            per_player[pid] = (ts, 1 + per_player.get(pid, (ts, 0))[1])
        else:
            per_player[pid] = (per_player[pid][0], per_player[pid][1] + 1)

    items: list[PitcherSummary] = []
    q_norm = q.lower().strip() if q else None
    for pid, (latest, n) in per_player.items():
        name = name_for_mlbam(pid)
        if q_norm:
            if not name or q_norm not in name.lower():
                continue
        items.append(PitcherSummary(
            pitcher_id=pid,
            pitcher_name=name,
            latest_asof_date=str(latest.date()),
            n_entries=int(n),
        ))
    items.sort(key=lambda s: (s.pitcher_name or "", s.pitcher_id))
    return PitcherListResponse(items=items[:limit], total=len(items))


@app.get("/pitcher/{pitcher_id}/profile", response_model=PitcherProfileResponse)
def pitcher_profile(
    pitcher_id: int,
    asof_date: Optional[str] = None,
) -> PitcherProfileResponse:
    """Decomposed 218-dim pitcher profile for the closest entry on/before ``asof_date``.

    No model inference — reads the v6 profile cache directly. League-mean
    blending is applied per ``ProfileCache.lookup`` semantics, so the response
    contains numeric values for every slot (no NaN).

    Args:
        pitcher_id: MLBAM player id.
        asof_date: ISO date "YYYY-MM-DD". If omitted, uses the latest entry.
    """
    nuisance = AppState.get_nuisance()
    cache = nuisance.pitcher_cache
    asof_ts, game_num = _resolve_pitcher_asof(pitcher_id, asof_date)
    looked = cache.lookup(pitcher_id, asof_ts, game_num)
    vec: np.ndarray = looked["vector"]
    source: str = looked["source"]

    def slot(name: str) -> float:
        idx = PITCHER_FEATURE_INDEX.get(name)
        if idx is None:
            return float("nan")
        return float(vec[idx])

    arsenal_pct = {pt: slot(f"arsenal_{pt}") for pt in PITCH_TYPES}
    has_pitch = {pt: slot(f"has_pitch_{pt}") > 0.5 for pt in PITCH_TYPES}
    mean_velo = {pt: slot(f"mean_velo_{pt}") for pt in PITCH_TYPES}
    mean_spin = {pt: slot(f"mean_spin_{pt}") for pt in PITCH_TYPES}
    mean_pfx_x = {pt: slot(f"mean_pfx_x_{pt}") for pt in PITCH_TYPES}
    mean_pfx_z = {pt: slot(f"mean_pfx_z_{pt}") for pt in PITCH_TYPES}
    arm_slot = {pt: slot(f"arm_slot_{pt}") for pt in PITCH_TYPES}

    arsenal_by_count: dict[str, list[float]] = {}
    for pt in PITCH_TYPES:
        arsenal_by_count[pt] = [slot(f"arsenal_{pt}_b{b}s{s}") for (b, s) in COUNT_STATES]
    arsenal_by_stand: dict[str, dict[str, float]] = {}
    for pt in PITCH_TYPES:
        arsenal_by_stand[pt] = {
            "L": slot(f"arsenal_{pt}_vsL"),
            "R": slot(f"arsenal_{pt}_vsR"),
        }
    heatmap_by_type: dict[str, list[float]] = {}
    for pt in PITCH_TYPES:
        heatmap_by_type[pt] = [slot(f"heatmap_{pt}_z{i}") for i in range(N_IN_ZONE_CELLS)]

    return PitcherProfileResponse(
        pitcher_id=pitcher_id,
        pitcher_name=name_for_mlbam(pitcher_id),
        fold_id=cache.fold_id,
        asof_date_requested=asof_date or str(asof_ts.date()),
        asof_date_used=str(asof_ts.date()),
        asof_game_num=int(game_num),
        source=source,  # type: ignore[arg-type]
        arsenal_pct=arsenal_pct,
        has_pitch=has_pitch,
        mean_velo=mean_velo,
        mean_spin=mean_spin,
        mean_pfx_x=mean_pfx_x,
        mean_pfx_z=mean_pfx_z,
        arm_slot=arm_slot,
        arsenal_by_count=arsenal_by_count,
        arsenal_by_stand=arsenal_by_stand,
        heatmap_by_type=heatmap_by_type,
        count_state_order=[f"{b}-{s}" for (b, s) in COUNT_STATES],
        in_zone_cell_order=list(range(N_IN_ZONE_CELLS)),
        recent_30d_xwoba=slot("recent_30d_xwoba"),
        recent_30d_n_pitches=slot("recent_30d_n_pitches"),
        days_since_last_appearance=slot("days_since_last_appearance"),
        recent_3starts_xwoba=slot("recent_3starts_xwoba"),
        recent_3starts_n=slot("recent_3starts_n"),
        profile_confidence=slot("profile_confidence"),
        long_window_span_days=slot("long_window_span_days"),
        long_window_pct_current_season=slot("long_window_pct_current_season"),
    )
