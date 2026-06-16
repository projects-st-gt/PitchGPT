"""Pydantic schemas for the PitchGPT demo API.

Three top-level shapes:

- :class:`QueryRequest` — what the frontend sends.
- :class:`QueryResponse` — what the API returns. Has THREE distinct states
  (``green`` / ``yellow`` / ``red``) per ADR 002's positivity gating; the
  causal-claim fields are populated for green/yellow and ``None`` for red,
  while ``refusal`` is populated only for red.
- :class:`AtBatSummary` — listing payload for the demo's AB-picker dropdown.

The frontend's three demo states map directly to ``QueryResponse.trust_state``.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ----- Request -----


class QueryRequest(BaseModel):
    """A single counterfactual query: 'what if pitch[k]'s type were a*?'"""

    game_pk: int = Field(..., description="MLB Statcast game_pk identifier")
    at_bat_number: int = Field(..., ge=1, description="AB number within the game")
    intervention_position: int = Field(
        ..., ge=0,
        description=(
            "Pitch index (0-based) where we counterfactually substitute the "
            "type. V2 model can predict at position 0 (from the start token)."
        ),
    )
    intervention_type: str = Field(
        ...,
        description="The counterfactual pitch type. Must be one of FF, SI, FC, SL, CU, CH, FS.",
    )
    intervention_zone: Optional[int] = Field(
        None, ge=0, le=12,
        description=(
            "Optional feature-zone for the intervention, in dense internal indexing: "
            "0..8 = 3x3 in-zone (SIS labels 1-9, top-left to bottom-right), "
            "9..12 = OOZ quadrants (SIS labels 11-14: upper-left, upper-right, "
            "lower-left, lower-right). If omitted, zone is sampled from the model. "
            "When set, the positivity check uses joint π̂(type, zone | history)."
        ),
    )
    n_paths: int = Field(
        200, ge=10, le=2000,
        description=(
            "Monte Carlo rollout paths. 200 gives a reasonably tight estimate "
            "in ~10 sec; 1000 tightens the CI ~2× at ~50 sec."
        ),
    )


# ----- Response: shared structure -----


class PitchSummary(BaseModel):
    """One pitch within an observed AB — used in QueryResponse.observed_ab."""

    pitch_index: int           # 0-indexed within AB
    type: str                  # one of PITCH_TYPES
    count_before: str          # e.g. "0-0" / "1-2"
    description: str           # raw Statcast `description`


class ObservedAB(BaseModel):
    """The real AB we're counterfactually intervening on."""

    game_pk: int
    at_bat_number: int
    n_pitches: int
    pitches: list[PitchSummary]
    terminal_event: Optional[str] = None    # raw `events` value, e.g. "strikeout"
    ab_outcome_class: Optional[str] = None  # K / BB / 1B / 2B / 3B / HR / out (or None)


class ExpectedDistribution(BaseModel):
    """Model's prediction at the intervention position, with NO intervention."""

    pitch_type_probs: dict[str, float]    # π̂(type | history through k-1)
    expected_ab_run_value: float          # μ̂ from ab_outcome head at last observed position


class CounterfactualResult(BaseModel):
    """ALWAYS returned — the rollout always runs, even when positivity is weak.

    The ``is_causal_claim`` field distinguishes "this is a defensible causal
    estimate" (green) from "we ran the rollout but the data doesn't strongly
    support the intervention" (yellow / red). The numbers are computed and
    returned in all cases; the frontend uses ``is_causal_claim`` to decide
    how to label/style them (full confidence vs. "low support" badge).
    """

    is_causal_claim: bool                 # True when both type AND zone clear their τ_green
    support_level: Literal["high", "moderate", "low"]  # mirrors green/yellow/red

    effect_runs: float                    # τ̂ = E[Y | do(A=a*)] - E[Y | do(A=observed)]
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    e_value_point: float                  # E-value for the point estimate
    e_value_ci_limit: float               # E-value for the CI limit closer to the null
    intervention_mean_run_value: float    # E[Y | do(A=a*)]
    intervention_mean_ab_length: float    # average # of pitches the AB ends in under do(A=a*)
    intervention_outcome_distribution: dict[str, float]  # {K, BB, 1B, 2B, 3B, HR, out}
    baseline_mean_run_value: float        # baseline reference E[Y]
    baseline_mean_ab_length: float
    baseline_outcome_distribution: dict[str, float]  # same shape, "what would have happened naturally"
    n_paths: int
    n_truncated_paths: int                # paths that didn't reach a natural terminal


class RefusalInfo(BaseModel):
    """Populated only when trust_state == 'red'."""

    reason: str = "out_of_support"
    rationale: str                        # human-readable explanation
    threshold_tau: float                  # the positivity floor that fired
    # We STILL surface a model rollout under the rare intervention so the
    # demo can show "this is what the model expects, but not a causal claim."
    rollout_only_mean_run_value: float
    rollout_only_mean_ab_length: float
    rollout_only_outcome_distribution: dict[str, float]


class QueryResponse(BaseModel):
    """Top-level response. Trust state determines which optional blocks are populated."""

    trust_state: Literal["green", "yellow", "red"]
    p_hat_intervention: float             # π̂(a* | history) — joint over (type, zone) if zone set
    p_hat_type: float                     # π̂(type | history) — marginal
    p_hat_zone: Optional[float] = None    # π̂(zone | history) — marginal (only when zone set)
    intervention_type: str
    intervention_zone: Optional[int] = None  # echo back for the frontend's strike zone
    intervention_position: int
    rationale: str                        # short human-readable rationale (from PositivityGate)
    observed_ab: ObservedAB
    expected_distribution: ExpectedDistribution
    counterfactual: Optional[CounterfactualResult] = None
    refusal: Optional[RefusalInfo] = None
    timing_seconds: float                 # how long this query took to compute


# ----- Listing endpoint -----


class AtBatSummary(BaseModel):
    """A row for the demo's AB-picker dropdown."""

    game_pk: int
    at_bat_number: int
    game_date: str                        # "YYYY-MM-DD"
    pitcher_id: int
    pitcher_name: Optional[str] = None    # "First Last" from Chadwick register; None if lookup failed
    batter_id: int
    batter_name: Optional[str] = None
    batter_stand: Optional[str] = None    # "R" or "L" — for the silhouette direction
    pitcher_throws: Optional[str] = None  # "R" or "L"
    n_pitches: int
    pitch_types: list[str]                # e.g. ["FF", "SL", "CH"]
    pitch_zones: list[int] = []           # 0..12 feature_zone per pitch (SIS 14-zone, dense internal)
    terminal_event: Optional[str] = None


class AtBatListResponse(BaseModel):
    """Response for GET /at-bats."""

    items: list[AtBatSummary]
    total: int


class GameSummary(BaseModel):
    """A row for the demo's game-picker dropdown."""

    game_pk: int
    game_date: str
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    n_at_bats: int                          # AB count we have for this game


class GameListResponse(BaseModel):
    """Response for GET /games."""

    items: list[GameSummary]
    total: int


# ----- Pitcher profile inspector (Tab 1) -----


class PitcherSummary(BaseModel):
    """A row for the pitcher search/picker."""

    pitcher_id: int
    pitcher_name: Optional[str] = None
    latest_asof_date: str               # "YYYY-MM-DD" — the most recent entry we have
    n_entries: int                      # how many (date, game_num) entries exist in the cache


class PitcherListResponse(BaseModel):
    """Response for GET /pitchers."""

    items: list[PitcherSummary]
    total: int


class PitcherProfileResponse(BaseModel):
    """The 218-dim pitcher profile vector decomposed into semantic groups (Tab 1).

    Used by the frontend's Pitcher Profile Inspector. Numbers come straight from
    the v6 profile cache for the (pitcher, asof_date, asof_game_num) key — no
    model inference. League-mean blending is applied per the profile-cache
    ``lookup`` semantics (see ``data/profile_cache_loader.py``).
    """

    pitcher_id: int
    pitcher_name: Optional[str] = None
    fold_id: int

    asof_date_requested: str            # the date the caller asked for
    asof_date_used: str                 # the actual cache-entry date (closest on/before)
    asof_game_num: int                  # game number of the matched entry
    source: Literal["per_player_blended", "league_only", "zero_fallback"]

    # Per-pitch-type aggregates (one float per type in PITCH_TYPES order).
    arsenal_pct: dict[str, float]       # usage fraction — sums to ~1 across types thrown
    has_pitch: dict[str, bool]
    mean_velo: dict[str, float]         # mph
    mean_spin: dict[str, float]         # rpm
    mean_pfx_x: dict[str, float]        # horizontal break, feet
    mean_pfx_z: dict[str, float]        # vertical break, feet
    arm_slot: dict[str, float]          # release-side arm angle, degrees

    # 2-D blocks. Rows keyed by pitch type; column order documented in
    # ``count_state_order`` / handedness keys.
    arsenal_by_count: dict[str, list[float]]              # {pt: [12 floats]}; column order in count_state_order
    arsenal_by_stand: dict[str, dict[str, float]]         # {pt: {"L": .., "R": ..}}
    heatmap_by_type: dict[str, list[float]]               # {pt: [9 floats]} — 3x3 in-zone usage, row-major (top-left→bot-right)

    count_state_order: list[str]        # ["0-0", "0-1", ..., "3-2"]; matches arsenal_by_count cols
    in_zone_cell_order: list[int]       # [0..8] — index → 3x3 cell, row-major

    # Scalars / recent form.
    recent_30d_xwoba: float
    recent_30d_n_pitches: float
    days_since_last_appearance: float
    recent_3starts_xwoba: float
    recent_3starts_n: float
    profile_confidence: float
    long_window_span_days: float
    long_window_pct_current_season: float


# ----- Health -----


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "loading"]
    nuisance_checkpoint: Optional[str] = None
    nuisance_loaded: bool = False
    val_pitches_loaded: bool = False
    val_at_bats_count: Optional[int] = None


# ----- AB context — fetched once per AB, drives the reactive frontend -----


class PositionInfo(BaseModel):
    """All the information for one pitch position within an AB.

    Returned per-position so the frontend can update instantly as the user
    clicks different pitches — no rollout needed for these fields.
    """

    position: int                                  # 0-indexed pitch index
    pitch_type: str                                # observed type at this pitch
    feature_zone: int                              # observed feature_zone (0..12, SIS 14-zone)
    count_before: str                              # e.g. "0-0", "1-2"
    description: str                               # raw Statcast description
    # Model's predicted distribution for THIS pitch given history through k-1.
    # Available for k >= 1 (the model doesn't autoregressively predict pitch[0]).
    expected_pitch_type_probs: Optional[dict[str, float]] = None
    expected_zone_probs: Optional[list[float]] = None  # length 13, indexed by feature_zone
    expected_ab_run_value: Optional[float] = None  # AB-outcome head's expectation here

    # ----- A.3 rollout-viewer fields -----
    # Result-head prediction for THIS pitch (ball / called_strike /
    # swinging_strike / foul / in_play_out / in_play_hit / in_play_hr). The
    # result head reads the pitch's own intended action, so this is available
    # for ALL k including k=0 — unlike the type propensity, which needs history.
    expected_result_probs: Optional[dict[str, float]] = None
    actual_result: Optional[str] = None            # observed result class (from result_id)
    # Type-propensity-derived signals. Available for k >= 1 only — same
    # availability as expected_pitch_type_probs.
    model_confidence: Optional[float] = None       # max π̂ over pitch types — "how sure was the model"
    actual_pitch_surprisal: Optional[float] = None  # -log2 π̂(actual type), in bits
    position_entropy: Optional[float] = None       # -Σ π̂ log2 π̂ over pitch types, in bits
    # True when the model gave the actual pitch < OFF_MODEL_PROB_FLOOR (10%)
    # probability — a genuine "off-model" pitch. Fixed floor, not entropy-based.
    is_surprising: Optional[bool] = None

    # Raw Statcast location — DISPLAY ONLY, not a model input. The model
    # consumes the discretized 13-cell feature_zone; these continuous coords
    # are passed through purely so the frontend can plot precise pitch dots.
    # plate_x/plate_z in feet (plate_x: 0 = center, + = catcher's right).
    # sz_top/sz_bot are this batter's strike-zone bounds (feet) for scaling.
    # None when Statcast didn't record the pitch's location.
    plate_x: Optional[float] = None
    plate_z: Optional[float] = None
    sz_top: Optional[float] = None
    sz_bot: Optional[float] = None


class ABContextResponse(BaseModel):
    """Per-AB context — call once when user picks an AB; cache on the frontend."""

    game_pk: int
    at_bat_number: int
    game_date: str
    pitcher_id: int
    pitcher_name: Optional[str] = None
    batter_id: int
    batter_name: Optional[str] = None
    pitcher_throws: Optional[str] = None
    batter_stand: Optional[str] = None
    n_pitches: int
    terminal_event: Optional[str] = None
    ab_outcome_class: Optional[str] = None

    # Pitcher arsenal (from his profile) — which pitch types he actually throws.
    # Frontend uses this to disable buttons for never-thrown types.
    pitcher_arsenal_pct: dict[str, float]          # e.g. {"FF": 0.55, "SL": 0.30, "CH": 0.10, "CU": 0.05}
    pitcher_has_pitch: dict[str, bool]             # binary: "FF": True, "FS": False, etc.

    # Batter zone-grid weaknesses (from his profile) — 9-cell whiff%, swing%, xBA.
    # Frontend overlays these as a heatmap on the strike zone (3×3 in-zone grid).
    batter_whiff_grid: list[Optional[float]]       # length 9 (in-zone cells; SIS 1-9)
    batter_swing_grid: list[Optional[float]]       # length 9

    # Per-position info: index 0..n_pitches-1, in pitch order.
    positions: list[PositionInfo]

    # Scoreboard state at the start of the AB.
    home_team: Optional[str] = None        # 3-letter abbr e.g. "NYY"
    away_team: Optional[str] = None
    inning: Optional[int] = None
    inning_half: Optional[str] = None      # "Top" or "Bot"
    home_score: Optional[int] = None       # at start of AB
    away_score: Optional[int] = None
    outs_before_ab: Optional[int] = None
    runner_on_1b: bool = False
    runner_on_2b: bool = False
    runner_on_3b: bool = False
    batting_team: Optional[str] = None     # which abbr is at bat (home or away)
