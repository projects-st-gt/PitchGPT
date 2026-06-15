// Pydantic-aligned TypeScript types for the demo API.
// Must stay in sync with ``inference/schemas.py``.

export type TrustState = "green" | "yellow" | "red";

export type PitchType = "FF" | "SI" | "FC" | "SL" | "CU" | "CH" | "FS";
export const PITCH_TYPES: PitchType[] = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"];

export const PITCH_TYPE_NAMES: Record<PitchType, string> = {
  FF: "4-Seam",
  SI: "Sinker",
  FC: "Cutter",
  SL: "Slider",
  CU: "Curveball",
  CH: "Changeup",
  FS: "Splitter",
};

// Color + glyph (color alone fails ~8% of male users).
export const PITCH_TYPE_GLYPHS: Record<PitchType, { color: string; glyph: string }> = {
  FF: { color: "#dc2626", glyph: "●" }, // red ●
  SI: { color: "#ea580c", glyph: "◆" }, // orange ◆
  FC: { color: "#ca8a04", glyph: "■" }, // amber ■
  SL: { color: "#0d9488", glyph: "▲" }, // teal ▲
  CU: { color: "#2563eb", glyph: "★" }, // blue ★
  CH: { color: "#7c3aed", glyph: "◐" }, // purple ◐
  FS: { color: "#475569", glyph: "✕" }, // slate ✕
};

export type ABOutcome = "K" | "BB" | "1B" | "2B" | "3B" | "HR" | "out";
export const AB_OUTCOMES: ABOutcome[] = ["K", "BB", "1B", "2B", "3B", "HR", "out"];

export interface PitchSummary {
  pitch_index: number;
  type: string;          // one of PITCH_TYPES (may be "PAD" in degenerate cases)
  count_before: string;  // e.g. "1-2"
  description: string;   // raw Statcast description
}

export interface ObservedAB {
  game_pk: number;
  at_bat_number: number;
  n_pitches: number;
  pitches: PitchSummary[];
  terminal_event: string | null;
  ab_outcome_class: ABOutcome | null;
}

export interface ExpectedDistribution {
  pitch_type_probs: Record<string, number>;
  expected_ab_run_value: number;
}

export interface CounterfactualResult {
  is_causal_claim: boolean;
  support_level: "high" | "moderate" | "low";
  effect_runs: number;
  ci_lower: number;
  ci_upper: number;
  ci_level: number;
  e_value_point: number;
  e_value_ci_limit: number;
  intervention_mean_run_value: number;
  intervention_mean_ab_length: number;
  intervention_outcome_distribution: Record<ABOutcome, number>;
  baseline_mean_run_value: number;
  baseline_mean_ab_length: number;
  baseline_outcome_distribution: Record<ABOutcome, number>;
  n_paths: number;
  n_truncated_paths: number;
}

export interface RefusalInfo {
  reason: string;
  rationale: string;
  threshold_tau: number;
  rollout_only_mean_run_value: number;
  rollout_only_mean_ab_length: number;
  rollout_only_outcome_distribution: Record<ABOutcome, number>;
}

export interface QueryResponse {
  trust_state: TrustState;
  p_hat_intervention: number;     // joint over (type, zone) when zone is set
  p_hat_type: number;             // marginal π̂(type | history)
  p_hat_zone: number | null;      // marginal π̂(zone | history), only when zone is set
  intervention_type: PitchType;
  intervention_zone: number | null;  // 0..12 feature_zone (v5 SIS 14-zone, dense internal index), echoed back
  intervention_position: number;
  rationale: string;
  observed_ab: ObservedAB;
  expected_distribution: ExpectedDistribution;
  counterfactual: CounterfactualResult | null;
  refusal: RefusalInfo | null;
  timing_seconds: number;
}

export interface AtBatSummary {
  game_pk: number;
  at_bat_number: number;
  game_date: string;
  pitcher_id: number;
  pitcher_name: string | null;
  batter_id: number;
  batter_name: string | null;
  batter_stand: "R" | "L" | null;
  pitcher_throws: "R" | "L" | null;
  n_pitches: number;
  pitch_types: string[];
  pitch_zones: number[];          // 0..12 feature_zone per pitch (v5 SIS 14-zone, dense internal index)
  terminal_event: string | null;
}

export interface AtBatListResponse {
  items: AtBatSummary[];
  total: number;
}

export interface GameSummary {
  game_pk: number;
  game_date: string;
  home_team: string | null;
  away_team: string | null;
  n_at_bats: number;
}

export interface GameListResponse {
  items: GameSummary[];
  total: number;
}

// ----- /ab-context — per-AB context, one fetch drives the reactive frontend -----

export interface PositionInfo {
  position: number;
  pitch_type: string;
  feature_zone: number;
  count_before: string;
  description: string;
  expected_pitch_type_probs: Record<string, number> | null;
  expected_zone_probs: number[] | null;
  expected_ab_run_value: number | null;

  // A.3 rollout-viewer fields.
  expected_result_probs: Record<string, number> | null;  // available for all k
  actual_result: string | null;                          // observed result class
  model_confidence: number | null;                       // max π̂ over types; k >= 1
  actual_pitch_surprisal: number | null;                 // -log2 π̂(actual type), bits; k >= 1
  position_entropy: number | null;                       // -Σ π̂ log2 π̂, bits; k >= 1
  is_surprising: boolean | null;                          // π̂(actual) < 10% floor; k >= 1

  // Raw Statcast location — display only, not a model input.
  plate_x: number | null;   // feet, 0 = center, + = catcher's right
  plate_z: number | null;   // feet, absolute height
  sz_top: number | null;    // batter's strike-zone top, feet
  sz_bot: number | null;    // batter's strike-zone bottom, feet
}

export interface ABContextResponse {
  game_pk: number;
  at_bat_number: number;
  game_date: string;
  pitcher_id: number;
  pitcher_name: string | null;
  batter_id: number;
  batter_name: string | null;
  pitcher_throws: "R" | "L" | null;
  batter_stand: "R" | "L" | null;
  n_pitches: number;
  terminal_event: string | null;
  ab_outcome_class: ABOutcome | null;
  pitcher_arsenal_pct: Record<string, number>;
  pitcher_has_pitch: Record<string, boolean>;
  batter_whiff_grid: (number | null)[];
  batter_swing_grid: (number | null)[];
  positions: PositionInfo[];

  // Scoreboard fields
  home_team: string | null;
  away_team: string | null;
  inning: number | null;
  inning_half: "Top" | "Bot" | null;
  home_score: number | null;
  away_score: number | null;
  outs_before_ab: number | null;
  runner_on_1b: boolean;
  runner_on_2b: boolean;
  runner_on_3b: boolean;
  batting_team: string | null;
}

export interface QueryRequest {
  game_pk: number;
  at_bat_number: number;
  intervention_position: number;
  intervention_type: PitchType;
  intervention_zone?: number | null;   // 0..12 feature_zone (v5 SIS: 0..8 = 3x3 in-zone, 9..12 = OOZ quadrants); null = sample from model
  n_paths?: number;
}

// ----- /pitcher/{id}/profile + /pitchers — Tab 1 (Pitcher Profile Inspector) -----

export interface PitcherSummary {
  pitcher_id: number;
  pitcher_name: string | null;
  latest_asof_date: string;          // "YYYY-MM-DD"
  n_entries: number;
}

export interface PitcherListResponse {
  items: PitcherSummary[];
  total: number;
}

export type ProfileSource = "per_player_blended" | "league_only" | "zero_fallback";

export interface PitcherProfileResponse {
  pitcher_id: number;
  pitcher_name: string | null;
  fold_id: number;

  asof_date_requested: string;
  asof_date_used: string;
  asof_game_num: number;
  source: ProfileSource;

  arsenal_pct: Record<string, number>;
  has_pitch: Record<string, boolean>;
  mean_velo: Record<string, number>;
  mean_spin: Record<string, number>;
  mean_pfx_x: Record<string, number>;
  mean_pfx_z: Record<string, number>;
  arm_slot: Record<string, number>;

  arsenal_by_count: Record<string, number[]>;        // {pt: 12 floats; columns in count_state_order}
  arsenal_by_stand: Record<string, { L: number; R: number }>;
  heatmap_by_type: Record<string, number[]>;          // {pt: 9 floats; row-major 3x3 in-zone}

  count_state_order: string[];        // ["0-0", "0-1", ..., "3-2"]
  in_zone_cell_order: number[];

  recent_30d_xwoba: number;
  recent_30d_n_pitches: number;
  days_since_last_appearance: number;
  recent_3starts_xwoba: number;
  recent_3starts_n: number;
  profile_confidence: number;
  long_window_span_days: number;
  long_window_pct_current_season: number;
}

// ----- Strike-zone helpers (v5 SIS 14-zone scheme) -----
//
// Internal feature_zone indices (from data/zones.py SIS_TO_INTERNAL):
//   In-zone 3x3 grid → indices 0..8, reading rows top→bottom, left→right:
//     0 1 2   (top row,   SIS labels 1, 2, 3)
//     3 4 5   (middle,    SIS labels 4, 5, 6)
//     6 7 8   (bottom,    SIS labels 7, 8, 9)
//   OOZ quadrants → indices 9..12:
//     9  = upper-left  OOZ (SIS 11)
//     10 = upper-right OOZ (SIS 12)
//     11 = lower-left  OOZ (SIS 13)
//     12 = lower-right OOZ (SIS 14)
//
// Display convention (catcher's view, batter facing camera):
//   - display row 0 = top of strike zone
//   - display row 2 = bottom of strike zone
//   - display col 0 = catcher's left (image-left)
//   - display col 2 = catcher's right (image-right)
//   - RHB batter stands on catcher's left → silhouette on LEFT of strike zone
//   - LHB batter stands on catcher's right → silhouette on RIGHT of strike zone

export const N_IN_ZONE_CELLS = 9;
export const N_FEATURE_ZONES = 13;
export const FEATURE_ZONE_OOZ_BASE = 9;  // first OOZ index; OOZ range is [9, 13).

// OOZ quadrant identifiers (positions outside the 3x3 in-zone grid).
export type OOZQuadrant = "UL" | "UR" | "LL" | "LR";
export const OOZ_QUADRANT_TO_FEATURE_ZONE: Record<OOZQuadrant, number> = {
  UL: 9,
  UR: 10,
  LL: 11,
  LR: 12,
};
export const FEATURE_ZONE_TO_OOZ_QUADRANT: Record<number, OOZQuadrant> = {
  9: "UL",
  10: "UR",
  11: "LL",
  12: "LR",
};

export function gridCellToFeatureZone(displayRow: number, displayCol: number): number {
  // 3x3 in-zone: row 0 (top) → indices 0..2; row 2 (bottom) → 6..8.
  return 3 * displayRow + displayCol;
}

export function featureZoneToGridCell(zone: number): { row: number; col: number } | null {
  if (zone < 0 || zone >= N_IN_ZONE_CELLS) return null;  // OOZ has no in-zone grid cell
  return { row: Math.floor(zone / 3), col: zone % 3 };
}

export function isOOZ(zone: number): boolean {
  return zone >= FEATURE_ZONE_OOZ_BASE && zone < N_FEATURE_ZONES;
}

export function oozQuadrantOf(zone: number): OOZQuadrant | null {
  return FEATURE_ZONE_TO_OOZ_QUADRANT[zone] ?? null;
}

// ============================================================
// MCSim App B — matchup-card predictions + actuals overlay
// (backend: inference/mcsim_api.py)
// ============================================================

export interface McsimPredictionSummary {
  game_pk: number;
  prediction_date: string;
  home_team: string | null;
  away_team: string | null;
  made_at: string | null;
  n_cells: number | null;
  has_actual: boolean;
}

export interface McsimPredictionsForDate {
  date: string;
  count: number;
  predictions: McsimPredictionSummary[];
}

export interface McsimActualEvent {
  pitcher_id: number;
  batter_id: number;
  inning: number | null;
  half: string | null;
  event: string | null;
  event_type: string | null;
  runs_scored: number | null;
  pitch_types: string[];
  bases_start: string[];
  outs_start: number | null;
}

export interface McsimCell {
  batter_id: number;
  batter_name: string;
  predicted_rv_median: number | null;
  predicted_rv_p05: number | null;
  predicted_rv_p95: number | null;
  predicted_top1_outcome: ABOutcome;
  predicted_outcome_dist: Record<ABOutcome, number>;
  predicted_obp?: number | null;
  predicted_slg?: number | null;
  predicted_ops?: number | null;
  modal_type: PitchType;
  p_hat_top_type: number;
  trust_state: TrustState;
  n_paths: number;
  n_truncated: number;
  actual?: { pa_count: number; events: McsimActualEvent[] } | null;
}

export interface McsimRow {
  pitcher_id: number;
  name: string;
  team: string;
  throws: string;
  is_starter: boolean;
  cells: McsimCell[];
}

export interface McsimCard {
  game_pk: number;
  game_date: string;
  home_team: string;
  away_team: string;
  starter_home: { pitcher_id: number; name: string } | null;
  starter_away: { pitcher_id: number; name: string } | null;
  rows: McsimRow[];
  n_cells: number;
  n_paths_per_cell: number;
}

export interface McsimCardResponse {
  game_pk: number;
  prediction_date: string;
  made_at: string | null;
  model_ckpt_hash: string | null;
  has_actual: boolean;
  final_score_home: number | null;
  final_score_away: number | null;
  winner: string | null;
  unmatched_event_count: number;
  card: McsimCard;
}

// ============================================================
// Score Prediction (game sim) — Monte Carlo full-game simulation
// (backend: inference/mcsim_api.py /gamesim endpoints)
// ============================================================

export interface GameSimSummary {
  game_pk: number;
  prediction_date: string;
  home_team: string;
  away_team: string;
  win_prob_home: number;
  win_prob_away: number;
  projected_home: number;
  projected_away: number;
  projected_total: number;
  home_starter_name: string | null;
  away_starter_name: string | null;
  has_actual: boolean;
  final_score_home: number | null;
  final_score_away: number | null;
  winner: string | null;
  predicted_winner_correct: boolean | null;
}

export interface GameSimListResponse {
  date: string;
  count: number;
  games: GameSimSummary[];
}

export interface StarterHookStats {
  hooked_pct: number;
  avg_hook_inning?: number;
  median_hook_inning?: number;
  avg_bf_at_hook?: number;
  hook_inning_dist?: Record<string, number>;
}

export interface RelieverUsage {
  pitcher_id: number;
  appearances: number;
  pct: number;
}

export interface BullpenTeamStats {
  starter_hook: StarterHookStats;
  relievers_used: RelieverUsage[];
}

export interface PitcherStaff {
  pitcher_id: number;
  name: string;
  throws: string;
  is_starter: boolean;
  workload_bf?: number | null;
}

export interface GameSimDetail {
  game_pk: number;
  prediction_date: string;
  home_team: string;
  away_team: string;
  sim: {
    n_sims: number;
    home_team: string;
    away_team: string;
    win_prob_home: number;
    win_prob_away: number;
    projected_score: { home: number; away: number };
    median_score: { home: number; away: number };
    projected_total_runs: number;
    total_runs_90pct_band: [number, number];
    n_backstop: number;
    inning_runs_home: number[];
    inning_runs_away: number[];
    home_starter_name?: string;
    away_starter_name?: string;
    home_starter_workload?: number;
    away_starter_workload?: number;
    total_runs_dist?: Record<string, number>;
    margin_dist?: Record<string, number>;
    bullpen_stats?: {
      home: BullpenTeamStats;
      away: BullpenTeamStats;
    };
  };
  has_actual: boolean;
  final_score_home: number | null;
  final_score_away: number | null;
  winner: string | null;
  home_staff: PitcherStaff[];
  away_staff: PitcherStaff[];
}
