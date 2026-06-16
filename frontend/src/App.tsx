import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getABContext, listAtBats, listGames, postQuery } from "./api";
import MCSimTab from "./MCSimTab";
import PitcherProfileTab from "./PitcherProfileTab";
import RolloutViewerTab from "./RolloutViewerTab";
import ScorePredictionTab from "./ScorePredictionTab";
import { ScoreboardHeader } from "./Scoreboard";
import { Caption, PitchBadge, Section, SectionLabel } from "./shared/ui";
import { StrikeZone } from "./StrikeZone";
import {
  AB_OUTCOMES,
  PITCH_TYPES,
  PITCH_TYPE_GLYPHS,
  PITCH_TYPE_NAMES,
  type ABContextResponse,
  type ABOutcome,
  type AtBatSummary,
  type GameSummary,
  type PitchType,
  type QueryResponse,
  type TrustState,
} from "./types";

function ProbBar({
  value,
  label,
  width = "w-32",
}: {
  value: number;
  label?: string;
  width?: string;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={`flex items-center gap-3 ${width}`}>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gray-900 rounded-full transition-state"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-small font-medium text-gray-900 tabular-nums w-12 text-right">
        {label ?? value.toFixed(3)}
      </div>
    </div>
  );
}

function TrustGauge({
  state,
  pHatType,
  pHatZone,
  interventionType,
}: {
  state: TrustState;
  pHatType: number;
  pHatZone: number | null;
  interventionType?: string;
}) {
  const colors: Record<TrustState, { bg: string; fg: string; label: string }> = {
    green: {
      bg: "bg-accent-subtle",
      fg: "text-accent",
      label: "High support — causal claim is defensible",
    },
    yellow: {
      bg: "bg-amber-50",
      fg: "text-amber-700",
      label: "Moderate support — interpret with care",
    },
    red: {
      bg: "bg-red-50",
      fg: "text-red-700",
      label: "Low support — shown for context, not as a causal claim",
    },
  };
  const c = colors[state];

  // Translate the raw probabilities into plain-English statements about how
  // often the pitcher would actually choose this. No math notation.
  const typePctLabel = (p: number) => {
    const pct = (p * 100).toFixed(1);
    if (p >= 0.20) return `${pct}% — common`;
    if (p >= 0.05) return `${pct}% — sometimes`;
    if (p >= 0.01) return `${pct}% — rare`;
    return `${pct}% — very rare`;
  };
  const zonePctLabel = (p: number) => {
    const pct = (p * 100).toFixed(2);
    if (p >= 0.05) return `${pct}% — a common spot for him`;
    if (p >= 0.02) return `${pct}% — sometimes a target zone`;
    if (p >= 0.005) return `${pct}% — rarely lives there`;
    return `${pct}% — essentially never goes there`;
  };

  return (
    <div className={`flex items-start gap-4 px-5 py-4 rounded-md ${c.bg} transition-state`}>
      <div className={`h-2.5 w-2.5 mt-2 rounded-full flex-shrink-0 ${
        state === "green" ? "bg-accent" : state === "yellow" ? "bg-amber-500" : "bg-red-600"
      }`} />
      <div className="flex-1">
        <div className={`text-body font-medium ${c.fg} leading-snug`}>{c.label}</div>
        <div className="text-small text-gray-700 mt-2 leading-relaxed space-y-1">
          <div className="text-gray-500 uppercase tracking-wide font-medium text-xs">
            How often the pitcher would actually choose this:
          </div>
          <div>
            <span className="text-gray-500">Throwing a {interventionType ?? "this pitch"}:</span>{" "}
            <span className="text-gray-900 font-medium tabular-nums">{typePctLabel(pHatType)}</span>
          </div>
          {pHatZone !== null && (
            <div>
              <span className="text-gray-500">In that specific zone:</span>{" "}
              <span className="text-gray-900 font-medium tabular-nums">{zonePctLabel(pHatZone)}</span>
            </div>
          )}
          {pHatZone !== null && (
            <div className="text-gray-500 italic pt-1">
              Both have to be common enough for us to make a strong causal claim. The lower of the two
              decides the gauge color.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Pickers — two-step: game first, then at-bat within game
// ============================================================

function GamePicker({
  games,
  selected,
  onSelect,
  loading,
}: {
  games: GameSummary[];
  selected: GameSummary | null;
  onSelect: (g: GameSummary) => void;
  loading: boolean;
}) {
  if (loading) return <div className="text-small text-gray-500">Loading games…</div>;
  if (games.length === 0) return <div className="text-small text-gray-500">No games available.</div>;
  return (
    <select
      className="w-full px-4 py-3 text-body border-hairline rounded-md bg-white hover:border-gray-300 focus:outline-none focus:border-accent transition-state"
      value={selected ? String(selected.game_pk) : ""}
      onChange={(e) => {
        const id = Number(e.target.value);
        const found = games.find((g) => g.game_pk === id);
        if (found) onSelect(found);
      }}
    >
      {games.map((g) => (
        <option key={g.game_pk} value={g.game_pk}>
          {g.game_date} · {g.away_team ?? "?"} @ {g.home_team ?? "?"} · {g.n_at_bats} AB
          {g.n_at_bats === 1 ? "" : "s"}
        </option>
      ))}
    </select>
  );
}

function ABPicker({
  atBats,
  selected,
  onSelect,
  loading,
}: {
  atBats: AtBatSummary[];
  selected: AtBatSummary | null;
  onSelect: (ab: AtBatSummary) => void;
  loading: boolean;
}) {
  if (loading) return <div className="text-small text-gray-500">Loading at-bats…</div>;
  if (atBats.length === 0)
    return <div className="text-small text-gray-500">No at-bats in this game (or filtered out).</div>;
  return (
    <select
      className="w-full px-4 py-3 text-body border-hairline rounded-md bg-white hover:border-gray-300 focus:outline-none focus:border-accent transition-state"
      value={selected ? String(selected.at_bat_number) : ""}
      onChange={(e) => {
        const abNum = Number(e.target.value);
        const found = atBats.find((x) => x.at_bat_number === abNum);
        if (found) onSelect(found);
      }}
    >
      {atBats.map((ab) => (
        <option key={ab.at_bat_number} value={ab.at_bat_number}>
          AB #{ab.at_bat_number} · {ab.pitcher_name ?? `P${ab.pitcher_id}`} ({ab.pitcher_throws ?? "?"}HP)
          {" → "}{ab.batter_name ?? `B${ab.batter_id}`} ({ab.batter_stand ?? "?"}HB) ·{" "}
          {ab.pitch_types.join(" → ")} → {ab.terminal_event ?? "?"}
        </option>
      ))}
    </select>
  );
}

// ============================================================
// Observed AB display — clickable pitch positions for intervention
// ============================================================

function ObservedABDisplay({
  ab,
  abContext,
  interventionPosition,
  onPick,
}: {
  ab: AtBatSummary;
  abContext: ABContextResponse | null;
  interventionPosition: number;
  onPick: (pos: number) => void;
}) {
  return (
    <div>
      <div className="text-small text-gray-500 mb-3">
        AB {ab.at_bat_number} · {ab.game_date} · {ab.n_pitches} pitches · ended in{" "}
        <span className="font-medium text-gray-900">{ab.terminal_event ?? "?"}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {ab.pitch_types.map((pt, i) => {
          const disabled = i === 0;
          const selected = i === interventionPosition;
          const count = abContext?.positions[i]?.count_before;
          const result = abContext?.positions[i]?.description;
          return (
            <button
              key={i}
              disabled={disabled}
              onClick={() => onPick(i)}
              className={[
                "px-3.5 py-2 rounded-md text-small font-medium border-hairline transition-state flex flex-col items-start gap-0.5 text-left",
                disabled
                  ? "bg-gray-50 text-gray-400 cursor-not-allowed"
                  : selected
                  ? "bg-accent text-accent-fg border-accent"
                  : "bg-white text-gray-900 hover:border-gray-300 cursor-pointer",
              ].join(" ")}
              title={
                disabled
                  ? "Pitch 0 isn't predictable from no history"
                  : `Intervene at pitch ${i + 1}`
              }
            >
              <div className="flex items-center gap-2">
                <span className={selected ? "text-white opacity-75" : "text-gray-500"}>
                  #{i}
                </span>
                {count && (
                  <span
                    className={`text-xs tabular-nums ${
                      selected ? "text-white opacity-75" : "text-gray-400"
                    }`}
                  >
                    {count}
                  </span>
                )}
                <PitchBadge type={pt} />
              </div>
              {result && (
                <span
                  className={`text-xs italic ${
                    selected ? "text-white opacity-75" : "text-gray-400"
                  }`}
                >
                  → {result}
                </span>
              )}
            </button>
          );
        })}
      </div>
      <Caption>
        Each card shows the pitch position, the count before it, the pitch type, and the actual
        result. Click any (except the first) to set the intervention point.
      </Caption>
    </div>
  );
}

// ============================================================
// Intervention controls (type + run button)
// ============================================================

function InterventionControls({
  interventionType,
  onTypeChange,
  nPaths,
  onPathsChange,
  onRun,
  running,
  abContext,
}: {
  interventionType: PitchType;
  onTypeChange: (t: PitchType) => void;
  nPaths: number;
  onPathsChange: (n: number) => void;
  onRun: () => void;
  running: boolean;
  abContext: ABContextResponse | null;
}) {
  // Pitch types the pitcher actually throws (his arsenal). Types where he
  // either doesn't have the pitch OR throws it < 1% of the time are disabled
  // (the gate would refuse anyway; we surface that at the UI level so the
  // user doesn't waste a click + 15-sec wait).
  function isAvailable(pt: PitchType): boolean {
    if (!abContext) return true;
    const usage = abContext.pitcher_arsenal_pct[pt] ?? 0;
    const has = abContext.pitcher_has_pitch[pt] ?? false;
    return has && usage >= 0.01;
  }
  return (
    <div>
      <div className="text-body text-gray-900 mb-3">What if instead the pitcher had thrown…</div>
      <div className="flex flex-wrap gap-2 mb-4">
        {PITCH_TYPES.map((pt) => {
          const selected = pt === interventionType;
          const available = isAvailable(pt);
          const usage = abContext?.pitcher_arsenal_pct[pt];
          const { color, glyph } = PITCH_TYPE_GLYPHS[pt];
          return (
            <button
              key={pt}
              onClick={() => available && onTypeChange(pt)}
              disabled={!available}
              title={
                available
                  ? usage !== undefined
                    ? `This pitcher throws ${pt} ${(usage * 100).toFixed(0)}% of the time`
                    : ""
                  : `This pitcher rarely or never throws ${pt}${
                      usage !== undefined ? ` (${(usage * 100).toFixed(0)}%)` : ""
                    }`
              }
              className={[
                "px-3.5 py-2 rounded-md text-small font-medium border-hairline transition-state",
                !available
                  ? "bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed line-through opacity-60"
                  : selected
                  ? "bg-accent text-accent-fg border-accent"
                  : "bg-white text-gray-900 hover:border-gray-300",
              ].join(" ")}
            >
              <span
                aria-hidden="true"
                className="mr-1.5"
                style={{ color: selected ? "white" : !available ? "#9ca3af" : color }}
              >
                {glyph}
              </span>
              <span className="mr-1">{pt}</span>
              <span className={selected ? "text-white opacity-75" : "text-gray-500"}>
                · {PITCH_TYPE_NAMES[pt]}
                {available && usage !== undefined && usage > 0 && (
                  <span className="ml-1 opacity-70">{(usage * 100).toFixed(0)}%</span>
                )}
              </span>
            </button>
          );
        })}
      </div>
      {abContext && (
        <Caption>
          Percentages are this pitcher's arsenal mix (from his last-1000-pitches profile).
          Pitches he doesn't throw are disabled — the gate would refuse them anyway.
        </Caption>
      )}
      <div className="flex items-center gap-4 mb-6">
        <label className="text-small text-gray-500">Monte Carlo paths</label>
        <select
          className="px-3 py-1.5 text-small border-hairline rounded-md bg-white hover:border-gray-300 transition-state"
          value={nPaths}
          onChange={(e) => onPathsChange(Number(e.target.value))}
        >
          <option value={100}>100 (~6s)</option>
          <option value={200}>200 (~13s) — default</option>
          <option value={500}>500 (~30s)</option>
          <option value={1000}>1000 (~60s)</option>
        </select>
      </div>
      <button
        onClick={onRun}
        disabled={running}
        className={[
          "px-6 py-3 text-body font-medium rounded-md transition-state",
          running
            ? "bg-gray-200 text-gray-500 cursor-wait"
            : "bg-gray-900 text-white hover:bg-gray-700",
        ].join(" ")}
      >
        {running ? "Running…" : "Run counterfactual"}
      </button>
    </div>
  );
}

// ============================================================
// Result panel
// ============================================================

function OutcomeDistribution({
  dist,
  baselineDist,
  highlight = false,
}: {
  dist: Record<ABOutcome, number>;
  baselineDist?: Record<ABOutcome, number>;
  highlight?: boolean;
}) {
  return (
    <div className="grid grid-cols-4 sm:grid-cols-7 gap-1.5 sm:gap-2">
      {AB_OUTCOMES.map((o) => {
        const v = dist[o] ?? 0;
        const vb = baselineDist?.[o] ?? null;
        const delta = vb !== null ? v - vb : null;
        return (
          <div key={o} className="flex flex-col items-center">
            <div className="text-small text-gray-500 mb-1">{o}</div>
            <div className="w-full h-20 bg-gray-100 rounded relative overflow-hidden">
              {/* Solid bar — the intervention distribution */}
              <div
                className={`absolute bottom-0 left-0 right-0 transition-state ${
                  highlight ? "bg-accent" : "bg-gray-400"
                }`}
                style={{ height: `${v * 100}%` }}
              />
              {/* Dotted outline — the baseline distribution (what would have happened naturally) */}
              {vb !== null && vb > 0 && (
                <div
                  className="absolute left-0 right-0 transition-state"
                  style={{
                    bottom: 0,
                    height: `${vb * 100}%`,
                    borderTop: "2px dashed #1f2937",
                    pointerEvents: "none",
                  }}
                />
              )}
            </div>
            <div className="text-small font-medium text-gray-900 mt-1 tabular-nums">
              {(v * 100).toFixed(0)}%
            </div>
            {delta !== null && (
              <div
                className={`text-small tabular-nums ${
                  Math.abs(delta) < 0.005
                    ? "text-gray-400"
                    : delta > 0
                    ? "text-accent"
                    : "text-gray-500"
                }`}
              >
                {delta >= 0 ? "+" : ""}
                {(delta * 100).toFixed(0)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ResultPanel({ resp }: { resp: QueryResponse }) {
  return (
    <div className="space-y-8">
      <TrustGauge
        state={resp.trust_state}
        pHatType={resp.p_hat_type}
        pHatZone={resp.p_hat_zone}
        interventionType={PITCH_TYPE_NAMES[resp.intervention_type] ?? resp.intervention_type}
      />

      <div>
        <SectionLabel>What the model expects at this position</SectionLabel>
        <div className="space-y-1.5">
          {Object.entries(resp.expected_distribution.pitch_type_probs)
            .sort((a, b) => b[1] - a[1])
            .map(([pt, p]) => (
              <div key={pt} className="flex items-center gap-4">
                <div className="w-20">
                  <PitchBadge type={pt} />
                </div>
                <ProbBar value={p} label={p.toFixed(3)} width="w-48" />
              </div>
            ))}
        </div>
        <Caption>
          The model's predicted pitch mix given the situation — type marginal across the 7 canonical pitches.
        </Caption>
      </div>

      {resp.counterfactual && (
        <div>
          <SectionLabel>
            What if: {resp.intervention_type}
            {resp.intervention_zone !== null && (
              <span className="text-gray-400 normal-case ml-2">
                · zone {resp.intervention_zone}
                {resp.intervention_zone === 25 ? " (OOZ)" : ""}
              </span>
            )}{" "}
            substituted at this position
          </SectionLabel>
          <div className="space-y-6">
            {/* Low-support banner — informational, not a block */}
            {resp.counterfactual.support_level !== "high" && (
              <div
                className={`text-small rounded-md px-4 py-3 leading-snug ${
                  resp.counterfactual.support_level === "low"
                    ? "bg-red-50 text-red-900 border border-red-100"
                    : "bg-amber-50 text-amber-900 border border-amber-100"
                }`}
              >
                <span className="font-medium">
                  {resp.counterfactual.support_level === "low" ? "Low support" : "Moderate support"}
                  :
                </span>{" "}
                {resp.counterfactual.support_level === "low"
                  ? "this intervention is rare in similar situations. The numbers below are model-rollout outputs, not a defensible causal claim. Read them as 'here's what the model expects if forced to play it out.'"
                  : "this intervention is somewhat rare. Treat the effect estimate with care; the underlying rollout is still informative."}
              </div>
            )}
            <div>
              <div className="text-h2 font-semibold text-gray-900 tabular-nums">
                {resp.counterfactual.effect_runs >= 0 ? "+" : ""}
                {resp.counterfactual.effect_runs.toFixed(4)} runs
                <span className="text-small text-gray-500 font-normal ml-3">
                  95% CI [{resp.counterfactual.ci_lower.toFixed(4)},{" "}
                  {resp.counterfactual.ci_upper.toFixed(4)}]
                </span>
                {!resp.counterfactual.is_causal_claim && (
                  <span className="ml-3 text-small text-gray-400 font-normal italic">
                    rollout estimate, not a causal claim
                  </span>
                )}
              </div>
              <Caption>
                {resp.counterfactual.is_causal_claim
                  ? "Effect on expected AB run value vs what the pitcher would normally do. Positive = batter-favored; negative = pitcher-favored."
                  : "Difference between two model rollouts (intervention vs. natural pick). Without enough support in the data, this isn't a causal effect — just a comparison of two simulated futures."}
              </Caption>
            </div>

            <div>
              <div className="text-body font-medium text-gray-900 tabular-nums">
                E-value: {resp.counterfactual.e_value_point.toFixed(2)}{" "}
                <span className="text-small text-gray-500 font-normal">
                  (CI-limit: {resp.counterfactual.e_value_ci_limit.toFixed(2)})
                </span>
              </div>
              <Caption>
                An unmeasured factor would have to have at least this much association with both
                the pitch choice and the outcome to nullify the effect.{" "}
                {resp.counterfactual.e_value_point < 1.5
                  ? "Fragile — any modest hidden factor could explain it."
                  : resp.counterfactual.e_value_point < 3
                  ? "Moderate robustness."
                  : "Robust — only very strong unmeasured confounding could undo this."}
              </Caption>
            </div>

            <div>
              <div className="text-body font-medium text-gray-900 tabular-nums">
                AB length: {resp.counterfactual.baseline_mean_ab_length.toFixed(2)} →{" "}
                {resp.counterfactual.intervention_mean_ab_length.toFixed(2)} pitches
                <span className="text-small text-gray-500 font-normal ml-3">
                  Δ ={" "}
                  {(
                    resp.counterfactual.intervention_mean_ab_length -
                    resp.counterfactual.baseline_mean_ab_length
                  ).toFixed(2)}
                </span>
              </div>
              <Caption>
                Average # of pitches the AB ends in — falls out of the simulation naturally.
              </Caption>
            </div>

            <div>
              <SectionLabel>Outcome distribution under intervention</SectionLabel>
              <OutcomeDistribution
                dist={resp.counterfactual.intervention_outcome_distribution}
                baselineDist={resp.counterfactual.baseline_outcome_distribution}
                highlight
              />
              <Caption>
                Solid bars = intervention; dotted outline = baseline (what would have happened
                naturally). Δ row below shows the pp shift. {resp.counterfactual.n_paths} paths each.
              </Caption>
            </div>
          </div>
        </div>
      )}

      {/* Refusal panel is gone — the continuous-trust-gauge approach (ADR 002
          Option D) folds the "low support" case into the always-shown
          counterfactual block with a labeled banner above. */}

      <div className="text-small text-gray-400">Computed in {resp.timing_seconds}s.</div>
    </div>
  );
}

// ============================================================
// Main app
// ============================================================

function CounterfactualExplorer() {
  const [games, setGames] = useState<GameSummary[]>([]);
  const [loadingGames, setLoadingGames] = useState(true);
  const [selectedGame, setSelectedGame] = useState<GameSummary | null>(null);

  const [atBats, setAtBats] = useState<AtBatSummary[]>([]);
  const [loadingABs, setLoadingABs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selectedAB, setSelectedAB] = useState<AtBatSummary | null>(null);
  const [abContext, setAbContext] = useState<ABContextResponse | null>(null);
  const [loadingContext, setLoadingContext] = useState(false);

  const [interventionPosition, setInterventionPosition] = useState(1);
  const [interventionType, setInterventionType] = useState<PitchType>("SL");
  const [interventionZone, setInterventionZone] = useState<number | null>(null);
  const [nPaths, setNPaths] = useState(200);

  const [running, setRunning] = useState(false);
  const [resp, setResp] = useState<QueryResponse | null>(null);

  // Fetch games once at startup.
  useEffect(() => {
    listGames(200)
      .then((data) => {
        setGames(data.items);
        if (data.items.length > 0) setSelectedGame(data.items[0]);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingGames(false));
  }, []);

  // When the selected game changes, refetch the ABs filtered to that game.
  useEffect(() => {
    if (!selectedGame) {
      setAtBats([]);
      setSelectedAB(null);
      return;
    }
    setLoadingABs(true);
    setSelectedAB(null);
    setAbContext(null);
    setResp(null);
    listAtBats({ limit: 200, min_pitches: 3, max_pitches: 12, game_pk: selectedGame.game_pk })
      .then((data) => {
        setAtBats(data.items);
        if (data.items.length > 0) setSelectedAB(data.items[0]);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingABs(false));
  }, [selectedGame?.game_pk]);

  // Whenever the selected AB changes, fetch the per-AB context (one forward
  // pass through the model giving per-position expected distributions, the
  // pitcher's arsenal, and the batter's whiff heatmap). All subsequent UI
  // reactivity reads from this cache without making more API calls.
  useEffect(() => {
    if (!selectedAB) {
      setAbContext(null);
      return;
    }
    setLoadingContext(true);
    setResp(null); // stale result for a different AB
    setInterventionZone(null);
    setInterventionPosition(1);
    getABContext(selectedAB.game_pk, selectedAB.at_bat_number)
      .then((ctx) => {
        setAbContext(ctx);
        // If the current intervention_type is one the pitcher doesn't throw,
        // auto-switch to the most-common alternative.
        const has = ctx.pitcher_has_pitch[interventionType] ?? false;
        const usage = ctx.pitcher_arsenal_pct[interventionType] ?? 0;
        if (!has || usage < 0.01) {
          const fallback = PITCH_TYPES.find(
            (pt) => (ctx.pitcher_has_pitch[pt] ?? false) && (ctx.pitcher_arsenal_pct[pt] ?? 0) >= 0.05,
          );
          if (fallback) setInterventionType(fallback);
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingContext(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAB?.game_pk, selectedAB?.at_bat_number]);

  const canRun = useMemo(
    () => selectedAB && interventionPosition >= 1 && interventionPosition < selectedAB.n_pitches,
    [selectedAB, interventionPosition],
  );

  const observedPitches = useMemo(() => {
    if (!selectedAB) return [];
    return selectedAB.pitch_types.map((t, i) => ({
      zone: selectedAB.pitch_zones[i] ?? 25,
      type: t,
    }));
  }, [selectedAB]);

  // The position info for the currently-selected intervention position.
  // Drives the "model expects" panel WITHOUT needing a full rollout.
  const currentPositionInfo = useMemo(() => {
    if (!abContext) return null;
    return abContext.positions[interventionPosition] ?? null;
  }, [abContext, interventionPosition]);

  async function onRun() {
    if (!selectedAB) return;
    setRunning(true);
    setError(null);
    try {
      const r = await postQuery({
        game_pk: selectedAB.game_pk,
        at_bat_number: selectedAB.at_bat_number,
        intervention_position: interventionPosition,
        intervention_type: interventionType,
        intervention_zone: interventionZone,
        n_paths: nPaths,
      });
      setResp(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-3 sm:px-6 py-6 sm:py-16">
      <header className="mb-12">
        <div className="text-small font-medium text-gray-500 uppercase tracking-wide mb-2">
          PitchGPT — counterfactual pitch strategy
        </div>
        <h1 className="text-h1 font-semibold text-gray-900 leading-tight">
          What would have happened if the pitcher had chosen differently?
        </h1>
        <p className="text-body text-gray-500 mt-4 max-w-prose leading-relaxed">
          Pick a real at-bat. Pick a pitch in it. Pick an alternative pitch type — and optionally a
          location on the strike zone. We simulate the rest of the at-bat 200 ways and report the
          effect, with honest uncertainty. The system{" "}
          <span className="text-gray-900 font-medium">refuses</span> when the intervention is too
          rare for a causal claim.
        </p>
      </header>

      {/* Top — two-step picker: game first, then AB within that game */}
      <Section>
        <SectionLabel>1. Pick a game</SectionLabel>
        <GamePicker
          games={games}
          selected={selectedGame}
          onSelect={setSelectedGame}
          loading={loadingGames}
        />
      </Section>

      {selectedGame && (
        <Section>
          <SectionLabel>2. Pick an at-bat in this game</SectionLabel>
          <ABPicker
            atBats={atBats}
            selected={selectedAB}
            onSelect={setSelectedAB}
            loading={loadingABs}
          />
        </Section>
      )}

      {/* Scoreboard — minimal Apple-style strip with all the game context */}
      {abContext && (
        <Section>
          <ScoreboardHeader ctx={abContext} />
        </Section>
      )}

      {/* Middle — two-column: controls on left, strike zone on right */}
      {selectedAB && (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_minmax(380px,440px)] gap-12 mt-4">
          <div className="space-y-10">
            <div>
              <SectionLabel>3. The actual pitches</SectionLabel>
              <ObservedABDisplay
                ab={selectedAB}
                abContext={abContext}
                interventionPosition={interventionPosition}
                onPick={setInterventionPosition}
              />
            </div>
            <div>
              <SectionLabel>4. Choose an alternative</SectionLabel>
              <InterventionControls
                interventionType={interventionType}
                onTypeChange={setInterventionType}
                nPaths={nPaths}
                onPathsChange={setNPaths}
                onRun={onRun}
                running={running || !canRun}
                abContext={abContext}
              />
            </div>
            {/* Live, no-rollout preview of what the model expects at this position */}
            {currentPositionInfo?.expected_pitch_type_probs && (
              <div>
                <SectionLabel>
                  Live: model's prediction at pitch #{interventionPosition}
                </SectionLabel>
                <div className="space-y-1.5">
                  {Object.entries(currentPositionInfo.expected_pitch_type_probs)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([pt, p]) => (
                      <div key={pt} className="flex items-center gap-4">
                        <div className="w-20">
                          <PitchBadge type={pt} />
                        </div>
                        <ProbBar value={p} label={p.toFixed(3)} width="w-48" />
                      </div>
                    ))}
                </div>
                <Caption>
                  Updates instantly as you click a different observed pitch above. No rollout
                  needed for this — just one forward pass cached from when you picked the AB.
                </Caption>
              </div>
            )}
          </div>
          <div className="lg:sticky lg:top-8 self-start">
            <StrikeZone
              selected={interventionZone}
              onSelect={setInterventionZone}
              batterStand={selectedAB.batter_stand}
              observedPitches={observedPitches}
              highlightedPitchIndex={interventionPosition}
              batterWhiffGrid={abContext?.batter_whiff_grid}
              disabled={running}
            />
            {loadingContext && (
              <div className="text-small text-gray-500 mt-3 text-center">Loading context…</div>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="my-8 bg-red-50 border border-red-100 rounded-md p-4 text-small text-red-900">
          {error}
        </div>
      )}

      {resp && (
        <Section>
          <SectionLabel>5. Result</SectionLabel>
          <ResultPanel resp={resp} />
        </Section>
      )}

      <footer className="mt-24 pt-8 border-t border-gray-200 text-small text-gray-400">
        Calibrated · cross-fit ready · honest about uncertainty.
      </footer>
    </div>
  );
}

// ============================================================
// Top-level: tab switcher between the counterfactual explorer and Tab 1
// (Pitcher Profile Inspector). Sprint 1 adds the first sibling-tab; subsequent
// sprints add tabs 2-5. The switcher is intentionally minimal — no router yet.
// ============================================================

type TopTab = "counterfactual" | "pitcher_profile" | "rollout_viewer" | "mcsim" | "score_prediction";

function TabSwitcher({ tab, onChange }: { tab: TopTab; onChange: (t: TopTab) => void }) {
  const tabs: { id: TopTab; label: string }[] = [
    { id: "counterfactual", label: "Counterfactual explorer" },
    { id: "rollout_viewer", label: "AB rollout viewer" },
    { id: "pitcher_profile", label: "Pitcher profile" },
    { id: "mcsim", label: "Matchup cards" },
    { id: "score_prediction", label: "Score predictions" },
  ];
  const [showFade, setShowFade] = useState(true);
  const checkFade = useCallback((el: HTMLElement | null) => {
    if (!el) return;
    setShowFade(el.scrollLeft + el.clientWidth < el.scrollWidth - 8);
  }, []);
  const navRef = useRef<HTMLElement | null>(null);
  return (
    <div className="relative mb-8">
      <nav
        className="border-b border-gray-200 overflow-x-auto"
        style={{ scrollbarWidth: "none" }}
        ref={(el) => { navRef.current = el; checkFade(el); }}
        onScroll={(e) => checkFade(e.currentTarget)}
      >
        <div className="max-w-6xl mx-auto px-3 sm:px-6 flex gap-1 sm:gap-2 min-w-max">
          {tabs.map((t) => {
            const active = t.id === tab;
            return (
              <button
                key={t.id}
                onClick={() => onChange(t.id)}
                className={[
                  "px-2.5 sm:px-4 py-3 text-xs sm:text-body font-medium transition-state border-b-2 whitespace-nowrap",
                  active
                    ? "text-gray-900 border-accent"
                    : "text-gray-500 border-transparent hover:text-gray-900",
                ].join(" ")}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </nav>
      {showFade && (
        <div
          className="pointer-events-none absolute right-0 top-0 bottom-0 w-10 sm:hidden flex items-center justify-end pr-1"
          style={{ background: "linear-gradient(to right, rgba(255,255,255,0), white 70%)" }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-gray-400">
            <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<TopTab>("counterfactual");
  return (
    <>
      <TabSwitcher tab={tab} onChange={setTab} />
      {tab === "counterfactual" && <CounterfactualExplorer />}
      {tab === "rollout_viewer" && (
        <div className="max-w-6xl mx-auto px-3 sm:px-6 py-4 sm:py-8">
          <RolloutViewerTab />
        </div>
      )}
      {tab === "pitcher_profile" && (
        <div className="max-w-6xl mx-auto px-3 sm:px-6 py-4 sm:py-8">
          <PitcherProfileTab />
        </div>
      )}
      {tab === "mcsim" && <MCSimTab />}
      {tab === "score_prediction" && <ScorePredictionTab />}
    </>
  );
}
