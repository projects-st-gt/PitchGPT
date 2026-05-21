import { useEffect, useMemo, useState } from "react";
import { getABContext, listAtBats, listGames } from "./api";
import { Caption, PitchBadge, Section, SectionLabel } from "./shared/ui";
import { PitchZonePlot, type ZoneDot } from "./shared/PitchZonePlot";
import {
  PITCH_TYPES,
  PITCH_TYPE_GLYPHS,
  type ABContextResponse,
  type AtBatSummary,
  type GameSummary,
  type PositionInfo,
} from "./types";

// Human-readable labels for the 7 result classes (RESULT_CLASSES, data/dataset.py).
const RESULT_LABELS: Record<string, string> = {
  ball: "Ball",
  called_strike: "Called strike",
  swinging_strike: "Swinging strike",
  foul: "Foul",
  in_play_out: "In play — out",
  in_play_hit: "In play — hit",
  in_play_hr: "In play — home run",
};

const OFF_MODEL_PCT = 10; // mirrors OFF_MODEL_PROB_FLOOR (0.10) in inference/api.py

// ============================================================
// Helpers
// ============================================================

function topPick(probs: Record<string, number> | null): string | null {
  if (!probs) return null;
  let best: string | null = null;
  let bestV = -1;
  for (const [k, v] of Object.entries(probs)) {
    if (v > bestV) {
      bestV = v;
      best = k;
    }
  }
  return best;
}

function Bar({ value, highlight = false }: { value: number; highlight?: boolean }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-state"
        style={{ width: `${pct}%`, backgroundColor: highlight ? "#0d9488" : "#9ca3af" }}
      />
    </div>
  );
}

function DistributionBars({
  probs,
  order,
  labelFor,
  glyphFor,
  actual,
}: {
  probs: Record<string, number>;
  order: string[];
  labelFor: (k: string) => string;
  glyphFor?: (k: string) => { color: string; glyph: string } | null;
  actual: string | null;
}) {
  const sorted = [...order].sort((a, b) => (probs[b] ?? 0) - (probs[a] ?? 0));
  return (
    <div className="space-y-1.5">
      {sorted.map((key) => {
        const v = probs[key] ?? 0;
        const isActual = key === actual;
        const g = glyphFor?.(key);
        return (
          <div key={key} className="flex items-center gap-3">
            <div className="w-36 flex items-center gap-1.5 text-small">
              {g && (
                <span aria-hidden="true" style={{ color: g.color }}>
                  {g.glyph}
                </span>
              )}
              <span className={isActual ? "text-gray-900 font-medium" : "text-gray-600"}>
                {labelFor(key)}
              </span>
              {isActual && <span className="text-xs text-accent font-medium">· actual</span>}
            </div>
            <Bar value={v} highlight={isActual} />
            <span className="text-small tabular-nums text-gray-500 w-12 text-right">
              {(v * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Pitch strip (scrubber)
// ============================================================

function PitchStrip({
  positions,
  cursor,
  onPick,
}: {
  positions: PositionInfo[];
  cursor: number;
  onPick: (k: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {positions.map((p) => {
        const k = p.position;
        const isCursor = k === cursor;
        const predictable = k >= 1;
        const known = p.pitch_type in PITCH_TYPE_GLYPHS;
        const { color, glyph } = known
          ? PITCH_TYPE_GLYPHS[p.pitch_type as (typeof PITCH_TYPES)[number]]
          : { color: "#9ca3af", glyph: "·" };
        return (
          <button
            key={k}
            disabled={!predictable}
            onClick={() => onPick(k)}
            title={
              predictable
                ? `Pitch #${k}`
                : "Pitch #0 has no prediction — no prior-pitch history"
            }
            className={[
              "px-2.5 py-1.5 rounded-md border-hairline transition-state flex items-center gap-1.5",
              !predictable
                ? "bg-gray-50 cursor-not-allowed opacity-60"
                : isCursor
                ? "border-accent bg-accent-subtle"
                : "bg-white hover:border-gray-300",
            ].join(" ")}
          >
            <span className="text-xs text-gray-400 tabular-nums">#{k}</span>
            <span aria-hidden="true" style={{ color }}>
              {glyph}
            </span>
            <span className="text-xs font-medium text-gray-900">{p.pitch_type}</span>
            {p.is_surprising && (
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" aria-hidden="true" />
            )}
          </button>
        );
      })}
    </div>
  );
}

// ============================================================
// The two panes
// ============================================================

function ActualPane({
  positions,
  cursor,
  szTop,
  szBot,
}: {
  positions: PositionInfo[];
  cursor: number;
  szTop: number;
  szBot: number;
}) {
  const shown = positions.filter((p) => p.position <= cursor);
  const dots: ZoneDot[] = shown
    .filter((p) => p.plate_x !== null && p.plate_z !== null)
    .map((p) => ({
      x: p.plate_x as number,
      z: p.plate_z as number,
      type: p.pitch_type,
      label: String(p.position),
      focus: p.position === cursor,
    }));
  return (
    <div className="border-hairline rounded-md p-4 bg-white">
      <div className="text-small font-medium text-gray-900 mb-1">
        Actual — pitches through #{cursor}
      </div>
      <div className="text-xs text-gray-500 mb-3">
        precise location · the focused pitch is solid, earlier pitches faded
      </div>
      <PitchZonePlot szTop={szTop} szBot={szBot} dots={dots} />
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        {shown.map((p) => (
          <div
            key={p.position}
            className={p.position === cursor ? "" : "opacity-50"}
          >
            <span className="text-xs text-gray-400 tabular-nums mr-1">#{p.position}</span>
            <PitchBadge type={p.pitch_type} />
          </div>
        ))}
      </div>
    </div>
  );
}

function PredictedPane({ pos }: { pos: PositionInfo }) {
  return (
    <div className="border-hairline rounded-md p-4 bg-white">
      <div className="text-small font-medium text-gray-900 mb-1">
        Predicted — pitch #{pos.position}
      </div>
      <div className="text-xs text-gray-500 mb-3">
        the model's call given the count and pitches #0–{pos.position - 1}
      </div>
      {pos.expected_zone_probs ? (
        <PitchZonePlot
          szTop={3.5}
          szBot={1.6}
          heatmap={pos.expected_zone_probs}
        />
      ) : (
        <div className="text-small text-gray-400">no zone prediction</div>
      )}
      {pos.expected_pitch_type_probs && (
        <div className="mt-3">
          <DistributionBars
            probs={pos.expected_pitch_type_probs}
            order={PITCH_TYPES}
            labelFor={(k) => k}
            glyphFor={(k) =>
              k in PITCH_TYPE_GLYPHS
                ? PITCH_TYPE_GLYPHS[k as (typeof PITCH_TYPES)[number]]
                : null
            }
            actual={pos.pitch_type}
          />
        </div>
      )}
    </div>
  );
}

// ============================================================
// Verdict + detail
// ============================================================

function Verdict({ pos }: { pos: PositionInfo }) {
  const modelTop = topPick(pos.expected_pitch_type_probs);
  const modelTopP = modelTop ? pos.expected_pitch_type_probs?.[modelTop] ?? 0 : 0;
  const actualP =
    pos.expected_pitch_type_probs?.[pos.pitch_type] ?? null;
  const matched = modelTop === pos.pitch_type;

  return (
    <div
      className={[
        "rounded-md px-4 py-3 text-small leading-relaxed",
        pos.is_surprising ? "bg-amber-50 text-amber-900" : "bg-gray-50 text-gray-700",
      ].join(" ")}
    >
      <span className="font-medium">Model's call:</span>{" "}
      {modelTop ?? "—"} ({(modelTopP * 100).toFixed(0)}%).{" "}
      <span className="font-medium">Thrown:</span> {pos.pitch_type}
      {actualP !== null && <> (model gave it {(actualP * 100).toFixed(0)}%)</>}.{" "}
      {matched ? (
        <span className="font-medium">The model's top pick matched.</span>
      ) : pos.is_surprising ? (
        <span className="font-medium">
          Off-model — the model gave this pitch under {OFF_MODEL_PCT}%. A surprise
          like this is evidence that something the model can't see (a catcher's
          read, an in-game adjustment) was in play — not a reason the model can name.
        </span>
      ) : (
        <span className="font-medium">
          On-model — not the top pick, but a likely enough pitch that nothing here
          needs explaining beyond normal variation.
        </span>
      )}
    </div>
  );
}

// ============================================================
// Pickers
// ============================================================

function GameSelect({
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
  return (
    <select
      className="w-full px-4 py-3 text-body border-hairline rounded-md bg-white hover:border-gray-300 focus:outline-none focus:border-accent transition-state"
      value={selected ? String(selected.game_pk) : ""}
      onChange={(e) => {
        const g = games.find((x) => x.game_pk === Number(e.target.value));
        if (g) onSelect(g);
      }}
    >
      {games.map((g) => (
        <option key={g.game_pk} value={g.game_pk}>
          {g.game_date} · {g.away_team ?? "?"} @ {g.home_team ?? "?"} · {g.n_at_bats} AB
        </option>
      ))}
    </select>
  );
}

function ABSelect({
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
    return <div className="text-small text-gray-500">No at-bats in this game.</div>;
  return (
    <select
      className="w-full px-4 py-3 text-body border-hairline rounded-md bg-white hover:border-gray-300 focus:outline-none focus:border-accent transition-state"
      value={selected ? String(selected.at_bat_number) : ""}
      onChange={(e) => {
        const ab = atBats.find((x) => x.at_bat_number === Number(e.target.value));
        if (ab) onSelect(ab);
      }}
    >
      {atBats.map((ab) => (
        <option key={ab.at_bat_number} value={ab.at_bat_number}>
          AB #{ab.at_bat_number} · {ab.pitcher_name ?? `P${ab.pitcher_id}`} →{" "}
          {ab.batter_name ?? `B${ab.batter_id}`} · {ab.pitch_types.join(" ")} →{" "}
          {ab.terminal_event ?? "?"}
        </option>
      ))}
    </select>
  );
}

// ============================================================
// Main tab
// ============================================================

export default function RolloutViewerTab() {
  const [games, setGames] = useState<GameSummary[]>([]);
  const [loadingGames, setLoadingGames] = useState(true);
  const [selectedGame, setSelectedGame] = useState<GameSummary | null>(null);

  const [atBats, setAtBats] = useState<AtBatSummary[]>([]);
  const [loadingABs, setLoadingABs] = useState(false);
  const [selectedAB, setSelectedAB] = useState<AtBatSummary | null>(null);

  const [ctx, setCtx] = useState<ABContextResponse | null>(null);
  const [loadingCtx, setLoadingCtx] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // cursor = the pitch index currently being predicted / focused (>= 1).
  const [cursor, setCursor] = useState(1);

  useEffect(() => {
    listGames(200)
      .then((d) => {
        setGames(d.items);
        if (d.items.length > 0) setSelectedGame(d.items[0]);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingGames(false));
  }, []);

  useEffect(() => {
    if (!selectedGame) return;
    setLoadingABs(true);
    setSelectedAB(null);
    setCtx(null);
    listAtBats({ limit: 200, min_pitches: 3, max_pitches: 12, game_pk: selectedGame.game_pk })
      .then((d) => {
        setAtBats(d.items);
        if (d.items.length > 0) setSelectedAB(d.items[0]);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingABs(false));
  }, [selectedGame?.game_pk]);

  useEffect(() => {
    if (!selectedAB) return;
    setLoadingCtx(true);
    setError(null);
    setCursor(1);
    getABContext(selectedAB.game_pk, selectedAB.at_bat_number)
      .then(setCtx)
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingCtx(false));
  }, [selectedAB?.game_pk, selectedAB?.at_bat_number]);

  // AB-average strike-zone bounds — Statcast measures sz_top/sz_bot per pitch
  // and they wobble slightly; one box for the whole AB reads cleaner.
  const { szTop, szBot } = useMemo(() => {
    if (!ctx) return { szTop: 3.5, szBot: 1.6 };
    const tops = ctx.positions.map((p) => p.sz_top).filter((v): v is number => v !== null);
    const bots = ctx.positions.map((p) => p.sz_bot).filter((v): v is number => v !== null);
    return {
      szTop: tops.length ? tops.reduce((a, b) => a + b, 0) / tops.length : 3.5,
      szBot: bots.length ? bots.reduce((a, b) => a + b, 0) / bots.length : 1.6,
    };
  }, [ctx]);

  const summary = useMemo(() => {
    if (!ctx) return null;
    const scored = ctx.positions.filter((p) => p.model_confidence !== null);
    const matched = scored.filter(
      (p) => topPick(p.expected_pitch_type_probs) === p.pitch_type,
    ).length;
    const surprises = scored.filter((p) => p.is_surprising).length;
    return { scored: scored.length, matched, surprises };
  }, [ctx]);

  const nPitches = ctx?.positions.length ?? 0;
  const focused = ctx?.positions[cursor] ?? null;

  return (
    <div>
      <header className="mb-10">
        <div className="text-small font-medium text-gray-500 uppercase tracking-wide mb-2">
          PitchGPT — single at-bat rollout viewer
        </div>
        <h1 className="text-h1 font-semibold text-gray-900 leading-tight">
          Watch the model call an at-bat, pitch by pitch
        </h1>
        <p className="text-body text-gray-500 mt-4 max-w-prose leading-relaxed">
          Step through a real at-bat. The left pane shows the pitches actually
          thrown, plotted where they crossed the zone. The right pane shows what
          the model expected for the focused pitch — type and location. The
          off-model flag fires only when the model gave the actual pitch under
          10%.
        </p>
      </header>

      <Section>
        <SectionLabel>1. Pick a game</SectionLabel>
        <GameSelect
          games={games}
          selected={selectedGame}
          onSelect={setSelectedGame}
          loading={loadingGames}
        />
      </Section>

      {selectedGame && (
        <Section>
          <SectionLabel>2. Pick an at-bat</SectionLabel>
          <ABSelect
            atBats={atBats}
            selected={selectedAB}
            onSelect={setSelectedAB}
            loading={loadingABs}
          />
        </Section>
      )}

      {error && (
        <div className="my-8 bg-red-50 border border-red-100 rounded-md p-4 text-small text-red-900">
          {error}
        </div>
      )}

      {loadingCtx && (
        <div className="my-8 text-small text-gray-500">
          Loading at-bat — first request loads the model, ~10s…
        </div>
      )}

      {ctx && !loadingCtx && focused && (
        <>
          <Section>
            <div className="text-small text-gray-500">
              <span className="text-gray-900 font-medium">{ctx.pitcher_name ?? "Pitcher"}</span>{" "}
              ({ctx.pitcher_throws ?? "?"}HP) vs{" "}
              <span className="text-gray-900 font-medium">{ctx.batter_name ?? "Batter"}</span>{" "}
              ({ctx.batter_stand ?? "?"}HB) · {ctx.game_date} · ended in{" "}
              <span className="text-gray-900 font-medium">{ctx.terminal_event ?? "?"}</span>
            </div>
            {summary && (
              <div className="mt-3 flex flex-wrap gap-3">
                <div className="border-hairline rounded-md px-4 py-2 bg-white">
                  <span className="text-small text-gray-500">model top pick matched </span>
                  <span className="text-small font-medium text-gray-900 tabular-nums">
                    {summary.matched}/{summary.scored}
                  </span>
                </div>
                <div className="border-hairline rounded-md px-4 py-2 bg-white">
                  <span className="text-small text-gray-500">off-model pitches </span>
                  <span className="text-small font-medium text-gray-900 tabular-nums">
                    {summary.surprises}/{summary.scored}
                  </span>
                </div>
              </div>
            )}
          </Section>

          <Section>
            <SectionLabel>3. Step through the at-bat</SectionLabel>
            <div className="flex items-center gap-3 mb-4">
              <button
                onClick={() => setCursor((c) => Math.max(1, c - 1))}
                disabled={cursor <= 1}
                className="px-3 py-1.5 text-small rounded-md border-hairline bg-white hover:border-gray-300 disabled:opacity-40 disabled:cursor-not-allowed transition-state"
              >
                ◄ prev
              </button>
              <span className="text-small text-gray-500 tabular-nums">
                pitch #{cursor} of {nPitches - 1}
              </span>
              <button
                onClick={() => setCursor((c) => Math.min(nPitches - 1, c + 1))}
                disabled={cursor >= nPitches - 1}
                className="px-3 py-1.5 text-small rounded-md border-hairline bg-white hover:border-gray-300 disabled:opacity-40 disabled:cursor-not-allowed transition-state"
              >
                next ►
              </button>
              <div className="ml-2">
                <PitchStrip positions={ctx.positions} cursor={cursor} onPick={setCursor} />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ActualPane
                positions={ctx.positions}
                cursor={cursor}
                szTop={szTop}
                szBot={szBot}
              />
              <PredictedPane pos={focused} />
            </div>

            <div className="mt-4">
              <Verdict pos={focused} />
            </div>
            <Caption>
              The predicted-location heatmap is the model's 13-cell zone
              distribution — coarser than the precise dots on the left, because
              13 cells is genuinely all the model predicts. Sharp dots next to a
              blocky heatmap is the honest picture.
            </Caption>
          </Section>

          {focused.expected_result_probs && (
            <Section>
              <SectionLabel>What result the model expected for pitch #{cursor}</SectionLabel>
              <DistributionBars
                probs={focused.expected_result_probs}
                order={Object.keys(RESULT_LABELS)}
                labelFor={(k) => RESULT_LABELS[k] ?? k}
                actual={focused.actual_result}
              />
              <Caption>
                The result head reads the pitch's own type and location. The bar
                marked "actual" is the result that occurred.
              </Caption>
            </Section>
          )}
        </>
      )}

      <footer className="mt-24 pt-8 border-t border-gray-200 text-small text-gray-400">
        Actual pitches at precise plate_x/plate_z · predicted location is the
        model's coarse 13-cell zone · off-model flag = model probability under 10%.
      </footer>
    </div>
  );
}
