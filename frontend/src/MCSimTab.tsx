import { useEffect, useMemo, useState } from "react";

import { getMcsimCard, listMcsimPredictions } from "./api";
import { PitchGlyph } from "./shared/ui";
import type {
  McsimCard,
  McsimCardResponse,
  McsimCell,
  McsimPredictionSummary,
  McsimRow,
  PitchType,
} from "./types";

// Headline metric is projected OPS (a stronger cross-cell discriminator than
// median run value, which pins to the out-value when most paths are outs).
// Tint cells red (pitcher-favorable, low OPS) -> white (~league avg) -> green (hitter-favorable).
function opsTint(ops: number | null | undefined): string {
  if (ops == null) return "transparent";
  const lo = 0.4;
  const hi = 1.1;
  const mid = 0.7;
  if (ops <= mid) {
    const a = (((mid - ops) / (mid - lo)) * 0.3).toFixed(3);
    return `rgba(230,90,90,${a})`; // light red — pitcher dominates
  }
  const a = (((ops - mid) / (hi - mid)) * 0.3).toFixed(3);
  return `rgba(74,190,120,${a})`; // light green — hitter-favorable
}

function reachedBase(eventType: string | null): boolean {
  if (!eventType) return false;
  return ["single", "double", "triple", "home_run", "walk", "hit_by_pitch"].includes(
    eventType,
  );
}

// AVG / OBP / SLG / OPS / BB% / K% derived from the per-PA outcome distribution
// (the same 7 class probabilities the rollout produces). AB = 1 - P(BB);
// strikeouts (K) and balls-in-play outs (out) are both at-bats.
export function deriveStats(d: Record<string, number> | undefined) {
  const g = (k: string) => (d && d[k] != null ? d[k] : 0);
  const bb = g("BB");
  const k = g("K");
  const hits = g("1B") + g("2B") + g("3B") + g("HR");
  const ab = 1 - bb;
  const tb = g("1B") + 2 * g("2B") + 3 * g("3B") + 4 * g("HR");
  const avg = ab > 0 ? hits / ab : 0;
  const obp = hits + bb;
  const slg = ab > 0 ? tb / ab : 0;
  return { avg, obp, slg, ops: obp + slg, bb, k };
}

// Baseball convention: drop the leading zero for sub-1.000 rate stats (.275),
// keep it for OPS that can exceed 1 (1.232).
function fmt3(x: number): string {
  const s = x.toFixed(3);
  return x < 1 ? s.replace(/^0/, "") : s;
}
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

function CellButton({
  cell,
  selected,
  onClick,
}: {
  cell: McsimCell;
  selected: boolean;
  onClick: () => void;
}) {
  const ops = cell.predicted_ops;
  const s = deriveStats(cell.predicted_outcome_dist as Record<string, number>);
  const acted = cell.actual && cell.actual.pa_count > 0 ? cell.actual : null;
  return (
    <button
      onClick={onClick}
      style={{ backgroundColor: opsTint(ops) }}
      className={[
        "relative w-full h-[56px] sm:h-[68px] px-1 sm:px-1.5 py-0.5 sm:py-1 flex flex-col items-center justify-center gap-0 sm:gap-0.5 border-r border-b border-gray-100 transition-state tabular-nums leading-none",
        selected ? "ring-2 ring-accent ring-inset" : "hover:brightness-95",
      ].join(" ")}
      title={`${cell.batter_name}: AVG ${fmt3(s.avg)} OPS ${fmt3(s.ops)} BB ${pct(s.bb)} K ${pct(s.k)}`}
    >
      <span className="text-[11px] sm:text-small font-medium text-gray-900">{fmt3(s.avg)}</span>
      <span className="text-[11px] sm:text-[13px] font-semibold text-gray-900">{fmt3(s.ops)}</span>
      <span className="text-[8px] sm:text-[10px] text-gray-500">
        BB {pct(s.bb)} · K {pct(s.k)}
      </span>
      <span className="flex items-center gap-0.5 sm:gap-1">
        <PitchGlyph type={cell.modal_type as PitchType} dim />
        {acted ? (
          <span className="flex gap-0.5">
            {acted.events.slice(0, 4).map((e, i) => (
              <span
                key={i}
                className="inline-block w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full"
                style={{
                  backgroundColor: reachedBase(e.event_type) ? "#FF6B35" : "#86868B",
                }}
              />
            ))}
          </span>
        ) : null}
      </span>
    </button>
  );
}

function StaffGrid({
  title,
  rows,
  selectedKey,
  onSelect,
}: {
  title: string;
  rows: McsimRow[];
  selectedKey: string | null;
  onSelect: (row: McsimRow, cell: McsimCell) => void;
}) {
  if (rows.length === 0) return null;
  const hitters = rows[0].cells.map((c) => c.batter_name);
  return (
    <div className="mb-10">
      <div className="text-data-label mb-2">{title}</div>
      <div className="overflow-x-auto border border-gray-200 rounded-xl">
        <table className="border-collapse w-full">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-white text-left text-[11px] sm:text-small font-medium text-gray-500 px-2 sm:px-3 py-1.5 sm:py-2 border-r border-b border-gray-200">
                Pitcher
              </th>
              {hitters.map((h, i) => (
                <th
                  key={i}
                  className="text-[11px] sm:text-small font-medium text-gray-500 px-1 sm:px-2 py-1.5 sm:py-2 border-r border-b border-gray-200 whitespace-nowrap min-w-[88px] sm:min-w-[112px]"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.pitcher_id}>
                <td className="sticky left-0 z-10 bg-white text-[11px] sm:text-small text-gray-900 px-2 sm:px-3 py-1 border-r border-b border-gray-100 whitespace-nowrap">
                  {row.is_starter ? <span className="text-accent mr-1">★</span> : null}
                  {row.name}
                  <span className="text-gray-400 ml-1">{row.throws}HP</span>
                </td>
                {row.cells.map((cell) => {
                  const key = `${row.pitcher_id}:${cell.batter_id}`;
                  return (
                    <td key={cell.batter_id} className="p-0 min-w-[88px] sm:min-w-[112px]">
                      <CellButton
                        cell={cell}
                        selected={selectedKey === key}
                        onClick={() => onSelect(row, cell)}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CellDetail({
  row,
  cell,
}: {
  row: McsimRow;
  cell: McsimCell;
}) {
  const fmt = (v: number | null | undefined, d = 3) =>
    v == null ? "—" : v.toFixed(d);
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-card">
      <div className="text-body font-medium text-gray-900 mb-4">
        {row.is_starter ? <span className="text-accent mr-1">★</span> : null}
        {row.name} <span className="text-gray-400">({row.throws}HP)</span> vs {cell.batter_name}
      </div>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <Metric label="Proj. OPS" value={fmt(cell.predicted_ops, 3)} />
        <Metric label="Proj. OBP" value={fmt(cell.predicted_obp, 3)} />
        <Metric label="Proj. SLG" value={fmt(cell.predicted_slg, 3)} />
      </div>
      <div className="text-small text-gray-500 mb-1">
        Run value (median): <span className="tabular-nums text-gray-900">{fmt(cell.predicted_rv_median, 3)}</span>{" "}
        <span className="text-gray-400">
          ({fmt(cell.predicted_rv_p05, 2)} … {fmt(cell.predicted_rv_p95, 2)})
        </span>
      </div>
      <div className="text-small text-gray-500 mb-1">
        Most likely: <span className="text-gray-900">{cell.predicted_top1_outcome}</span> · modal pitch{" "}
        <span className="inline-flex align-middle"><PitchGlyph type={cell.modal_type as PitchType} /></span>{" "}
        at π̂ {cell.p_hat_top_type.toFixed(2)} · trust {cell.trust_state}
      </div>
      {cell.actual && cell.actual.pa_count > 0 ? (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="text-data-label mb-2">Actual ({cell.actual.pa_count} PA)</div>
          {cell.actual.events.map((e, i) => (
            <div key={i} className="text-small text-gray-900 mb-1">
              inn {e.inning}
              {e.half ? e.half[0] : ""} · <span className="font-medium">{e.event}</span>
              {e.pitch_types.length ? (
                <span className="text-gray-400"> · {e.pitch_types.join(" ")}</span>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-4 pt-4 border-t border-gray-200 text-small text-gray-400">
          No plate appearance yet (game not played, or this matchup didn't occur).
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-data-label">{label}</div>
      <div className="text-stat text-gray-900 tabular-nums" style={{ fontSize: "28px" }}>
        {value}
      </div>
    </div>
  );
}

export default function MCSimTab() {
  const [date, setDate] = useState("2026-06-04");
  const [list, setList] = useState<McsimPredictionSummary[] | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [selectedPk, setSelectedPk] = useState<number | null>(null);
  const [resp, setResp] = useState<McsimCardResponse | null>(null);
  const [loadingCard, setLoadingCard] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sel, setSel] = useState<{ row: McsimRow; cell: McsimCell } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoadingList(true);
    setError(null);
    setList(null);
    setSelectedPk(null);
    setResp(null);
    setSel(null);
    listMcsimPredictions(date)
      .then((r) => {
        if (cancelled) return;
        setList(r.predictions);
        if (r.predictions.length > 0) setSelectedPk(r.predictions[0].game_pk);
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoadingList(false));
    return () => {
      cancelled = true;
    };
  }, [date]);

  useEffect(() => {
    if (selectedPk == null) return;
    let cancelled = false;
    setLoadingCard(true);
    setResp(null);
    setSel(null);
    getMcsimCard(selectedPk, date)
      .then((r) => !cancelled && setResp(r))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoadingCard(false));
    return () => {
      cancelled = true;
    };
  }, [selectedPk, date]);

  const card: McsimCard | null = resp?.card ?? null;
  const homeStaff = useMemo(
    () => (card ? card.rows.filter((r) => r.team === card.home_team) : []),
    [card],
  );
  const awayStaff = useMemo(
    () => (card ? card.rows.filter((r) => r.team === card.away_team) : []),
    [card],
  );
  const selectedKey = sel ? `${sel.row.pitcher_id}:${sel.cell.batter_id}` : null;

  return (
    <div className="w-full px-3 sm:px-6 py-4 sm:py-8">
      <div className="text-section mb-1">Matchup cards</div>
      <p className="text-body text-gray-500 mb-4">
        Pre-game scouting grid — every rostered pitcher against every opposing hitter, in a
        neutral count. Cells show projected OPS. Once a game finishes, the real plate appearances
        are stamped on each cell.
      </p>

      <div className="flex items-center gap-3 mb-6">
        <label className="text-data-label">Date</label>
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="border border-gray-200 rounded-lg px-3 py-1.5 text-body"
        />
      </div>

      {error ? <div className="text-small text-gauge-red mb-4">{error}</div> : null}
      {loadingList ? <div className="text-small text-gray-400">Loading games…</div> : null}
      {list && list.length === 0 ? (
        <div className="text-body text-gray-400">No matchup cards stored for {date}.</div>
      ) : null}

      {list && list.length > 0 ? (
        <div className="flex gap-2 overflow-x-auto pb-2 mb-8">
          {list.map((g) => {
            const active = g.game_pk === selectedPk;
            return (
              <button
                key={g.game_pk}
                onClick={() => setSelectedPk(g.game_pk)}
                className={[
                  "shrink-0 text-left px-4 py-3 rounded-xl border transition-state",
                  active
                    ? "border-accent bg-accent-subtle"
                    : "border-gray-200 hover:border-gray-300",
                ].join(" ")}
              >
                <div className="text-small font-medium text-gray-900 whitespace-nowrap">
                  {g.away_team} @ {g.home_team}
                </div>
                <div className="text-small text-gray-400">
                  {g.n_cells ?? "?"} cells {g.has_actual ? "· final" : ""}
                </div>
              </button>
            );
          })}
        </div>
      ) : null}

      {loadingCard ? <div className="text-small text-gray-400">Loading card…</div> : null}

      {resp && card ? (
        <>
          {resp.has_actual ? (
            <div className="bg-white border border-gray-200 rounded-xl p-4 mb-6 shadow-card">
              <span className="text-data-label mr-3">Final</span>
              <span className="text-body text-gray-900">
                {card.away_team} {resp.final_score_away} — {resp.final_score_home} {card.home_team}
              </span>
              {resp.winner ? (
                <span className="text-small text-gray-400 ml-3">
                  ({resp.winner === "home" ? card.home_team : resp.winner === "away" ? card.away_team : "tie"})
                </span>
              ) : null}
              {resp.unmatched_event_count > 0 ? (
                <span className="text-small text-gray-400 ml-3">
                  · {resp.unmatched_event_count} PA outside the roster grid
                </span>
              ) : null}
            </div>
          ) : (
            <div className="text-small text-gray-400 mb-6">
              Game not yet played — predictions only.
            </div>
          )}

          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-small text-gray-500 mb-4">
            <span className="font-medium text-gray-600">Cell tint (proj. OPS):</span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-4 rounded" style={{ background: "rgba(230,90,90,0.25)" }} />
              Pitcher-dominant
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-4 rounded border border-gray-200" style={{ background: "white" }} />
              Neutral
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-4 rounded" style={{ background: "rgba(74,190,120,0.25)" }} />
              Hitter-favorable
            </span>
            <span className="mx-1 text-gray-300">|</span>
            <span className="font-medium text-gray-600">Modal pitch:</span>
            <span className="flex items-center gap-1">
              <span style={{ color: "#dc2626" }}>●</span> FF
              <span className="ml-1" style={{ color: "#ea580c" }}>◆</span> SI
              <span className="ml-1" style={{ color: "#ca8a04" }}>■</span> FC
              <span className="ml-1" style={{ color: "#0d9488" }}>▲</span> SL
              <span className="ml-1" style={{ color: "#2563eb" }}>★</span> CU
              <span className="ml-1" style={{ color: "#7c3aed" }}>◐</span> CH
              <span className="ml-1" style={{ color: "#475569" }}>✕</span> FS
            </span>
            <span className="mx-1 text-gray-300">|</span>
            <span className="font-medium text-gray-600">Actual PAs:</span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "#FF6B35" }} />
              Reached base
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "#86868B" }} />
              Out
            </span>
          </div>
          <StaffGrid
            title={`${card.home_team} pitchers vs ${card.away_team} hitters`}
            rows={homeStaff}
            selectedKey={selectedKey}
            onSelect={(row, cell) => setSel({ row, cell })}
          />
          <StaffGrid
            title={`${card.away_team} pitchers vs ${card.home_team} hitters`}
            rows={awayStaff}
            selectedKey={selectedKey}
            onSelect={(row, cell) => setSel({ row, cell })}
          />
          {sel ? (
            <div className="mt-4 max-w-xl">
              <CellDetail row={sel.row} cell={sel.cell} />
            </div>
          ) : (
            <div className="text-small text-gray-400 mt-2">
              Select a cell to see the predicted distribution and what actually happened.
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
