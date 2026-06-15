import { useEffect, useMemo, useState } from "react";

import { getGameSimDetail, listGameSimDates, listGameSims } from "./api";
import type { BullpenTeamStats, GameSimDetail, GameSimSummary, PitcherStaff } from "./types";

function wpBar(wp: number): string {
  return `${Math.max(0, Math.min(1, wp)) * 100}%`;
}

function confidenceLabel(wp: number): { text: string; cls: string } {
  const margin = Math.abs(wp - 0.5);
  if (margin < 0.02) return { text: "Toss-up", cls: "bg-gray-100 text-gray-500" };
  if (margin < 0.07) return { text: "Lean", cls: "bg-amber-50 text-amber-700" };
  return { text: "Confident", cls: "bg-green-50 text-green-700" };
}

function GameCard({
  game,
  active,
  onClick,
}: {
  game: GameSimSummary;
  active: boolean;
  onClick: () => void;
}) {
  const wpH = game.win_prob_home;
  const favHome = wpH > 0.5;
  const conf = confidenceLabel(wpH);
  return (
    <button
      onClick={onClick}
      className={[
        "shrink-0 text-left px-4 py-3 rounded-xl border transition-state min-w-[180px]",
        active ? "border-accent bg-accent-subtle" : "border-gray-200 hover:border-gray-300",
      ].join(" ")}
    >
      <div className="flex justify-between items-baseline mb-1">
        <span className={`text-small font-medium ${!favHome ? "text-gray-900" : "text-gray-500"}`}>
          {game.away_team}
        </span>
        {game.has_actual && game.final_score_away != null ? (
          <span className="text-small font-semibold text-gray-900 tabular-nums ml-2">
            {game.final_score_away}
          </span>
        ) : (
          <span className="text-small text-gray-400 tabular-nums ml-2">
            {game.projected_away.toFixed(1)}
          </span>
        )}
      </div>
      <div className="flex justify-between items-baseline mb-2">
        <span className={`text-small font-medium ${favHome ? "text-gray-900" : "text-gray-500"}`}>
          {game.home_team}
        </span>
        {game.has_actual && game.final_score_home != null ? (
          <span className="text-small font-semibold text-gray-900 tabular-nums ml-2">
            {game.final_score_home}
          </span>
        ) : (
          <span className="text-small text-gray-400 tabular-nums ml-2">
            {game.projected_home.toFixed(1)}
          </span>
        )}
      </div>
      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden flex">
        <div className="h-full bg-gray-700 rounded-l-full transition-all" style={{ width: wpBar(game.win_prob_away) }} />
        <div className="h-full bg-accent rounded-r-full transition-all" style={{ width: wpBar(game.win_prob_home) }} />
      </div>
      <div className="flex justify-between items-center mt-1.5">
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${conf.cls}`}>{conf.text}</span>
        {game.has_actual && game.predicted_winner_correct != null ? (
          <span className={`text-[10px] font-medium ${game.predicted_winner_correct ? "text-green-600" : "text-red-500"}`}>
            {game.predicted_winner_correct ? "Correct" : "Wrong"}
          </span>
        ) : (
          <span className="text-[10px] text-gray-400 tabular-nums">
            {(Math.max(game.win_prob_home, game.win_prob_away) * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </button>
  );
}

function inningHeat(runs: number): string {
  if (runs <= 0.3) return "rgba(74,144,217,0.12)";
  if (runs <= 0.5) return "rgba(74,144,217,0.06)";
  if (runs <= 0.7) return "transparent";
  if (runs <= 1.0) return "rgba(255,107,53,0.10)";
  return `rgba(255,107,53,${Math.min(0.35, 0.10 + (runs - 1.0) * 0.12).toFixed(3)})`;
}

function Linescore({ detail }: { detail: GameSimDetail }) {
  const sim = detail.sim;
  const nInnings = Math.max(sim.inning_runs_home.length, 9);
  const innings = Array.from({ length: nInnings }, (_, i) => i);

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-small tabular-nums w-full">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left text-gray-500 font-medium px-3 py-2 w-36">Team</th>
            {innings.map((i) => (
              <th key={i} className="text-center text-gray-400 font-medium px-1.5 py-2 w-10">{i + 1}</th>
            ))}
            <th className="text-center text-gray-500 font-medium px-3 py-2 border-l border-gray-200 w-12">R</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-gray-100">
            <td className="px-3 py-2 text-gray-900 font-medium">{sim.away_team}</td>
            {innings.map((i) => {
              const v = i < sim.inning_runs_away.length ? sim.inning_runs_away[i] : 0;
              return (
                <td key={i} className="text-center px-1.5 py-2 text-gray-700 font-medium" style={{ backgroundColor: inningHeat(v) }}>
                  {v > 0 ? v.toFixed(1) : <span className="text-gray-300">—</span>}
                </td>
              );
            })}
            <td className="text-center px-3 py-2 border-l border-gray-200 font-semibold text-gray-900">
              {sim.projected_score.away.toFixed(1)}
            </td>
          </tr>
          <tr className="border-b border-gray-100">
            <td className="px-3 py-2 text-gray-900 font-medium">{sim.home_team}</td>
            {innings.map((i) => {
              const v = i < sim.inning_runs_home.length ? sim.inning_runs_home[i] : 0;
              return (
                <td key={i} className="text-center px-1.5 py-2 text-gray-700 font-medium" style={{ backgroundColor: inningHeat(v) }}>
                  {v > 0 ? v.toFixed(1) : <span className="text-gray-300">—</span>}
                </td>
              );
            })}
            <td className="text-center px-3 py-2 border-l border-gray-200 font-semibold text-gray-900">
              {sim.projected_score.home.toFixed(1)}
            </td>
          </tr>
          {detail.has_actual && detail.final_score_away != null ? (
            <>
              <tr className="border-t-2 border-gray-300">
                <td colSpan={nInnings + 2} className="px-3 py-1 text-[10px] text-gray-400 font-medium tracking-wider uppercase">
                  Actual final
                </td>
              </tr>
              <tr className="border-b border-gray-100 bg-gray-50">
                <td className="px-3 py-2 text-gray-900 font-medium">{detail.away_team}</td>
                <td colSpan={nInnings} />
                <td className="text-center px-3 py-2 border-l border-gray-200 font-semibold text-gray-900">
                  {detail.final_score_away}
                </td>
              </tr>
              <tr className="bg-gray-50">
                <td className="px-3 py-2 text-gray-900 font-medium">{detail.home_team}</td>
                <td colSpan={nInnings} />
                <td className="text-center px-3 py-2 border-l border-gray-200 font-semibold text-gray-900">
                  {detail.final_score_home}
                </td>
              </tr>
            </>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

function DistBar({ entries, label, highlightKey }: {
  entries: [string, number][];
  label: string;
  highlightKey?: string | null;
}) {
  if (entries.length === 0) return null;
  const maxProb = Math.max(...entries.map(([, p]) => p));
  return (
    <div>
      <div className="text-data-label mb-2">{label}</div>
      <div className="flex items-end gap-px h-20">
        {entries.map(([k, p]) => {
          const highlighted = highlightKey != null && k === highlightKey;
          return (
            <div key={k} className="flex-1 flex flex-col items-center justify-end h-full min-w-0">
              <div
                className={`w-full rounded-t transition-all ${highlighted ? "bg-accent" : "bg-gray-300"}`}
                style={{ height: `${(p / maxProb) * 100}%`, minHeight: p > 0 ? "2px" : 0 }}
                title={`${k}: ${(p * 100).toFixed(1)}%`}
              />
              <span className="text-[9px] text-gray-400 tabular-nums mt-0.5 leading-none">{k}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TotalRunsDist({ detail }: { detail: GameSimDetail }) {
  const dist = detail.sim.total_runs_dist;
  if (!dist) return null;
  const entries: [string, number][] = Object.entries(dist)
    .map(([k, v]) => [k, v] as [string, number])
    .sort((a, b) => Number(a[0]) - Number(b[0]));
  const lo = Math.max(0, Number(entries[0][0]) - 1);
  const hi = Number(entries[entries.length - 1][0]) + 1;
  const filled: [string, number][] = [];
  for (let i = lo; i <= hi; i++) {
    const key = String(i);
    filled.push([key, dist[key] ?? 0]);
  }
  const actualTotal = detail.has_actual && detail.final_score_home != null && detail.final_score_away != null
    ? String((detail.final_score_home ?? 0) + (detail.final_score_away ?? 0))
    : null;
  return <DistBar entries={filled} label="Total runs distribution" highlightKey={actualTotal} />;
}

function MarginDist({ detail }: { detail: GameSimDetail }) {
  const dist = detail.sim.margin_dist;
  if (!dist) return null;
  const entries: [string, number][] = Object.entries(dist)
    .map(([k, v]) => [k, v] as [string, number])
    .sort((a, b) => Number(a[0]) - Number(b[0]));
  const lo = Number(entries[0][0]);
  const hi = Number(entries[entries.length - 1][0]);
  const filled: [string, number][] = [];
  for (let i = lo; i <= hi; i++) {
    const key = String(i);
    filled.push([key, dist[key] ?? 0]);
  }
  const actualMargin = detail.has_actual && detail.final_score_home != null && detail.final_score_away != null
    ? String((detail.final_score_home ?? 0) - (detail.final_score_away ?? 0))
    : null;
  return <DistBar entries={filled} label={`Score margin (+ = ${detail.sim.home_team})`} highlightKey={actualMargin} />;
}

function BigMetric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-data-label">{label}</div>
      <div className="text-stat text-gray-900 tabular-nums" style={{ fontSize: "28px" }}>{value}</div>
      {sub ? <div className="text-small text-gray-400">{sub}</div> : null}
    </div>
  );
}

function ScoreError({ detail }: { detail: GameSimDetail }) {
  if (!detail.has_actual || detail.final_score_home == null || detail.final_score_away == null) return null;
  const sim = detail.sim;
  const errH = sim.projected_score.home - detail.final_score_home;
  const errA = sim.projected_score.away - detail.final_score_away;
  const errTotal = (sim.projected_score.home + sim.projected_score.away) - (detail.final_score_home + detail.final_score_away);
  const fmt = (v: number) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1));
  return (
    <div className="mt-4 pt-4 border-t border-gray-200">
      <div className="text-data-label mb-2">Prediction error</div>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <div className="text-small text-gray-500">{sim.away_team}</div>
          <div className="text-body tabular-nums text-gray-900">
            {sim.projected_score.away.toFixed(1)} vs {detail.final_score_away}{" "}
            <span className={Math.abs(errA) <= 1.5 ? "text-green-600" : Math.abs(errA) <= 3 ? "text-amber-600" : "text-red-500"}>
              ({fmt(errA)})
            </span>
          </div>
        </div>
        <div>
          <div className="text-small text-gray-500">{sim.home_team}</div>
          <div className="text-body tabular-nums text-gray-900">
            {sim.projected_score.home.toFixed(1)} vs {detail.final_score_home}{" "}
            <span className={Math.abs(errH) <= 1.5 ? "text-green-600" : Math.abs(errH) <= 3 ? "text-amber-600" : "text-red-500"}>
              ({fmt(errH)})
            </span>
          </div>
        </div>
        <div>
          <div className="text-small text-gray-500">Total</div>
          <div className="text-body tabular-nums text-gray-900">
            {sim.projected_total_runs.toFixed(1)} vs {detail.final_score_home + detail.final_score_away}{" "}
            <span className={Math.abs(errTotal) <= 2 ? "text-green-600" : Math.abs(errTotal) <= 4 ? "text-amber-600" : "text-red-500"}>
              ({fmt(errTotal)})
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function HookInningDist({ dist }: { dist: Record<string, number> }) {
  const entries = Object.entries(dist).sort((a, b) => Number(a[0]) - Number(b[0]));
  const maxPct = Math.max(...entries.map(([, p]) => p));
  return (
    <div className="flex items-end gap-px h-10 mt-1">
      {entries.map(([inn, pct]) => (
        <div key={inn} className="flex-1 flex flex-col items-center justify-end h-full min-w-0">
          <div
            className="w-full rounded-t bg-gray-400"
            style={{ height: `${(pct / maxPct) * 100}%`, minHeight: pct > 0 ? "2px" : 0 }}
            title={`Inn ${inn}: ${(pct * 100).toFixed(0)}%`}
          />
          <span className="text-[8px] text-gray-400 mt-0.5 leading-none">{inn}</span>
        </div>
      ))}
    </div>
  );
}

function PitchingStaffPanel({
  teamName,
  staff,
  starterName,
  starterWorkload,
  bullpenStats,
}: {
  teamName: string;
  staff: PitcherStaff[];
  starterName?: string;
  starterWorkload?: number;
  bullpenStats?: BullpenTeamStats;
}) {
  const starter = staff.find((p) => p.is_starter);
  const relievers = staff.filter((p) => !p.is_starter);
  const hook = bullpenStats?.starter_hook;
  const relieverUsage = bullpenStats?.relievers_used ?? [];

  const relieverNames: Record<number, string> = {};
  for (const rp of relievers) {
    relieverNames[rp.pitcher_id] = rp.name;
  }

  return (
    <div>
      <div className="text-small text-gray-500 mb-2">{teamName}</div>
      {/* Starter */}
      <div className="mb-3">
        <div className="text-body font-medium text-gray-900">
          {starter?.name ?? starterName ?? "Unknown"}{" "}
          <span className="text-small text-gray-400">
            ({starter?.throws ?? "?"}HP)
          </span>
        </div>
        {hook && hook.hooked_pct > 0 ? (
          <div className="text-small text-gray-500 mt-1">
            Hooked in <span className="font-medium text-gray-700">
              inn {hook.median_hook_inning ?? "?"}</span> (median)
            {" · "}{hook.avg_bf_at_hook ?? starterWorkload} BF
          </div>
        ) : starterWorkload ? (
          <div className="text-small text-gray-400">
            ~{starterWorkload} BF before bullpen
          </div>
        ) : null}
        {hook?.hook_inning_dist ? (
          <div className="mt-1">
            <div className="text-[9px] text-gray-400">Hook inning distribution</div>
            <HookInningDist dist={hook.hook_inning_dist} />
          </div>
        ) : null}
      </div>
      {/* Bullpen usage */}
      {relieverUsage.length > 0 ? (
        <div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">
            Bullpen usage (across {((relieverUsage[0]?.appearances ?? 0) / (relieverUsage[0]?.pct || 1)).toLocaleString(undefined, {maximumFractionDigits: 0})} sims)
          </div>
          <div className="space-y-0.5">
            {relieverUsage.filter((r) => r.pct >= 0.01).map((r) => {
              const name = relieverNames[r.pitcher_id] ?? `#${r.pitcher_id}`;
              const rp = relievers.find((p) => p.pitcher_id === r.pitcher_id);
              return (
                <div key={r.pitcher_id} className="flex items-center gap-2">
                  <div className="text-small text-gray-600 w-36 truncate">
                    {name} <span className="text-gray-400">({rp?.throws ?? "?"}HP)</span>
                  </div>
                  <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gray-400 rounded-full"
                      style={{ width: `${r.pct * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-gray-400 tabular-nums w-8 text-right">
                    {(r.pct * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ) : relievers.length > 0 ? (
        <div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">
            Bullpen ({relievers.length})
          </div>
          <div className="space-y-0.5">
            {relievers.map((rp) => (
              <div key={rp.pitcher_id} className="text-small text-gray-600">
                {rp.name} <span className="text-gray-400">({rp.throws}HP)</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function GameDetail({ detail }: { detail: GameSimDetail }) {
  const sim = detail.sim;
  const wpH = sim.win_prob_home;
  const wpA = sim.win_prob_away;
  const favHome = wpH > 0.5;
  const fav = favHome ? sim.home_team : sim.away_team;
  const dog = favHome ? sim.away_team : sim.home_team;
  const conf = confidenceLabel(wpH);

  return (
    <div className="space-y-6">
      {/* Win probability */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-card">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-data-label">Win probability</span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${conf.cls}`}>{conf.text}</span>
        </div>
        <div className="flex items-center gap-4 mb-3">
          <div className="flex-1">
            <div className="flex justify-between mb-1">
              <span className="text-body font-medium text-gray-900">
                {sim.away_team} <span className="tabular-nums">{(wpA * 100).toFixed(1)}%</span>
              </span>
              <span className="text-body font-medium text-gray-900">
                <span className="tabular-nums">{(wpH * 100).toFixed(1)}%</span> {sim.home_team}
              </span>
            </div>
            <div className="h-4 bg-gray-100 rounded-full overflow-hidden flex">
              <div className="h-full bg-gray-600 rounded-l-full transition-all flex items-center justify-end pr-1" style={{ width: wpBar(wpA) }}>
                {wpA > 0.25 ? <span className="text-[10px] text-white font-medium">{(wpA * 100).toFixed(0)}%</span> : null}
              </div>
              <div className="h-full bg-accent rounded-r-full transition-all flex items-center pl-1" style={{ width: wpBar(wpH) }}>
                {wpH > 0.25 ? <span className="text-[10px] text-white font-medium">{(wpH * 100).toFixed(0)}%</span> : null}
              </div>
            </div>
          </div>
        </div>
        <div className="text-small text-gray-500">
          {fav} favored · {dog} is the underdog
        </div>
        {detail.has_actual && detail.winner ? (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <span className="text-small font-medium text-gray-900">
              Result: {detail.winner === "home" ? detail.home_team : detail.away_team} won{" "}
              {detail.final_score_away}–{detail.final_score_home}
            </span>
            {" "}
            <span className={`text-small font-medium ${
              (detail.winner === "home") === favHome ? "text-green-600" : "text-red-500"
            }`}>
              ({(detail.winner === "home") === favHome ? "Predicted correctly" : "Upset"})
            </span>
          </div>
        ) : null}
      </div>

      {/* Projected scores */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-card">
        <div className="text-data-label mb-4">Projected score</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          <BigMetric label={sim.away_team} value={sim.projected_score.away.toFixed(1)} sub={`median ${sim.median_score.away.toFixed(0)}`} />
          <BigMetric label={sim.home_team} value={sim.projected_score.home.toFixed(1)} sub={`median ${sim.median_score.home.toFixed(0)}`} />
          <BigMetric label="Total" value={sim.projected_total_runs.toFixed(1)} sub={`90% band: ${sim.total_runs_90pct_band[0]}–${sim.total_runs_90pct_band[1]}`} />
          <BigMetric label="Sims" value={sim.n_sims.toLocaleString()} sub={sim.n_backstop > 0 ? `${sim.n_backstop} hit backstop` : undefined} />
        </div>
        <ScoreError detail={detail} />
      </div>

      {/* Distributions */}
      {(sim.total_runs_dist || sim.margin_dist) ? (
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-card">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
            <TotalRunsDist detail={detail} />
            <MarginDist detail={detail} />
          </div>
          {detail.has_actual ? (
            <div className="text-[10px] text-gray-400 mt-3">
              Orange bar = actual result
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Pitching staff */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-card">
        <div className="text-data-label mb-4">Pitching staff</div>
        <div className="grid grid-cols-2 gap-6">
          <PitchingStaffPanel
            teamName={sim.away_team}
            staff={detail.away_staff}
            starterName={sim.away_starter_name}
            starterWorkload={sim.away_starter_workload}
            bullpenStats={sim.bullpen_stats?.away}
          />
          <PitchingStaffPanel
            teamName={sim.home_team}
            staff={detail.home_staff}
            starterName={sim.home_starter_name}
            starterWorkload={sim.home_starter_workload}
            bullpenStats={sim.bullpen_stats?.home}
          />
        </div>
      </div>

      {/* Inning-by-inning linescore */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-card">
        <div className="text-data-label mb-1">Inning-by-inning scoring</div>
        <div className="text-[11px] text-gray-400 mb-4">
          Average runs per inning across {sim.n_sims.toLocaleString()} simulations.
          Warmer = more runs expected in that inning.
        </div>
        <Linescore detail={detail} />
      </div>
    </div>
  );
}

function DaySummary({ games }: { games: GameSimSummary[] }) {
  const withActuals = useMemo(() => games.filter((g) => g.has_actual && g.predicted_winner_correct != null), [games]);
  if (withActuals.length === 0) {
    return (
      <div className="text-small text-gray-400 mb-4">
        {games.length} games — no results yet
      </div>
    );
  }
  const correct = withActuals.filter((g) => g.predicted_winner_correct).length;
  const pct = correct / withActuals.length;
  const avgError = useMemo(() => {
    const errs = withActuals
      .filter((g) => g.final_score_home != null)
      .map((g) => Math.abs(g.projected_home + g.projected_away - (g.final_score_home ?? 0) - (g.final_score_away ?? 0)));
    return errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
  }, [withActuals]);
  return (
    <div className="flex items-center gap-4 text-small text-gray-500 mb-4">
      <span>
        Winner: <span className="font-medium text-gray-900">{correct}/{withActuals.length}</span>{" "}
        ({(pct * 100).toFixed(0)}%)
      </span>
      {avgError != null ? (
        <span>
          Avg total runs error: <span className="font-medium text-gray-900">{avgError.toFixed(1)}</span>
        </span>
      ) : null}
    </div>
  );
}

function AggregateStats({ allGames }: { allGames: Map<string, GameSimSummary[]> }) {
  const all = useMemo(() => {
    const flat: GameSimSummary[] = [];
    allGames.forEach((gs) => flat.push(...gs));
    return flat;
  }, [allGames]);
  const withActuals = useMemo(() => all.filter((g) => g.has_actual && g.predicted_winner_correct != null), [all]);
  if (withActuals.length === 0) return null;

  const correct = withActuals.filter((g) => g.predicted_winner_correct).length;
  const nonTossup = withActuals.filter((g) => Math.abs(g.win_prob_home - 0.5) >= 0.02);
  const correctNT = nonTossup.filter((g) => g.predicted_winner_correct).length;
  const errs = withActuals
    .filter((g) => g.final_score_home != null)
    .map((g) => Math.abs(g.projected_home + g.projected_away - (g.final_score_home ?? 0) - (g.final_score_away ?? 0)));
  const avgErr = errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : null;

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-xl px-5 py-3 mb-6 flex flex-wrap items-center gap-x-6 gap-y-1 text-small">
      <span className="text-gray-500">
        Backtest: <span className="font-semibold text-gray-900">{withActuals.length}</span> games across{" "}
        <span className="font-semibold text-gray-900">{allGames.size}</span> dates
      </span>
      <span className="text-gray-400">|</span>
      <span className="text-gray-500">
        Winner: <span className="font-semibold text-gray-900">{correct}/{withActuals.length}</span>{" "}
        ({(correct / withActuals.length * 100).toFixed(0)}%)
      </span>
      {nonTossup.length > 0 ? (
        <>
          <span className="text-gray-400">|</span>
          <span className="text-gray-500">
            Excl toss-ups: <span className="font-semibold text-gray-900">{correctNT}/{nonTossup.length}</span>{" "}
            ({(correctNT / nonTossup.length * 100).toFixed(0)}%)
          </span>
        </>
      ) : null}
      {avgErr != null ? (
        <>
          <span className="text-gray-400">|</span>
          <span className="text-gray-500">
            Avg total error: <span className="font-semibold text-gray-900">{avgErr.toFixed(1)} runs</span>
          </span>
        </>
      ) : null}
    </div>
  );
}

export default function ScorePredictionTab() {
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [games, setGames] = useState<GameSimSummary[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedPk, setSelectedPk] = useState<number | null>(null);
  const [detail, setDetail] = useState<GameSimDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [allGamesCache, setAllGamesCache] = useState<Map<string, GameSimSummary[]>>(new Map());

  useEffect(() => {
    listGameSimDates()
      .then((r) => {
        setAvailableDates(r.dates);
        if (r.default_date && !date) setDate(r.default_date);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!date) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setGames(null);
    setSelectedPk(null);
    setDetail(null);
    listGameSims(date)
      .then((r) => {
        if (cancelled) return;
        setGames(r.games);
        setAllGamesCache((prev) => new Map(prev).set(date, r.games));
        if (r.games.length > 0) setSelectedPk(r.games[0].game_pk);
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [date]);

  useEffect(() => {
    if (selectedPk == null || !date) return;
    let cancelled = false;
    setLoadingDetail(true);
    setDetail(null);
    getGameSimDetail(selectedPk, date)
      .then((r) => !cancelled && setDetail(r))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoadingDetail(false));
    return () => { cancelled = true; };
  }, [selectedPk, date]);

  // Pre-fetch all dates for aggregate stats
  useEffect(() => {
    for (const d of availableDates) {
      if (!allGamesCache.has(d)) {
        listGameSims(d).then((r) => {
          setAllGamesCache((prev) => new Map(prev).set(d, r.games));
        }).catch(() => {});
      }
    }
  }, [availableDates]);

  const formatDateLabel = (d: string) => {
    const dt = new Date(d + "T12:00:00");
    const day = dt.toLocaleDateString("en-US", { weekday: "short" });
    const md = dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return `${day} ${md}`;
  };

  const dateAccuracy = (d: string): string | null => {
    const gs = allGamesCache.get(d);
    if (!gs) return null;
    const scored = gs.filter((g) => g.has_actual && g.predicted_winner_correct != null);
    if (scored.length === 0) return null;
    const correct = scored.filter((g) => g.predicted_winner_correct).length;
    return `${correct}/${scored.length}`;
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="text-section mb-1">Score predictions</div>
      <p className="text-body text-gray-500 mb-6">
        Full-game Monte Carlo simulation — 10,000 games per matchup using pitchGPT matchup
        distributions, times-through-order adjustments, park factors, and platoon-aware bullpen.
      </p>

      {allGamesCache.size > 1 ? <AggregateStats allGames={allGamesCache} /> : null}

      <div className="flex flex-wrap items-center gap-2 mb-6">
        {availableDates.length > 0 ? (
          availableDates.slice().sort().map((d) => {
            const acc = dateAccuracy(d);
            return (
              <button
                key={d}
                onClick={() => setDate(d)}
                className={[
                  "px-3 py-1.5 rounded-lg text-small font-medium transition-colors",
                  d === date
                    ? "bg-gray-900 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200",
                ].join(" ")}
              >
                {formatDateLabel(d)}
                {acc ? (
                  <span className={d === date ? "text-gray-400 ml-1.5" : "text-gray-400 ml-1.5"}>
                    {acc}
                  </span>
                ) : null}
              </button>
            );
          })
        ) : (
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="border border-gray-200 rounded-lg px-3 py-1.5 text-body"
          />
        )}
      </div>

      {error ? <div className="text-small text-gauge-red mb-4">{error}</div> : null}
      {loading ? <div className="text-small text-gray-400">Loading games...</div> : null}
      {games && games.length === 0 ? (
        <div className="text-body text-gray-400">No score predictions for {date}.</div>
      ) : null}

      {games && games.length > 0 ? (
        <>
          <DaySummary games={games} />
          <div className="flex gap-2 overflow-x-auto pb-2 mb-8">
            {games.map((g) => (
              <GameCard
                key={g.game_pk}
                game={g}
                active={g.game_pk === selectedPk}
                onClick={() => setSelectedPk(g.game_pk)}
              />
            ))}
          </div>
        </>
      ) : null}

      {loadingDetail ? <div className="text-small text-gray-400">Loading simulation...</div> : null}
      {detail ? <GameDetail detail={detail} /> : null}
    </div>
  );
}
