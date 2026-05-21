// ScoreboardHeader — Apple-minimalist mini scoreboard.
//
// Single horizontal row of small caps + tabular nums. No graphics chrome,
// just hairline borders and generous whitespace. Renders:
//
//   AWAY @ HOME   ·   Top 7th   ·   3 - 2   ·   1 out   ·   ◇◆◇ (runners 1st, 2nd)   ·   Pitcher: ...  vs  Batter: ...
//
// All the data is in the AB context. Nothing fancy.

import type { ABContextResponse } from "./types";

// Hardcoded team primary colors for a subtle visual identity. The 30 MLB teams.
// Pulled from team-official palettes (approximate hex). Used as a thin accent
// strip under the team abbreviation — never a heavy fill.
const TEAM_COLORS: Record<string, string> = {
  ARI: "#a71930", ATL: "#ce1141", BAL: "#df4601", BOS: "#bd3039",
  CHC: "#0e3386", CHW: "#27251f", CIN: "#c6011f", CLE: "#0c2340",
  COL: "#33006f", DET: "#0c2340", HOU: "#002d62", KC:  "#004687",
  LAA: "#ba0021", LAD: "#005a9c", MIA: "#00a3e0", MIL: "#0a2351",
  MIN: "#002b5c", NYM: "#002d72", NYY: "#0c2340", OAK: "#003831",
  PHI: "#e81828", PIT: "#fdb827", SD:  "#2f241d", SF:  "#fd5a1e",
  SEA: "#0c2c56", STL: "#c41e3a", TB:  "#092c5c", TEX: "#003278",
  TOR: "#134a8e", WSH: "#ab0003",
};

// ESPN CDN serves clean MLB team logos at standard URLs. Lower-case abbr.
// Format: https://a.espncdn.com/i/teamlogos/mlb/500/{abbr}.png — 500×500 PNG.
function teamLogoUrl(abbr: string): string {
  return `https://a.espncdn.com/i/teamlogos/mlb/500/${abbr.toLowerCase()}.png`;
}

function TeamBadge({ abbr, isBatting }: { abbr: string | null; isBatting: boolean }) {
  if (!abbr) return <span className="text-gray-400 text-small">—</span>;
  const color = TEAM_COLORS[abbr] ?? "#6b7280";
  return (
    <div className="flex flex-col items-center gap-1">
      <img
        src={teamLogoUrl(abbr)}
        alt={`${abbr} logo`}
        className={`h-8 w-8 transition-state ${isBatting ? "opacity-100" : "opacity-50"}`}
        loading="lazy"
      />
      <span
        className={`font-semibold tracking-wide tabular-nums text-small ${
          isBatting ? "text-gray-900" : "text-gray-500"
        }`}
      >
        {abbr}
      </span>
      <span
        className="h-0.5 w-6 rounded-full"
        style={{ backgroundColor: color, opacity: isBatting ? 1 : 0.4 }}
      />
    </div>
  );
}

function BaseDiamond({
  on1,
  on2,
  on3,
}: {
  on1: boolean;
  on2: boolean;
  on3: boolean;
}) {
  // Three small diamonds in a triangle: 2nd at top, 3rd at left, 1st at right.
  const Diamond = ({ filled, x, y }: { filled: boolean; x: number; y: number }) => (
    <rect
      x={x - 6}
      y={y - 6}
      width={12}
      height={12}
      transform={`rotate(45 ${x} ${y})`}
      fill={filled ? "#0d9488" : "white"}
      stroke="#9ca3af"
      strokeWidth={1.2}
    />
  );
  return (
    <svg width={48} height={36} viewBox="0 0 48 36" aria-label={`Runners ${[on1 && "1st", on2 && "2nd", on3 && "3rd"].filter(Boolean).join(", ") || "none"}`}>
      <Diamond filled={on2} x={24} y={10} />
      <Diamond filled={on3} x={10} y={22} />
      <Diamond filled={on1} x={38} y={22} />
    </svg>
  );
}

function OutsIndicator({ outs }: { outs: number | null }) {
  const n = outs ?? 0;
  return (
    <span className="inline-flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className={`h-2 w-2 rounded-full ${i < n ? "bg-gray-900" : "bg-gray-200"}`}
        />
      ))}
      <span className="ml-1.5 text-small text-gray-500 tabular-nums">
        {n} {n === 1 ? "out" : "outs"}
      </span>
    </span>
  );
}

export function ScoreboardHeader({ ctx }: { ctx: ABContextResponse }) {
  const inningStr =
    ctx.inning !== null && ctx.inning_half
      ? `${ctx.inning_half} ${ctx.inning}${ordinalSuffix(ctx.inning)}`
      : "—";
  return (
    <div className="bg-white border-hairline rounded-md p-5">
      <div className="flex items-center justify-between gap-6 flex-wrap">
        {/* Teams + score */}
        <div className="flex items-center gap-4">
          <TeamBadge abbr={ctx.away_team} isBatting={ctx.inning_half === "Top"} />
          <span className="text-h2 font-semibold text-gray-900 tabular-nums">
            {ctx.away_score ?? "—"}
          </span>
          <span className="text-small text-gray-400 px-1">@</span>
          <span className="text-h2 font-semibold text-gray-900 tabular-nums">
            {ctx.home_score ?? "—"}
          </span>
          <TeamBadge abbr={ctx.home_team} isBatting={ctx.inning_half === "Bot"} />
        </div>

        {/* Inning */}
        <div className="flex flex-col items-center">
          <span className="text-small font-medium text-gray-500 uppercase tracking-wide">Inning</span>
          <span className="text-body font-medium text-gray-900 tabular-nums mt-0.5">{inningStr}</span>
        </div>

        {/* Outs + bases */}
        <div className="flex items-center gap-4">
          <OutsIndicator outs={ctx.outs_before_ab} />
          <BaseDiamond on1={ctx.runner_on_1b} on2={ctx.runner_on_2b} on3={ctx.runner_on_3b} />
        </div>
      </div>

      {/* Pitcher vs Batter line */}
      <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between gap-6 flex-wrap text-small">
        <div>
          <span className="text-gray-500 uppercase tracking-wide font-medium mr-2">Pitcher</span>
          <span className="text-gray-900 font-medium">
            {ctx.pitcher_name ?? `MLBAM ${ctx.pitcher_id}`}
          </span>
          {ctx.pitcher_throws && (
            <span className="text-gray-400 ml-1">({ctx.pitcher_throws}HP)</span>
          )}
        </div>
        <div className="text-gray-300">vs</div>
        <div>
          <span className="text-gray-500 uppercase tracking-wide font-medium mr-2">Batter</span>
          <span className="text-gray-900 font-medium">
            {ctx.batter_name ?? `MLBAM ${ctx.batter_id}`}
          </span>
          {ctx.batter_stand && (
            <span className="text-gray-400 ml-1">({ctx.batter_stand}HB)</span>
          )}
        </div>
      </div>
    </div>
  );
}

function ordinalSuffix(n: number): string {
  const j = n % 10;
  const k = n % 100;
  if (j === 1 && k !== 11) return "st";
  if (j === 2 && k !== 12) return "nd";
  if (j === 3 && k !== 13) return "rd";
  return "th";
}
