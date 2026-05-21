import { useEffect, useMemo, useState } from "react";
import { getPitcherProfile, listPitchers } from "./api";
import { Caption, PitchGlyph, Section, SectionLabel } from "./shared/ui";
import {
  PITCH_TYPES,
  PITCH_TYPE_GLYPHS,
  PITCH_TYPE_NAMES,
  type PitchType,
  type PitcherProfileResponse,
  type PitcherSummary,
} from "./types";

// ============================================================
// Local primitives (shared ones live in ./shared/ui).
// ============================================================

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
}) {
  return (
    <div className="border-hairline rounded-md px-4 py-3 bg-white">
      <div className="text-small text-gray-500 uppercase tracking-wide font-medium text-xs">
        {label}
      </div>
      <div className="text-h2 font-medium text-gray-900 tabular-nums leading-tight mt-1">
        {value}
      </div>
      {hint && <div className="text-small text-gray-500 mt-0.5 leading-snug">{hint}</div>}
    </div>
  );
}

// Format a possibly-NaN number, returning "—" when not finite.
function fmt(x: number, decimals = 2, suffix = ""): string {
  if (!Number.isFinite(x)) return "—";
  return `${x.toFixed(decimals)}${suffix}`;
}
function fmtInt(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return String(Math.round(x));
}

// ============================================================
// Heatmap (3x3 in-zone usage per pitch type)
// ============================================================

function HeatCell({ v, max, color }: { v: number; max: number; color: string }) {
  // 3x3 cell shaded by usage fraction within this type. max normalizes across
  // the type's own 9 cells (relative concentration), not across types.
  const valid = Number.isFinite(v);
  const alpha = valid && max > 0 ? Math.min(1, v / max) : 0;
  return (
    <div className="aspect-square border border-gray-200 flex items-center justify-center relative">
      <div
        className="absolute inset-0"
        style={{ backgroundColor: color, opacity: alpha * 0.85 }}
      />
      <span
        className="relative tabular-nums text-xs font-medium"
        style={{ color: alpha > 0.45 ? "#fff" : "#1f2937" }}
      >
        {valid ? Math.round(v * 100) : "—"}
      </span>
    </div>
  );
}

function PitchHeatmapCard({
  type,
  cells,
  usagePct,
}: {
  type: PitchType;
  cells: number[];
  usagePct: number;
}) {
  const max = cells.reduce(
    (acc, x) => (Number.isFinite(x) && x > acc ? x : acc),
    0,
  );
  const { color } = PITCH_TYPE_GLYPHS[type];
  return (
    <div className="border-hairline rounded-md p-3 bg-white">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5 text-small font-medium text-gray-900">
          <PitchGlyph type={type} />
          <span>{type}</span>
          <span className="text-gray-500">· {PITCH_TYPE_NAMES[type]}</span>
        </div>
        <div className="text-small text-gray-500 tabular-nums">
          {fmt(usagePct * 100, 1, "%")}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-0.5">
        {cells.map((v, i) => (
          <HeatCell key={i} v={v} max={max} color={color} />
        ))}
      </div>
      <div className="text-xs text-gray-400 text-center mt-1.5">in-zone usage %</div>
    </div>
  );
}

// ============================================================
// Arsenal-by-count matrix (7×12)
// ============================================================

function ArsenalByCountTable({
  arsenalByCount,
  countOrder,
  hasPitch,
}: {
  arsenalByCount: Record<string, number[]>;
  countOrder: string[];
  hasPitch: Record<string, boolean>;
}) {
  return (
    <div className="overflow-x-auto border-hairline rounded-md">
      <table className="text-small w-full">
        <thead>
          <tr className="bg-gray-50">
            <th className="text-left px-3 py-2 text-gray-500 font-medium">type</th>
            {countOrder.map((c) => (
              <th key={c} className="px-2 py-2 text-gray-500 font-medium tabular-nums">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PITCH_TYPES.map((pt) => {
            const row = arsenalByCount[pt] ?? [];
            const has = hasPitch[pt] ?? false;
            return (
              <tr key={pt} className={has ? "" : "opacity-40"}>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5 text-gray-900 font-medium">
                    <PitchGlyph type={pt} dim={!has} />
                    {pt}
                  </span>
                </td>
                {row.map((v, i) => {
                  const valid = Number.isFinite(v) && v > 0;
                  const intensity = valid ? Math.min(1, v) : 0;
                  return (
                    <td
                      key={i}
                      className="px-2 py-2 text-center tabular-nums"
                      style={{
                        backgroundColor: valid
                          ? `rgba(17, 24, 39, ${0.04 + intensity * 0.6})`
                          : undefined,
                        color: intensity > 0.55 ? "#fff" : "#1f2937",
                      }}
                    >
                      {valid ? `${Math.round(v * 100)}` : "·"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================
// Arsenal-by-stand mini-table
// ============================================================

function ArsenalByStandTable({
  arsenalByStand,
  hasPitch,
}: {
  arsenalByStand: Record<string, { L: number; R: number }>;
  hasPitch: Record<string, boolean>;
}) {
  return (
    <div className="border-hairline rounded-md overflow-hidden">
      <table className="text-small w-full">
        <thead>
          <tr className="bg-gray-50">
            <th className="text-left px-3 py-2 text-gray-500 font-medium">type</th>
            <th className="px-3 py-2 text-gray-500 font-medium">vs LHB</th>
            <th className="px-3 py-2 text-gray-500 font-medium">vs RHB</th>
            <th className="px-3 py-2 text-gray-500 font-medium">Δ (L−R)</th>
          </tr>
        </thead>
        <tbody>
          {PITCH_TYPES.map((pt) => {
            const has = hasPitch[pt] ?? false;
            const v = arsenalByStand[pt] ?? { L: NaN, R: NaN };
            const delta = v.L - v.R;
            return (
              <tr key={pt} className={has ? "border-t border-gray-100" : "opacity-40 border-t border-gray-100"}>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5 text-gray-900 font-medium">
                    <PitchGlyph type={pt} dim={!has} />
                    {pt}
                  </span>
                </td>
                <td className="px-3 py-2 tabular-nums">{fmt(v.L * 100, 1, "%")}</td>
                <td className="px-3 py-2 tabular-nums">{fmt(v.R * 100, 1, "%")}</td>
                <td
                  className="px-3 py-2 tabular-nums"
                  style={{
                    color: Number.isFinite(delta)
                      ? Math.abs(delta) < 0.005
                        ? "#9ca3af"
                        : delta > 0
                        ? "#0d9488"
                        : "#dc2626"
                      : "#9ca3af",
                  }}
                >
                  {Number.isFinite(delta)
                    ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp`
                    : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================
// Arsenal-overview rows (per-type velo / arm slot / break)
// ============================================================

function ArsenalOverview({ profile }: { profile: PitcherProfileResponse }) {
  return (
    <div className="border-hairline rounded-md overflow-hidden">
      <table className="text-small w-full">
        <thead>
          <tr className="bg-gray-50">
            <th className="text-left px-3 py-2 text-gray-500 font-medium">type</th>
            <th className="text-right px-3 py-2 text-gray-500 font-medium">usage</th>
            <th className="text-right px-3 py-2 text-gray-500 font-medium">velo</th>
            <th className="text-right px-3 py-2 text-gray-500 font-medium">spin</th>
            <th className="text-right px-3 py-2 text-gray-500 font-medium">arm slot</th>
            <th className="text-right px-3 py-2 text-gray-500 font-medium">pfx_x</th>
            <th className="text-right px-3 py-2 text-gray-500 font-medium">pfx_z</th>
          </tr>
        </thead>
        <tbody>
          {PITCH_TYPES.map((pt) => {
            const has = profile.has_pitch[pt] ?? false;
            const usage = profile.arsenal_pct[pt] ?? 0;
            return (
              <tr key={pt} className={has ? "border-t border-gray-100" : "opacity-40 border-t border-gray-100"}>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5 text-gray-900 font-medium">
                    <PitchGlyph type={pt} dim={!has} />
                    {pt}
                    <span className="text-gray-500 font-normal">{PITCH_TYPE_NAMES[pt]}</span>
                  </span>
                </td>
                <td className="text-right px-3 py-2 tabular-nums">
                  {has ? `${(usage * 100).toFixed(1)}%` : "—"}
                </td>
                <td className="text-right px-3 py-2 tabular-nums">
                  {fmt(profile.mean_velo[pt], 1, " mph")}
                </td>
                <td className="text-right px-3 py-2 tabular-nums">
                  {fmtInt(profile.mean_spin[pt])}{" rpm"}
                </td>
                <td className="text-right px-3 py-2 tabular-nums">
                  {fmt(profile.arm_slot[pt], 1, "°")}
                </td>
                <td className="text-right px-3 py-2 tabular-nums">
                  {fmt(profile.mean_pfx_x[pt], 2, "")}
                </td>
                <td className="text-right px-3 py-2 tabular-nums">
                  {fmt(profile.mean_pfx_z[pt], 2, "")}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================
// Pitcher search dropdown
// ============================================================

function PitcherSearch({
  onPick,
  picked,
}: {
  onPick: (p: PitcherSummary) => void;
  picked: PitcherSummary | null;
}) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<PitcherSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  // Debounce the query.
  useEffect(() => {
    const handle = setTimeout(() => {
      if (!q.trim()) {
        setResults([]);
        return;
      }
      setLoading(true);
      listPitchers(q, 12)
        .then((r) => setResults(r.items))
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(handle);
  }, [q]);

  return (
    <div className="relative">
      <input
        type="text"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        placeholder={
          picked
            ? `Currently: ${picked.pitcher_name ?? `id ${picked.pitcher_id}`}`
            : "Search by name (e.g. Gerrit Cole)"
        }
        className="w-full px-4 py-3 text-body border-hairline rounded-md bg-white hover:border-gray-300 focus:outline-none focus:border-accent transition-state"
      />
      {open && (results.length > 0 || loading) && (
        <div className="absolute z-10 mt-1 w-full max-h-72 overflow-y-auto bg-white border-hairline rounded-md shadow-sm">
          {loading && (
            <div className="px-4 py-2 text-small text-gray-500">Searching…</div>
          )}
          {results.map((p) => (
            <button
              key={p.pitcher_id}
              onClick={() => {
                onPick(p);
                setOpen(false);
                setQ("");
              }}
              className="block w-full text-left px-4 py-2.5 text-body hover:bg-gray-50 transition-state border-b border-gray-100 last:border-b-0"
            >
              <div className="text-gray-900 font-medium">
                {p.pitcher_name ?? `id ${p.pitcher_id}`}
              </div>
              <div className="text-small text-gray-500">
                latest profile {p.latest_asof_date} · {p.n_entries} entries
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================
// Source badge
// ============================================================

function SourceBadge({ source }: { source: PitcherProfileResponse["source"] }) {
  const map: Record<PitcherProfileResponse["source"], { bg: string; fg: string; label: string }> = {
    per_player_blended: { bg: "bg-emerald-50", fg: "text-emerald-700", label: "per-player profile" },
    league_only: { bg: "bg-amber-50", fg: "text-amber-700", label: "league-mean only" },
    zero_fallback: { bg: "bg-red-50", fg: "text-red-700", label: "no data" },
  };
  const c = map[source];
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium ${c.bg} ${c.fg}`}>
      {c.label}
    </span>
  );
}

// ============================================================
// Main tab
// ============================================================

export default function PitcherProfileTab() {
  const [picked, setPicked] = useState<PitcherSummary | null>(null);
  const [asofDate, setAsofDate] = useState<string>("");      // YYYY-MM-DD, "" = latest
  const [profile, setProfile] = useState<PitcherProfileResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!picked) return;
    setLoading(true);
    setError(null);
    getPitcherProfile(picked.pitcher_id, asofDate || undefined)
      .then(setProfile)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [picked?.pitcher_id, asofDate]);

  // When a new pitcher is picked, default asof to their latest entry.
  useEffect(() => {
    if (picked) setAsofDate(picked.latest_asof_date);
  }, [picked?.pitcher_id]);

  const totalActiveTypes = useMemo(() => {
    if (!profile) return 0;
    return PITCH_TYPES.filter((pt) => profile.has_pitch[pt]).length;
  }, [profile]);

  return (
    <div>
      <header className="mb-10">
        <div className="text-small font-medium text-gray-500 uppercase tracking-wide mb-2">
          PitchGPT — pitcher profile inspector
        </div>
        <h1 className="text-h1 font-semibold text-gray-900 leading-tight">
          What does the model know about this pitcher?
        </h1>
        <p className="text-body text-gray-500 mt-4 max-w-prose leading-relaxed">
          The 218-dim profile vector — arsenal mix, per-count tendencies,
          per-stand splits, release/arm-slot/break by type, recent form —
          that the model reads on every prediction. Built fold-aware (no
          leakage) and only from pitches strictly before the as-of date.
        </p>
      </header>

      <Section>
        <SectionLabel>1. Pick a pitcher</SectionLabel>
        <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-3">
          <PitcherSearch onPick={setPicked} picked={picked} />
          <input
            type="date"
            value={asofDate}
            onChange={(e) => setAsofDate(e.target.value)}
            disabled={!picked}
            className="px-4 py-3 text-body border-hairline rounded-md bg-white hover:border-gray-300 focus:outline-none focus:border-accent transition-state disabled:bg-gray-50 disabled:text-gray-400"
            title={
              picked
                ? `Pick the as-of date. Defaults to this pitcher's latest profile entry (${picked.latest_asof_date}).`
                : "Pick a pitcher first."
            }
          />
        </div>
        <Caption>
          The system matches the closest profile entry on or before this date.
        </Caption>
      </Section>

      {error && (
        <div className="my-8 bg-red-50 border border-red-100 rounded-md p-4 text-small text-red-900">
          {error}
        </div>
      )}

      {loading && !profile && (
        <div className="my-8 text-small text-gray-500">Loading profile…</div>
      )}

      {profile && (
        <>
          <Section>
            <div className="flex items-baseline gap-4 mb-2">
              <h2 className="text-h2 font-semibold text-gray-900 leading-tight">
                {profile.pitcher_name ?? `id ${profile.pitcher_id}`}
              </h2>
              <SourceBadge source={profile.source} />
              <span className="text-small text-gray-500 tabular-nums">
                fold {profile.fold_id}
              </span>
            </div>
            <div className="text-small text-gray-500">
              Profile as of <span className="text-gray-900 font-medium">{profile.asof_date_used}</span>
              {profile.asof_date_used !== profile.asof_date_requested && (
                <span className="text-gray-400">
                  {" "}(requested {profile.asof_date_requested}; nearest earlier entry)
                </span>
              )}{" "}· game #{profile.asof_game_num}
            </div>
          </Section>

          <Section>
            <SectionLabel>Quick stats</SectionLabel>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard
                label="active types"
                value={`${totalActiveTypes}/7`}
                hint="pitch types he has actually thrown in the trailing window"
              />
              <StatCard
                label="recent 30d xwOBA"
                value={fmt(profile.recent_30d_xwoba, 3)}
                hint={`${fmtInt(profile.recent_30d_n_pitches)} pitches in window`}
              />
              <StatCard
                label="last 3 starts xwOBA"
                value={fmt(profile.recent_3starts_xwoba, 3)}
                hint={`n_starts = ${fmtInt(profile.recent_3starts_n)}`}
              />
              <StatCard
                label="days since last app."
                value={fmt(profile.days_since_last_appearance, 1)}
                hint={`profile confidence ${fmt(profile.profile_confidence, 2)}`}
              />
            </div>
            <Caption>
              The "long window" (last 1000 pitches) spans {fmtInt(profile.long_window_span_days)} days
              and is {fmt(profile.long_window_pct_current_season * 100, 0, "%")} from the current season.
              When that share is low, the profile is dominated by prior-season pitches.
            </Caption>
          </Section>

          <Section>
            <SectionLabel>Arsenal — per-type aggregates</SectionLabel>
            <ArsenalOverview profile={profile} />
            <Caption>
              Negative pfx_x = arm-side break (catcher's view); positive pfx_z = induced ride above
              gravity. Velo in mph, arm slot in degrees from horizontal at release. League-mean
              backfill applies where the pitcher has never thrown the type.
            </Caption>
          </Section>

          <Section>
            <SectionLabel>Per-zone usage by pitch type (3×3 in-zone)</SectionLabel>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {PITCH_TYPES.filter((pt) => profile.has_pitch[pt]).map((pt) => (
                <PitchHeatmapCard
                  key={pt}
                  type={pt}
                  cells={profile.heatmap_by_type[pt] ?? []}
                  usagePct={profile.arsenal_pct[pt] ?? 0}
                />
              ))}
            </div>
            <Caption>
              Each cell is the share of this pitcher's <i>{`{pitch type, in-zone}`}</i> pitches that
              landed in that 3×3 cell (rows top→bottom from the umpire's view, cols left→right).
              Color intensity is normalized within each pitch type — not comparable across pitches.
            </Caption>
          </Section>

          <Section>
            <SectionLabel>Usage by count (rows = pitch type, columns = balls-strikes)</SectionLabel>
            <ArsenalByCountTable
              arsenalByCount={profile.arsenal_by_count}
              countOrder={profile.count_state_order}
              hasPitch={profile.has_pitch}
            />
            <Caption>
              Each cell is the share of pitches in that count state that were this type — e.g.
              "FF at 3-0" = "out of all pitches he's thrown in 3-0 counts, what fraction were
              fastballs." Reading down a column shows the conditional mix; reading across a row
              shows how a pitch type's usage changes with the count.
            </Caption>
          </Section>

          <Section>
            <SectionLabel>Usage by batter handedness</SectionLabel>
            <ArsenalByStandTable
              arsenalByStand={profile.arsenal_by_stand}
              hasPitch={profile.has_pitch}
            />
            <Caption>
              Δ (L−R) flags handedness asymmetry: a strong positive Δ means the pitcher leans on
              that type more against LHB than RHB.
            </Caption>
          </Section>
        </>
      )}

      <footer className="mt-24 pt-8 border-t border-gray-200 text-small text-gray-400">
        v6 profile schema · 218 dims · fold-aware · trailing-window only.
      </footer>
    </div>
  );
}
