// Reusable strike-zone SVG plot.
//
// Two render modes, both optional and composable:
//  - `dots`    — actual pitches at precise (plate_x, plate_z) coordinates.
//  - `heatmap` — a 13-cell feature_zone distribution (the model's coarse
//                location prediction): 9 in-zone cells + 4 OOZ quadrants.
//
// Built so the live-game tracker can reuse it unchanged: the actual pane
// passes `dots`, the predicted pane passes `heatmap`.

import { PITCH_TYPE_GLYPHS, type PitchType } from "../types";

export interface ZoneDot {
  x: number; // plate_x, feet (0 = center of plate, + = catcher's right)
  z: number; // plate_z, feet (absolute height)
  type: string; // pitch type — drives color + glyph
  label: string; // short label, e.g. the pitch number
  focus?: boolean; // emphasize this dot (the focused pitch)
}

// Effective strike-zone half-width: plate (17 in) + ball radius ≈ 0.83 ft.
// Matches PLATE_HALF_WIDTH_FT in data/zones.py.
const PLATE_HALF_FT = 0.83;

const VIEW_W = 200;
const VIEW_H = 248;
const BOX_L = 62;
const BOX_R = 138;
const BOX_T = 58;
const BOX_B = 198;

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

export function PitchZonePlot({
  szTop,
  szBot,
  dots,
  heatmap,
}: {
  szTop: number;
  szBot: number;
  dots?: ZoneDot[];
  heatmap?: number[]; // length 13, feature_zone order
}) {
  const zRange = szTop - szBot > 0.1 ? szTop - szBot : 2.0;
  const mapX = (px: number) =>
    clamp(100 + (px / PLATE_HALF_FT) * ((BOX_R - BOX_L) / 2), 8, VIEW_W - 8);
  const mapZ = (pz: number) =>
    clamp(BOX_T + ((szTop - pz) / zRange) * (BOX_B - BOX_T), 8, VIEW_H - 20);

  const cellW = (BOX_R - BOX_L) / 3;
  const cellH = (BOX_B - BOX_T) / 3;
  const maxHeat = heatmap ? Math.max(...heatmap, 1e-6) : 1;

  // 13-cell heatmap regions: 0..8 = in-zone 3×3 (row-major), 9..12 = OOZ
  // quadrants UL / UR / LL / LR.
  const heatRegions: { i: number; x: number; y: number; w: number; h: number }[] = [];
  if (heatmap) {
    for (let i = 0; i < 9; i++) {
      const r = Math.floor(i / 3);
      const c = i % 3;
      heatRegions.push({
        i,
        x: BOX_L + c * cellW,
        y: BOX_T + r * cellH,
        w: cellW,
        h: cellH,
      });
    }
    heatRegions.push({ i: 9, x: 8, y: 8, w: BOX_L - 8, h: BOX_T - 8 }); // UL
    heatRegions.push({ i: 10, x: BOX_R, y: 8, w: VIEW_W - 8 - BOX_R, h: BOX_T - 8 }); // UR
    heatRegions.push({ i: 11, x: 8, y: BOX_B, w: BOX_L - 8, h: VIEW_H - 20 - BOX_B }); // LL
    heatRegions.push({
      i: 12,
      x: BOX_R,
      y: BOX_B,
      w: VIEW_W - 8 - BOX_R,
      h: VIEW_H - 20 - BOX_B,
    }); // LR
  }

  return (
    <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="w-full" role="img">
      {/* heatmap cells (drawn first, under everything) */}
      {heatmap &&
        heatRegions.map((reg) => (
          <rect
            key={`h${reg.i}`}
            x={reg.x}
            y={reg.y}
            width={reg.w}
            height={reg.h}
            fill="#0d9488"
            opacity={Math.min(0.85, (heatmap[reg.i] / maxHeat) * 0.85)}
          />
        ))}

      {/* strike-zone box */}
      <rect
        x={BOX_L}
        y={BOX_T}
        width={BOX_R - BOX_L}
        height={BOX_B - BOX_T}
        fill="none"
        stroke="#1f2937"
        strokeWidth={1.5}
      />
      {/* 3×3 grid lines */}
      {[1, 2].map((i) => (
        <line
          key={`v${i}`}
          x1={BOX_L + i * cellW}
          y1={BOX_T}
          x2={BOX_L + i * cellW}
          y2={BOX_B}
          stroke="#d1d5db"
          strokeWidth={0.75}
        />
      ))}
      {[1, 2].map((i) => (
        <line
          key={`hl${i}`}
          x1={BOX_L}
          y1={BOX_T + i * cellH}
          x2={BOX_R}
          y2={BOX_T + i * cellH}
          stroke="#d1d5db"
          strokeWidth={0.75}
        />
      ))}

      {/* heatmap cell labels (percent) */}
      {heatmap &&
        heatRegions.map((reg) => {
          const pct = Math.round(heatmap[reg.i] * 100);
          if (pct < 1) return null;
          return (
            <text
              key={`ht${reg.i}`}
              x={reg.x + reg.w / 2}
              y={reg.y + reg.h / 2 + 3}
              textAnchor="middle"
              fontSize={8}
              fill={heatmap[reg.i] / maxHeat > 0.5 ? "#ffffff" : "#6b7280"}
            >
              {pct}
            </text>
          );
        })}

      {/* actual pitch dots */}
      {dots &&
        dots.map((d, idx) => {
          const cx = mapX(d.x);
          const cy = mapZ(d.z);
          const known = d.type in PITCH_TYPE_GLYPHS;
          const { color, glyph } = known
            ? PITCH_TYPE_GLYPHS[d.type as PitchType]
            : { color: "#9ca3af", glyph: "·" };
          const r = d.focus ? 11 : 7;
          return (
            <g key={idx} opacity={d.focus ? 1 : 0.45}>
              {d.focus && (
                <circle cx={cx} cy={cy} r={r + 3} fill="none" stroke={color} strokeWidth={1.5} />
              )}
              <circle cx={cx} cy={cy} r={r} fill={color} />
              <text
                x={cx}
                y={cy + (d.focus ? 4 : 3)}
                textAnchor="middle"
                fontSize={d.focus ? 11 : 8}
                fill="#ffffff"
                aria-hidden="true"
              >
                {glyph}
              </text>
              <text
                x={cx + r + 2}
                y={cy - r + 2}
                fontSize={7}
                fill="#1f2937"
                fontWeight={600}
              >
                {d.label}
              </text>
            </g>
          );
        })}

      {/* axis hint */}
      <text x={VIEW_W / 2} y={VIEW_H - 4} textAnchor="middle" fontSize={7} fill="#9ca3af">
        catcher's view · strike zone box
      </text>
    </svg>
  );
}
