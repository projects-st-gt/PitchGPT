// StrikeZone widget — interactive 3x3 in-zone grid + 4 OOZ quadrants + batter silhouette.
// v5 SIS 14-zone scheme; see data/zones.py and ../types.ts for the SIS_TO_INTERNAL map.
//
// Catcher's perspective (batter facing camera):
//   - Top row = high pitches (display row 0); bottom row = low (display row 2)
//   - Left col = catcher's left (display col 0); right col = catcher's right (display col 2)
//   - OOZ quadrants live OUTSIDE the strike zone box (UL/UR/LL/LR)
//   - RHB silhouette on LEFT (batter stands on catcher's left)
//   - LHB silhouette on RIGHT
//
// The widget calls `onSelect(zone)` with a dense feature_zone id (0..12 under
// v5: 0..8 = 3x3 in-zone, 9..12 = OOZ quadrants UL/UR/LL/LR). `selected` is
// the same id, or null for "not selected — sample from the model."

import {
  OOZ_QUADRANT_TO_FEATURE_ZONE,
  type OOZQuadrant,
  featureZoneToGridCell,
  gridCellToFeatureZone,
  isOOZ,
  oozQuadrantOf,
} from "./types";

interface Props {
  selected: number | null;
  onSelect: (zone: number | null) => void;
  batterStand: "R" | "L" | null;
  // Optional: dot markers for where observed pitches landed.
  observedPitches?: { zone: number; type: string }[];
  // The pitch index the user has chosen as the intervention position — its
  // marker on the zone is highlighted to anchor "you're replacing THIS one".
  highlightedPitchIndex?: number | null;
  // Optional: batter whiff% per in-zone cell (length 9, indices 0..8 in
  // feature_zone numbering — v5 SIS 3x3 in-zone). Rendered as a faint heatmap
  // underneath the grid so the user can see "this batter whiffs in the
  // up-and-in corner".
  batterWhiffGrid?: (number | null)[];
  // Whether the selection is enabled (e.g., disable while a query is running).
  disabled?: boolean;
}

export function StrikeZone({
  selected,
  onSelect,
  batterStand,
  observedPitches = [],
  highlightedPitchIndex = null,
  batterWhiffGrid,
  disabled = false,
}: Props) {
  const isLHB = batterStand === "L";
  // SVG viewBox: 600×620. Strike zone is 200×240 cells (MLB aspect ~0.7).
  // CRITICALLY: the strike zone has to be properly proportioned relative to
  // the batter — real MLB zone top is at the chest/letters, bottom at the
  // hollow of the knee. That means the zone vertically spans only ~45% of
  // the batter's height (chest at ~30% from head, knee at ~75%), NOT the
  // full body. The batter is ~2.5× taller than the zone.
  const SZ_W = 200;
  const SZ_H = 240;
  const SZ_X = 220;
  const SZ_Y = 220;      // top of zone aligns with batter's chest
  const cellW = SZ_W / 3;
  const cellH = SZ_H / 3;
  const OOZ_PAD = 40;

  const selectedGrid = selected !== null && !isOOZ(selected)
    ? featureZoneToGridCell(selected)
    : null;
  const selectedOOZQuadrant: OOZQuadrant | null =
    selected !== null ? oozQuadrantOf(selected) : null;

  return (
    <div className="flex flex-col items-center">
      <div className="text-small font-medium text-gray-500 uppercase tracking-wide mb-3">
        Strike zone (catcher's view) ·{" "}
        {batterStand ? (
          <span className="text-gray-900">{batterStand === "R" ? "RHB" : "LHB"}</span>
        ) : (
          <span className="text-gray-400">batter handedness unknown</span>
        )}
      </div>
      <svg
        viewBox="0 0 600 620"
        width="100%"
        className="max-w-lg"
        style={{ aspectRatio: "600/620" }}
      >
        {/* OOZ — 4 quadrants surrounding the strike zone, each clickable.
            Layout (UL=upper-left, UR=upper-right, LL=lower-left, LR=lower-right):
                [UL] [   top edge   ] [UR]
                [  left ] [   zone   ] [right]
                [LL] [   bot edge  ] [LR]
            The "edges" (top/bottom/left/right of the zone but inside the OOZ pad)
            are split between adjacent quadrants by halfway lines, so the UL
            quadrant gets the top-left edge wedge, etc. Cleanest is to just
            split the surrounding ring with the strike-zone's center lines
            extended outward — that's what the SIS scheme does. */}
        {(["UL", "UR", "LL", "LR"] as OOZQuadrant[]).map((quad) => {
          const zoneId = OOZ_QUADRANT_TO_FEATURE_ZONE[quad];
          const isSelected = selectedOOZQuadrant === quad;
          // Each quadrant spans half-width × half-height of (zone + OOZ_PAD).
          const xStart = SZ_X - OOZ_PAD;
          const yStart = SZ_Y - OOZ_PAD;
          const halfW = (SZ_W + 2 * OOZ_PAD) / 2;
          const halfH = (SZ_H + 2 * OOZ_PAD) / 2;
          const left = quad === "UL" || quad === "LL";
          const top = quad === "UL" || quad === "UR";
          const qx = left ? xStart : xStart + halfW;
          const qy = top ? yStart : yStart + halfH;
          return (
            <rect
              key={`ooz-${quad}`}
              x={qx}
              y={qy}
              width={halfW}
              height={halfH}
              fill={isSelected ? "#0d9488" : "#fafafa"}
              fillOpacity={isSelected ? 0.15 : 1}
              stroke="#e5e7eb"
              strokeWidth={1}
              onClick={() => !disabled && onSelect(zoneId)}
              style={{ cursor: disabled ? "default" : "pointer" }}
            />
          );
        })}

        {/* OOZ quadrant divider lines — subtle visual cue that the 4 OOZ
            regions are distinct, even when none is selected. The lines extend
            the strike-zone midlines outward through the OOZ pad. */}
        {(() => {
          const xMid = SZ_X + SZ_W / 2;
          const yMid = SZ_Y + SZ_H / 2;
          const x0 = SZ_X - OOZ_PAD;
          const x1 = SZ_X + SZ_W + OOZ_PAD;
          const y0 = SZ_Y - OOZ_PAD;
          const y1 = SZ_Y + SZ_H + OOZ_PAD;
          return (
            <g pointerEvents="none">
              {/* Vertical midline through top OOZ band */}
              <line x1={xMid} y1={y0} x2={xMid} y2={SZ_Y}
                    stroke="#d1d5db" strokeWidth={1} strokeDasharray="4 4" />
              {/* Vertical midline through bottom OOZ band */}
              <line x1={xMid} y1={SZ_Y + SZ_H} x2={xMid} y2={y1}
                    stroke="#d1d5db" strokeWidth={1} strokeDasharray="4 4" />
              {/* Horizontal midline through left OOZ band */}
              <line x1={x0} y1={yMid} x2={SZ_X} y2={yMid}
                    stroke="#d1d5db" strokeWidth={1} strokeDasharray="4 4" />
              {/* Horizontal midline through right OOZ band */}
              <line x1={SZ_X + SZ_W} y1={yMid} x2={x1} y2={yMid}
                    stroke="#d1d5db" strokeWidth={1} strokeDasharray="4 4" />
            </g>
          );
        })()}

        {/* Batter silhouette — clean side-view, composed of simple primitives.
            Local coords:
              - x axis: 0 = back of batter, +x = toward the plate (which is at
                x = FIG_BACK_TO_PLATE = 90).
              - y axis: 0 = top of helmet, +y down to feet (~360 = ground).
            For RHB: place the figure to the LEFT of the zone, NO flip
            (already faces right, toward the plate on its right).
            For LHB: place the figure to the RIGHT of the zone, FLIP (so it
            faces left, toward the plate on its left).
            The figure's "front" (toward plate) is at x_local=90; "back" is at
            x_local=0 (with the bat extending further to negative x). */}
        {batterStand && (() => {
          // MLB-style compact batter silhouette. Anatomical anchors:
          //   y=0    helmet top
          //   y=110  chest/letters (zone top)
          //   y=290  hollow of knee (zone bottom)
          //   y=370  cleats
          // Zone span 110→290 = 180 local units maps to SZ_H=240 world units.
          const scale = SZ_H / 180;
          const CHEST_Y_LOCAL = 110;
          const ty = SZ_Y - CHEST_Y_LOCAL * scale;
          const FRONT_GAP = 10;
          const FIG_W = 80 * scale;
          const tx = isLHB
            ? SZ_X + SZ_W + FRONT_GAP + FIG_W
            : SZ_X - FRONT_GAP - FIG_W;
          const flip = isLHB ? -1 : 1;
          const SIL = "#1f2937";

          return (
            <g transform={`translate(${tx}, ${ty}) scale(${flip * scale}, ${scale})`}>
              {/* Bat */}
              <line x1={22} y1={58} x2={-20} y2={-10}
                stroke={SIL} strokeWidth={4.5} strokeLinecap="round" />
              <ellipse cx={-20} cy={-12} rx={4} ry={3.5} fill={SIL} />

              {/* Helmet */}
              <ellipse cx={44} cy={20} rx={20} ry={22} fill={SIL} />
              <path d="M 58 26 Q 68 24 70 28 L 68 34 Q 60 36 58 32 Z" fill={SIL} />
              <circle cx={58} cy={20} r={1.5} fill="#f9fafb" opacity={0.7} />

              {/* Neck */}
              <path d="M 36 40 L 52 40 L 50 50 L 38 50 Z" fill={SIL} />

              {/* Torso — athletic build, wider chest tapering to waist */}
              <path d="M 10 54 Q 44 48 78 54 L 74 130 Q 44 134 18 130 Z" fill={SIL} />

              {/* Back arm + hand on bat */}
              <path d="M 14 58 Q 0 64 -8 56 Q -12 48 -4 40 L 4 44 Q -2 52 6 56 Z" fill={SIL} />
              <path d="M -6 42 L -18 -6 L -12 -8 L -2 38 Z" fill={SIL} />
              <ellipse cx={-14} cy={-4} rx={5} ry={4} fill={SIL} />

              {/* Front arm across body to grip */}
              <path d="M 74 58 Q 64 76 46 68 Q 28 72 14 64 L 18 56 Q 32 64 46 60 Q 60 66 70 52 Z" fill={SIL} />

              {/* Belt/waist */}
              <path d="M 20 128 L 72 128 L 70 142 L 22 142 Z" fill={SIL} />

              {/* Front leg — slightly open stance, athletic */}
              <path d="M 48 142 Q 56 200 58 260 Q 58 300 62 360 L 76 362 Q 74 300 72 260 Q 72 200 68 142 Z" fill={SIL} />

              {/* Back leg — weight loaded */}
              <path d="M 22 142 Q 16 200 18 260 Q 18 300 20 360 L 36 362 Q 38 300 38 260 Q 40 200 42 142 Z" fill={SIL} />

              {/* Cleats */}
              <ellipse cx={68} cy={366} rx={14} ry={4.5} fill={SIL} />
              <ellipse cx={28} cy={366} rx={14} ry={4.5} fill={SIL} />
            </g>
          );
        })()}

        {/* Batter whiff% heatmap underlay — render BEFORE the clickable cells
            so cells stack on top and remain clickable. Higher whiff% → darker
            teal (pitcher-friendly zone — batter swings and misses here).
            Cells WITH data but 0% whiff get a faint baseline tint so the
            heatmap looks continuous; cells with NO DATA (true None) stay blank. */}
        {batterWhiffGrid &&
          Array.from({ length: 3 }, (_, row) =>
            Array.from({ length: 3 }, (_, col) => {
              const cellZone = gridCellToFeatureZone(row, col);
              const w = batterWhiffGrid[cellZone];
              if (w === null || w === undefined || !isFinite(w)) {
                // No data — render a very faint gray to indicate "we don't know"
                return (
                  <rect
                    key={`heat-${row}-${col}`}
                    x={SZ_X + col * cellW}
                    y={SZ_Y + row * cellH}
                    width={cellW}
                    height={cellH}
                    fill="#9ca3af"
                    fillOpacity={0.05}
                    pointerEvents="none"
                  />
                );
              }
              // Map whiff [0, 0.5] → opacity [0.06, 0.6]. Floor at 0.06 so 0%
              // whiffs still get a visible baseline tint and the heatmap
              // covers every cell continuously.
              const opacity = Math.max(0.06, Math.min(0.6, w / 0.5 * 0.55));
              return (
                <rect
                  key={`heat-${row}-${col}`}
                  x={SZ_X + col * cellW}
                  y={SZ_Y + row * cellH}
                  width={cellW}
                  height={cellH}
                  fill="#0d9488"
                  fillOpacity={opacity}
                  pointerEvents="none"
                />
              );
            }),
          )}

        {/* 3x3 in-zone grid (clickable, on top of heatmap) */}
        {Array.from({ length: 3 }, (_, row) =>
          Array.from({ length: 3 }, (_, col) => {
            const cellZone = gridCellToFeatureZone(row, col);
            const isSelected = selectedGrid?.row === row && selectedGrid?.col === col;
            return (
              <rect
                key={`${row}-${col}`}
                x={SZ_X + col * cellW}
                y={SZ_Y + row * cellH}
                width={cellW}
                height={cellH}
                fill={isSelected ? "#0d9488" : "transparent"}
                fillOpacity={isSelected ? 1 : 0}
                stroke="#d1d5db"
                strokeWidth={isSelected ? 2 : 1}
                onClick={() => !disabled && onSelect(cellZone)}
                style={{ cursor: disabled ? "default" : "pointer" }}
                className="transition-state"
              />
            );
          }),
        )}

        {/* Strike-zone outer border (thicker, sits over the cells' borders) */}
        <rect
          x={SZ_X}
          y={SZ_Y}
          width={SZ_W}
          height={SZ_H}
          fill="none"
          stroke="#374151"
          strokeWidth={2}
          pointerEvents="none"
        />

        {/* Observed pitch markers — numbered dots at each pitch's cell center.
            Currently-selected pitch (highlightedPitchIndex) is larger and
            accent-colored so the user can see which one they're replacing. */}
        {observedPitches.map((p, i) => {
          const isHighlight = i === highlightedPitchIndex;
          let cx: number;
          let cy: number;
          if (isOOZ(p.zone)) {
            // OOZ pitches: place in the corresponding quadrant. We use the
            // center of each quadrant minus a small inset so the marker doesn't
            // overlap the quadrant's stroke.
            const quad = oozQuadrantOf(p.zone);
            if (quad === null) return null;
            const left = quad === "UL" || quad === "LL";
            const top = quad === "UL" || quad === "UR";
            cx = left ? SZ_X - OOZ_PAD / 2 : SZ_X + SZ_W + OOZ_PAD / 2;
            cy = top ? SZ_Y - OOZ_PAD / 2 : SZ_Y + SZ_H + OOZ_PAD / 2;
          } else {
            const gc = featureZoneToGridCell(p.zone);
            if (!gc) return null;
            cx = SZ_X + gc.col * cellW + cellW / 2;
            cy = SZ_Y + gc.row * cellH + cellH / 2;
          }
          return (
            <g key={i}>
              <circle
                cx={cx}
                cy={cy}
                r={isHighlight ? 13 : 9}
                fill={isHighlight ? "#fde68a" : "white"}
                stroke={isHighlight ? "#b45309" : "#1f2937"}
                strokeWidth={isHighlight ? 2.5 : 1.5}
                className="transition-state"
              />
              <text
                x={cx}
                y={cy + (isHighlight ? 4.5 : 3.5)}
                textAnchor="middle"
                fontSize={isHighlight ? 11 : 9}
                fontWeight={isHighlight ? 600 : 500}
                fill="#1f2937"
              >
                {i + 1}
              </text>
            </g>
          );
        })}

        {/* Plate (home-plate pentagon) — minimal, at the bottom */}
        <polygon
          points={`${SZ_X + SZ_W / 2 - 35},${SZ_Y + SZ_H + 35}
                   ${SZ_X + SZ_W / 2 + 35},${SZ_Y + SZ_H + 35}
                   ${SZ_X + SZ_W / 2 + 30},${SZ_Y + SZ_H + 60}
                   ${SZ_X + SZ_W / 2},${SZ_Y + SZ_H + 70}
                   ${SZ_X + SZ_W / 2 - 30},${SZ_Y + SZ_H + 60}`}
          fill="#f3f4f6"
          stroke="#9ca3af"
          strokeWidth={1}
          pointerEvents="none"
        />
      </svg>

      <div className="mt-4 flex items-center gap-3 text-small text-gray-500">
        <button
          onClick={() => !disabled && onSelect(null)}
          disabled={disabled}
          className={`px-3 py-1.5 rounded-md border-hairline transition-state ${
            selected === null
              ? "bg-gray-900 text-white border-gray-900"
              : "bg-white hover:border-gray-300"
          }`}
        >
          Let model pick the zone
        </button>
        {selected !== null && (
          <span>
            Selected:{" "}
            <span className="text-gray-900 font-medium">
              {isOOZ(selected)
                ? `OOZ ${oozQuadrantOf(selected) ?? ""}`
                : `cell ${selected}`}
            </span>
          </span>
        )}
      </div>
    </div>
  );
}
